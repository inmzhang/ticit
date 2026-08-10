//! Selected detector postselection: fired detectors mark shots dead
//! without touching the amplitude buffers; dead lanes are physically
//! compacted away lazily — always before an impure instruction (measurements
//! draw randomness and must see only live shots), otherwise only once enough
//! of the batch is dead to be worth the move.

use super::runtime::{
    BatchSignSource, detector_outcome_bits, execute_batch_component_instruction,
    execute_batch_instruction, execute_shot_major_rotation_run, expression_slice_word,
};
use super::symbols::write_batch_detector_record;
use super::{
    BatchFactoredExecutorState, batch_record_offset, batch_shot_mask, batch_shot_word,
    batch_word_count, live_word_mask_for_shots,
};
use crate::active::active_length;
use crate::component_plan::ActiveComponentStepKind;
use crate::errors::{Result, TicitError};
use crate::factored::{FactoredInstruction, FactoredInstructionProgram, RecordDetector};
use crate::presampled_expression::{PresampledExpressionBlock, PresampledExpressionPlan};

const EXPENSIVE_PURE_COMPACTION_DENOMINATOR: usize = 64;

/// Reusable postselection workspace.
///
/// The C++ carries a fourth word buffer (`scratch`) that nothing reads; it is
/// deliberately not ported.
#[derive(Debug, Default)]
pub struct BatchDetectorPostselectionScratch {
    pub dead_bits: Vec<u64>,
    pub keep_bits: Vec<u64>,
    pub compact_scratch: Vec<u64>,
    /// Materialized, compactable copy of the expression block:
    /// `[block_expression * batch_words + word]`.
    pub expression_words: Vec<u64>,
    pub live_sources: Vec<usize>,
    pub condition_last_use_by_index: Vec<i32>,
    pub record_last_use_by_index: Vec<i32>,
    /// Cache keys for the two last-use tables; recomputed when either the
    /// program or the retained-record set changes.
    metadata_program_key: usize,
    metadata_retained_key: u64,
    metadata_valid: bool,
    pub dead_count: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchDetectorPostselectionResult {
    pub discarded: usize,
    pub accepted: usize,
}

#[derive(Clone, Debug)]
pub struct BatchDetectorPostselectionOptions<'a> {
    /// Compaction fires once at least `1/denominator` of shots are dead.
    pub mask_dead_shots_min_fraction_denominator: usize,
    /// Record groups (the logical observable's) pinned live to program end so
    /// the final compaction preserves them for accumulation.
    pub retained_record_uses: Option<&'a [Vec<i32>]>,
    /// Noiseless detector bits XORed into postselection outcomes.
    pub expected_detectors: &'a [u8],
}

impl Default for BatchDetectorPostselectionOptions<'_> {
    fn default() -> Self {
        Self {
            mask_dead_shots_min_fraction_denominator: 2,
            retained_record_uses: None,
            expected_detectors: &[],
        }
    }
}

// ==============================================================================
// Last-use tables
// ==============================================================================

fn mark_condition_uses(
    last_use: &mut [i32],
    plan: &crate::symbolic::SymbolicBoolEvaluationPlan,
    instruction_index: i32,
) {
    for &condition in &plan.conditions {
        if condition > 0 && (condition as usize) < last_use.len() {
            let slot = &mut last_use[condition as usize];
            *slot = (*slot).max(instruction_index);
        }
    }
}

fn condition_last_uses(program: &FactoredInstructionProgram) -> Vec<i32> {
    let mut last_use = vec![-1i32; program.nsymbols + 1];
    for (idx, instruction) in program.instructions.iter().enumerate() {
        let idx = idx as i32;
        match instruction {
            FactoredInstruction::ApplyPrecomputedActivePauliRotation(inst) => {
                mark_condition_uses(&mut last_use, &inst.sign_plan, idx);
            }
            FactoredInstruction::PromoteDormantRotation(inst) => {
                mark_condition_uses(&mut last_use, &inst.sign_plan, idx);
            }
            FactoredInstruction::RecordDetector(inst) => {
                if inst.records.is_empty() && !inst.outcome_plan.conditions.is_empty() {
                    mark_condition_uses(&mut last_use, &inst.outcome_plan, idx);
                }
            }
            FactoredInstruction::RecordMeasurement(inst) => {
                mark_condition_uses(&mut last_use, &inst.outcome_plan, idx);
            }
            FactoredInstruction::MeasurePrecomputedActivePauli(inst) => {
                mark_condition_uses(&mut last_use, &inst.outcome_plan, idx);
            }
            FactoredInstruction::IntroduceDormantMeasurementBranch(inst) => {
                mark_condition_uses(&mut last_use, &inst.outcome_plan, idx);
            }
        }
    }
    last_use
}

fn measurement_record_last_uses(
    program: &FactoredInstructionProgram,
    retained_record_uses: &[Vec<i32>],
    final_instruction_index: i32,
) -> Result<Vec<i32>> {
    let nrecords = program.nrecords;
    let mut last_use = vec![-1i32; nrecords + 1];
    for (idx, instruction) in program.instructions.iter().enumerate() {
        let FactoredInstruction::RecordDetector(detector) = instruction else {
            continue;
        };
        for &record in &detector.records {
            if record <= 0 || record as usize > nrecords {
                return Err(TicitError::new(
                    "detector references an out-of-range measurement record",
                ));
            }
            let slot = &mut last_use[record as usize];
            *slot = (*slot).max(idx as i32);
        }
    }
    for records in retained_record_uses {
        for &record in records {
            if record <= 0 || record as usize > nrecords {
                return Err(TicitError::new(
                    "retained record use references an out-of-range measurement record",
                ));
            }
            let slot = &mut last_use[record as usize];
            *slot = (*slot).max(final_instruction_index);
        }
    }
    Ok(last_use)
}

fn retained_key(retained: Option<&[Vec<i32>]>) -> u64 {
    // Cheap content fingerprint standing in for the C++ pointer-identity key;
    // logical-record sets are tiny.
    let mut hash = 0xcbf29ce484222325u64;
    if let Some(groups) = retained {
        for group in groups {
            hash = hash.wrapping_mul(0x100000001b3) ^ group.len() as u64;
            for &record in group {
                hash = hash.wrapping_mul(0x100000001b3) ^ record as u64;
            }
        }
    }
    hash
}

// ==============================================================================
// Purity tables
// ==============================================================================

fn instruction_is_pure_over_dead(instruction: &FactoredInstruction) -> bool {
    !matches!(
        instruction,
        FactoredInstruction::MeasurePrecomputedActivePauli(_)
            | FactoredInstruction::IntroduceDormantMeasurementBranch(_)
    )
}

fn instruction_is_expensive_over_dead(instruction: &FactoredInstruction) -> bool {
    !matches!(
        instruction,
        FactoredInstruction::RecordMeasurement(_) | FactoredInstruction::RecordDetector(_)
    )
}

fn should_compact_dead_before_instruction(
    runtime: &BatchFactoredExecutorState,
    scratch: &BatchDetectorPostselectionScratch,
    options: &BatchDetectorPostselectionOptions<'_>,
    instruction: &FactoredInstruction,
) -> bool {
    if runtime.active_shots == 0 || scratch.dead_count == 0 {
        return false;
    }
    if !instruction_is_pure_over_dead(instruction) {
        return true;
    }
    // Shot-major dense skips dead lanes per shot instead of compacting.
    if runtime.dense_shot_major_active && !runtime.active_components_enabled {
        return false;
    }
    let mut denominator = options.mask_dead_shots_min_fraction_denominator.max(1);
    if instruction_is_expensive_over_dead(instruction) {
        denominator = denominator.max(EXPENSIVE_PURE_COMPACTION_DENOMINATOR);
    }
    scratch.dead_count * denominator >= runtime.active_shots
}

// ==============================================================================
// Bit compression
// ==============================================================================

fn compress_bits_portable(bits: u64, mut keep_mask: u64) -> u64 {
    let mut out = 0u64;
    let mut dest = 1u64;
    while keep_mask != 0 {
        let bit = keep_mask & keep_mask.wrapping_neg();
        if bits & bit != 0 {
            out |= dest;
        }
        keep_mask &= keep_mask - 1;
        dest <<= 1;
    }
    out
}

/// Executes one BMI2 `PEXT`; the caller must prove the current CPU has BMI2.
///
/// This function has no pointer or memory operands, so the target-feature
/// precondition is its only safety invariant.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[allow(unsafe_code)]
unsafe fn compress_bits_bmi2(bits: u64, keep_mask: u64) -> u64 {
    core::arch::x86_64::_pext_u64(bits, keep_mask)
}

#[inline]
#[allow(unsafe_code)]
fn compress_bits(bits: u64, keep_mask: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("bmi2") {
        // SAFETY: the branch immediately above proves the sole target-feature
        // precondition; the intrinsic operates only on its two integer values.
        return unsafe { compress_bits_bmi2(bits, keep_mask) };
    }
    compress_bits_portable(bits, keep_mask)
}

fn append_compressed_bits(out: &mut [u64], dest_bit: &mut usize, compressed: u64, nbits: usize) {
    if nbits == 0 {
        return;
    }
    let word = *dest_bit >> 6;
    let shift = *dest_bit & 63;
    out[word] |= compressed << shift;
    if shift != 0 && nbits > 64 - shift {
        out[word + 1] |= compressed >> (64 - shift);
    }
    *dest_bit += nbits;
}

fn count_live_bits(bits: &[u64], shots: usize) -> usize {
    let nwords = batch_word_count(shots);
    let mut count = 0usize;
    for word in 0..nwords {
        count += (bits[word] & live_word_mask_for_shots(shots, word)).count_ones() as usize;
    }
    count
}

fn collect_live_sources(
    live_sources: &mut Vec<usize>,
    old_shots: usize,
    survivor_count: usize,
    keep_bits: &[u64],
) -> Result<()> {
    live_sources.clear();
    live_sources.reserve(survivor_count);
    let nwords = batch_word_count(old_shots);
    for word in 0..nwords {
        let mut mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            live_sources.push((word << 6) + bit);
            mask &= mask - 1;
        }
    }
    if live_sources.len() != survivor_count {
        return Err(TicitError::internal(
            "internal live source compaction count mismatch",
        ));
    }
    Ok(())
}

/// Compresses one family of bit-plane columns down to the surviving shots.
fn compact_bit_columns(
    columns: &mut [u64],
    live_column_bases: impl Iterator<Item = usize>,
    stride_words: usize,
    old_shots: usize,
    survivor_count: usize,
    keep_bits: &[u64],
    scratch: &mut Vec<u64>,
) -> Result<()> {
    if stride_words == 0 || old_shots == 0 {
        return Ok(());
    }
    if old_shots <= 64 {
        let keep_mask = if keep_bits.is_empty() {
            0
        } else {
            keep_bits[0] & live_word_mask_for_shots(old_shots, 0)
        };
        for base in live_column_bases {
            columns[base] = compress_bits(columns[base], keep_mask);
            columns[base + 1..base + stride_words].fill(0);
        }
        return Ok(());
    }
    if scratch.len() < stride_words {
        scratch.resize(stride_words, 0);
    }
    let nwords = batch_word_count(old_shots);
    for base in live_column_bases {
        scratch[..stride_words].fill(0);
        let mut dest_bit = 0usize;
        for word in 0..nwords {
            let keep_mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            if keep_mask == 0 {
                continue;
            }
            let kept = keep_mask.count_ones() as usize;
            append_compressed_bits(
                scratch,
                &mut dest_bit,
                compress_bits(columns[base + word], keep_mask),
                kept,
            );
        }
        if dest_bit != survivor_count {
            return Err(TicitError::internal(
                "internal bit-plane compaction count mismatch",
            ));
        }
        columns[base..base + stride_words].copy_from_slice(&scratch[..stride_words]);
    }
    Ok(())
}

fn compact_active_columns(
    runtime: &mut BatchFactoredExecutorState,
    survivor_count: usize,
    live_sources: &[usize],
) -> Result<()> {
    let mut first_moved = 0usize;
    while first_moved < survivor_count && live_sources[first_moved] == first_moved {
        first_moved += 1;
    }
    if first_moved == survivor_count {
        return Ok(());
    }

    let pitch = runtime.active_pitch;
    let shot_major = runtime.dense_shot_major_active;
    if runtime.active_components_enabled {
        for component in &mut runtime.active_components {
            if !component.active {
                continue;
            }
            let dim = active_length(component.k)?;
            if shot_major {
                for dst in first_moved..survivor_count {
                    let src = live_sources[dst];
                    let src_base = src * component.stride;
                    let dst_base = dst * component.stride;
                    component.re.copy_within(src_base..src_base + dim, dst_base);
                    component.im.copy_within(src_base..src_base + dim, dst_base);
                }
            } else {
                for basis in 0..dim {
                    let row = basis * pitch;
                    for dst in first_moved..survivor_count {
                        let src = live_sources[dst];
                        component.re[row + dst] = component.re[row + src];
                        component.im[row + dst] = component.im[row + src];
                    }
                }
            }
        }
        return Ok(());
    }

    let dim = active_length(runtime.k)?;
    if shot_major {
        let stride = runtime.active_stride;
        for dst in first_moved..survivor_count {
            let src = live_sources[dst];
            let src_base = src * stride;
            let dst_base = dst * stride;
            // Disjoint because dim <= stride and src != dst rows.
            runtime
                .active_re
                .copy_within(src_base..src_base + dim, dst_base);
            runtime
                .active_im
                .copy_within(src_base..src_base + dim, dst_base);
        }
        return Ok(());
    }

    for basis in 0..dim {
        let row = basis * pitch;
        for dst in first_moved..survivor_count {
            let src = live_sources[dst];
            runtime.active_re[row + dst] = runtime.active_re[row + src];
            runtime.active_im[row + dst] = runtime.active_im[row + src];
        }
    }
    Ok(())
}

fn compact_dense_columns<T: Copy>(
    columns: &mut [T],
    column_count: usize,
    stride: usize,
    survivor_count: usize,
    live_sources: &[usize],
) {
    for column in 0..column_count {
        let base = column * stride;
        for destination in 0..survivor_count {
            columns[base + destination] = columns[base + live_sources[destination]];
        }
    }
}

struct CompactionTables<'a> {
    condition_last_use: &'a [i32],
    record_last_use: &'a [i32],
    expression_last_use: Option<&'a [i32]>,
}

#[allow(clippy::too_many_arguments)]
fn compact_surviving_shots(
    runtime: &mut BatchFactoredExecutorState,
    keep_bits: &[u64],
    tables: &CompactionTables<'_>,
    instruction_index: i32,
    include_current_record_use: bool,
    scratch: &mut Vec<u64>,
    live_sources: &mut Vec<usize>,
    expression_words: Option<&mut Vec<u64>>,
) -> Result<()> {
    let old_shots = runtime.active_shots;
    let survivor_count = count_live_bits(keep_bits, old_shots);
    if survivor_count == old_shots {
        return Ok(());
    }
    if survivor_count == 0 {
        runtime.active_shots = 0;
        return Ok(());
    }
    collect_live_sources(live_sources, old_shots, survivor_count, keep_bits)?;
    compact_active_columns(runtime, survivor_count, live_sources)?;
    compact_dense_columns(
        &mut runtime.exp_values,
        runtime.nexpvals,
        runtime.batches,
        survivor_count,
        live_sources,
    );
    let stride_words = runtime.batch_words;
    if let Some(expression_words) = expression_words {
        let expression_last_use = tables.expression_last_use.ok_or_else(|| {
            TicitError::internal("internal expression compaction last-use table is missing")
        })?;
        let bases = expression_last_use
            .iter()
            .enumerate()
            .filter(|&(_, &last_use)| last_use > instruction_index)
            .map(|(expression, _)| expression * stride_words);
        compact_bit_columns(
            expression_words,
            bases,
            stride_words,
            old_shots,
            survivor_count,
            keep_bits,
            scratch,
        )?;
    }
    // Symbols: only assigned conditions still used after this instruction.
    let assigned = runtime.assigned_words.clone();
    let symbol_bases = {
        let mut bases = Vec::new();
        for (word, &word_bits) in assigned.iter().enumerate() {
            let mut bits = word_bits;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let condition = (word << 6) + bit + 1;
                if condition < tables.condition_last_use.len()
                    && tables.condition_last_use[condition] > instruction_index
                {
                    bases.push((condition - 1) * stride_words);
                }
                bits &= bits - 1;
            }
        }
        bases
    };
    compact_bit_columns(
        &mut runtime.value_words,
        symbol_bases.into_iter(),
        stride_words,
        old_shots,
        survivor_count,
        keep_bits,
        scratch,
    )?;
    let record_bases = (1..tables.record_last_use.len())
        .filter(|&record| {
            let last_use = tables.record_last_use[record];
            if include_current_record_use {
                last_use >= instruction_index
            } else {
                last_use > instruction_index
            }
        })
        .map(|record| (record - 1) * stride_words);
    compact_bit_columns(
        &mut runtime.measurement_words,
        record_bases,
        stride_words,
        old_shots,
        survivor_count,
        keep_bits,
        scratch,
    )?;
    if runtime.store_detector_records {
        let detector_bases = (0..runtime.ndetectors).map(|detector| detector * stride_words);
        compact_bit_columns(
            &mut runtime.detector_words,
            detector_bases,
            stride_words,
            old_shots,
            survivor_count,
            keep_bits,
            scratch,
        )?;
    }
    runtime.active_shots = survivor_count;
    Ok(())
}

fn compact_dead_shots_if_needed(
    runtime: &mut BatchFactoredExecutorState,
    scratch: &mut BatchDetectorPostselectionScratch,
    instruction_index: i32,
    include_current_record_use: bool,
    expression_last_use: Option<&[i32]>,
    compact_expressions: bool,
) -> Result<()> {
    if runtime.active_shots == 0 || scratch.dead_count == 0 {
        return Ok(());
    }
    let nwords = batch_word_count(runtime.active_shots);
    for word in 0..nwords {
        scratch.keep_bits[word] =
            !scratch.dead_bits[word] & live_word_mask_for_shots(runtime.active_shots, word);
    }
    scratch.keep_bits[nwords..].fill(0);
    // Split-borrow dance: lift the buffers out of the scratch so the runtime
    // and the scratch tables can be borrowed independently.
    let keep_bits = std::mem::take(&mut scratch.keep_bits);
    let mut compact_scratch = std::mem::take(&mut scratch.compact_scratch);
    let mut live_sources = std::mem::take(&mut scratch.live_sources);
    let mut expression_words = std::mem::take(&mut scratch.expression_words);
    let tables = CompactionTables {
        condition_last_use: &scratch.condition_last_use_by_index,
        record_last_use: &scratch.record_last_use_by_index,
        expression_last_use,
    };
    let result = compact_surviving_shots(
        runtime,
        &keep_bits,
        &tables,
        instruction_index,
        include_current_record_use,
        &mut compact_scratch,
        &mut live_sources,
        compact_expressions.then_some(&mut expression_words),
    );
    scratch.keep_bits = keep_bits;
    scratch.compact_scratch = compact_scratch;
    scratch.live_sources = live_sources;
    scratch.expression_words = expression_words;
    result?;
    scratch.dead_bits.fill(0);
    scratch.dead_count = 0;
    Ok(())
}

// ==============================================================================
// Dead marking
// ==============================================================================

fn mark_dead_words(
    runtime: &BatchFactoredExecutorState,
    scratch: &mut BatchDetectorPostselectionScratch,
    fired_at: impl Fn(usize, u64) -> u64,
) -> usize {
    let nwords = batch_word_count(runtime.active_shots);
    let mut discarded_now = 0usize;
    for word in 0..nwords {
        let live = live_word_mask_for_shots(runtime.active_shots, word);
        let fired = fired_at(word, live);
        let newly_dead = fired & !scratch.dead_bits[word];
        scratch.dead_bits[word] |= fired;
        discarded_now += newly_dead.count_ones() as usize;
    }
    scratch.dead_count += discarded_now;
    discarded_now
}

fn mark_dead_from_detector_bits(
    runtime: &BatchFactoredExecutorState,
    detector_bits: &[u64],
    scratch: &mut BatchDetectorPostselectionScratch,
    expected: bool,
) -> usize {
    if runtime.active_shots == 0 {
        return 0;
    }
    mark_dead_words(runtime, scratch, |word, live| {
        if expected {
            !detector_bits[word] & live
        } else {
            detector_bits[word] & live
        }
    })
}

fn mark_dead_from_constant_detector(
    runtime: &BatchFactoredExecutorState,
    raw_fired: bool,
    scratch: &mut BatchDetectorPostselectionScratch,
    expected: bool,
) -> usize {
    let fired = raw_fired != expected;
    if !fired || runtime.active_shots == 0 {
        return 0;
    }
    mark_dead_words(runtime, scratch, |_, live| live)
}

fn mark_dead_from_detector_records(
    runtime: &BatchFactoredExecutorState,
    instruction: &RecordDetector,
    scratch: &mut BatchDetectorPostselectionScratch,
    expected: bool,
) -> Result<usize> {
    if runtime.active_shots == 0 || instruction.records.is_empty() {
        return Ok(0);
    }
    for &record in &instruction.records {
        if record <= 0 || record as usize > runtime.nrecords {
            return Err(TicitError::new(
                "detector references an out-of-range measurement record",
            ));
        }
    }
    if instruction.records.len() == 1 {
        let base = batch_record_offset(runtime, instruction.records[0], 0);
        return Ok(mark_dead_words(runtime, scratch, |word, live| {
            let raw = runtime.measurement_words[base + word];
            if expected { !raw & live } else { raw & live }
        }));
    }
    Ok(mark_dead_words(runtime, scratch, |word, live| {
        let mut fired = 0u64;
        for &record in &instruction.records {
            fired ^= runtime.measurement_words[batch_record_offset(runtime, record, word)];
        }
        if expected {
            !fired & live
        } else {
            fired & live
        }
    }))
}

// ==============================================================================
// Postselected shot-skipping kernels (shot-major dense)
// ==============================================================================

fn postselected_shot_is_dead(scratch: &BatchDetectorPostselectionScratch, shot: usize) -> bool {
    scratch.dead_count != 0
        && (scratch.dead_bits[batch_shot_word(shot)] & batch_shot_mask(shot)) != 0
}

fn rotate_shot_major_postselected(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &crate::active::PrecomputedActivePauliRotationKernel,
    sign_bits: &[u64],
    scratch: &BatchDetectorPostselectionScratch,
) -> Result<()> {
    if !runtime.dense_shot_major_active || scratch.dead_count == 0 {
        return super::active::rotate_pauli_batch(runtime, kernel, sign_bits);
    }
    if kernel.action.nqubits != runtime.k {
        return Err(TicitError::new(
            "rotation kernel dimension does not match batch active state",
        ));
    }
    let dim = active_length(runtime.k)?;
    for shot in 0..runtime.active_shots {
        if postselected_shot_is_dead(scratch, shot) {
            continue;
        }
        let base = shot * runtime.active_stride;
        let range = base..base + runtime.active_stride;
        let (re, im) = (
            &mut runtime.active_re[range.clone()],
            &mut runtime.active_im[range],
        );
        crate::contiguous::rotate_contiguous_active(
            re,
            im,
            dim,
            kernel,
            super::active::batch_bit_at(sign_bits, shot),
        );
    }
    Ok(())
}

fn promote_shot_major_postselected(
    runtime: &mut BatchFactoredExecutorState,
    kernel_angle: f64,
    sign_bits: &[u64],
    scratch: &BatchDetectorPostselectionScratch,
) -> Result<()> {
    if !runtime.dense_shot_major_active || scratch.dead_count == 0 {
        return super::active::promote_first_dormant_rotation_batch(
            runtime,
            kernel_angle,
            sign_bits,
        );
    }
    if runtime.ndormant == 0 {
        return Err(TicitError::new(
            "cannot promote a dormant qubit when none remain",
        ));
    }
    let dim = active_length(runtime.k)?;
    let promoted_dim = 2 * dim;
    if runtime.active_stride < promoted_dim {
        return Err(TicitError::new(
            "batch active shot-major stride is too short for dormant promotion",
        ));
    }
    let c = kernel_angle.cos();
    let s = kernel_angle.sin();
    for shot in 0..runtime.active_shots {
        if postselected_shot_is_dead(scratch, shot) {
            continue;
        }
        let q = if super::active::batch_bit_at(sign_bits, shot) {
            s
        } else {
            -s
        };
        let base = shot * runtime.active_stride;
        let range = base..base + runtime.active_stride;
        let (re, im) = (
            &mut runtime.active_re[range.clone()],
            &mut runtime.active_im[range],
        );
        crate::contiguous::promote_contiguous_active(re, im, dim, c, q);
    }
    runtime.k += 1;
    runtime.ndormant -= 1;
    Ok(())
}

// ==============================================================================
// Preparation and the main loop
// ==============================================================================

pub fn prepare_batch_detector_postselection_scratch(
    scratch: &mut BatchDetectorPostselectionScratch,
    runtime: &BatchFactoredExecutorState,
) {
    let nwords = runtime.batch_words;
    if scratch.dead_bits.len() < nwords {
        scratch.dead_bits.resize(nwords, 0);
    }
    if scratch.keep_bits.len() < nwords {
        scratch.keep_bits.resize(nwords, 0);
    }
    if scratch.compact_scratch.len() < nwords {
        scratch.compact_scratch.resize(nwords, 0);
    }
    if scratch.live_sources.capacity() < runtime.batches {
        let additional = runtime.batches - scratch.live_sources.len();
        scratch.live_sources.reserve(additional);
    }
}

pub fn prepare_batch_detector_postselection_scratch_for_program(
    scratch: &mut BatchDetectorPostselectionScratch,
    runtime: &BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
    options: &BatchDetectorPostselectionOptions<'_>,
) -> Result<()> {
    prepare_batch_detector_postselection_scratch(scratch, runtime);
    let program_key = std::ptr::from_ref(program) as usize;
    let retained_key = retained_key(options.retained_record_uses);
    if scratch.metadata_valid
        && scratch.metadata_program_key == program_key
        && scratch.metadata_retained_key == retained_key
    {
        return Ok(());
    }
    scratch.condition_last_use_by_index = condition_last_uses(program);
    scratch.record_last_use_by_index = measurement_record_last_uses(
        program,
        options.retained_record_uses.unwrap_or(&[]),
        program.instructions.len() as i32,
    )?;
    scratch.metadata_program_key = program_key;
    scratch.metadata_retained_key = retained_key;
    scratch.metadata_valid = true;
    Ok(())
}

/// Runs the program with selected detectors marking and compacting
/// dead shots. `accepted` is the surviving live-shot count.
pub fn execute_batch_postselected_in_place(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
    expression_plan: &PresampledExpressionPlan,
    expression_block: &PresampledExpressionBlock,
    first_sample_shot: usize,
    scratch: &mut BatchDetectorPostselectionScratch,
    options: &BatchDetectorPostselectionOptions<'_>,
) -> Result<BatchDetectorPostselectionResult> {
    if runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n {
        return Err(TicitError::new(
            "batch executor state does not match program",
        ));
    }
    if expression_plan.instruction_expressions.len() != program.instructions.len() {
        return Err(TicitError::new(
            "batch presampled expression plan does not match program",
        ));
    }
    if expression_plan.block_expression_last_use_by_index.len()
        != expression_plan.block_expressions.len()
    {
        return Err(TicitError::new(
            "batch presampled expression last-use table does not match expression plan",
        ));
    }
    prepare_batch_detector_postselection_scratch_for_program(scratch, runtime, program, options)?;
    scratch.dead_bits.fill(0);
    scratch.dead_count = 0;
    if runtime.active_shots == 0 {
        return Ok(BatchDetectorPostselectionResult::default());
    }

    let mut discarded = 0usize;
    let mut workspace_materialized = false;
    let mut work = std::mem::take(&mut runtime.eval_scratch);
    let result = (|| -> Result<()> {
        let mut idx = 0usize;
        while idx < program.instructions.len() {
            if runtime.active_shots == 0 {
                break;
            }
            if should_compact_dead_before_instruction(
                runtime,
                scratch,
                options,
                &program.instructions[idx],
            ) {
                if !workspace_materialized {
                    materialize_expression_workspace(
                        scratch,
                        expression_plan,
                        expression_block,
                        runtime.batch_words,
                        runtime.active_shots,
                        first_sample_shot,
                    )?;
                    workspace_materialized = true;
                }
                compact_dead_shots_if_needed(
                    runtime,
                    scratch,
                    idx as i32 - 1,
                    false,
                    Some(&expression_plan.block_expression_last_use_by_index),
                    true,
                )?;
                if runtime.active_shots == 0 {
                    break;
                }
            }
            // Borrow the right source for this iteration: the materialized
            // workspace once compaction has rebased shot indices, else the
            // immutable chunk block.
            let expression_words_view;
            let source = if workspace_materialized {
                expression_words_view = std::mem::take(&mut scratch.expression_words);
                BatchSignSource::Workspace {
                    plan: expression_plan,
                    words: &expression_words_view,
                    stride_words: runtime.batch_words,
                }
            } else {
                expression_words_view = Vec::new();
                BatchSignSource::Block {
                    plan: expression_plan,
                    block: expression_block,
                    first_sample_shot,
                }
            };
            let step = execute_postselected_step(
                runtime, program, &source, idx, scratch, options, &mut work,
            );
            if workspace_materialized {
                scratch.expression_words = expression_words_view;
            }
            let (consumed, discarded_now) = step?;
            discarded += discarded_now;
            if scratch.dead_count >= runtime.active_shots {
                runtime.active_shots = 0;
                break;
            }
            idx += consumed;
        }
        Ok(())
    })();
    runtime.eval_scratch = work;
    result?;
    compact_dead_shots_if_needed(
        runtime,
        scratch,
        program.instructions.len() as i32,
        true,
        workspace_materialized.then_some(&expression_plan.block_expression_last_use_by_index),
        workspace_materialized,
    )?;
    Ok(BatchDetectorPostselectionResult {
        discarded,
        accepted: runtime.active_shots,
    })
}

/// One postselected step; returns (instructions consumed, shots discarded).
fn execute_postselected_step(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
    source: &BatchSignSource<'_>,
    idx: usize,
    scratch: &mut BatchDetectorPostselectionScratch,
    options: &BatchDetectorPostselectionOptions<'_>,
    work: &mut Vec<u64>,
) -> Result<(usize, usize)> {
    if runtime.dense_shot_major_active && !runtime.active_components_enabled {
        let consumed = execute_shot_major_rotation_run(
            runtime,
            program,
            source,
            idx,
            (scratch.dead_count != 0).then_some(&scratch.dead_bits),
            work,
        )?;
        if consumed != 0 {
            return Ok((consumed, 0));
        }
    }
    let instruction = &program.instructions[idx];
    if let FactoredInstruction::RecordDetector(detector) = instruction
        && detector.postselect
    {
        let component_handled = runtime.active_components_enabled
            && program
                .active_component_plan
                .as_deref()
                .expect("component execution requires a plan")
                .instruction_steps[idx]
                .kind
                != ActiveComponentStepKind::None;
        if !component_handled {
            let expected = options
                .expected_detectors
                .get((detector.detector - 1) as usize)
                .is_some_and(|&bit| bit != 0);
            let discarded_now = if runtime.store_detector_records {
                detector_outcome_bits(runtime, detector, source, idx, work)?;
                write_batch_detector_record(runtime, detector.detector, work)?;
                mark_dead_from_detector_bits(runtime, work, scratch, expected)
            } else if !detector.records.is_empty() {
                mark_dead_from_detector_records(runtime, detector, scratch, expected)?
            } else if detector.outcome.conditions.is_empty() {
                mark_dead_from_constant_detector(
                    runtime,
                    detector.outcome.constant,
                    scratch,
                    expected,
                )
            } else {
                source.eval(idx, runtime, work)?;
                mark_dead_from_detector_bits(runtime, work, scratch, expected)
            };
            return Ok((1, discarded_now));
        }
    }
    match instruction {
        FactoredInstruction::ApplyPrecomputedActivePauliRotation(inst)
            if !runtime.active_components_enabled =>
        {
            source.eval(idx, runtime, work)?;
            rotate_shot_major_postselected(runtime, &inst.rotation_kernel, work, scratch)?;
            Ok((1, 0))
        }
        FactoredInstruction::PromoteDormantRotation(inst) if !runtime.active_components_enabled => {
            source.eval(idx, runtime, work)?;
            promote_shot_major_postselected(runtime, inst.kernel_angle, work, scratch)?;
            Ok((1, 0))
        }
        _ => {
            // Component rotations run on dead lanes here (pure, so harmless);
            // the C++ additionally dead-skips them inside the component scope.
            if runtime.active_components_enabled {
                execute_batch_component_instruction(runtime, program, source, idx, work)?;
            } else {
                execute_batch_instruction(runtime, instruction, source, idx, work)?;
            }
            Ok((1, 0))
        }
    }
}

fn materialize_expression_workspace(
    scratch: &mut BatchDetectorPostselectionScratch,
    expression_plan: &PresampledExpressionPlan,
    expression_block: &PresampledExpressionBlock,
    stride_words: usize,
    active_shots: usize,
    first_sample_shot: usize,
) -> Result<()> {
    if expression_block.nshots < first_sample_shot
        || active_shots > expression_block.nshots - first_sample_shot
    {
        return Err(TicitError::new(
            "batch presampled expression source range is out of bounds",
        ));
    }
    let expression_count = expression_plan.block_expressions.len();
    let total_words = expression_count * stride_words;
    if scratch.expression_words.len() < total_words {
        scratch.expression_words.resize(total_words, 0);
    }
    for expression in 0..expression_count {
        let base = expression * stride_words;
        for word in 0..stride_words {
            scratch.expression_words[base + word] = expression_slice_word(
                expression_block,
                expression,
                first_sample_shot,
                active_shots,
                word,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compress_bits_reference(bits: u64, keep_mask: u64) -> u64 {
        let mut out = 0;
        let mut dest = 0;
        for source in 0..64 {
            if keep_mask >> source & 1 != 0 {
                out |= (bits >> source & 1) << dest;
                dest += 1;
            }
        }
        out
    }

    #[test]
    fn dispatched_bit_compression_matches_reference() {
        let mut bits = 0x0123_4567_89ab_cdefu64;
        let mut mask = 0xf0f0_00ff_aaaa_5555u64;
        for _ in 0..4096 {
            let expected = compress_bits_reference(bits, mask);
            assert_eq!(compress_bits_portable(bits, mask), expected);
            assert_eq!(compress_bits(bits, mask), expected);
            bits = bits.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17);
            mask = mask.wrapping_add(0xd1b5_4a32_d192_ed03).rotate_right(11);
        }
        assert_eq!(compress_bits(u64::MAX, 0), 0);
        assert_eq!(compress_bits(bits, u64::MAX), bits);
    }
}
