//! Batch executor: runs a planned [`FactoredInstructionProgram`] over many
//! shots at once, with symbols, records and detectors stored as bit-planes of
//! 64 shots per word.
//!
//! Two amplitude layouts share one state struct, selected by
//! `dense_shot_major_active`: shot-major (`re[shot * active_stride + basis]`,
//! each shot a contiguous vector — the production layout) and basis-major
//! (`re[basis * active_pitch + shot]`,
//! vectorizing across shots). `batch_words` counts *capacity* words
//! (`batches`); loops must bound themselves by the *live* word count from
//! `runtime_batch_word_count` — mixing the two reads stale lanes.

// The word-indexed loops throughout this module mirror the C++ form and
// frequently index several parallel bit-planes; iterator rewrites obscure
// that parity without changing codegen.
#![allow(clippy::needless_range_loop)]

mod active;
mod postselect;
mod runtime;
mod symbols;

pub use postselect::{
    BatchDetectorPostselectionOptions, BatchDetectorPostselectionScratch,
    execute_batch_postselected_in_place, prepare_batch_detector_postselection_scratch_for_program,
};
pub use runtime::execute_batch_in_place_expressions;
#[cfg(test)]
pub(crate) use runtime::sample_measurements_batch;

use crate::active::active_length;
use crate::bits::symbol_word_count;
use crate::errors::{Result, TicitError};
use crate::factored::FactoredInstructionProgram;
use crate::random::next_random_u64;

pub(crate) const DEFAULT_BATCH_SHOTS: usize = 2048;
pub(crate) const DEFAULT_BATCH_ACTIVE_AMPLITUDES: usize = 1 << 15;
pub(crate) const XMASK_ROTATION_PAIR_THRESHOLD: usize = 64;
pub(crate) const BATCH_SCALAR_SYMBOLIC_EVAL_THRESHOLD: usize = 32;
pub(crate) const BATCH_ACTIVE_LANE_ALIGNMENT: usize = 4;

/// One tensor factor of the batch active state when component execution is on.
#[derive(Clone, Debug, Default)]
pub struct BatchActiveComponent {
    pub k: usize,
    pub active: bool,
    /// Allocated dim = `2^component_max_k`; doubles as the shot-major stride.
    pub stride: usize,
    pub re: Vec<f64>,
    pub im: Vec<f64>,
    pub scratch_re: Vec<f64>,
    pub scratch_im: Vec<f64>,
}

/// Batched runtime state. Reused across blocks via [`reset_batch_executor`].
#[derive(Clone, Debug, Default)]
pub struct BatchFactoredExecutorState {
    pub n: usize,
    pub k: usize,
    pub ndormant: usize,
    /// Shot capacity, fixed at construction.
    pub batches: usize,
    /// Live shots in the current block; shrinks under postselection.
    pub active_shots: usize,
    /// Padded shot-lane count, derived from `batches` — never shrinks.
    pub active_pitch: usize,
    /// Shot-major row stride (dense: `2^max_k`; component scope: its stride).
    pub active_stride: usize,
    pub nsymbols: usize,
    pub nrecords: usize,
    pub ndetectors: usize,
    pub nexpvals: usize,
    pub max_k: usize,
    /// Capacity words = `ceil(batches / 64)`; the allocation stride of every
    /// bit-plane. Live loops use `runtime_batch_word_count` instead.
    pub batch_words: usize,
    /// When false only `detector_any_words` is maintained.
    pub store_detector_records: bool,
    pub dense_shot_major_active: bool,
    pub active_re: Vec<f64>,
    pub active_im: Vec<f64>,
    pub scratch_re: Vec<f64>,
    pub scratch_im: Vec<f64>,
    /// `[(condition - 1) * batch_words + word]`.
    pub value_words: Vec<u64>,
    /// Symbol-indexed (not shot-indexed): bit `(condition - 1)` set once the
    /// condition has concrete values for the whole batch.
    pub assigned_words: Vec<u64>,
    /// `[(record - 1) * batch_words + word]`.
    pub measurement_words: Vec<u64>,
    /// `[(detector - 1) * batch_words + word]`.
    pub detector_words: Vec<u64>,
    /// OR of every detector column.
    pub detector_any_words: Vec<u64>,
    /// `[exp_val * batches + shot]` — stride is `batches`, not a word count.
    pub exp_values: Vec<f64>,
    /// The shared working buffer for one evaluated expression. Functions take
    /// it out with `mem::take` while they also need `&mut` access to the rest
    /// of the state, and put it back before returning.
    pub eval_scratch: Vec<u64>,
    /// `[run_offset * live_words + word]` for the rotation-run fusion.
    pub rotation_run_sign_words: Vec<u64>,
    /// Per-shot ±coefficient fan-out; only `[0, active_shots)` is written.
    pub shot_coefficient_scalars: Vec<f64>,
    pub branch_prob_true: Vec<f64>,
    /// Tail `[active_shots, active_pitch)` is deliberately kept at 1.0.
    pub branch_invnorms: Vec<f64>,
    pub active_components_enabled: bool,
    pub active_components: Vec<BatchActiveComponent>,
    /// One shared SplitMix64 state for the whole batch.
    pub rng_state: u64,
}

pub fn default_batch_count(max_k: usize) -> Result<usize> {
    let dim = active_length(max_k)?;
    Ok(DEFAULT_BATCH_SHOTS.min((DEFAULT_BATCH_ACTIVE_AMPLITUDES / dim).max(1)))
}

// ==============================================================================
// Word/offset helpers
// ==============================================================================

pub(crate) fn batch_word_count(shots: usize) -> usize {
    shots.div_ceil(64)
}

pub(crate) fn padded_batch_active_pitch(shot_capacity: usize) -> usize {
    if shot_capacity <= 2 {
        return shot_capacity.max(1);
    }
    shot_capacity
        .max(BATCH_ACTIVE_LANE_ALIGNMENT)
        .div_ceil(BATCH_ACTIVE_LANE_ALIGNMENT)
        * BATCH_ACTIVE_LANE_ALIGNMENT
}

pub(crate) fn low_bits_mask(nbits: i64) -> u64 {
    if nbits <= 0 {
        0
    } else if nbits >= 64 {
        u64::MAX
    } else {
        (1u64 << nbits) - 1
    }
}

pub(crate) fn live_word_mask_for_shots(shots: usize, word: usize) -> u64 {
    low_bits_mask(shots as i64 - ((word as i64) << 6))
}

pub(crate) fn batch_live_word_mask(runtime: &BatchFactoredExecutorState, word: usize) -> u64 {
    live_word_mask_for_shots(runtime.active_shots, word)
}

pub(crate) fn runtime_batch_word_count(runtime: &BatchFactoredExecutorState) -> usize {
    batch_word_count(runtime.active_shots)
}

pub(crate) fn batch_shot_word(shot: usize) -> usize {
    shot >> 6
}

pub(crate) fn batch_shot_mask(shot: usize) -> u64 {
    1u64 << (shot & 63)
}

pub(crate) fn batch_condition_offset(
    runtime: &BatchFactoredExecutorState,
    condition: i32,
    word: usize,
) -> usize {
    (condition - 1) as usize * runtime.batch_words + word
}

pub(crate) fn batch_record_offset(
    runtime: &BatchFactoredExecutorState,
    record: i32,
    word: usize,
) -> usize {
    (record - 1) as usize * runtime.batch_words + word
}

pub(crate) fn batch_detector_offset(
    runtime: &BatchFactoredExecutorState,
    detector: i32,
    word: usize,
) -> usize {
    (detector - 1) as usize * runtime.batch_words + word
}

pub(crate) fn fill_batch_bits(
    bits: &mut Vec<u64>,
    runtime: &BatchFactoredExecutorState,
    value: bool,
) {
    let nwords = runtime_batch_word_count(runtime);
    if bits.len() < runtime.batch_words {
        bits.resize(runtime.batch_words, 0);
    }
    let fill = if value { u64::MAX } else { 0 };
    for word in 0..nwords {
        bits[word] = fill & batch_live_word_mask(runtime, word);
    }
    bits[nwords..].fill(0);
}

pub(crate) fn mask_batch_bits(bits: &mut [u64], runtime: &BatchFactoredExecutorState) {
    let nwords = runtime_batch_word_count(runtime);
    if nwords == 0 {
        return;
    }
    let live_bits = runtime.active_shots & 63;
    if live_bits != 0 {
        bits[nwords - 1] &= low_bits_mask(live_bits as i64);
    }
}

pub(crate) fn invert_batch_bits(bits: &mut [u64], runtime: &BatchFactoredExecutorState) {
    let nwords = runtime_batch_word_count(runtime);
    for word in 0..nwords {
        bits[word] = !bits[word] & batch_live_word_mask(runtime, word);
    }
    bits[nwords..].fill(0);
}

/// Fair coins for the whole batch: one raw draw per live word, never per shot.
pub(crate) fn fill_batch_random_half_bits(
    bits: &mut Vec<u64>,
    runtime: &mut BatchFactoredExecutorState,
) {
    let nwords = runtime_batch_word_count(runtime);
    if bits.len() < runtime.batch_words {
        bits.resize(runtime.batch_words, 0);
    }
    if runtime.active_shots & 63 == 0 {
        for word in bits.iter_mut().take(nwords) {
            *word = next_random_u64(&mut runtime.rng_state);
        }
    } else {
        for word in 0..nwords {
            bits[word] =
                next_random_u64(&mut runtime.rng_state) & batch_live_word_mask(runtime, word);
        }
    }
    bits[nwords..].fill(0);
}

// ==============================================================================
// Construction / reset
// ==============================================================================

fn configure_batch_components(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
) -> Result<()> {
    if program.use_active_components {
        let plan_matches = program
            .active_component_plan
            .as_deref()
            .is_some_and(|plan| plan.instruction_steps.len() == program.instructions.len());
        if !plan_matches {
            return Err(TicitError::new(
                "program does not contain an executable active component plan",
            ));
        }
    }
    let enabled = program.use_active_components && program.active_component_plan.is_some();
    runtime.active_components_enabled = enabled;
    if !enabled {
        runtime.active_components.clear();
        return Ok(());
    }
    let plan = program
        .active_component_plan
        .as_deref()
        .expect("checked above");
    if plan.component_count != plan.component_max_k.len() {
        return Err(TicitError::new(
            "active component plan has inconsistent component capacities",
        ));
    }
    runtime
        .active_components
        .resize_with(plan.component_count, BatchActiveComponent::default);
    let pitch = runtime.active_pitch;
    for component in 0..plan.component_count {
        let buffer = &mut runtime.active_components[component];
        let capacity = active_length(plan.component_max_k[component])?;
        buffer.stride = capacity;
        let storage = capacity * pitch;
        buffer.re.resize(storage, 0.0);
        buffer.im.resize(storage, 0.0);
        buffer.scratch_re.resize(storage, 0.0);
        buffer.scratch_im.resize(storage, 0.0);
    }
    // The dense buffers are dead weight in component mode; free them.
    runtime.active_re = Vec::new();
    runtime.active_im = Vec::new();
    runtime.scratch_re = Vec::new();
    runtime.scratch_im = Vec::new();
    Ok(())
}

fn reset_batch_components(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
) -> Result<()> {
    if program.use_active_components {
        let plan_matches = program
            .active_component_plan
            .as_deref()
            .is_some_and(|plan| plan.instruction_steps.len() == program.instructions.len());
        if !plan_matches {
            return Err(TicitError::new(
                "program does not contain an executable active component plan",
            ));
        }
    }
    let should_enable = program.use_active_components && program.active_component_plan.is_some();
    let component_count = program
        .active_component_plan
        .as_deref()
        .map_or(0, |plan| plan.component_max_k.len());
    if runtime.active_components_enabled != should_enable
        || (should_enable && runtime.active_components.len() != component_count)
    {
        configure_batch_components(runtime, program)?;
    }
    if !runtime.active_components_enabled {
        return Ok(());
    }
    let plan = program
        .active_component_plan
        .as_deref()
        .expect("enabled implies a plan");
    for component in &mut runtime.active_components {
        component.k = 0;
        component.active = false;
    }
    if plan.initial_components != program.initial_k {
        return Err(TicitError::new(
            "active component plan has the wrong initial component count",
        ));
    }
    let pitch = runtime.active_pitch;
    let shot_major = runtime.dense_shot_major_active;
    let active_shots = runtime.active_shots;
    for component_id in 0..plan.initial_components {
        let component = &mut runtime.active_components[component_id];
        if component.stride < 2 {
            return Err(TicitError::new(
                "initial active component has insufficient storage",
            ));
        }
        component.k = 1;
        component.active = true;
        if shot_major {
            for shot in 0..active_shots {
                let base = shot * component.stride;
                component.re[base] = 1.0;
                component.im[base] = 0.0;
                component.re[base + 1] = 0.0;
                component.im[base + 1] = 0.0;
            }
        } else {
            for shot in 0..active_shots {
                component.re[shot] = 1.0;
                component.im[shot] = 0.0;
                component.re[pitch + shot] = 0.0;
                component.im[pitch + shot] = 0.0;
            }
        }
    }
    Ok(())
}

fn ensure_dense_batch_active_storage(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
) -> Result<()> {
    runtime.active_components_enabled = false;
    runtime.active_components.clear();
    let max_dim = active_length(program.max_k)?;
    runtime.active_stride = max_dim;
    let active_size = max_dim * runtime.active_pitch;
    runtime.active_re.resize(active_size, 0.0);
    runtime.active_im.resize(active_size, 0.0);
    runtime.scratch_re.resize(active_size, 0.0);
    runtime.scratch_im.resize(active_size, 0.0);
    Ok(())
}

/// Only the first `2^initial_k` entries of each row are zeroed; the rest stays
/// stale from the previous block, which is safe because promotion fully writes
/// the newly exposed upper half.
fn initialize_dense_batch_active(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
) -> Result<()> {
    let dim = active_length(program.initial_k)?;
    if runtime.dense_shot_major_active {
        for shot in 0..runtime.active_shots {
            let base = shot * runtime.active_stride;
            runtime.active_re[base..base + dim].fill(0.0);
            runtime.active_im[base..base + dim].fill(0.0);
            runtime.active_re[base] = 1.0;
        }
        return Ok(());
    }
    let pitch = runtime.active_pitch;
    for basis in 0..dim {
        let base = basis * pitch;
        runtime.active_re[base..base + pitch].fill(0.0);
        runtime.active_im[base..base + pitch].fill(0.0);
    }
    for shot in 0..runtime.active_shots {
        runtime.active_re[shot] = 1.0;
    }
    Ok(())
}

impl BatchFactoredExecutorState {
    /// `batches == 0` selects [`default_batch_count`].
    pub fn new(program: &FactoredInstructionProgram, batches: usize, seed: u64) -> Result<Self> {
        let batches = if batches > 0 {
            batches
        } else {
            default_batch_count(program.max_k)?
        };
        let mut runtime = Self {
            batches,
            rng_state: seed,
            ..Self::default()
        };
        reset_batch_executor(&mut runtime, program, batches)?;
        Ok(runtime)
    }
}

/// Rewinds the runtime for a block of `shots <= batches` live shots.
pub fn reset_batch_executor(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
    shots: usize,
) -> Result<()> {
    if shots > runtime.batches {
        return Err(TicitError::new("active batch shot count is out of range"));
    }
    runtime.n = program.n;
    runtime.k = program.initial_k;
    runtime.ndormant = program.n - program.initial_k;
    runtime.active_shots = shots;
    runtime.active_pitch = padded_batch_active_pitch(runtime.batches);
    runtime.nsymbols = program.nsymbols;
    runtime.nrecords = program.nrecords;
    runtime.ndetectors = program.ndetectors;
    runtime.nexpvals = program.nexpvals;
    runtime.max_k = program.max_k;
    runtime.batch_words = batch_word_count(runtime.batches);

    reset_batch_components(runtime, program)?;
    if !runtime.active_components_enabled {
        ensure_dense_batch_active_storage(runtime, program)?;
    }

    let symbol_size = program.nsymbols * runtime.batch_words;
    if runtime.value_words.len() != symbol_size {
        runtime.value_words.resize(symbol_size, 0);
    }
    let assigned_size = symbol_word_count(program.nsymbols);
    if runtime.assigned_words.len() != assigned_size {
        runtime.assigned_words.resize(assigned_size, 0);
    }
    runtime.assigned_words.fill(0);
    let measurement_size = program.nrecords * runtime.batch_words;
    if runtime.measurement_words.len() != measurement_size {
        runtime.measurement_words.resize(measurement_size, 0);
    }
    runtime.measurement_words.fill(0);
    if runtime.store_detector_records {
        let detector_size = program.ndetectors * runtime.batch_words;
        if runtime.detector_words.len() != detector_size {
            runtime.detector_words.resize(detector_size, 0);
        }
        runtime.detector_words.fill(0);
    }
    if runtime.detector_any_words.len() != runtime.batch_words {
        runtime.detector_any_words.resize(runtime.batch_words, 0);
    }
    runtime.detector_any_words.fill(0);
    let expectation_size = program.nexpvals * runtime.batches;
    if runtime.exp_values.len() != expectation_size {
        runtime.exp_values.resize(expectation_size, 0.0);
    }
    runtime.exp_values.fill(0.0);
    if runtime.eval_scratch.len() != runtime.batch_words {
        runtime.eval_scratch.resize(runtime.batch_words, 0);
    }
    runtime.eval_scratch.fill(0);
    if runtime.shot_coefficient_scalars.len() < runtime.active_pitch {
        runtime
            .shot_coefficient_scalars
            .resize(runtime.active_pitch, 0.0);
    }
    if runtime.branch_prob_true.len() < runtime.active_pitch {
        runtime.branch_prob_true.resize(runtime.active_pitch, 0.0);
    }
    if runtime.branch_invnorms.len() < runtime.active_pitch {
        runtime.branch_invnorms.resize(runtime.active_pitch, 0.0);
    }

    if !runtime.active_components_enabled {
        initialize_dense_batch_active(runtime, program)?;
    }
    Ok(())
}

// ==============================================================================
// Component scope + merges
// ==============================================================================

pub(crate) fn swap_batch_component_buffers(
    runtime: &mut BatchFactoredExecutorState,
    component_index: usize,
) {
    let component = &mut runtime.active_components[component_index];
    std::mem::swap(&mut runtime.active_re, &mut component.re);
    std::mem::swap(&mut runtime.active_im, &mut component.im);
    std::mem::swap(&mut runtime.scratch_re, &mut component.scratch_re);
    std::mem::swap(&mut runtime.scratch_im, &mut component.scratch_im);
}

/// Runs `body` with the component's buffers, `k` and stride swapped in as the
/// dense state, restoring the global bookkeeping afterwards even on error.
pub(crate) fn with_batch_component<R>(
    runtime: &mut BatchFactoredExecutorState,
    component_index: usize,
    body: impl FnOnce(&mut BatchFactoredExecutorState) -> Result<R>,
) -> Result<R> {
    if !runtime.active_components[component_index].active {
        return Err(TicitError::new(
            "cannot execute an instruction on an inactive component",
        ));
    }
    let global_k = runtime.k;
    let global_ndormant = runtime.ndormant;
    let global_stride = runtime.active_stride;
    swap_batch_component_buffers(runtime, component_index);
    runtime.k = runtime.active_components[component_index].k;
    runtime.active_stride = runtime.active_components[component_index].stride;

    let result = body(runtime);

    runtime.active_components[component_index].k = runtime.k;
    runtime.active_components[component_index].stride = runtime.active_stride;
    swap_batch_component_buffers(runtime, component_index);
    runtime.k = global_k;
    runtime.ndormant = global_ndormant;
    runtime.active_stride = global_stride;
    result
}

pub(crate) fn merge_batch_components(
    runtime: &mut BatchFactoredExecutorState,
    merge_components: &[usize],
    merge_offset: usize,
    merge_count: usize,
    expected_target: usize,
) -> Result<()> {
    if merge_count == 0 || merge_offset + merge_count > merge_components.len() {
        return Err(TicitError::new(
            "active component instruction has an invalid merge range",
        ));
    }
    let target_id = merge_components[merge_offset];
    if target_id != expected_target || target_id >= runtime.active_components.len() {
        return Err(TicitError::new(
            "active component merge has an invalid target",
        ));
    }
    if !runtime.active_components[target_id].active {
        return Err(TicitError::new("active component merge target is inactive"));
    }
    let pitch = runtime.active_pitch;
    let shot_major = runtime.dense_shot_major_active;
    let active_shots = runtime.active_shots;
    for source_index in 1..merge_count {
        let source_id = merge_components[merge_offset + source_index];
        if source_id >= runtime.active_components.len() || source_id == target_id {
            return Err(TicitError::new(
                "active component merge has an invalid source",
            ));
        }
        // Split the borrow: lift the source out, merge, put it back.
        let mut source = std::mem::take(&mut runtime.active_components[source_id]);
        let merge = (|| -> Result<()> {
            if !source.active {
                return Err(TicitError::new("active component merge source is inactive"));
            }
            let target = &mut runtime.active_components[target_id];
            let target_dim = active_length(target.k)?;
            let source_dim = active_length(source.k)?;
            let merged_k = target.k + source.k;
            let merged_dim = active_length(merged_k)?;
            if target.stride < merged_dim {
                return Err(TicitError::new(
                    "active component merge target has insufficient storage",
                ));
            }
            // Tensor in place: walking source_basis downwards means the
            // expanded output never lands on an unread source row.
            if shot_major {
                for shot in 0..active_shots {
                    let target_shot = shot * target.stride;
                    let source_shot = shot * source.stride;
                    for source_basis in (0..source_dim).rev() {
                        let source_re = source.re[source_shot + source_basis];
                        let source_im = source.im[source_shot + source_basis];
                        let output_base = target_shot + (source_basis << target.k);
                        for target_basis in 0..target_dim {
                            let target_re = target.re[target_shot + target_basis];
                            let target_im = target.im[target_shot + target_basis];
                            let output = output_base + target_basis;
                            target.re[output] = target_re * source_re - target_im * source_im;
                            target.im[output] = target_re * source_im + target_im * source_re;
                        }
                    }
                }
            } else {
                for source_basis in (0..source_dim).rev() {
                    let source_base = source_basis * pitch;
                    let output_basis_base = source_basis << target.k;
                    for target_basis in 0..target_dim {
                        let target_base = target_basis * pitch;
                        let output_base = (output_basis_base + target_basis) * pitch;
                        for shot in 0..active_shots {
                            let target_re = target.re[target_base + shot];
                            let target_im = target.im[target_base + shot];
                            let source_re = source.re[source_base + shot];
                            let source_im = source.im[source_base + shot];
                            target.re[output_base + shot] =
                                target_re * source_re - target_im * source_im;
                            target.im[output_base + shot] =
                                target_re * source_im + target_im * source_re;
                        }
                    }
                }
            }
            target.k = merged_k;
            source.k = 0;
            source.active = false;
            Ok(())
        })();
        runtime.active_components[source_id] = source;
        merge?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Batch executor and layout-agreement tests.

    use super::*;
    use crate::circuit::{parse_ticit_text, plan_ticit_factored_program};
    use crate::sampler::batch::runtime::{
        sample_measurements_and_expectations_batch, sample_measurements_batch,
    };
    use crate::sampler::exogenous::presample_exogenous_packed;
    use crate::sampler::presampled_expression::{
        PresampledExpressionBlock, PresampledExpressionPlan, evaluate_presampled_expression_block,
        prepare_presampled_expression_plan,
    };

    fn planned(text: &str) -> FactoredInstructionProgram {
        let parsed = parse_ticit_text(text).expect("test circuit parses");
        plan_ticit_factored_program(&parsed).expect("test circuit plans")
    }

    fn record_bit(row: &[u64], record: usize) -> bool {
        (row[record >> 6] >> (record & 63)) & 1 != 0
    }

    #[test]
    fn default_constants_match_the_cpp() {
        assert_eq!(default_batch_count(10).expect("k=10 valid"), 32);
        assert_eq!(default_batch_count(0).expect("k=0 valid"), 2048);
        assert_eq!(default_batch_count(15).expect("k=15 valid"), 1);
    }

    /// `test_batch_sampler`: H/T/M at 200 shots, seed 23, batch 32 — the ones
    /// count sits in the C++ statistical window.
    #[test]
    fn batch_sampler_t_gate_window() {
        let program = planned("H 0\nT 0\nM 0\n");
        let records = sample_measurements_batch(&program, 200, 32, 23).expect("sampling succeeds");
        assert_eq!(records.len(), 200);
        let ones = records.iter().filter(|row| record_bit(row, 0)).count();
        assert!(
            ones > 50 && ones < 150,
            "ones {ones} outside the (50, 150) window"
        );
    }

    /// Identical seeds reproduce bit-identically; the batch path's RNG is
    /// self-contained.
    #[test]
    fn batch_sampler_is_seed_deterministic() {
        let program = planned("X_ERROR(0.25) 0\nH 1\nT 1\nH 1\nM 0 1\n");
        let first = sample_measurements_batch(&program, 300, 0, 99).expect("sampling succeeds");
        let second = sample_measurements_batch(&program, 300, 0, 99).expect("sampling succeeds");
        assert_eq!(first, second);
        let other = sample_measurements_batch(&program, 300, 0, 100).expect("sampling succeeds");
        assert_ne!(first, other, "a different seed should move some record");
    }

    /// `test_batch_expectation_sampler`: exact expectation values through the
    /// probe circuit, and the probe is non-destructive.
    #[test]
    fn batch_expectation_sampler_matches_exact_values() {
        let program = planned("EXP_VAL Z0 X0\nH 0\nT 0\nEXP_VAL X0 Y0 Z0\nT_DAG 0\nH 0\nM 0\n");
        let out = sample_measurements_and_expectations_batch(&program, 5, 0, 71)
            .expect("sampling succeeds");
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let expected = [1.0, 0.0, inv_sqrt2, inv_sqrt2, 0.0];
        for (shot, expectations) in out.expectations.iter().enumerate() {
            assert_eq!(expectations.len(), 5);
            for (i, (&actual, &want)) in expectations.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - want).abs() <= 1e-12,
                    "shot {shot} exp {i}: {actual} != {want}"
                );
            }
        }
        for row in &out.measurements {
            assert!(!record_bit(row, 0), "the probe must be non-destructive");
        }
    }

    // ==============================================================================
    // Postselection
    // ==============================================================================

    struct PostselectRun {
        program: FactoredInstructionProgram,
        plan: PresampledExpressionPlan,
        block: PresampledExpressionBlock,
    }

    fn prepare_postselect(text: &str, shots: usize, seed: u64) -> PostselectRun {
        let program = planned(text);
        let samples =
            presample_exogenous_packed(&program, shots, seed).expect("presample succeeds");
        let mut plan = PresampledExpressionPlan::default();
        prepare_presampled_expression_plan(&mut plan, &program, &samples).expect("plan matches");
        let mut block = PresampledExpressionBlock::default();
        evaluate_presampled_expression_block(&mut block, &plan, &samples).expect("block evaluates");
        PostselectRun {
            program,
            plan,
            block,
        }
    }

    fn run_postselected(
        run: &PostselectRun,
        shots: usize,
        seed: u64,
        shot_major: bool,
        denominator: usize,
    ) -> (usize, usize, Vec<u64>) {
        let mut runtime =
            BatchFactoredExecutorState::new(&run.program, shots, seed).expect("runtime builds");
        runtime.dense_shot_major_active = shot_major;
        reset_batch_executor(&mut runtime, &run.program, shots).expect("reset succeeds");
        runtime.rng_state = seed;
        let mut scratch = BatchDetectorPostselectionScratch::default();
        let options = BatchDetectorPostselectionOptions {
            mask_dead_shots_min_fraction_denominator: denominator,
            retained_record_uses: None,
        };
        let result = execute_batch_postselected_in_place(
            &mut runtime,
            &run.program,
            &run.plan,
            &run.block,
            0,
            &mut scratch,
            &options,
        )
        .expect("postselected execution succeeds");
        (
            result.discarded,
            result.accepted,
            runtime.measurement_words.clone(),
        )
    }

    /// `test_batch_postselection` block 1: an always-fired detector discards all.
    #[test]
    fn postselection_discards_every_shot_of_a_fired_detector() {
        let run = prepare_postselect("M !0\nDISCARD rec[-1]\n", 8, 3);
        let (discarded, accepted, _) = run_postselected(&run, 8, 3, false, 2);
        assert_eq!(discarded, 8);
        assert_eq!(accepted, 0);
    }

    /// Block 2: a quiet detector keeps all shots and the record column is zero.
    #[test]
    fn postselection_keeps_every_shot_of_a_quiet_detector() {
        let run = prepare_postselect("M 0\nDISCARD rec[-1]\n", 8, 3);
        let (discarded, accepted, measurement_words) = run_postselected(&run, 8, 3, false, 2);
        assert_eq!(discarded, 0);
        assert_eq!(accepted, 8);
        assert_eq!(measurement_words[0], 0);
    }

    /// Block 3: both compaction strategies (and both layouts) agree exactly —
    /// compaction timing never changes which shots draw randomness.
    #[test]
    fn postselection_compaction_strategies_agree() {
        let text = "X_ERROR(0.125) 0\nM 0\nDISCARD rec[-1]\nH 1\nT 1\nT_DAG 1\nH 1\nM 1\n";
        let run = prepare_postselect(text, 130, 41);
        let default_strategy = run_postselected(&run, 130, 41, false, 2);
        let eager_strategy = run_postselected(&run, 130, 41, false, 1);
        assert!(default_strategy.0 > 0, "some shots should discard");
        assert_eq!(default_strategy.0, eager_strategy.0);
        assert_eq!(default_strategy.1, eager_strategy.1);
        // Record columns agree over the surviving (compacted) shots.
        let live = default_strategy.1;
        let live_words = live.div_ceil(64);
        let nrecords = run.program.nrecords;
        let batch_words = default_strategy.2.len() / nrecords.max(1);
        for record in 0..nrecords {
            let base = record * batch_words;
            for word in 0..live_words {
                let remaining = live - word * 64;
                let mask = if remaining >= 64 {
                    u64::MAX
                } else {
                    (1u64 << remaining) - 1
                };
                assert_eq!(
                    default_strategy.2[base + word] & mask,
                    eager_strategy.2[base + word] & mask,
                    "record {record} word {word} differs between compaction strategies"
                );
            }
        }

        let shot_major = run_postselected(&run, 130, 41, true, 2);
        assert_eq!(shot_major.0, default_strategy.0);
        assert_eq!(shot_major.1, default_strategy.1);
    }
}
