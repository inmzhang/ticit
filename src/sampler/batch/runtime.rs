//! Execution drivers: inline-exogenous and presampled-expression paths,
//! component dispatch, and the shot-major rotation-run fusion.

use super::active::{
    batch_bit_at, measure_precomputed_active_pauli_branch_batch,
    measure_precomputed_active_pauli_expectation_batch, promote_first_dormant_rotation_batch,
    rotate_pauli_batch,
};
#[cfg(test)]
use super::reset_batch_executor;
use super::symbols::{
    assign_batch_symbol, eval_symbolic_bool_batch, write_batch_detector_record,
    write_batch_measurement_record, write_direct_branch_measurement_record,
    xor_symbolic_bool_batch_into,
};
use super::{
    BatchFactoredExecutorState, batch_live_word_mask, batch_record_offset, batch_shot_mask,
    batch_shot_word, fill_batch_random_half_bits, live_word_mask_for_shots, merge_batch_components,
    runtime_batch_word_count, with_batch_component,
};
use crate::active::active_length;
#[cfg(test)]
use crate::bits::symbol_word_count;
use crate::component_plan::ActiveComponentStepKind;
use crate::errors::{Result, TicitError};
#[cfg(test)]
use crate::exogenous::{
    PackedPresampledExogenous, prepare_presampled_exogenous_packed,
    resample_prepared_exogenous_packed_in_place,
};
use crate::factored::{FactoredInstruction, FactoredInstructionProgram, RecordDetector};
use crate::presampled_expression::{
    PresampledExpressionBlock, PresampledExpressionPlan, presampled_expression_block_offset,
};
#[cfg(test)]
use crate::presampled_expression::{
    evaluate_presampled_expression_block, prepare_presampled_expression_plan,
};

pub(crate) const SHOT_MAJOR_ROTATION_RUN_LIMIT: usize = 32;

// ==============================================================================
// Expectation writes
// ==============================================================================

fn write_batch_expectation(
    runtime: &mut BatchFactoredExecutorState,
    exp_val: i32,
    outcome_bits: &[u64],
) -> Result<()> {
    if exp_val < 0 || exp_val as usize >= runtime.nexpvals {
        return Err(TicitError::new("expectation value index is out of range"));
    }
    let base = exp_val as usize * runtime.batches;
    for shot in 0..runtime.active_shots {
        runtime.exp_values[base + shot] = if batch_bit_at(outcome_bits, shot) {
            -1.0
        } else {
            1.0
        };
    }
    Ok(())
}

fn write_zero_batch_expectation(
    runtime: &mut BatchFactoredExecutorState,
    exp_val: i32,
) -> Result<()> {
    if exp_val < 0 || exp_val as usize >= runtime.nexpvals {
        return Err(TicitError::new("expectation value index is out of range"));
    }
    let base = exp_val as usize * runtime.batches;
    runtime.exp_values[base..base + runtime.active_shots].fill(0.0);
    Ok(())
}

// ==============================================================================
// Sign sources
// ==============================================================================

/// Where a batch instruction's expression bits come from: the inline plan, the
/// immutable chunk-wide expression block (bit-sliced per block), or the
/// materialized postselection workspace.
pub(crate) enum BatchSignSource<'a> {
    Block {
        plan: &'a PresampledExpressionPlan,
        block: &'a PresampledExpressionBlock,
        first_sample_shot: usize,
    },
    Workspace {
        plan: &'a PresampledExpressionPlan,
        words: &'a [u64],
        stride_words: usize,
    },
}

pub(crate) fn expression_slice_word(
    block: &PresampledExpressionBlock,
    block_expression_index: usize,
    first_sample_shot: usize,
    active_shots: usize,
    dest_word: usize,
) -> u64 {
    let bit_offset = first_sample_shot & 63;
    let src_word = (first_sample_shot >> 6) + dest_word;
    let mut out = 0u64;
    if src_word < block.shot_words {
        out = block.expression_words
            [presampled_expression_block_offset(block, block_expression_index, src_word)]
            >> bit_offset;
    }
    if bit_offset != 0 && src_word + 1 < block.shot_words {
        out |= block.expression_words
            [presampled_expression_block_offset(block, block_expression_index, src_word + 1)]
            << (64 - bit_offset);
    }
    out & live_word_mask_for_shots(active_shots, dest_word)
}

impl BatchSignSource<'_> {
    /// Fills `out` with the instruction's expression bits: base bits from the
    /// source, then the residual XOR against the runtime.
    pub(crate) fn eval(
        &self,
        instruction_index: usize,
        runtime: &BatchFactoredExecutorState,
        out: &mut Vec<u64>,
    ) -> Result<()> {
        let plan = match self {
            BatchSignSource::Block { plan, .. } | BatchSignSource::Workspace { plan, .. } => plan,
        };
        if instruction_index >= plan.instruction_expressions.len() {
            return Err(TicitError::new(
                "batch presampled expression plan does not match program",
            ));
        }
        let expression = &plan.instruction_expressions[instruction_index];
        if expression.block_expression_index >= plan.block_expressions.len() {
            return Err(TicitError::new(
                "batch presampled expression references an out-of-range block expression",
            ));
        }
        let block_expression_index = expression.block_expression_index;
        let nwords = runtime_batch_word_count(runtime);
        if out.len() < runtime.batch_words {
            out.resize(runtime.batch_words, 0);
        }
        match self {
            BatchSignSource::Block {
                block,
                first_sample_shot,
                ..
            } => {
                for word in 0..nwords {
                    out[word] = expression_slice_word(
                        block,
                        block_expression_index,
                        *first_sample_shot,
                        runtime.active_shots,
                        word,
                    );
                }
            }
            BatchSignSource::Workspace {
                words,
                stride_words,
                ..
            } => {
                let base = block_expression_index * stride_words;
                if words.len() < base + stride_words {
                    return Err(TicitError::new(
                        "batch presampled expression workspace is too short",
                    ));
                }
                for word in 0..nwords {
                    out[word] = words[base + word] & batch_live_word_mask(runtime, word);
                }
            }
        }
        out[nwords..].fill(0);
        if !expression.residual_plan.conditions.is_empty() {
            xor_symbolic_bool_batch_into(out, &expression.residual_plan, runtime)?;
        }
        Ok(())
    }
}

// ==============================================================================
// Per-instruction execution
// ==============================================================================

fn detector_record_outcome_bits(
    runtime: &BatchFactoredExecutorState,
    instruction: &RecordDetector,
    out: &mut Vec<u64>,
) -> Result<()> {
    if instruction.records.is_empty() {
        return eval_symbolic_bool_batch(out, &instruction.outcome_plan, runtime);
    }
    if out.len() < runtime.batch_words {
        out.resize(runtime.batch_words, 0);
    }
    let nwords = runtime_batch_word_count(runtime);
    out[..nwords].fill(0);
    for &record in &instruction.records {
        if record <= 0 || record as usize > runtime.nrecords {
            return Err(TicitError::new(
                "detector references an out-of-range measurement record",
            ));
        }
        let base = batch_record_offset(runtime, record, 0);
        for word in 0..nwords {
            out[word] ^= runtime.measurement_words[base + word];
        }
    }
    out[nwords..].fill(0);
    Ok(())
}

/// Detector outcome bits per the presampled convention: record-XOR and
/// constant outcomes come from the runtime, everything else from the source.
fn detector_outcome_bits(
    runtime: &BatchFactoredExecutorState,
    instruction: &RecordDetector,
    source: &BatchSignSource<'_>,
    instruction_index: usize,
    out: &mut Vec<u64>,
) -> Result<()> {
    if !instruction.records.is_empty() || instruction.outcome.conditions.is_empty() {
        return detector_record_outcome_bits(runtime, instruction, out);
    }
    source.eval(instruction_index, runtime, out)
}

pub(crate) fn execute_batch_instruction(
    runtime: &mut BatchFactoredExecutorState,
    instruction: &FactoredInstruction,
    source: &BatchSignSource<'_>,
    instruction_index: usize,
    work: &mut Vec<u64>,
) -> Result<()> {
    match instruction {
        FactoredInstruction::ApplyPrecomputedActivePauliRotation(inst) => {
            source.eval(instruction_index, runtime, work)?;
            rotate_pauli_batch(runtime, &inst.rotation_kernel, work)
        }
        FactoredInstruction::PromoteDormantRotation(inst) => {
            source.eval(instruction_index, runtime, work)?;
            promote_first_dormant_rotation_batch(runtime, inst.kernel_angle, work)
        }
        FactoredInstruction::RecordMeasurement(inst) => {
            source.eval(instruction_index, runtime, work)?;
            if let Some(exp_val) = inst.exp_val {
                return write_batch_expectation(runtime, exp_val, work);
            }
            write_batch_measurement_record(runtime, inst.record, work, inst.record_condition)
        }
        FactoredInstruction::RecordDetector(inst) => {
            detector_outcome_bits(runtime, inst, source, instruction_index, work)?;
            write_batch_detector_record(runtime, inst.detector, work)
        }
        FactoredInstruction::MeasurePrecomputedActivePauli(inst) => {
            if let Some(exp_val) = inst.exp_val {
                source.eval(instruction_index, runtime, work)?;
                return measure_precomputed_active_pauli_expectation_batch(
                    runtime,
                    &inst.kernel,
                    work,
                    exp_val,
                );
            }
            measure_precomputed_active_pauli_branch_batch(
                runtime,
                &inst.kernel,
                inst.branch,
                work,
            )?;
            if write_direct_branch_measurement_record(
                runtime,
                work,
                inst.branch,
                &inst.outcome_plan,
                inst.record,
                inst.record_condition,
            )? {
                return Ok(());
            }
            source.eval(instruction_index, runtime, work)?;
            write_batch_measurement_record(runtime, inst.record, work, inst.record_condition)
        }
        FactoredInstruction::IntroduceDormantMeasurementBranch(inst) => {
            if let Some(exp_val) = inst.exp_val {
                return write_zero_batch_expectation(runtime, exp_val);
            }
            fill_batch_random_half_bits(work, runtime);
            assign_batch_symbol(runtime, inst.branch, work)?;
            if write_direct_branch_measurement_record(
                runtime,
                work,
                inst.branch,
                &inst.outcome_plan,
                inst.record,
                inst.record_condition,
            )? {
                return Ok(());
            }
            source.eval(instruction_index, runtime, work)?;
            write_batch_measurement_record(runtime, inst.record, work, inst.record_condition)
        }
    }
}

// ==============================================================================
// Component dispatch
// ==============================================================================

pub(crate) fn execute_batch_component_instruction(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
    source: &BatchSignSource<'_>,
    instruction_index: usize,
    work: &mut Vec<u64>,
) -> Result<()> {
    let plan = program
        .active_component_plan
        .as_deref()
        .expect("component execution requires a plan");
    let step = plan.instruction_steps[instruction_index];
    match step.kind {
        ActiveComponentStepKind::IgnoredGlobalPhase => Ok(()),
        ActiveComponentStepKind::Rotation => {
            let FactoredInstruction::ApplyPrecomputedActivePauliRotation(_) =
                &program.instructions[instruction_index]
            else {
                return Err(TicitError::new(
                    "component rotation step does not match its instruction",
                ));
            };
            source.eval(instruction_index, runtime, work)?;
            let rotation = &plan.rotations[step.payload];
            merge_batch_components(
                runtime,
                &plan.merge_components,
                rotation.merge_offset,
                rotation.merge_count,
                rotation.component,
            )?;
            let kernel = rotation.kernel;
            with_batch_component(runtime, rotation.component, |scoped| {
                rotate_pauli_batch(scoped, &kernel, work)
            })
        }
        ActiveComponentStepKind::Promotion => {
            let FactoredInstruction::PromoteDormantRotation(instruction) =
                &program.instructions[instruction_index]
            else {
                return Err(TicitError::new(
                    "component promotion step does not match its instruction",
                ));
            };
            source.eval(instruction_index, runtime, work)?;
            execute_batch_component_promotion(
                runtime,
                plan.promotions[step.payload].component,
                instruction.kernel_angle,
                work,
            )
        }
        ActiveComponentStepKind::Measurement => {
            let FactoredInstruction::MeasurePrecomputedActivePauli(instruction) =
                &program.instructions[instruction_index]
            else {
                return Err(TicitError::new(
                    "component measurement step does not match its instruction",
                ));
            };
            let measurement = &plan.measurements[step.payload];
            merge_batch_components(
                runtime,
                &plan.merge_components,
                measurement.merge_offset,
                measurement.merge_count,
                measurement.component,
            )?;
            let kernel = measurement.kernel;
            let branch_condition = instruction.branch;
            with_batch_component(runtime, measurement.component, |scoped| {
                measure_precomputed_active_pauli_branch_batch(
                    scoped,
                    &kernel,
                    branch_condition,
                    work,
                )
            })?;
            // The component scope restored the global k; the measurement's
            // global effect is applied here, on top of the component-local one.
            runtime.k -= 1;
            runtime.ndormant += 1;
            let component = &mut runtime.active_components[measurement.component];
            if measurement.deactivate_after {
                if component.k != 0 {
                    return Err(TicitError::new(
                        "active component measurement did not remove its last coordinate",
                    ));
                }
                component.active = false;
            }
            if write_direct_branch_measurement_record(
                runtime,
                work,
                instruction.branch,
                &instruction.outcome_plan,
                instruction.record,
                instruction.record_condition,
            )? {
                return Ok(());
            }
            source.eval(instruction_index, runtime, work)?;
            write_batch_measurement_record(
                runtime,
                instruction.record,
                work,
                instruction.record_condition,
            )
        }
        ActiveComponentStepKind::None => execute_batch_instruction(
            runtime,
            &program.instructions[instruction_index],
            source,
            instruction_index,
            work,
        ),
    }
}

fn execute_batch_component_promotion(
    runtime: &mut BatchFactoredExecutorState,
    component_index: usize,
    kernel_angle: f64,
    sign_bits: &[u64],
) -> Result<()> {
    if runtime.ndormant == 0 || component_index >= runtime.active_components.len() {
        return Err(TicitError::new("active component promotion is invalid"));
    }
    let pitch = runtime.active_pitch;
    let shot_major = runtime.dense_shot_major_active;
    let active_shots = runtime.active_shots;
    let component = &mut runtime.active_components[component_index];
    if component.active || component.stride < 2 {
        return Err(TicitError::new(
            "active component promotion target is unavailable",
        ));
    }
    let c = kernel_angle.cos();
    let s = kernel_angle.sin();
    component.k = 1;
    component.active = true;
    if shot_major {
        for shot in 0..active_shots {
            let base = shot * component.stride;
            component.re[base] = c;
            component.im[base] = 0.0;
            component.re[base + 1] = 0.0;
            component.im[base + 1] = if batch_bit_at(sign_bits, shot) { s } else { -s };
        }
    } else {
        for shot in 0..active_shots {
            component.re[shot] = c;
            component.im[shot] = 0.0;
            component.re[pitch + shot] = 0.0;
            component.im[pitch + shot] = if batch_bit_at(sign_bits, shot) { s } else { -s };
        }
    }
    runtime.k += 1;
    runtime.ndormant -= 1;
    Ok(())
}

// ==============================================================================
// Shot-major rotation-run fusion
// ==============================================================================

fn shot_major_rotation_run_length(
    program: &FactoredInstructionProgram,
    first_index: usize,
) -> usize {
    let mut run_len = 0;
    while first_index + run_len < program.instructions.len()
        && run_len < SHOT_MAJOR_ROTATION_RUN_LIMIT
        && matches!(
            program.instructions[first_index + run_len],
            FactoredInstruction::ApplyPrecomputedActivePauliRotation(_)
        )
    {
        run_len += 1;
    }
    run_len
}

/// Length of the dormant-promotion prefix a register run can absorb: the
/// consecutive promotions starting at `first_index` that carry the state
/// from its current dimension exactly to 16 with rotations following.
/// Returns 0 when the shape does not match (the plain path then runs them).
/// Constant-time: exactly `log2(16 / dim)` instructions are probed, so
/// promotion chains on circuits this fusion never applies to cost nothing.
fn shot_major_promotion_prefix_length(
    program: &FactoredInstructionProgram,
    first_index: usize,
    dim: usize,
    ndormant: usize,
) -> usize {
    let needed = match dim {
        2 => 3,
        4 => 2,
        8 => 1,
        _ => return 0,
    };
    if needed > ndormant || first_index + needed > program.instructions.len() {
        return 0;
    }
    for offset in 0..needed {
        if !matches!(
            program.instructions[first_index + offset],
            FactoredInstruction::PromoteDormantRotation(_)
        ) {
            return 0;
        }
    }
    needed
}

fn rotation_run_sign_at(
    sign_words: &[u64],
    stride_words: usize,
    run_offset: usize,
    shot: usize,
) -> bool {
    (sign_words[run_offset * stride_words + batch_shot_word(shot)] & batch_shot_mask(shot)) != 0
}

/// Detects a run of consecutive rotations, evaluates all their sign vectors
/// up front, then loops shot-outer / rotation-inner so each shot's vector is
/// touched once across the whole run. Returns the number of instructions
/// consumed (0 = no run).
pub(crate) fn execute_shot_major_rotation_run(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
    source: &BatchSignSource<'_>,
    first_index: usize,
    dead_bits: Option<&[u64]>,
    work: &mut Vec<u64>,
) -> Result<usize> {
    if matches!(
        program.instructions.get(first_index),
        Some(FactoredInstruction::PromoteDormantRotation(_))
    ) {
        return execute_shot_major_promotion_rotation_run(
            runtime,
            program,
            source,
            first_index,
            dead_bits,
            work,
        );
    }
    let run_len = shot_major_rotation_run_length(program, first_index);
    if run_len <= 1 || runtime.active_shots == 0 {
        return Ok(0);
    }

    let active_words = runtime_batch_word_count(runtime);
    let total_words = run_len * active_words;
    if runtime.rotation_run_sign_words.len() < total_words {
        runtime.rotation_run_sign_words.resize(total_words, 0);
    }
    for run_offset in 0..run_len {
        let FactoredInstruction::ApplyPrecomputedActivePauliRotation(rotation) =
            &program.instructions[first_index + run_offset]
        else {
            unreachable!("run length counted only rotations");
        };
        if rotation.rotation_kernel.action.nqubits != runtime.k {
            return Err(TicitError::new(
                "rotation kernel dimension does not match batch active state",
            ));
        }
        source.eval(first_index + run_offset, runtime, work)?;
        let dest = run_offset * active_words;
        runtime.rotation_run_sign_words[dest..dest + active_words]
            .copy_from_slice(&work[..active_words]);
    }

    let dim = active_length(runtime.k)?;
    let sign_words = std::mem::take(&mut runtime.rotation_run_sign_words);

    if dim == 32
        && run_len == 5
        && execute_diagonal_rotation_run_dim32(
            runtime,
            program,
            first_index,
            dead_bits,
            &sign_words,
            active_words,
        )
    {
        runtime.rotation_run_sign_words = sign_words;
        return Ok(run_len);
    }

    // Register-resident fast path for the msc shape: every rotation in the
    // run is a dim-16 uniform imaginary pair rotation, so one shot's whole
    // state can stay in registers across the run. Both `imag` variants are
    // pre-resolved here; the per-shot sign bit picks one. The kernel's
    // arithmetic is bit-identical to the sequential per-rotation calls.
    let mut register_run: Option<
        [crate::contiguous::UniformImagRunStep; SHOT_MAJOR_ROTATION_RUN_LIMIT],
    > = None;
    if dim == 16 && crate::contiguous::has_uniform_imag_run_dim16_backend() {
        let mut base_steps = [crate::contiguous::UniformImagRunStep {
            xmask: 0,
            cos: 0.0,
            imag_false: 0.0,
            imag_true: 0.0,
        }; SHOT_MAJOR_ROTATION_RUN_LIMIT];
        let mut qualified = true;
        for run_offset in 0..run_len {
            let FactoredInstruction::ApplyPrecomputedActivePauliRotation(rotation) =
                &program.instructions[first_index + run_offset]
            else {
                unreachable!("run length counted only rotations");
            };
            let kernel = &rotation.rotation_kernel;
            let xmask = kernel.action.xmask;
            if kernel.is_diagonal
                || !kernel.uniform_imag_pairs
                || xmask == 0
                || xmask > 15
                || kernel.pair_bit != xmask.ilog2()
            {
                qualified = false;
                break;
            }
            base_steps[run_offset] = crate::contiguous::UniformImagRunStep {
                xmask,
                cos: kernel.cos_kernel_angle,
                imag_false: kernel.coefficient(0, false).im,
                imag_true: kernel.coefficient(0, true).im,
            };
        }
        if qualified {
            register_run = Some(base_steps);
        }
    }

    for shot in 0..runtime.active_shots {
        if let Some(dead) = dead_bits
            && batch_bit_at(dead, shot)
        {
            continue;
        }
        let base = shot * runtime.active_stride;
        if let Some(base_steps) = &register_run
            && rotate_register_run_shot(
                runtime,
                base,
                &base_steps[..run_len],
                &sign_words,
                active_words,
                shot,
            )
        {
            continue;
        }
        for run_offset in 0..run_len {
            let FactoredInstruction::ApplyPrecomputedActivePauliRotation(rotation) =
                &program.instructions[first_index + run_offset]
            else {
                unreachable!("run length counted only rotations");
            };
            let sign = rotation_run_sign_at(&sign_words, active_words, run_offset, shot);
            rotate_contiguous_shot(runtime, base, dim, &rotation.rotation_kernel, sign);
        }
    }
    runtime.rotation_run_sign_words = sign_words;
    Ok(run_len)
}

/// Applies one shot's whole register-resident rotation run. The run's step
/// table is shared across shots; only the per-rotation sign bits are
/// gathered here, into one mask word the kernel decodes. Outlined so the
/// fast path does not perturb the shared runner's code layout.
#[inline(never)]
fn rotate_register_run_shot(
    runtime: &mut BatchFactoredExecutorState,
    base: usize,
    steps: &[crate::contiguous::UniformImagRunStep],
    sign_words: &[u64],
    active_words: usize,
    shot: usize,
) -> bool {
    let sign_mask = rotation_run_sign_mask(steps.len(), sign_words, active_words, shot);
    let range = base..base + runtime.active_stride;
    crate::contiguous::rotate_uniform_imag_run_dim16(
        &mut runtime.active_re[range.clone()],
        &mut runtime.active_im[range],
        steps,
        sign_mask,
    )
}

#[cold]
#[inline(never)]
fn execute_diagonal_rotation_run_dim32(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
    first_index: usize,
    dead_bits: Option<&[u64]>,
    sign_words: &[u64],
    active_words: usize,
) -> bool {
    if !crate::contiguous::has_diagonal_run_dim32_backend() {
        return false;
    }
    let mut steps = [crate::contiguous::DiagonalRunStep::default(); 5];
    for (run_offset, step) in steps.iter_mut().enumerate() {
        let FactoredInstruction::ApplyPrecomputedActivePauliRotation(rotation) =
            &program.instructions[first_index + run_offset]
        else {
            unreachable!("run length counted only rotations");
        };
        let kernel = &rotation.rotation_kernel;
        if !kernel.is_diagonal {
            return false;
        }
        *step = crate::contiguous::DiagonalRunStep::new(
            kernel.action.zmask,
            kernel.cos_kernel_angle,
            kernel.minus_even_coefficient,
        );
    }
    for shot in 0..runtime.active_shots {
        if let Some(dead) = dead_bits
            && batch_bit_at(dead, shot)
        {
            continue;
        }
        let sign_mask = rotation_run_sign_mask(steps.len(), sign_words, active_words, shot);
        let base = shot * runtime.active_stride;
        let range = base..base + runtime.active_stride;
        let applied = crate::contiguous::rotate_diagonal_run_dim32(
            &mut runtime.active_re[range.clone()],
            &mut runtime.active_im[range],
            &steps,
            sign_mask,
        );
        debug_assert!(applied);
    }
    true
}

#[inline(always)]
fn rotation_run_sign_mask(
    run_len: usize,
    sign_words: &[u64],
    active_words: usize,
    shot: usize,
) -> u32 {
    let shot_word = batch_shot_word(shot);
    let shot_mask = batch_shot_mask(shot);
    let mut sign_mask = 0u32;
    for run_offset in 0..run_len {
        if sign_words[run_offset * active_words + shot_word] & shot_mask != 0 {
            sign_mask |= 1 << run_offset;
        }
    }
    sign_mask
}

fn rotate_contiguous_shot(
    runtime: &mut BatchFactoredExecutorState,
    base: usize,
    dim: usize,
    kernel: &crate::active::PrecomputedActivePauliRotationKernel,
    sign: bool,
) {
    let range = base..base + runtime.active_stride;
    let (re, im) = (
        &mut runtime.active_re[range.clone()],
        &mut runtime.active_im[range],
    );
    crate::contiguous::rotate_contiguous_active(re, im, dim, kernel, sign);
}

/// Detects a dormant-promotion prefix that carries the state exactly to
/// dim 16 with a qualifying uniform-imaginary rotation run behind it, and
/// executes the whole sequence register-resident per shot: the pre-promotion
/// state (2/4/8 amplitudes) is loaded once, promoted and rotated in
/// registers, and stored once at dim 16. Returns instructions consumed
/// (0 = shape not fused; the plain per-instruction path runs it).
///
/// Dead shots are skipped like the rotation-run path skips them; the
/// postselected promotion path already skips dead shots, so the fused and
/// unfused schedules agree.
///
/// Outlined and `#[cold]`: the caller runs once per instruction, and this
/// body sitting in the hot text region measurably regressed promotion-free
/// circuits (d05_p0/d07_p0 +1.5-3% cycles) through code layout alone —
/// their profiles shifted in *unchanged* functions. `cold` moves it to
/// `.text.unlikely`; on msc-shaped circuits it runs a few times per block,
/// so residency there costs nothing measurable.
#[cold]
#[inline(never)]
fn execute_shot_major_promotion_rotation_run(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
    source: &BatchSignSource<'_>,
    first_index: usize,
    dead_bits: Option<&[u64]>,
    work: &mut Vec<u64>,
) -> Result<usize> {
    if runtime.active_shots == 0
        || runtime.active_stride < 16
        || !crate::contiguous::has_uniform_imag_run_dim16_backend()
    {
        return Ok(0);
    }
    let dim = active_length(runtime.k)?;
    let promo_len = shot_major_promotion_prefix_length(program, first_index, dim, runtime.ndormant);
    if promo_len == 0 {
        return Ok(0);
    }
    let rot_len = shot_major_rotation_run_length(program, first_index + promo_len)
        .min(SHOT_MAJOR_ROTATION_RUN_LIMIT - promo_len);
    if rot_len == 0 {
        return Ok(0);
    }

    let mut rot_steps = [crate::contiguous::UniformImagRunStep {
        xmask: 0,
        cos: 0.0,
        imag_false: 0.0,
        imag_true: 0.0,
    }; SHOT_MAJOR_ROTATION_RUN_LIMIT];
    for run_offset in 0..rot_len {
        let FactoredInstruction::ApplyPrecomputedActivePauliRotation(rotation) =
            &program.instructions[first_index + promo_len + run_offset]
        else {
            unreachable!("run length counted only rotations");
        };
        let kernel = &rotation.rotation_kernel;
        let xmask = kernel.action.xmask;
        if kernel.action.nqubits != runtime.k + promo_len
            || kernel.is_diagonal
            || !kernel.uniform_imag_pairs
            || xmask == 0
            || xmask > 15
            || kernel.pair_bit != xmask.ilog2()
        {
            return Ok(0);
        }
        rot_steps[run_offset] = crate::contiguous::UniformImagRunStep {
            xmask,
            cos: kernel.cos_kernel_angle,
            imag_false: kernel.coefficient(0, false).im,
            imag_true: kernel.coefficient(0, true).im,
        };
    }
    // The promotion sign selects `q = sin` (bit set) or `q = -sin`, exactly
    // `promote_first_dormant_rotation_batch`'s convention.
    let mut promo_steps = [crate::contiguous::PromotionRunStep {
        cos: 0.0,
        imag_false: 0.0,
        imag_true: 0.0,
    }; 3];
    for promo_offset in 0..promo_len {
        let FactoredInstruction::PromoteDormantRotation(instruction) =
            &program.instructions[first_index + promo_offset]
        else {
            unreachable!("prefix length counted only promotions");
        };
        let s = instruction.kernel_angle.sin();
        promo_steps[promo_offset] = crate::contiguous::PromotionRunStep {
            cos: instruction.kernel_angle.cos(),
            imag_false: -s,
            imag_true: s,
        };
    }

    let run_len = promo_len + rot_len;
    let active_words = runtime_batch_word_count(runtime);
    let total_words = run_len * active_words;
    if runtime.rotation_run_sign_words.len() < total_words {
        runtime.rotation_run_sign_words.resize(total_words, 0);
    }
    for run_offset in 0..run_len {
        source.eval(first_index + run_offset, runtime, work)?;
        let dest = run_offset * active_words;
        runtime.rotation_run_sign_words[dest..dest + active_words]
            .copy_from_slice(&work[..active_words]);
    }

    let sign_words = std::mem::take(&mut runtime.rotation_run_sign_words);
    for shot in 0..runtime.active_shots {
        if let Some(dead) = dead_bits
            && batch_bit_at(dead, shot)
        {
            continue;
        }
        promote_rotate_register_run_shot(
            runtime,
            shot * runtime.active_stride,
            dim,
            &promo_steps[..promo_len],
            &rot_steps[..rot_len],
            &sign_words,
            active_words,
            shot,
        );
    }
    runtime.rotation_run_sign_words = sign_words;
    runtime.k += promo_len;
    runtime.ndormant -= promo_len;
    Ok(run_len)
}

/// One shot of the fused promotion+rotation run. Gathers the shot's sign
/// bits (promotions first, rotations after) into the kernel's mask word;
/// falls back to the sequential per-instruction arithmetic if the register
/// kernel declines (it cannot after the caller's qualification, but the
/// fallback keeps state exact rather than silently wrong). Outlined like
/// [`rotate_register_run_shot`] to protect the runner's code layout, and
/// `#[cold]` so it lands in `.text.unlikely` next to its only caller —
/// per-shot execution keeps it cache-resident there on msc-shaped runs.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn promote_rotate_register_run_shot(
    runtime: &mut BatchFactoredExecutorState,
    base: usize,
    start_dim: usize,
    promotions: &[crate::contiguous::PromotionRunStep],
    steps: &[crate::contiguous::UniformImagRunStep],
    sign_words: &[u64],
    active_words: usize,
    shot: usize,
) {
    let shot_word = batch_shot_word(shot);
    let shot_mask = batch_shot_mask(shot);
    let run_len = promotions.len() + steps.len();
    let mut sign_mask = 0u32;
    for run_offset in 0..run_len {
        if sign_words[run_offset * active_words + shot_word] & shot_mask != 0 {
            sign_mask |= 1 << run_offset;
        }
    }
    let range = base..base + runtime.active_stride;
    let (re, im) = (
        &mut runtime.active_re[range.clone()],
        &mut runtime.active_im[range],
    );
    if crate::contiguous::promote_rotate_uniform_imag_run_dim16(
        re, im, start_dim, promotions, steps, sign_mask,
    ) {
        return;
    }
    let mut dim = start_dim;
    for (promo_offset, step) in promotions.iter().enumerate() {
        let q = if (sign_mask >> promo_offset) & 1 == 1 {
            step.imag_true
        } else {
            step.imag_false
        };
        crate::contiguous::promote_contiguous_active(re, im, dim, step.cos, q);
        dim *= 2;
    }
    for (index, step) in steps.iter().enumerate() {
        let sign = (sign_mask >> (promotions.len() + index)) & 1 == 1;
        let q = if sign {
            step.imag_true
        } else {
            step.imag_false
        };
        crate::contiguous::rotate_uniform_imag_pairs_soa(
            re,
            im,
            dim,
            step.xmask,
            step.xmask.ilog2(),
            step.cos,
            q,
        );
    }
}

// ==============================================================================
// Whole-batch drivers
// ==============================================================================

fn check_batch_runtime_matches(
    runtime: &BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
) -> Result<()> {
    if runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n {
        return Err(TicitError::new(
            "batch executor state does not match program",
        ));
    }
    Ok(())
}

fn run_batch_instructions(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
    source: &BatchSignSource<'_>,
) -> Result<()> {
    let mut work = std::mem::take(&mut runtime.eval_scratch);
    let result = (|| -> Result<()> {
        if runtime.active_components_enabled {
            for idx in 0..program.instructions.len() {
                execute_batch_component_instruction(runtime, program, source, idx, &mut work)?;
            }
            return Ok(());
        }
        let mut idx = 0usize;
        while idx < program.instructions.len() {
            if runtime.dense_shot_major_active {
                let consumed = execute_shot_major_rotation_run(
                    runtime, program, source, idx, None, &mut work,
                )?;
                if consumed != 0 {
                    idx += consumed;
                    continue;
                }
            }
            execute_batch_instruction(runtime, &program.instructions[idx], source, idx, &mut work)?;
            idx += 1;
        }
        Ok(())
    })();
    runtime.eval_scratch = work;
    result
}

/// Expression-plan fast path — the production route with rotation-run fusion.
pub fn execute_batch_in_place_expressions(
    runtime: &mut BatchFactoredExecutorState,
    program: &FactoredInstructionProgram,
    expression_plan: &PresampledExpressionPlan,
    expression_block: &PresampledExpressionBlock,
    first_sample_shot: usize,
) -> Result<()> {
    check_batch_runtime_matches(runtime, program)?;
    if expression_plan.instruction_expressions.len() != program.instructions.len() {
        return Err(TicitError::new(
            "batch presampled expression plan does not match program",
        ));
    }
    if runtime.active_shots == 0 {
        return Ok(());
    }
    let source = BatchSignSource::Block {
        plan: expression_plan,
        block: expression_block,
        first_sample_shot,
    };
    run_batch_instructions(runtime, program, &source)
}

// ==============================================================================
// Standalone drivers
// ==============================================================================

#[cfg(test)]
pub struct MeasurementExpectationSamples {
    pub measurements: Vec<Vec<u64>>,
    pub expectations: Vec<Vec<f64>>,
}

#[cfg(test)]
fn test_expression_block(
    program: &FactoredInstructionProgram,
    shots: usize,
    seed: u64,
) -> Result<(PresampledExpressionPlan, PresampledExpressionBlock)> {
    let mut samples = PackedPresampledExogenous::default();
    prepare_presampled_exogenous_packed(&mut samples, program)?;
    resample_prepared_exogenous_packed_in_place(&mut samples, program, shots, seed)?;
    let mut plan = PresampledExpressionPlan::default();
    prepare_presampled_expression_plan(&mut plan, program, &samples)?;
    let mut block = PresampledExpressionBlock::default();
    evaluate_presampled_expression_block(&mut block, &plan, &samples)?;
    Ok((plan, block))
}

#[cfg(test)]
fn collect_shot_records(runtime: &BatchFactoredExecutorState, shot: usize, row: &mut [u64]) {
    for record in 1..=runtime.nrecords as i32 {
        let word = batch_shot_word(shot);
        let bit = (runtime.measurement_words[batch_record_offset(runtime, record, word)]
            & batch_shot_mask(shot))
            != 0;
        if bit {
            let record_bit = (record - 1) as usize;
            row[record_bit >> 6] |= 1u64 << (record_bit & 63);
        }
    }
}

/// Samples `shots` shots block by block. One continuous `rng_state` carries
/// through every block — no per-block reseed, unlike the prepared sampler.
#[cfg(test)]
pub fn sample_measurements_batch(
    program: &FactoredInstructionProgram,
    shots: usize,
    batches: usize,
    seed: u64,
) -> Result<Vec<Vec<u64>>> {
    let (expression_plan, expression_block) = test_expression_block(program, shots, seed)?;
    let mut runtime = BatchFactoredExecutorState::new(program, batches, seed)?;
    let mut out = vec![vec![0u64; symbol_word_count(program.nrecords)]; shots];
    let mut offset = 0usize;
    while offset < shots {
        let block = runtime.batches.min(shots - offset);
        reset_batch_executor(&mut runtime, program, block)?;
        execute_batch_in_place_expressions(
            &mut runtime,
            program,
            &expression_plan,
            &expression_block,
            offset,
        )?;
        for shot in 0..block {
            collect_shot_records(&runtime, shot, &mut out[offset + shot]);
        }
        offset += block;
    }
    Ok(out)
}

/// Like [`sample_measurements_batch`], also returning per-shot expectations.
#[cfg(test)]
pub fn sample_measurements_and_expectations_batch(
    program: &FactoredInstructionProgram,
    shots: usize,
    batches: usize,
    seed: u64,
) -> Result<MeasurementExpectationSamples> {
    let (expression_plan, expression_block) = test_expression_block(program, shots, seed)?;
    let mut runtime = BatchFactoredExecutorState::new(program, batches, seed)?;
    let mut out = MeasurementExpectationSamples {
        measurements: vec![vec![0u64; symbol_word_count(program.nrecords)]; shots],
        expectations: vec![vec![0.0; program.nexpvals]; shots],
    };
    let mut offset = 0usize;
    while offset < shots {
        let block = runtime.batches.min(shots - offset);
        reset_batch_executor(&mut runtime, program, block)?;
        execute_batch_in_place_expressions(
            &mut runtime,
            program,
            &expression_plan,
            &expression_block,
            offset,
        )?;
        for shot in 0..block {
            collect_shot_records(&runtime, shot, &mut out.measurements[offset + shot]);
            for exp_val in 0..runtime.nexpvals {
                out.expectations[offset + shot][exp_val] =
                    runtime.exp_values[exp_val * runtime.batches + shot];
            }
        }
        offset += block;
    }
    Ok(out)
}
