//! Symbol values, records and detectors: one bit-plane of shots per
//! condition/record/detector, XOR-combined a word (64 shots) at a time.

use super::{
    BATCH_SCALAR_SYMBOLIC_EVAL_THRESHOLD, BatchFactoredExecutorState, batch_condition_offset,
    batch_detector_offset, batch_live_word_mask, batch_record_offset, fill_batch_bits,
    invert_batch_bits, mask_batch_bits, runtime_batch_word_count,
};
use crate::bits::{symbol_bit_mask, symbol_word_index};
use crate::errors::{Result, TicitError};
#[cfg(test)]
use crate::factored::FactoredInstructionProgram;
use crate::symbolic::SymbolicBoolEvaluationPlan;

pub(crate) fn check_batch_symbol_slot(
    runtime: &BatchFactoredExecutorState,
    condition: i32,
) -> Result<()> {
    if condition <= 0 || condition as usize > runtime.nsymbols {
        return Err(TicitError::new(
            "symbolic condition exceeds batch executor symbol table",
        ));
    }
    Ok(())
}

pub(crate) fn batch_symbol_assigned(
    runtime: &BatchFactoredExecutorState,
    condition: i32,
) -> Result<bool> {
    check_batch_symbol_slot(runtime, condition)?;
    let word = symbol_word_index(condition);
    Ok(word < runtime.assigned_words.len()
        && (runtime.assigned_words[word] & symbol_bit_mask(condition)) != 0)
}

fn mark_batch_symbol_assigned_unchecked(runtime: &mut BatchFactoredExecutorState, condition: i32) {
    runtime.assigned_words[symbol_word_index(condition)] |= symbol_bit_mask(condition);
}

fn batch_symbol_matches_bits(
    runtime: &BatchFactoredExecutorState,
    condition: i32,
    bits: &[u64],
) -> Result<bool> {
    let nwords = runtime_batch_word_count(runtime);
    if bits.len() < nwords {
        return Err(TicitError::new("batch bit vector is too short"));
    }
    for word in 0..nwords {
        let mask = batch_live_word_mask(runtime, word);
        let actual = runtime.value_words[batch_condition_offset(runtime, condition, word)];
        if (actual ^ bits[word]) & mask != 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn copy_bits_to_batch_symbol_unchecked(
    runtime: &mut BatchFactoredExecutorState,
    condition: i32,
    bits: &[u64],
) -> Result<()> {
    let nwords = runtime_batch_word_count(runtime);
    if bits.len() < nwords {
        return Err(TicitError::new("batch bit vector is too short"));
    }
    let base = batch_condition_offset(runtime, condition, 0);
    if nwords == runtime.batch_words && runtime.active_shots & 63 == 0 {
        runtime.value_words[base..base + nwords].copy_from_slice(&bits[..nwords]);
        return Ok(());
    }
    for word in 0..nwords {
        runtime.value_words[base + word] = bits[word] & batch_live_word_mask(runtime, word);
    }
    for word in nwords..runtime.batch_words {
        runtime.value_words[base + word] = 0;
    }
    Ok(())
}

/// Idempotent-with-check assignment: re-assigning must match the stored bits.
pub(crate) fn assign_batch_symbol(
    runtime: &mut BatchFactoredExecutorState,
    condition: i32,
    bits: &[u64],
) -> Result<()> {
    check_batch_symbol_slot(runtime, condition)?;
    if batch_symbol_assigned(runtime, condition)? {
        if !batch_symbol_matches_bits(runtime, condition, bits)? {
            return Err(TicitError::new(
                "symbolic condition was assigned inconsistent concrete batch values",
            ));
        }
        return Ok(());
    }
    mark_batch_symbol_assigned_unchecked(runtime, condition);
    copy_bits_to_batch_symbol_unchecked(runtime, condition, bits)
}

pub(crate) fn assign_batch_symbol_opt(
    runtime: &mut BatchFactoredExecutorState,
    condition: Option<i32>,
    bits: &[u64],
) -> Result<()> {
    match condition {
        Some(condition) => assign_batch_symbol(runtime, condition, bits),
        None => Ok(()),
    }
}

// ==============================================================================
// Batched symbolic evaluation
// ==============================================================================

fn check_condition_assigned(runtime: &BatchFactoredExecutorState, condition: i32) -> Result<()> {
    if !batch_symbol_assigned(runtime, condition)? {
        return Err(TicitError::new(
            "symbolic condition expression has no concrete batch value",
        ));
    }
    Ok(())
}

fn check_batch_symbolic_plan_assigned(
    plan: &SymbolicBoolEvaluationPlan,
    runtime: &BatchFactoredExecutorState,
) -> Result<()> {
    if plan.word_indices.is_empty() {
        for &condition in &plan.conditions {
            check_condition_assigned(runtime, condition)?;
        }
        return Ok(());
    }
    let max_word = *plan.word_indices.last().expect("nonempty checked above");
    if max_word >= runtime.assigned_words.len() {
        return Err(TicitError::new(
            "symbolic condition expression has no concrete batch value",
        ));
    }
    let mut missing = 0u64;
    for (i, &word) in plan.word_indices.iter().enumerate() {
        missing |= plan.word_masks[i] & !runtime.assigned_words[word];
    }
    if missing != 0 {
        return Err(TicitError::new(
            "symbolic condition expression has no concrete batch value",
        ));
    }
    Ok(())
}

/// Evaluates `plan` for the whole batch: fills `out` with the constant, then
/// XORs one condition column per condition.
///
/// The `nwords ∈ {1, 2}` shapes are unrolled with register accumulators, and
/// small plans check each condition's assignment inline while large plans do
/// one bulk word-mask test first.
pub fn eval_symbolic_bool_batch(
    out: &mut Vec<u64>,
    plan: &SymbolicBoolEvaluationPlan,
    runtime: &BatchFactoredExecutorState,
) -> Result<()> {
    fill_batch_bits(out, runtime, plan.constant);
    if plan.conditions.is_empty() {
        return Ok(());
    }
    let nwords = runtime_batch_word_count(runtime);
    let checked_per_condition = plan.word_indices.is_empty()
        || plan.conditions.len() <= BATCH_SCALAR_SYMBOLIC_EVAL_THRESHOLD;
    if !checked_per_condition {
        check_batch_symbolic_plan_assigned(plan, runtime)?;
    }
    match nwords {
        1 => {
            let mut out0 = out[0];
            if runtime.batch_words == 1 {
                for &condition in &plan.conditions {
                    if checked_per_condition {
                        check_condition_assigned(runtime, condition)?;
                    }
                    out0 ^= runtime.value_words[(condition - 1) as usize];
                }
            } else {
                for &condition in &plan.conditions {
                    if checked_per_condition {
                        check_condition_assigned(runtime, condition)?;
                    }
                    out0 ^= runtime.value_words[batch_condition_offset(runtime, condition, 0)];
                }
            }
            out[0] = out0;
        }
        2 => {
            let mut out0 = out[0];
            let mut out1 = out[1];
            for &condition in &plan.conditions {
                if checked_per_condition {
                    check_condition_assigned(runtime, condition)?;
                }
                let base = batch_condition_offset(runtime, condition, 0);
                out0 ^= runtime.value_words[base];
                out1 ^= runtime.value_words[base + 1];
            }
            out[0] = out0;
            out[1] = out1;
        }
        _ => {
            for &condition in &plan.conditions {
                if checked_per_condition {
                    check_condition_assigned(runtime, condition)?;
                }
                let base = batch_condition_offset(runtime, condition, 0);
                for word in 0..nwords {
                    out[word] ^= runtime.value_words[base + word];
                }
            }
        }
    }
    mask_batch_bits(out, runtime);
    Ok(())
}

/// Like [`eval_symbolic_bool_batch`] but XOR-accumulates into an existing
/// buffer — the residual half of a presampled expression.
pub(crate) fn xor_symbolic_bool_batch_into(
    out: &mut Vec<u64>,
    plan: &SymbolicBoolEvaluationPlan,
    runtime: &BatchFactoredExecutorState,
) -> Result<()> {
    let nwords = runtime_batch_word_count(runtime);
    if out.len() < runtime.batch_words {
        out.resize(runtime.batch_words, 0);
    }
    if plan.conditions.is_empty() {
        if plan.constant {
            invert_batch_bits(out, runtime);
        }
        return Ok(());
    }
    let checked_per_condition = plan.word_indices.is_empty()
        || plan.conditions.len() <= BATCH_SCALAR_SYMBOLIC_EVAL_THRESHOLD;
    if !checked_per_condition {
        check_batch_symbolic_plan_assigned(plan, runtime)?;
    }
    if nwords == 1 {
        let mut out0 = out[0]
            ^ if plan.constant {
                batch_live_word_mask(runtime, 0)
            } else {
                0
            };
        if runtime.batch_words == 1 {
            for &condition in &plan.conditions {
                if checked_per_condition {
                    check_condition_assigned(runtime, condition)?;
                }
                out0 ^= runtime.value_words[(condition - 1) as usize];
            }
        } else {
            for &condition in &plan.conditions {
                if checked_per_condition {
                    check_condition_assigned(runtime, condition)?;
                }
                out0 ^= runtime.value_words[batch_condition_offset(runtime, condition, 0)];
            }
        }
        out[0] = out0 & batch_live_word_mask(runtime, 0);
        out[1..].fill(0);
        return Ok(());
    }
    if nwords == 2 {
        let mut out0 = out[0]
            ^ if plan.constant {
                batch_live_word_mask(runtime, 0)
            } else {
                0
            };
        let mut out1 = out[1]
            ^ if plan.constant {
                batch_live_word_mask(runtime, 1)
            } else {
                0
            };
        for &condition in &plan.conditions {
            if checked_per_condition {
                check_condition_assigned(runtime, condition)?;
            }
            let base = batch_condition_offset(runtime, condition, 0);
            out0 ^= runtime.value_words[base];
            out1 ^= runtime.value_words[base + 1];
        }
        out[0] = out0 & batch_live_word_mask(runtime, 0);
        out[1] = out1 & batch_live_word_mask(runtime, 1);
        out[2..].fill(0);
        return Ok(());
    }
    if plan.constant {
        for word in 0..nwords {
            out[word] ^= batch_live_word_mask(runtime, word);
        }
    }
    for &condition in &plan.conditions {
        if checked_per_condition {
            check_condition_assigned(runtime, condition)?;
        }
        let base = batch_condition_offset(runtime, condition, 0);
        for word in 0..nwords {
            out[word] ^= runtime.value_words[base + word];
        }
    }
    mask_batch_bits(out, runtime);
    out[nwords..].fill(0);
    Ok(())
}

// ==============================================================================
// Records and detectors
// ==============================================================================

/// Growing the record count changes the column stride, so every existing
/// column must be recopied — a plain resize would interleave old columns.
fn ensure_batch_measurement_storage(
    runtime: &mut BatchFactoredExecutorState,
    record: i32,
) -> Result<()> {
    if record as usize <= runtime.nrecords {
        return Ok(());
    }
    let mut next = vec![0u64; record as usize * runtime.batch_words];
    for old_record in 1..=runtime.nrecords as i32 {
        let old_base = batch_record_offset(runtime, old_record, 0);
        let new_base = (old_record - 1) as usize * runtime.batch_words;
        next[new_base..new_base + runtime.batch_words]
            .copy_from_slice(&runtime.measurement_words[old_base..old_base + runtime.batch_words]);
    }
    runtime.nrecords = record as usize;
    runtime.measurement_words = next;
    Ok(())
}

fn ensure_batch_detector_storage(
    runtime: &mut BatchFactoredExecutorState,
    detector: i32,
) -> Result<()> {
    if detector as usize <= runtime.ndetectors {
        return Ok(());
    }
    let mut next = vec![0u64; detector as usize * runtime.batch_words];
    for old_detector in 1..=runtime.ndetectors as i32 {
        let old_base = batch_detector_offset(runtime, old_detector, 0);
        let new_base = (old_detector - 1) as usize * runtime.batch_words;
        next[new_base..new_base + runtime.batch_words]
            .copy_from_slice(&runtime.detector_words[old_base..old_base + runtime.batch_words]);
    }
    runtime.ndetectors = detector as usize;
    runtime.detector_words = next;
    Ok(())
}

pub(crate) fn write_batch_measurement_record(
    runtime: &mut BatchFactoredExecutorState,
    record: Option<i32>,
    outcome_bits: &[u64],
    record_condition: Option<i32>,
) -> Result<()> {
    let nwords = runtime_batch_word_count(runtime);
    if let Some(record) = record {
        if record <= 0 {
            return Err(TicitError::new("measurement record id must be positive"));
        }
        ensure_batch_measurement_storage(runtime, record)?;
        let base = batch_record_offset(runtime, record, 0);
        if nwords == runtime.batch_words && runtime.active_shots & 63 == 0 {
            runtime.measurement_words[base..base + nwords].copy_from_slice(&outcome_bits[..nwords]);
        } else {
            for word in 0..nwords {
                runtime.measurement_words[base + word] =
                    outcome_bits[word] & batch_live_word_mask(runtime, word);
            }
            for word in nwords..runtime.batch_words {
                runtime.measurement_words[base + word] = 0;
            }
        }
    }
    assign_batch_symbol_opt(runtime, record_condition, outcome_bits)
}

/// Fast path: an outcome plan that is exactly the branch condition (possibly
/// negated) can reuse the branch bits sitting in the caller's buffer instead
/// of re-evaluating the plan. Returns whether it fired.
pub(crate) fn write_direct_branch_measurement_record(
    runtime: &mut BatchFactoredExecutorState,
    branch_bits: &mut [u64],
    branch_condition: i32,
    outcome_plan: &SymbolicBoolEvaluationPlan,
    record: Option<i32>,
    record_condition: Option<i32>,
) -> Result<bool> {
    if outcome_plan.conditions.len() != 1 || outcome_plan.conditions[0] != branch_condition {
        return Ok(false);
    }
    if outcome_plan.constant {
        invert_batch_bits(branch_bits, runtime);
    }
    write_batch_measurement_record(runtime, record, branch_bits, record_condition)?;
    Ok(true)
}

pub(crate) fn write_batch_detector_record(
    runtime: &mut BatchFactoredExecutorState,
    detector: i32,
    outcome_bits: &[u64],
) -> Result<()> {
    let nwords = runtime_batch_word_count(runtime);
    if detector <= 0 {
        return Err(TicitError::new("detector id must be positive"));
    }
    if !runtime.store_detector_records {
        for word in 0..nwords {
            runtime.detector_any_words[word] |=
                outcome_bits[word] & batch_live_word_mask(runtime, word);
        }
        return Ok(());
    }
    ensure_batch_detector_storage(runtime, detector)?;
    let base = batch_detector_offset(runtime, detector, 0);
    if nwords == runtime.batch_words && runtime.active_shots & 63 == 0 {
        runtime.detector_words[base..base + nwords].copy_from_slice(&outcome_bits[..nwords]);
        for word in 0..nwords {
            runtime.detector_any_words[word] |= outcome_bits[word];
        }
    } else {
        for word in 0..nwords {
            let bits = outcome_bits[word] & batch_live_word_mask(runtime, word);
            runtime.detector_words[base + word] = bits;
            runtime.detector_any_words[word] |= bits;
        }
        for word in nwords..runtime.batch_words {
            runtime.detector_words[base + word] = 0;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::{SymbolicBool, SymbolicBoolEvaluationPlan};

    #[test]
    fn residual_xor_fast_shapes_match_wordwise_parity() {
        let program = FactoredInstructionProgram {
            nsymbols: 2,
            ..Default::default()
        };
        let plan = SymbolicBoolEvaluationPlan::new(&SymbolicBool::new(true, vec![1, 2]));
        for (capacity, shots) in [(64, 37), (128, 37), (128, 128)] {
            let mut runtime = BatchFactoredExecutorState::new(&program, capacity, 1)
                .expect("batch runtime builds");
            runtime.active_shots = shots;
            let first = [0x0f0f_0f0f_0f0f_0f0f, 0xaaaa_aaaa_aaaa_aaaa];
            let second = [0x3333_3333_3333_3333, 0x5555_5555_5555_5555];
            assign_batch_symbol(&mut runtime, 1, &first).expect("first symbol assigns");
            assign_batch_symbol(&mut runtime, 2, &second).expect("second symbol assigns");

            let mut actual = vec![0x0123_4567_89ab_cdef; runtime.batch_words];
            let mut expected = actual.clone();
            let nwords = runtime_batch_word_count(&runtime);
            for word in 0..nwords {
                expected[word] = (expected[word] ^ first[word] ^ second[word])
                    ^ batch_live_word_mask(&runtime, word);
                expected[word] &= batch_live_word_mask(&runtime, word);
            }
            expected[nwords..].fill(0);

            xor_symbolic_bool_batch_into(&mut actual, &plan, &runtime)
                .expect("residual expression evaluates");
            assert_eq!(actual, expected, "capacity={capacity}, shots={shots}");
        }
    }
}
