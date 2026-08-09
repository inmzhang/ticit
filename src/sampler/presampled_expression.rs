//! Evaluation of the exogenous half of every instruction's symbolic
//! expression.
//!
//! Each instruction's sign/outcome plan splits into an exogenous part (fully
//! determined by the presampled noise table) and a residual part (measurement
//! branches, known only at execution time). The exogenous partials are
//! interned — identical partials across instructions are computed once per
//! block — and evaluated through a parent-delta chain: each block expression
//! copies its cheapest predecessor and XORs only the symmetric difference.

use crate::bits::{symbol_bit_mask, symbol_word_count, symbol_word_index};
use crate::errors::{Result, TicitError};
use crate::exogenous::PackedPresampledExogenous;
use crate::factored::{FactoredInstruction, FactoredInstructionProgram};
use crate::symbolic::{SymbolicBool, SymbolicBoolEvaluationPlan};

#[derive(Clone, Debug, Default)]
pub struct PresampledExpression {
    pub constant: bool,
    pub block_expression_index: usize,
    /// Index of the earlier block expression this one deltas from, if any.
    pub parent_block_expression_index: Option<usize>,
    pub parent_delta_constant: bool,
    pub residual_plan: SymbolicBoolEvaluationPlan,
    pub exogenous_conditions: Vec<i32>,
    pub parent_delta_exogenous_conditions: Vec<i32>,
}

#[derive(Clone, Debug, Default)]
pub struct PresampledExpressionPlan {
    pub instruction_expressions: Vec<PresampledExpression>,
    pub block_expressions: Vec<PresampledExpression>,
    pub block_expression_last_use_by_index: Vec<i32>,
}

/// One evaluated block: `expression_words[expr * shot_words + shot_word]`.
#[derive(Clone, Debug, Default)]
pub struct PresampledExpressionBlock {
    pub nshots: usize,
    pub shot_words: usize,
    pub expression_words: Vec<u64>,
}

fn low_bits_mask(nbits: i64) -> u64 {
    if nbits <= 0 {
        return 0;
    }
    if nbits >= 64 {
        return u64::MAX;
    }
    (1u64 << nbits) - 1
}

fn live_shot_word_mask(shots: usize, word: usize) -> u64 {
    low_bits_mask(shots as i64 - ((word as i64) << 6))
}

fn split_presampled_expression(
    source: &SymbolicBoolEvaluationPlan,
    exogenous_assigned_words: &[u64],
) -> PresampledExpression {
    let mut out = PresampledExpression {
        constant: source.constant,
        ..PresampledExpression::default()
    };
    let mut residual_conditions = Vec::with_capacity(source.conditions.len());
    for &condition in &source.conditions {
        let word = symbol_word_index(condition);
        let exogenous = word < exogenous_assigned_words.len()
            && (exogenous_assigned_words[word] & symbol_bit_mask(condition)) != 0;
        if exogenous {
            out.exogenous_conditions.push(condition);
        } else {
            residual_conditions.push(condition);
        }
    }
    out.residual_plan =
        SymbolicBoolEvaluationPlan::new(&SymbolicBool::new(false, residual_conditions));
    out
}

/// Which plan an instruction evaluates in bulk, if any. A detector that reads
/// measurement records (or has a constant outcome) stays on the runtime path.
fn instruction_expression_plan(
    instruction: &FactoredInstruction,
) -> Option<&SymbolicBoolEvaluationPlan> {
    match instruction {
        FactoredInstruction::ApplyPrecomputedActivePauliRotation(inst) => Some(&inst.sign_plan),
        FactoredInstruction::PromoteDormantRotation(inst) => Some(&inst.sign_plan),
        FactoredInstruction::RecordMeasurement(inst) => Some(&inst.outcome_plan),
        FactoredInstruction::RecordDetector(inst) => {
            if !inst.records.is_empty() || inst.outcome.conditions.is_empty() {
                None
            } else {
                Some(&inst.outcome_plan)
            }
        }
        FactoredInstruction::MeasurePrecomputedActivePauli(inst) => Some(&inst.outcome_plan),
        FactoredInstruction::IntroduceDormantMeasurementBranch(inst) => Some(&inst.outcome_plan),
    }
}

/// First-encounter interning of `(constant, exogenous_conditions)` partials.
/// The map only accelerates lookup; indices are still assigned in encounter
/// order, so the resulting block list is identical to a linear-scan intern.
#[derive(Default)]
struct ExogenousPartialInterner {
    index_by_partial: std::collections::HashMap<(bool, Vec<i32>), usize>,
}

impl ExogenousPartialInterner {
    fn intern(
        &mut self,
        block_expressions: &mut Vec<PresampledExpression>,
        expression: &PresampledExpression,
    ) -> usize {
        let key = (expression.constant, expression.exogenous_conditions.clone());
        *self.index_by_partial.entry(key).or_insert_with(|| {
            block_expressions.push(PresampledExpression {
                constant: expression.constant,
                exogenous_conditions: expression.exogenous_conditions.clone(),
                ..PresampledExpression::default()
            });
            block_expressions.len() - 1
        })
    }
}

fn symmetric_difference_conditions(lhs: &[i32], rhs: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(lhs.len() + rhs.len());
    let mut lit = lhs.iter().peekable();
    let mut rit = rhs.iter().peekable();
    loop {
        match (lit.peek(), rit.peek()) {
            (None, None) => return out,
            (Some(&&l), None) => {
                out.push(l);
                lit.next();
            }
            (None, Some(&&r)) => {
                out.push(r);
                rit.next();
            }
            (Some(&&l), Some(&&r)) => {
                if l < r {
                    out.push(l);
                    lit.next();
                } else if r < l {
                    out.push(r);
                    rit.next();
                } else {
                    lit.next();
                    rit.next();
                }
            }
        }
    }
}

/// Size of the `parent -> child` delta (symmetric difference plus a constant
/// flip), or `None` as soon as it provably reaches `bound`.
fn bounded_parent_delta_cost(
    parent: &[i32],
    child: &[i32],
    delta_constant: bool,
    bound: usize,
) -> Option<usize> {
    let mut cost = usize::from(delta_constant);
    let mut lit = parent.iter().peekable();
    let mut rit = child.iter().peekable();
    loop {
        if cost >= bound {
            return None;
        }
        match (lit.peek(), rit.peek()) {
            (None, None) => return Some(cost),
            (Some(_), None) => {
                cost += 1;
                lit.next();
            }
            (None, Some(_)) => {
                cost += 1;
                rit.next();
            }
            (Some(&&l), Some(&&r)) => {
                if l < r {
                    cost += 1;
                    lit.next();
                } else if r < l {
                    cost += 1;
                    rit.next();
                } else {
                    lit.next();
                    rit.next();
                }
            }
        }
    }
}

/// Greedy parent choice: each block expression deltas from whichever earlier
/// one minimizes the XOR work (symmetric difference plus a constant flip).
///
/// Scanning every earlier expression is quadratic and dominated planning on
/// large-noise circuits, so only genuine contenders are examined: a parent
/// whose delta undercuts the root cost must share a condition with the child
/// (`|parent| - 2*shared + delta_const < child_const <= 1` forces
/// `shared >= 1` for nonempty parents) or be the interned `(true, [])`
/// partial for a constant-true child. Candidates are visited in ascending
/// index order with the same strict-improvement rule, so the selected parents
/// are identical to the full scan's; a test pins that equivalence.
fn prepare_block_expression_parent_deltas(block_expressions: &mut [PresampledExpression]) {
    let mut expressions_by_condition: std::collections::HashMap<i32, Vec<usize>> =
        std::collections::HashMap::new();
    let mut empty_true_index: Option<usize> = None;
    let mut candidates: Vec<usize> = Vec::new();
    for expression_index in 0..block_expressions.len() {
        let (earlier, rest) = block_expressions.split_at_mut(expression_index);
        let expression = &mut rest[0];
        expression.parent_block_expression_index = None;
        expression.parent_delta_constant = expression.constant;
        expression.parent_delta_exogenous_conditions = expression.exogenous_conditions.clone();

        candidates.clear();
        for condition in &expression.exogenous_conditions {
            if let Some(sharing) = expressions_by_condition.get(condition) {
                candidates.extend_from_slice(sharing);
            }
        }
        if expression.constant
            && let Some(index) = empty_true_index
        {
            candidates.push(index);
        }
        candidates.sort_unstable();
        candidates.dedup();

        let mut best_cost =
            expression.exogenous_conditions.len() + usize::from(expression.constant);
        let mut best: Option<(usize, bool)> = None;
        for &parent_index in &candidates {
            let parent = &earlier[parent_index];
            let delta_constant = parent.constant != expression.constant;
            if let Some(cost) = bounded_parent_delta_cost(
                &parent.exogenous_conditions,
                &expression.exogenous_conditions,
                delta_constant,
                best_cost,
            ) {
                best_cost = cost;
                best = Some((parent_index, delta_constant));
            }
        }
        if let Some((parent_index, delta_constant)) = best {
            expression.parent_block_expression_index = Some(parent_index);
            expression.parent_delta_constant = delta_constant;
            expression.parent_delta_exogenous_conditions = symmetric_difference_conditions(
                &earlier[parent_index].exogenous_conditions,
                &expression.exogenous_conditions,
            );
        }

        if expression.exogenous_conditions.is_empty() {
            if expression.constant {
                empty_true_index.get_or_insert(expression_index);
            }
        } else {
            for &condition in &expression.exogenous_conditions {
                expressions_by_condition
                    .entry(condition)
                    .or_default()
                    .push(expression_index);
            }
        }
    }
}

pub fn prepare_presampled_expression_plan(
    out: &mut PresampledExpressionPlan,
    program: &FactoredInstructionProgram,
    samples: &PackedPresampledExogenous,
) -> Result<()> {
    if samples.nsymbols != program.nsymbols
        || samples.exogenous_assigned_words.len() != symbol_word_count(program.nsymbols)
    {
        return Err(TicitError::new(
            "packed presampled exogenous storage was not prepared for this program",
        ));
    }
    prepare_presampled_expression_plan_from_words(out, program, &samples.exogenous_assigned_words);
    Ok(())
}

pub fn prepare_presampled_expression_plan_from_words(
    out: &mut PresampledExpressionPlan,
    program: &FactoredInstructionProgram,
    exogenous_assigned_words: &[u64],
) {
    out.instruction_expressions.clear();
    out.block_expressions.clear();
    out.block_expression_last_use_by_index.clear();
    out.instruction_expressions
        .reserve(program.instructions.len());
    let mut interner = ExogenousPartialInterner::default();
    for (instruction_index, instruction) in program.instructions.iter().enumerate() {
        let mut expression = match instruction_expression_plan(instruction) {
            None => PresampledExpression::default(),
            Some(source) => split_presampled_expression(source, exogenous_assigned_words),
        };
        expression.block_expression_index =
            interner.intern(&mut out.block_expressions, &expression);
        let block_index = expression.block_expression_index;
        if out.block_expression_last_use_by_index.len() < out.block_expressions.len() {
            out.block_expression_last_use_by_index
                .resize(out.block_expressions.len(), -1);
        }
        out.block_expression_last_use_by_index[block_index] = instruction_index as i32;
        out.instruction_expressions.push(expression);
    }
    prepare_block_expression_parent_deltas(&mut out.block_expressions);
}

pub fn presampled_expression_block_offset(
    block: &PresampledExpressionBlock,
    expression_index: usize,
    shot_word: usize,
) -> usize {
    expression_index * block.shot_words + shot_word
}

pub fn evaluate_presampled_expression_block(
    out: &mut PresampledExpressionBlock,
    plan: &PresampledExpressionPlan,
    samples: &PackedPresampledExogenous,
) -> Result<()> {
    if samples.value_words.len() != samples.nsymbols * samples.shot_words
        || samples.sparse_hit_offsets.len() != samples.nsymbols + 1
    {
        return Err(TicitError::new(
            "packed presampled exogenous values are not initialized",
        ));
    }
    out.nshots = samples.nshots;
    out.shot_words = samples.shot_words;
    let nexpressions = plan.block_expressions.len();
    let total_words = nexpressions * out.shot_words;
    if out.expression_words.len() != total_words {
        out.expression_words.resize(total_words, 0);
    }
    for expression_index in 0..nexpressions {
        let expression = &plan.block_expressions[expression_index];
        let expression_base = presampled_expression_block_offset(out, expression_index, 0);
        let (parent, constant, conditions) = match expression.parent_block_expression_index {
            Some(parent_index) => {
                if parent_index >= expression_index {
                    return Err(TicitError::new(
                        "presampled expression parent must be earlier in the block",
                    ));
                }
                (
                    Some(presampled_expression_block_offset(out, parent_index, 0)),
                    expression.parent_delta_constant,
                    &expression.parent_delta_exogenous_conditions,
                )
            }
            None => (None, expression.constant, &expression.exogenous_conditions),
        };
        for &condition in conditions {
            if condition <= 0 || condition as usize > samples.nsymbols {
                return Err(TicitError::new(
                    "presampled expression references an out-of-range exogenous condition",
                ));
            }
        }
        let (head, tail) = out.expression_words.split_at_mut(expression_base);
        let destination = &mut tail[..out.shot_words];
        let parent = parent.map(|parent_base| &head[parent_base..parent_base + out.shot_words]);
        accumulate_expression_row(destination, parent, constant, conditions, samples);
    }
    Ok(())
}

/// Writes one block-expression row in a single pass:
/// `destination = parent(or 0) ^ (constant ? live mask : 0) ^ XOR of rows`.
///
/// Sparse conditions apply their few hit bits directly; dense condition rows
/// XOR in whole words. The destination row stays L1-resident throughout, and
/// XOR is associative and commutative over identical words, so the result is
/// bit-identical to any other accumulation order.
#[inline(never)]
fn accumulate_expression_row(
    destination: &mut [u64],
    parent: Option<&[u64]>,
    constant: bool,
    conditions: &[i32],
    samples: &PackedPresampledExogenous,
) {
    match parent {
        Some(parent) => destination.copy_from_slice(parent),
        None => destination.fill(0),
    }
    if constant {
        for (word, slot) in destination.iter_mut().enumerate() {
            *slot ^= live_shot_word_mask(samples.nshots, word);
        }
    }
    for &condition in conditions {
        if samples.is_sparse_condition(condition) {
            for &shot in samples.sparse_condition_hits(condition) {
                destination[(shot >> 6) as usize] ^= 1u64 << (shot & 63);
            }
        } else {
            let row_base = (condition - 1) as usize * samples.shot_words;
            let row = &samples.value_words[row_base..row_base + destination.len()];
            for (slot, word) in destination.iter_mut().zip(row) {
                *slot ^= word;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The original full-scan greedy parent search. The production candidate
    /// search must select byte-identical parents and deltas.
    fn reference_parent_deltas(block_expressions: &mut [PresampledExpression]) {
        for expression_index in 0..block_expressions.len() {
            let (earlier, rest) = block_expressions.split_at_mut(expression_index);
            let expression = &mut rest[0];
            expression.parent_block_expression_index = None;
            expression.parent_delta_constant = expression.constant;
            expression.parent_delta_exogenous_conditions = expression.exogenous_conditions.clone();

            let mut best_cost =
                expression.exogenous_conditions.len() + usize::from(expression.constant);
            for (parent_index, parent) in earlier.iter().enumerate() {
                let delta_conditions = symmetric_difference_conditions(
                    &parent.exogenous_conditions,
                    &expression.exogenous_conditions,
                );
                let delta_constant = parent.constant != expression.constant;
                let cost = delta_conditions.len() + usize::from(delta_constant);
                if cost < best_cost {
                    best_cost = cost;
                    expression.parent_block_expression_index = Some(parent_index);
                    expression.parent_delta_constant = delta_constant;
                    expression.parent_delta_exogenous_conditions = delta_conditions;
                }
            }
        }
    }

    fn pseudo_random_partials(seed: &mut u64, count: usize) -> Vec<PresampledExpression> {
        // Interned inputs never repeat a (constant, conditions) pair, so the
        // generator dedups; sorted condition lists mirror plan invariants.
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        while out.len() < count {
            let bits = crate::random::next_random_u64(seed);
            let constant = bits & 1 != 0;
            let nconditions = ((bits >> 1) % 6) as usize;
            let mut conditions: Vec<i32> = (0..nconditions)
                .map(|_| (crate::random::next_random_u64(seed) % 9 + 1) as i32)
                .collect();
            conditions.sort_unstable();
            conditions.dedup();
            if seen.insert((constant, conditions.clone())) {
                out.push(PresampledExpression {
                    constant,
                    exogenous_conditions: conditions,
                    ..PresampledExpression::default()
                });
            }
        }
        out
    }

    #[test]
    fn candidate_parent_search_matches_full_scan() {
        let mut seed = 20260808;
        for case in 0..200 {
            let count = 1 + case % 40;
            let mut fast = pseudo_random_partials(&mut seed, count);
            let mut reference = fast.clone();
            prepare_block_expression_parent_deltas(&mut fast);
            reference_parent_deltas(&mut reference);
            for (fast, reference) in fast.iter().zip(&reference) {
                assert_eq!(
                    fast.parent_block_expression_index,
                    reference.parent_block_expression_index
                );
                assert_eq!(fast.parent_delta_constant, reference.parent_delta_constant);
                assert_eq!(
                    fast.parent_delta_exogenous_conditions,
                    reference.parent_delta_exogenous_conditions
                );
            }
        }
    }
}
