//! The planner: drains the pending-operation queue into a flat instruction
//! stream.
//!
//! # What the planner is for
//!
//! Everything expensive about simulating one of these circuits is decided here,
//! not at runtime. Three things happen per queued operation:
//!
//! 1. **Basis changes are resolved at plan time.** When a rotation reaches a
//!    dormant qubit, or a measurement retires an active one, the coordinates
//!    have to be rearranged. Rather than permuting the runtime amplitude
//!    vector, the planner builds a Clifford frame and rewrites the *remaining
//!    queued Paulis* through it. The runtime kernels then always act on
//!    coordinate `k - 1` or `k`, and the amplitude vector is touched only by the
//!    promote/measure instructions themselves.
//! 2. **Randomness is named, not drawn.** A measurement mints a fresh condition
//!    symbol for its branch and pushes that symbol's effect through every later
//!    operation, so the sampler evaluates XOR expressions instead of
//!    backtracking.
//! 3. **Those expressions are minimized.** Evaluating a symbolic bool costs one
//!    XOR per distinct 64-bit word it touches, so the planner rewrites
//!    expressions to touch fewer words, using the relations that measurement
//!    records imply.
//!
//! # Cost model
//!
//! [`symbolic_word_cost`] is the primary metric — distinct symbol *words*, not
//! terms — with term count as the tiebreak. Every rewrite below is gated on
//! strictly improving it, which is what keeps the fixpoint loops terminating.

use std::collections::HashMap;
use std::sync::Arc;

use crate::active::{ActivePauliAction, PrecomputedActivePauliMeasurementKernel};
use crate::bits::{normalize_xor_conditions, symbol_word_index};
use crate::component_plan::build_active_component_plan;
use crate::errors::{Result, TicitError};
use crate::factored::{
    ApplyPrecomputedActivePauliRotation, BernoulliSampleGroup, FactoredInstruction,
    FactoredInstructionProgram, IntroduceDormantMeasurementBranch, MeasurePrecomputedActivePauli,
    PendingClassicalRecord, PendingFactoredState, PendingOperation, PendingPauliMeasurement,
    PendingPauliRotation, PromoteDormantRotation, RareCategoricalSampleGroup, RecordMeasurement,
};
use crate::frames::{
    CliffordFrame, DormantState, SymbolicPauliString, coordinates_in_frame, preimage,
};
use crate::pauli::{
    PauliString, measurement_phase_sign, pauli_anticommutes, pauli_body_y_count,
    pauli_squares_to_identity, pauli_x, pauli_z,
};
use crate::pending_optimizer::optimize_pending_operations;
use crate::symbolic::{
    SymbolicBool, SymbolicBoolEvaluationPlan, SymbolicCategoricalDistribution, symbolic_bool,
    xor_bool, xor_bool_constant,
};

/// Below this, an exogenous symbol is sampled by geometric gaps rather than once
/// per shot.
pub const LOW_PROBABILITY_SAMPLE_THRESHOLD: f64 = 0.02;

// ==============================================================================
// Small state helpers
// ==============================================================================

/// Moves the active/dormant split. The dormant classical bits are rebuilt
/// empty: after a basis change the old bits no longer name anything.
fn set_planning_active_count(state: &mut PendingFactoredState, k: usize) -> Result<()> {
    if k > state.n {
        return Err(TicitError::new(
            "active qubit count exceeds total qubit count",
        ));
    }
    state.k = k;
    state.max_k = state.max_k.max(k);
    state.dormant = DormantState::new(state.n - state.k);
    Ok(())
}

fn push_instruction(
    state: &mut PendingFactoredState,
    instruction: FactoredInstruction,
) -> FactoredInstruction {
    state
        .context
        .bump_next_condition(instruction.max_condition());
    state.instructions.push(instruction);
    state
        .instructions
        .last()
        .expect("an instruction was just pushed")
        .clone()
}

/// Allocates the record slot an operation writes, if it writes one.
///
/// An operation with an explicit `record_condition` but no `record` is
/// referenced symbolically by later operations without appearing in the record
/// stream, so it takes no slot.
fn measurement_record(
    state: &mut PendingFactoredState,
    record: Option<i32>,
    record_condition: Option<i32>,
    exp_val: Option<i32>,
) -> Option<i32> {
    if exp_val.is_some() {
        return None;
    }
    match record {
        None if record_condition.is_some() => None,
        None => {
            let next = state.next_record;
            state.next_record += 1;
            Some(next)
        }
        Some(record) => {
            state.next_record = state.next_record.max(record + 1);
            Some(record)
        }
    }
}

/// Restricts a Pauli to the active qubits, in canonical positive form.
///
/// The dropped sign is not lost: callers take it from the *full* Pauli via
/// [`measurement_phase_sign`] before projecting.
fn project_active_body(pauli: &PauliString, k: usize) -> PauliString {
    let mut out = PauliString::new(k);
    let mut y_count = 0;
    for q in 0..k {
        let xbit = pauli.xbit(q);
        let zbit = pauli.zbit(q);
        out.set_xbit(q, xbit);
        out.set_zbit(q, zbit);
        if xbit && zbit {
            y_count += 1;
        }
    }
    out.set_phase(y_count);
    out
}

/// Widens an active-only Pauli back to the full register.
fn embed_active_pauli(n: usize, active_body: &PauliString) -> PauliString {
    let mut out = PauliString::new(n);
    for q in 0..active_body.nqubits {
        out.set_xbit(q, active_body.xbit(q));
        out.set_zbit(q, active_body.zbit(q));
    }
    out.set_phase(active_body.phase_exponent());
    out
}

// ==============================================================================
// Symbolic cost model
// ==============================================================================

/// Number of distinct 64-bit symbol words an expression touches, which is
/// exactly the number of XORs evaluating it costs at runtime.
pub fn symbolic_word_cost(expr: &SymbolicBool) -> usize {
    let mut count = 0;
    let mut previous = 0;
    for &condition in &expr.conditions {
        let word = symbol_word_index(condition);
        if count == 0 || word != previous {
            count += 1;
            previous = word;
        }
    }
    count
}

fn has_lower_sampling_cost(candidate: &SymbolicBool, current: &SymbolicBool) -> bool {
    let candidate_words = symbolic_word_cost(candidate);
    let current_words = symbolic_word_cost(current);
    if candidate_words != current_words {
        return candidate_words < current_words;
    }
    candidate.conditions.len() < current.conditions.len()
}

/// XORs `relation` into `expr`, but only when that strictly lowers the cost.
///
/// The symmetric difference is costed during a single merge scan, so the
/// rewrite is rejected before any allocation happens.
fn reduce_by_relation_once(expr: &SymbolicBool, relation: &SymbolicBool) -> SymbolicBool {
    if relation.conditions.is_empty() && !relation.constant {
        return expr.clone();
    }
    let mut lhs = 0;
    let mut rhs = 0;
    let mut candidate_size = 0;
    let mut candidate_words = 0;
    let mut previous_word = 0;
    let mut have_word = false;
    let mut overlap = false;
    let mut add_candidate_condition = |condition: i32| {
        candidate_size += 1;
        let word = symbol_word_index(condition);
        if !have_word || word != previous_word {
            candidate_words += 1;
            previous_word = word;
            have_word = true;
        }
    };
    while lhs < expr.conditions.len() || rhs < relation.conditions.len() {
        if rhs == relation.conditions.len()
            || (lhs < expr.conditions.len() && expr.conditions[lhs] < relation.conditions[rhs])
        {
            add_candidate_condition(expr.conditions[lhs]);
            lhs += 1;
        } else if lhs == expr.conditions.len() || relation.conditions[rhs] < expr.conditions[lhs] {
            add_candidate_condition(relation.conditions[rhs]);
            rhs += 1;
        } else {
            overlap = true;
            lhs += 1;
            rhs += 1;
        }
    }
    if !overlap {
        return expr.clone();
    }
    let current_words = symbolic_word_cost(expr);
    if candidate_words > current_words
        || (candidate_words == current_words && candidate_size >= expr.conditions.len())
    {
        return expr.clone();
    }
    xor_bool(expr, relation)
}

/// Applies a whole set of XOR relations to fixpoint.
///
/// Single-condition relations are special-cased into `fixed_conditions`: they
/// pin a symbol to a literal, so substituting them is unconditional rather than
/// cost-gated.
#[derive(Debug, Default)]
pub struct SymbolicRelationReducer {
    relations: Vec<SymbolicBool>,
    relation_index: HashMap<i32, Vec<usize>>,
    fixed_conditions: HashMap<i32, bool>,
}

impl SymbolicRelationReducer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, relation: SymbolicBool) {
        if relation.conditions.is_empty() {
            return;
        }
        if relation.conditions.len() == 1 {
            self.fixed_conditions
                .insert(relation.conditions[0], relation.constant);
            return;
        }
        let index = self.relations.len();
        for &condition in &relation.conditions {
            self.relation_index
                .entry(condition)
                .or_default()
                .push(index);
        }
        self.relations.push(relation);
    }

    pub fn reduce(&self, mut expression: SymbolicBool) -> SymbolicBool {
        loop {
            let mut changed = false;
            if !self.fixed_conditions.is_empty() {
                let mut remaining = Vec::with_capacity(expression.conditions.len());
                for &condition in &expression.conditions {
                    match self.fixed_conditions.get(&condition) {
                        Some(&fixed) => {
                            expression.constant ^= fixed;
                            changed = true;
                        }
                        None => remaining.push(condition),
                    }
                }
                expression.conditions = remaining;
            }
            if !expression.conditions.is_empty() && !self.relations.is_empty() {
                let mut candidates = Vec::new();
                for condition in &expression.conditions {
                    if let Some(found) = self.relation_index.get(condition) {
                        candidates.extend_from_slice(found);
                    }
                }
                candidates.sort_unstable();
                candidates.dedup();
                for relation in candidates {
                    let reduced = reduce_by_relation_once(&expression, &self.relations[relation]);
                    if reduced != expression {
                        expression = reduced;
                        changed = true;
                    }
                }
            }
            if !changed {
                return expression;
            }
        }
    }
}

/// The relation a recorded measurement establishes: `record_condition` XOR its
/// outcome is identically zero.
fn measurement_relation(record_condition: i32, outcome: &SymbolicBool) -> SymbolicBool {
    xor_bool(&symbolic_bool(record_condition), outcome)
}

// ==============================================================================
// Pushing a symbolic sign through the queue
// ==============================================================================

/// Builds the transposed occupancy bitset over queued Paulis.
///
/// Only expectation mode needs it: there the queue is consumed by cursor rather
/// than popped, so the same tail gets scanned once per operation, and the
/// bitset turns that into one word-XOR per 64 operations.
fn ensure_pending_operation_blocks(state: &mut PendingFactoredState) {
    if state.pending_operation_blocks_valid {
        return;
    }
    let blocks = state.pending_operations.len().div_ceil(64);
    let n = state.n;
    state.pending_x_operation_blocks = vec![0; blocks * n];
    state.pending_z_operation_blocks = vec![0; blocks * n];
    for (operation, queued) in state.pending_operations.iter().enumerate() {
        let Some(body) = queued.pauli().map(|pauli| &pauli.pauli) else {
            continue;
        };
        let base = (operation >> 6) * n;
        let mask = 1u64 << (operation & 63);
        for (word, (&x_word, &z_word)) in body.x.iter().zip(&body.z).enumerate() {
            let mut x_bits = x_word;
            while x_bits != 0 {
                let q = word * 64 + x_bits.trailing_zeros() as usize;
                if q < n {
                    state.pending_x_operation_blocks[base + q] |= mask;
                }
                x_bits &= x_bits - 1;
            }
            let mut z_bits = z_word;
            while z_bits != 0 {
                let q = word * 64 + z_bits.trailing_zeros() as usize;
                if q < n {
                    state.pending_z_operation_blocks[base + q] |= mask;
                }
                z_bits &= z_bits - 1;
            }
        }
    }
    state.pending_operation_blocks_valid = true;
}

/// [`push_symbolic_pauli_through_pending_from`] via the transposed bitset.
fn push_symbolic_pauli_through_indexed_pending(
    state: &mut PendingFactoredState,
    start: usize,
    pauli: &PauliString,
    sign: &SymbolicBool,
) {
    ensure_pending_operation_blocks(state);
    let PendingFactoredState {
        n,
        pending_operations,
        pending_x_operation_blocks,
        pending_z_operation_blocks,
        ..
    } = state;
    let n = *n;
    let blocks = pending_operations.len().div_ceil(64);
    for block in (start >> 6)..blocks {
        let base = block * n;
        let mut anticommuting = 0u64;
        for (word, (&x_word, &z_word)) in pauli.x.iter().zip(&pauli.z).enumerate() {
            let mut x_bits = x_word;
            while x_bits != 0 {
                let q = word * 64 + x_bits.trailing_zeros() as usize;
                if q < n {
                    anticommuting ^= pending_z_operation_blocks[base + q];
                }
                x_bits &= x_bits - 1;
            }
            let mut z_bits = z_word;
            while z_bits != 0 {
                let q = word * 64 + z_bits.trailing_zeros() as usize;
                if q < n {
                    anticommuting ^= pending_x_operation_blocks[base + q];
                }
                z_bits &= z_bits - 1;
            }
        }
        if block == (start >> 6) && (start & 63) != 0 {
            anticommuting &= !((1u64 << (start & 63)) - 1);
        }
        while anticommuting != 0 {
            let operation = block * 64 + anticommuting.trailing_zeros() as usize;
            if let Some(target) = pending_operations
                .get_mut(operation)
                .and_then(PendingOperation::pauli_mut)
            {
                target.sign = xor_bool(&target.sign, sign);
            }
            anticommuting &= anticommuting - 1;
        }
    }
}

/// XORs `sign` into every queued operation from `first_index_one_based` on whose
/// Pauli anticommutes with `pauli`.
///
/// This is how a freshly minted branch symbol reaches the operations it affects:
/// conjugating the rest of the queue by the (symbolically signed) Pauli
/// `X_pivot` is what makes the sampled branch consistent with everything that
/// comes after it.
fn push_symbolic_pauli_through_pending_from(
    state: &mut PendingFactoredState,
    first_index_one_based: usize,
    pauli: &PauliString,
    sign: &SymbolicBool,
) {
    state.context.bump_next_condition_for(sign);
    let start = if state.has_expectation {
        state.pending_operation_cursor
    } else {
        0
    } + first_index_one_based.saturating_sub(1);
    // In expectation mode queued Paulis were never rewritten by the basis
    // changes, so the pushed Pauli has to be pulled back through the deferred
    // frame instead.
    let stored_pauli = if state.has_expectation && state.pending_frame_active {
        Some(preimage(&state.pending_frame, pauli))
    } else {
        None
    };
    let pending_pauli = stored_pauli.as_ref().unwrap_or(pauli);
    if state.has_expectation {
        push_symbolic_pauli_through_indexed_pending(state, start, pending_pauli, sign);
        return;
    }

    // A single-X Pauli anticommutes with exactly those operations carrying a Z
    // on that qubit, which is one bit test instead of a full symplectic product.
    let mut single_x_qubit = None;
    let mut single_x = true;
    for (word, (&x_word, &z_word)) in pending_pauli.x.iter().zip(&pending_pauli.z).enumerate() {
        if z_word != 0 || (x_word != 0 && (x_word & (x_word - 1)) != 0) {
            single_x = false;
            break;
        }
        if x_word != 0 {
            if single_x_qubit.is_some() {
                single_x = false;
                break;
            }
            single_x_qubit = Some(word * 64 + x_word.trailing_zeros() as usize);
        }
    }
    if !single_x {
        single_x_qubit = None;
    }

    for queued in state.pending_operations.iter_mut().skip(start) {
        let Some(target) = queued.pauli_mut() else {
            continue;
        };
        let anticommutes = match single_x_qubit {
            Some(q) => (target.pauli.z[q >> 6] >> (q & 63)) & 1 != 0,
            None => pauli_anticommutes(pending_pauli, &target.pauli),
        };
        if anticommutes {
            target.sign = xor_bool(&target.sign, sign);
        }
    }
}

// ==============================================================================
// Symbolic minimization against measurement relations
// ==============================================================================

/// Expands recorded substitutions to fixpoint.
fn substitute_pending_symbols(
    expression: &mut SymbolicBool,
    substitutions: &HashMap<i32, SymbolicBool>,
) {
    if expression.conditions.is_empty() || substitutions.is_empty() {
        return;
    }
    let mut expanded = expression.conditions.clone();
    loop {
        let mut changed = false;
        let mut next = Vec::with_capacity(expanded.len());
        for condition in expanded {
            match substitutions.get(&condition) {
                Some(replacement) => {
                    changed = true;
                    expression.constant ^= replacement.constant;
                    next.extend_from_slice(&replacement.conditions);
                }
                None => next.push(condition),
            }
        }
        normalize_xor_conditions(&mut next);
        expanded = next;
        if !changed {
            break;
        }
    }
    expression.conditions = expanded;
}

/// Substitutes, then applies any relation that provably lowers the word cost.
fn reduce_pending_symbolic_bool(expression: &mut SymbolicBool, state: &PendingFactoredState) {
    substitute_pending_symbols(expression, &state.pending_substitutions);
    if expression.conditions.is_empty() || state.pending_relations.is_empty() {
        return;
    }
    // Sorted but *not* deduped: a run of equal indices counts how many of the
    // expression's conditions the relation covers.
    let mut candidates = Vec::new();
    for condition in &expression.conditions {
        if let Some(found) = state.pending_relation_index.get(condition) {
            candidates.extend_from_slice(found);
        }
    }
    candidates.sort_unstable();
    let mut expression_words: Vec<usize> = Vec::new();
    for &condition in &expression.conditions {
        let word = symbol_word_index(condition);
        if expression_words.last() != Some(&word) {
            expression_words.push(word);
        }
    }

    let mut start = 0;
    while start < candidates.len() {
        let mut end = start + 1;
        while end < candidates.len() && candidates[end] == candidates[start] {
            end += 1;
        }
        let relation_index = candidates[start];
        let overlap = end - start;
        let relation = &state.pending_relations[relation_index];
        // Either the relation cancels more than half of its own terms, or it
        // lands mostly in words the expression already touches.
        let mut can_lower_cost = 2 * overlap > relation.conditions.len();
        if !can_lower_cost {
            let relation_words = &state.pending_relation_words[relation_index];
            let mut expr_word = 0;
            let mut shared_words = 0;
            let mut new_words = 0;
            for &word in relation_words {
                while expr_word < expression_words.len() && expression_words[expr_word] < word {
                    expr_word += 1;
                }
                if expr_word < expression_words.len() && expression_words[expr_word] == word {
                    shared_words += 1;
                } else {
                    new_words += 1;
                }
            }
            can_lower_cost = shared_words > new_words;
        }
        if can_lower_cost {
            *expression = reduce_by_relation_once(expression, relation);
        }
        start = end;
    }
}

fn reduce_pending_operation_signs(operation: &mut PendingOperation, state: &PendingFactoredState) {
    match operation {
        PendingOperation::PauliRotation(rotation) => {
            reduce_pending_symbolic_bool(&mut rotation.pauli.sign, state);
        }
        PendingOperation::PauliMeasurement(measurement) => {
            reduce_pending_symbolic_bool(&mut measurement.pauli.sign, state);
        }
        PendingOperation::ClassicalRecord(record) => {
            reduce_pending_symbolic_bool(&mut record.outcome, state);
        }
    }
}

/// Records what a just-emitted measurement tells the planner about its
/// `record_condition`, as either a substitution or a relation.
///
/// A substitution is preferable — it rewrites uses away entirely — but it is
/// only sound when the outcome does not mention the condition itself, and only
/// profitable when the outcome is the cheaper of the two. Otherwise the equality
/// is kept as a relation and applied opportunistically.
///
/// The C++ takes a `first_index_one_based` here and ignores it: the equivalent
/// substitution over the queue tail is deferred rather than applied eagerly, so
/// the tail is not rescanned.
fn reduce_pending_signs_by_measurement_relation(
    state: &mut PendingFactoredState,
    record_condition: Option<i32>,
    outcome: &SymbolicBool,
) {
    let Some(pivot) = record_condition else {
        return;
    };
    let mut reduced_outcome = outcome.clone();
    substitute_pending_symbols(&mut reduced_outcome, &state.pending_substitutions);
    let relation = measurement_relation(pivot, &reduced_outcome);
    state.context.bump_next_condition_for(&relation);

    let self_reference = reduced_outcome.conditions.binary_search(&pivot).is_ok();
    if !self_reference
        && (reduced_outcome.conditions.len() <= 1
            || has_lower_sampling_cost(&reduced_outcome, &symbolic_bool(pivot)))
    {
        state.pending_substitutions.insert(pivot, reduced_outcome);
        return;
    }

    let relation_index = state.pending_relations.len();
    let mut relation_words: Vec<usize> = Vec::new();
    for &condition in &relation.conditions {
        state
            .pending_relation_index
            .entry(condition)
            .or_default()
            .push(relation_index);
        let word = symbol_word_index(condition);
        if relation_words.last() != Some(&word) {
            relation_words.push(word);
        }
    }
    state.pending_relations.push(relation);
    state.pending_relation_words.push(relation_words);
}

// ==============================================================================
// Plan-time tableau frames
// ==============================================================================

/// The dormant qubit a Pauli's X part reaches, highest first.
///
/// Highest wins so that the qubit promoted into the active set lands at the top
/// coordinate, which is the one the kernels' pivot convention expects.
fn highest_dormant_x_qubit(state: &PendingFactoredState, pauli: &PauliString) -> Option<usize> {
    (state.k..state.n)
        .rev()
        .find(|&q| pauli.xbit(q))
        .map(|q| q - state.k)
}

/// The sign a rotation carries, stripped out of its Pauli's phase.
fn rotation_sign_from_pauli(pauli: &SymbolicPauliString) -> Result<SymbolicBool> {
    Ok(xor_bool_constant(
        &pauli.sign,
        measurement_phase_sign(&pauli.pauli)?,
    ))
}

/// Same as [`rotation_sign_from_pauli`]; named for the measurement path, where
/// the stripped sign is the deterministic part of the outcome bit.
fn measurement_base_outcome(pauli: &SymbolicPauliString) -> Result<SymbolicBool> {
    rotation_sign_from_pauli(pauli)
}

fn positive_hermitian_body(mut pauli: PauliString) -> Result<PauliString> {
    if !pauli_squares_to_identity(&pauli) {
        return Err(TicitError::new(
            "Pauli frame row requires a Hermitian Pauli body",
        ));
    }
    pauli.set_phase(pauli_body_y_count(&pauli));
    Ok(pauli)
}

/// Fixes up a tableau row so it commutes with the Pauli being installed.
fn multiply_by_stabilizer_if_anticommutes(
    row: PauliString,
    measured_or_rotated: &PauliString,
    stabilizer: &PauliString,
) -> PauliString {
    if pauli_anticommutes(&row, measured_or_rotated) {
        let mut product = &row * stabilizer;
        product.set_phase(pauli_body_y_count(&product));
        return product;
    }
    row
}

/// Frame for promoting a dormant qubit so a rotation can act on it.
///
/// Installs the rotation's Pauli as `X_{old_k}` and the picked qubit's `Z` as
/// `Z_{old_k}`, then repacks the surviving dormant rows to close the gap. After
/// this the rotation is literally `X` on the new top coordinate.
fn dormant_rotation_promotion_tableau_frame(
    state: &PendingFactoredState,
    rotation_pauli: &PauliString,
    picked_dormant: usize,
) -> Result<CliffordFrame> {
    let old_k = state.k;
    let picked_q = old_k + picked_dormant;
    let stabilizer = pauli_z(state.n, picked_q);
    let promoted_x = positive_hermitian_body(rotation_pauli.clone())?;
    if !pauli_anticommutes(&promoted_x, &stabilizer) {
        return Err(TicitError::new(
            "dormant rotation promotion requires an anti-commuting fixed stabilizer",
        ));
    }

    let mut frame = CliffordFrame::new(state.n);
    for q in 0..old_k {
        let row = frame.zrow(q);
        frame.copy_pauli_to_row(
            row,
            &multiply_by_stabilizer_if_anticommutes(pauli_z(state.n, q), &promoted_x, &stabilizer),
        );
        let row = frame.xrow(q);
        frame.copy_pauli_to_row(
            row,
            &multiply_by_stabilizer_if_anticommutes(pauli_x(state.n, q), &promoted_x, &stabilizer),
        );
    }

    let row = frame.zrow(old_k);
    frame.copy_pauli_to_row(row, &stabilizer);
    let row = frame.xrow(old_k);
    frame.copy_pauli_to_row(row, &promoted_x);

    let mut new_q = old_k + 1;
    for old_q in old_k..state.n {
        if old_q == picked_q {
            continue;
        }
        let row = frame.zrow(new_q);
        frame.copy_pauli_to_row(
            row,
            &multiply_by_stabilizer_if_anticommutes(
                pauli_z(state.n, old_q),
                &promoted_x,
                &stabilizer,
            ),
        );
        let row = frame.xrow(new_q);
        frame.copy_pauli_to_row(
            row,
            &multiply_by_stabilizer_if_anticommutes(
                pauli_x(state.n, old_q),
                &promoted_x,
                &stabilizer,
            ),
        );
        new_q += 1;
    }
    if new_q != state.n {
        return Err(TicitError::new(
            "dormant rotation promotion frame did not repack dormant rows",
        ));
    }
    Ok(frame)
}

/// Frame for measuring a Pauli that reaches a dormant qubit.
///
/// Swaps the measured Pauli into the picked qubit's `Z` row; the measurement
/// then reads a qubit that is unbiased in that basis, so it costs a fair coin
/// and no amplitude work.
fn dormant_measurement_replacement_tableau_frame(
    state: &PendingFactoredState,
    measured_pauli: &PauliString,
    picked_dormant: usize,
) -> Result<CliffordFrame> {
    let picked_q = state.k + picked_dormant;
    let old_stabilizer = pauli_z(state.n, picked_q);
    let new_stabilizer = positive_hermitian_body(measured_pauli.clone())?;
    if !pauli_anticommutes(&new_stabilizer, &old_stabilizer) {
        return Err(TicitError::new(
            "dormant measurement replacement requires an anti-commuting fixed stabilizer",
        ));
    }

    let mut frame = CliffordFrame::new(state.n);
    for q in 0..state.n {
        if q == picked_q {
            continue;
        }
        let row = frame.zrow(q);
        frame.copy_pauli_to_row(
            row,
            &multiply_by_stabilizer_if_anticommutes(
                pauli_z(state.n, q),
                &new_stabilizer,
                &old_stabilizer,
            ),
        );
        let row = frame.xrow(q);
        frame.copy_pauli_to_row(
            row,
            &multiply_by_stabilizer_if_anticommutes(
                pauli_x(state.n, q),
                &new_stabilizer,
                &old_stabilizer,
            ),
        );
    }
    let row = frame.zrow(picked_q);
    frame.copy_pauli_to_row(row, &new_stabilizer);
    let row = frame.xrow(picked_q);
    frame.copy_pauli_to_row(row, &old_stabilizer);
    Ok(frame)
}

/// Frame for retiring the coordinate an active measurement consumes.
///
/// The measured Pauli becomes `Z_{k-1}` and an anticommuting partner becomes
/// `X_{k-1}`, so the sampler's kernel always collapses the top coordinate; the
/// remaining `k - 1` coordinates are relabelled around the dropped pivot.
fn active_measurement_coordinate_frame(
    state: &PendingFactoredState,
    active_body: &PauliString,
    kernel: &PrecomputedActivePauliMeasurementKernel,
) -> Result<CliffordFrame> {
    let mut frame = CliffordFrame::new(state.n);
    let k = state.k;
    let pivot = kernel.pivot;
    let measured = embed_active_pauli(state.n, active_body);
    let fixed_x = if kernel.is_diagonal {
        pauli_x(state.n, pivot)
    } else {
        pauli_z(state.n, pivot)
    };
    let row = frame.xrow(k - 1);
    frame.copy_pauli_to_row(row, &fixed_x);
    let row = frame.zrow(k - 1);
    frame.copy_pauli_to_row(row, &measured);

    let mut new_q = 0;
    for old_q in 0..k {
        if old_q == pivot {
            continue;
        }
        let mut zrow = pauli_z(state.n, old_q);
        let mut xrow = pauli_x(state.n, old_q);
        if kernel.is_diagonal {
            if active_body.zbit(old_q) {
                xrow = &xrow * &pauli_x(state.n, pivot);
            }
        } else {
            if active_body.xbit(old_q) {
                zrow = &zrow * &pauli_z(state.n, pivot);
            }
            if active_body.zbit(old_q) {
                xrow = &xrow * &pauli_z(state.n, pivot);
            }
        }
        let row = frame.xrow(new_q);
        frame.copy_pauli_to_row(row, &xrow);
        let row = frame.zrow(new_q);
        frame.copy_pauli_to_row(row, &zrow);
        new_q += 1;
    }
    if new_q != k - 1 {
        return Err(TicitError::new(
            "active measurement tableau dropped the wrong number of qubits",
        ));
    }
    Ok(frame)
}

fn transform_operation_by_frame(
    operation: &PendingOperation,
    frame: &CliffordFrame,
) -> Result<PendingOperation> {
    Ok(match operation {
        PendingOperation::PauliRotation(rotation) => PendingPauliRotation {
            kernel_angle: rotation.kernel_angle,
            pauli: SymbolicPauliString::with_sign(
                coordinates_in_frame(frame, &rotation.pauli.pauli)?,
                rotation.pauli.sign.clone(),
            ),
        }
        .into(),
        PendingOperation::PauliMeasurement(measurement) => PendingPauliMeasurement {
            pauli: SymbolicPauliString::with_sign(
                coordinates_in_frame(frame, &measurement.pauli.pauli)?,
                measurement.pauli.sign.clone(),
            ),
            ..measurement.clone()
        }
        .into(),
        PendingOperation::ClassicalRecord(_) => operation.clone(),
    })
}

/// Applies a basis change to everything still queued.
///
/// In expectation mode the queue may not be rewritten — a probe has to observe
/// the coordinates it was written in — so the frame is composed into
/// `pending_frame` and applied lazily as operations are pulled off.
fn transform_pending_operations_by_frame(
    state: &mut PendingFactoredState,
    frame: &CliffordFrame,
) -> Result<()> {
    if state.has_expectation {
        let rows = frame
            .rows
            .iter()
            .map(|row| preimage(&state.pending_frame, row))
            .collect();
        let mut composed = CliffordFrame::new(frame.nqubits);
        composed.rows = rows;
        composed.invalidate_support_cache();
        state.pending_frame = composed;
        state.pending_frame_active = true;
        return Ok(());
    }
    for index in 0..state.pending_operations.len() {
        state.pending_operations[index] =
            transform_operation_by_frame(&state.pending_operations[index], frame)?;
    }
    Ok(())
}

// ==============================================================================
// Per-operation planning
// ==============================================================================

fn active_rotation_instruction(
    active_body: &PauliString,
    kernel_angle: f64,
    sign: SymbolicBool,
) -> Result<ApplyPrecomputedActivePauliRotation> {
    let action = ActivePauliAction::new(active_body)?;
    Ok(ApplyPrecomputedActivePauliRotation {
        rotation_kernel: crate::active::PrecomputedActivePauliRotationKernel::new(
            &action,
            kernel_angle,
        )?,
        sign_plan: SymbolicBoolEvaluationPlan::new(&sign),
        sign,
    })
}

/// A rotation whose Pauli stays inside the active set: one dense kernel call.
fn process_diagonal_dormant_rotation(
    state: &mut PendingFactoredState,
    current: &PendingPauliRotation,
) -> Result<Option<FactoredInstruction>> {
    let active_body = project_active_body(&current.pauli.pauli, state.k);
    let sign = rotation_sign_from_pauli(&current.pauli)?;
    if !active_body.has_nonidentity_body() {
        // A rotation about the identity is a global phase.
        return Ok(None);
    }
    if !pauli_squares_to_identity(&active_body) {
        return Err(TicitError::new(
            "active rotation Pauli must square to identity",
        ));
    }
    let instruction = active_rotation_instruction(&active_body, current.kernel_angle, sign)?.into();
    Ok(Some(push_instruction(state, instruction)))
}

/// A rotation reaching a dormant qubit: promote it, doubling the state vector.
fn process_nondiagonal_dormant_rotation(
    state: &mut PendingFactoredState,
    current: &PendingPauliRotation,
    picked_dormant: usize,
) -> Result<Option<FactoredInstruction>> {
    let old_k = state.k;
    let frame =
        dormant_rotation_promotion_tableau_frame(state, &current.pauli.pauli, picked_dormant)?;
    let current = transform_operation_by_frame(&current.clone().into(), &frame)?;
    transform_pending_operations_by_frame(state, &frame)?;
    let PendingOperation::PauliRotation(current) = current else {
        unreachable!("a rotation stays a rotation under a basis change");
    };
    if !current.pauli.pauli.same_body(&pauli_x(state.n, old_k)) {
        return Err(TicitError::new(
            "dormant rotation tableau reduction did not expose promoted X",
        ));
    }
    let sign = rotation_sign_from_pauli(&current.pauli)?;
    let instruction = PromoteDormantRotation {
        kernel_angle: current.kernel_angle,
        sign_plan: SymbolicBoolEvaluationPlan::new(&sign),
        sign,
    };
    let pushed = push_instruction(state, instruction.into());
    set_planning_active_count(state, old_k + 1)?;
    Ok(Some(pushed))
}

pub fn process_pending_rotation(
    state: &mut PendingFactoredState,
    rotation: &PendingPauliRotation,
) -> Result<Option<FactoredInstruction>> {
    state
        .context
        .bump_next_condition(rotation.pauli.sign.max_condition());
    match highest_dormant_x_qubit(state, &rotation.pauli.pauli) {
        None => process_diagonal_dormant_rotation(state, rotation),
        Some(picked) => process_nondiagonal_dormant_rotation(state, rotation, picked),
    }
}

/// A measurement whose value is already determined: no sampling, just a record.
fn record_deterministic_measurement(
    state: &mut PendingFactoredState,
    measurement: &PendingPauliMeasurement,
    outcome: SymbolicBool,
) -> Result<Option<FactoredInstruction>> {
    let record = measurement_record(
        state,
        measurement.record,
        measurement.record_condition,
        measurement.exp_val,
    );
    let instruction = RecordMeasurement {
        outcome_plan: SymbolicBoolEvaluationPlan::new(&outcome),
        outcome: outcome.clone(),
        record,
        record_condition: measurement.record_condition,
        exp_val: measurement.exp_val,
    };
    let pushed = push_instruction(state, instruction.into());
    reduce_pending_signs_by_measurement_relation(state, measurement.record_condition, &outcome);
    Ok(Some(pushed))
}

/// A measurement reaching a dormant qubit: a fair coin, no amplitude work.
fn measure_dormant_xy_pauli(
    state: &mut PendingFactoredState,
    current: &PendingPauliMeasurement,
    picked_dormant: usize,
    queued_first: bool,
) -> Result<Option<FactoredInstruction>> {
    if current.exp_val.is_some() {
        // A probe of an unbiased qubit has expectation 0 with no state change.
        let outcome = SymbolicBool::default();
        let instruction = IntroduceDormantMeasurementBranch {
            branch: 0,
            outcome_plan: SymbolicBoolEvaluationPlan::new(&outcome),
            outcome,
            record: None,
            record_condition: None,
            exp_val: current.exp_val,
        };
        return Ok(Some(push_instruction(state, instruction.into())));
    }
    let picked_q = state.k + picked_dormant;
    let frame =
        dormant_measurement_replacement_tableau_frame(state, &current.pauli.pauli, picked_dormant)?;
    let current = transform_operation_by_frame(&current.clone().into(), &frame)?;
    transform_pending_operations_by_frame(state, &frame)?;
    let PendingOperation::PauliMeasurement(current) = current else {
        unreachable!("a measurement stays a measurement under a basis change");
    };
    if !current.pauli.pauli.same_body(&pauli_z(state.n, picked_q)) {
        return Err(TicitError::new(
            "dormant measurement tableau reduction did not expose fixed Z",
        ));
    }
    let base_outcome = measurement_base_outcome(&current.pauli)?;
    let branch = state.context.fresh_condition();
    let branch_bit = symbolic_bool(branch);
    push_symbolic_pauli_through_pending_from(
        state,
        if queued_first { 2 } else { 1 },
        &pauli_x(state.n, picked_q),
        &branch_bit,
    );
    let outcome = xor_bool(&base_outcome, &branch_bit);
    let record = measurement_record(state, current.record, current.record_condition, None);
    let instruction = IntroduceDormantMeasurementBranch {
        branch,
        outcome_plan: SymbolicBoolEvaluationPlan::new(&outcome),
        outcome: outcome.clone(),
        record,
        record_condition: current.record_condition,
        exp_val: None,
    };
    let pushed = push_instruction(state, instruction.into());
    reduce_pending_signs_by_measurement_relation(state, current.record_condition, &outcome);
    Ok(Some(pushed))
}

/// A non-destructive expectation probe of an active Pauli.
fn evaluate_active_pauli(
    state: &mut PendingFactoredState,
    current: &PendingPauliMeasurement,
    active_body: &PauliString,
    base_outcome: SymbolicBool,
) -> Result<Option<FactoredInstruction>> {
    if !pauli_squares_to_identity(active_body) {
        return Err(TicitError::new(
            "active expectation Pauli must square to identity",
        ));
    }
    let kernel = PrecomputedActivePauliMeasurementKernel::from_pauli(active_body)?;
    let instruction = MeasurePrecomputedActivePauli {
        kernel,
        branch: 0,
        outcome_plan: SymbolicBoolEvaluationPlan::new(&base_outcome),
        outcome: base_outcome,
        record: None,
        record_condition: None,
        exp_val: current.exp_val,
    };
    Ok(Some(push_instruction(state, instruction.into())))
}

/// A destructive measurement of an active Pauli: Born-rule sample, project, and
/// retire the pivot coordinate.
fn measure_active_pauli_branches(
    state: &mut PendingFactoredState,
    current: &PendingPauliMeasurement,
    active_body: &PauliString,
    base_outcome: SymbolicBool,
    queued_first: bool,
) -> Result<Option<FactoredInstruction>> {
    if !pauli_squares_to_identity(active_body) {
        return Err(TicitError::new(
            "active measurement Pauli must square to identity",
        ));
    }
    let kernel = PrecomputedActivePauliMeasurementKernel::from_pauli(active_body)?;
    let frame = active_measurement_coordinate_frame(state, active_body, &kernel)?;
    transform_pending_operations_by_frame(state, &frame)?;
    let branch = state.context.fresh_condition();
    let branch_bit = symbolic_bool(branch);
    push_symbolic_pauli_through_pending_from(
        state,
        if queued_first { 2 } else { 1 },
        &pauli_x(state.n, state.k - 1),
        &branch_bit,
    );
    let outcome = xor_bool(&base_outcome, &branch_bit);
    let record = measurement_record(
        state,
        current.record,
        current.record_condition,
        current.exp_val,
    );
    let instruction = MeasurePrecomputedActivePauli {
        kernel,
        branch,
        outcome_plan: SymbolicBoolEvaluationPlan::new(&outcome),
        outcome: outcome.clone(),
        record,
        record_condition: current.record_condition,
        exp_val: None,
    };
    let pushed = push_instruction(state, instruction.into());
    reduce_pending_signs_by_measurement_relation(state, current.record_condition, &outcome);
    set_planning_active_count(state, state.k - 1)?;
    Ok(Some(pushed))
}

pub fn process_pending_measurement(
    state: &mut PendingFactoredState,
    measurement: &PendingPauliMeasurement,
) -> Result<Option<FactoredInstruction>> {
    state.context.bump_next_condition(
        measurement
            .pauli
            .sign
            .max_condition()
            .max(measurement.record_condition.unwrap_or(0)),
    );
    // Whether the operation being planned is still sitting at the head of the
    // queue, in which case sign push-through must skip it.
    let queued_first = state.has_expectation
        || matches!(
            state.pending_operations.first(),
            Some(PendingOperation::PauliMeasurement(front)) if front == measurement
        );
    let active_body = project_active_body(&measurement.pauli.pauli, state.k);
    if let Some(picked) = highest_dormant_x_qubit(state, &measurement.pauli.pauli) {
        return measure_dormant_xy_pauli(state, measurement, picked, queued_first);
    }
    let base_outcome = measurement_base_outcome(&measurement.pauli)?;
    if !active_body.has_nonidentity_body() {
        return record_deterministic_measurement(state, measurement, base_outcome);
    }
    if measurement.exp_val.is_some() {
        return evaluate_active_pauli(state, measurement, &active_body, base_outcome);
    }
    measure_active_pauli_branches(state, measurement, &active_body, base_outcome, queued_first)
}

pub fn process_pending_classical_record(
    state: &mut PendingFactoredState,
    record: &PendingClassicalRecord,
) -> Result<Option<FactoredInstruction>> {
    state.context.bump_next_condition(
        record
            .outcome
            .max_condition()
            .max(record.record_condition.unwrap_or(0)),
    );
    let slot = measurement_record(state, record.record, record.record_condition, None);
    let instruction = RecordMeasurement {
        outcome_plan: SymbolicBoolEvaluationPlan::new(&record.outcome),
        outcome: record.outcome.clone(),
        record: slot,
        record_condition: record.record_condition,
        exp_val: None,
    };
    Ok(Some(push_instruction(state, instruction.into())))
}

pub fn has_pending_operations(state: &PendingFactoredState) -> bool {
    if state.has_expectation {
        state.pending_operation_cursor < state.pending_operations.len()
    } else {
        !state.pending_operations.is_empty()
    }
}

/// Plans exactly one queued operation.
///
/// Checkpoints the instruction index both before and after: the frontend
/// re-anchors detectors on these boundaries, and a detector placed between two
/// operations needs the index on both sides to survive optimization.
pub fn process_next_pending_operation(
    state: &mut PendingFactoredState,
) -> Result<Option<FactoredInstruction>> {
    if !has_pending_operations(state) {
        return Ok(None);
    }
    if state.pending_prefix_instruction_indices.is_empty() {
        state
            .pending_prefix_instruction_indices
            .push(state.instructions.len() as i32);
    }
    let index = if state.has_expectation {
        state.pending_operation_cursor
    } else {
        0
    };
    let mut operation = state.pending_operations[index].clone();
    reduce_pending_operation_signs(&mut operation, state);
    if state.has_expectation && state.pending_frame_active {
        operation = transform_operation_by_frame(&operation, &state.pending_frame)?;
    }

    let start = state.instructions.len();
    let result = match &operation {
        PendingOperation::PauliRotation(rotation) => process_pending_rotation(state, rotation)?,
        PendingOperation::PauliMeasurement(measurement) => {
            process_pending_measurement(state, measurement)?
        }
        PendingOperation::ClassicalRecord(record) => {
            process_pending_classical_record(state, record)?
        }
    };
    if state.has_expectation {
        state.pending_operation_cursor += 1;
    } else {
        state.pending_operations.remove(0);
    }
    state
        .pending_prefix_instruction_indices
        .push(state.instructions.len() as i32);
    if state.instructions.len() == start {
        return Ok(result);
    }
    Ok(state.instructions.last().cloned())
}

fn process_pending_operations_in_place(state: &mut PendingFactoredState) -> Result<()> {
    if !state.pending_operations_optimized
        && state.instructions.is_empty()
        && state.pending_prefix_instruction_indices.is_empty()
    {
        optimize_pending_operations(state, &[])?;
    }
    while has_pending_operations(state) {
        process_next_pending_operation(state)?;
    }
    Ok(())
}

// ==============================================================================
// Whole-program passes
// ==============================================================================

/// Final global minimization: every measurement relation discovered while
/// planning is applied to every later expression, then the evaluation plans are
/// recompiled against the rewritten expressions.
fn reduce_program_symbolic_expressions(instructions: &mut [FactoredInstruction]) {
    let mut reducer = SymbolicRelationReducer::new();
    for instruction in instructions.iter_mut() {
        if let Some(sign) = instruction.sign_mut() {
            *sign = reducer.reduce(std::mem::take(sign));
        }
        if let Some(outcome) = instruction.outcome_mut() {
            *outcome = reducer.reduce(std::mem::take(outcome));
        }
        if let (Some(record_condition), Some(outcome)) =
            (instruction.record_condition(), instruction.outcome())
        {
            let relation = measurement_relation(record_condition, outcome);
            if !relation.conditions.is_empty() || relation.constant {
                reducer.add(relation);
            }
        }
    }
}

// ==============================================================================
// Exogenous sampling plan
// ==============================================================================

struct RareInfo {
    event_probability: f64,
    event_rows: Vec<usize>,
    event_probabilities: Vec<f64>,
}

/// Classifies a categorical distribution as "rare": it has a do-nothing row and
/// the chance of anything else is below the threshold, so the sampler can skip
/// most shots with a geometric gap instead of drawing per shot.
///
/// The threshold test is negated on purpose, so that a NaN probability declines
/// the rare path rather than taking it.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn rare_categorical_sample_info(
    distribution: &SymbolicCategoricalDistribution,
) -> Option<RareInfo> {
    let false_row = distribution.assignments.iter().position(|assignment| {
        (0..distribution.nbits).all(|bit| !crate::bits::packed_bit(assignment, bit))
    })?;
    let event_probability = 1.0 - distribution.probabilities[false_row];
    if !(event_probability < LOW_PROBABILITY_SAMPLE_THRESHOLD) {
        return None;
    }
    let mut info = RareInfo {
        event_probability,
        event_rows: Vec::new(),
        event_probabilities: Vec::new(),
    };
    if event_probability > 0.0 {
        // Row weights are renormalized to be conditional on the event, so the
        // sampler picks a row only once it knows something happened.
        let inv_event = 1.0 / event_probability;
        for (row, &probability) in distribution.probabilities.iter().enumerate() {
            if row == false_row || probability <= 0.0 {
                continue;
            }
            info.event_rows.push(row);
            info.event_probabilities.push(probability * inv_event);
        }
    }
    Some(info)
}

/// Groups by *exact* `f64` equality, matching the C++. Probabilities come from
/// the same arithmetic on both sides, so bitwise equality is the intended test.
fn push_bernoulli_sample_group(
    groups: &mut Vec<BernoulliSampleGroup>,
    probability: f64,
    condition: i32,
) {
    for group in groups.iter_mut() {
        if group.probability == probability {
            group.conditions.push(condition);
            return;
        }
    }
    groups.push(BernoulliSampleGroup {
        probability,
        conditions: vec![condition],
    });
}

fn push_rare_categorical_sample_group(
    groups: &mut Vec<RareCategoricalSampleGroup>,
    distribution: &SymbolicCategoricalDistribution,
    info: &RareInfo,
) {
    for group in groups.iter_mut() {
        if group.event_probability == info.event_probability
            && group.nbits == distribution.nbits
            && group.assignments == distribution.assignments
            && group.probabilities == distribution.probabilities
            && group.event_rows == info.event_rows
            && group.event_probabilities == info.event_probabilities
        {
            group.conditions.push(distribution.conditions.clone());
            return;
        }
    }
    groups.push(RareCategoricalSampleGroup {
        event_probability: info.event_probability,
        nbits: distribution.nbits,
        conditions: vec![distribution.conditions.clone()],
        assignments: distribution.assignments.clone(),
        probabilities: distribution.probabilities.clone(),
        event_rows: info.event_rows.clone(),
        event_probabilities: info.event_probabilities.clone(),
    });
}

// ==============================================================================
// Program construction
// ==============================================================================

impl FactoredInstructionProgram {
    #[cfg(test)]
    pub fn new(
        n: usize,
        initial_k: usize,
        instructions: Vec<FactoredInstruction>,
        max_k: usize,
    ) -> Result<Self> {
        Self::with_context(
            n,
            initial_k,
            instructions,
            max_k,
            crate::symbolic::SymbolicContext::new(),
            Vec::new(),
        )
    }

    /// Builds a program, running the final symbolic-reduction pass and the
    /// exogenous sampling plan.
    pub fn with_context(
        n: usize,
        initial_k: usize,
        mut instructions: Vec<FactoredInstruction>,
        max_k: usize,
        mut context: crate::symbolic::SymbolicContext,
        pending_prefix_instruction_indices: Vec<i32>,
    ) -> Result<Self> {
        if initial_k > n || max_k > n || initial_k > max_k {
            return Err(TicitError::new(
                "invalid factored instruction program dimensions",
            ));
        }
        reduce_program_symbolic_expressions(&mut instructions);

        let mut record_count = 0;
        let mut detector_count = 0;
        let mut exp_val_count = 0;
        for instruction in instructions.iter_mut() {
            instruction.refresh_plan();
            context.bump_next_condition(instruction.max_condition());
            if let Some(record) = instruction.record() {
                record_count = record_count.max(record);
            }
            if let Some(detector) = instruction.detector() {
                detector_count = detector_count.max(detector);
            }
            if let Some(exp_val) = instruction.exp_val() {
                exp_val_count = exp_val_count.max(exp_val + 1);
            }
        }

        let mut program = Self {
            n,
            initial_k,
            max_k,
            instructions,
            pending_prefix_instruction_indices,
            nsymbols: (context.next_condition - 1).max(0) as usize,
            nrecords: record_count.max(0) as usize,
            ndetectors: detector_count.max(0) as usize,
            nexpvals: exp_val_count.max(0) as usize,
            context,
            ..Self::default()
        };

        for distribution in &program.context.categorical_distributions {
            match rare_categorical_sample_info(distribution) {
                Some(info) => push_rare_categorical_sample_group(
                    &mut program.sampled_rare_categorical_groups,
                    distribution,
                    &info,
                ),
                None => program
                    .sampled_categorical_distributions
                    .push(distribution.clone()),
            }
        }
        // Ascending condition order here is a reproducibility contract: it fixes
        // the order the sampler draws these symbols in.
        for (&condition, &probability) in &program.context.bernoulli_probabilities {
            if probability < LOW_PROBABILITY_SAMPLE_THRESHOLD {
                push_bernoulli_sample_group(
                    &mut program.sampled_low_probability_bernoulli_groups,
                    probability,
                    condition,
                );
            } else {
                program.sampled_bernoulli_conditions.push(condition);
                program.sampled_bernoulli_probabilities.push(probability);
            }
        }

        let plan = build_active_component_plan(&program)?;
        program.use_active_components = plan.selected;
        program.active_component_plan = Some(Arc::new(plan));
        Ok(program)
    }
}

/// Freezes a planned state into a program.
pub fn factored_instruction_program(
    state: PendingFactoredState,
) -> Result<FactoredInstructionProgram> {
    FactoredInstructionProgram::with_context(
        state.n,
        state.initial_k,
        state.instructions,
        state.max_k,
        state.context,
        state.pending_prefix_instruction_indices,
    )
}

/// Optimizes, plans, and freezes in one step.
pub fn plan_factored_updates(
    mut state: PendingFactoredState,
) -> Result<FactoredInstructionProgram> {
    process_pending_operations_in_place(&mut state)?;
    factored_instruction_program(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_cost_counts_words_not_terms() {
        assert_eq!(symbolic_word_cost(&SymbolicBool::new(false, vec![])), 0);
        assert_eq!(
            symbolic_word_cost(&SymbolicBool::new(false, vec![1, 2, 64])),
            1
        );
        assert_eq!(
            symbolic_word_cost(&SymbolicBool::new(false, vec![1, 65])),
            2
        );
    }

    #[test]
    fn relations_apply_only_when_they_shrink_the_expression() {
        let expr = SymbolicBool::new(false, vec![1, 2, 3]);
        // Cancels two of three terms: a strict win.
        let shrinking = SymbolicBool::new(false, vec![1, 2]);
        assert_eq!(
            reduce_by_relation_once(&expr, &shrinking),
            SymbolicBool::new(false, vec![3])
        );
        // Same word, more terms: rejected.
        let growing = SymbolicBool::new(false, vec![3, 4, 5]);
        assert_eq!(reduce_by_relation_once(&expr, &growing), expr);
        // No overlap at all: nothing to gain.
        let disjoint = SymbolicBool::new(false, vec![7]);
        assert_eq!(reduce_by_relation_once(&expr, &disjoint), expr);
    }

    #[test]
    fn fixed_conditions_are_substituted_unconditionally() {
        let mut reducer = SymbolicRelationReducer::new();
        reducer.add(SymbolicBool::new(true, vec![4]));
        assert_eq!(
            reducer.reduce(SymbolicBool::new(false, vec![4, 9])),
            SymbolicBool::new(true, vec![9])
        );
    }

    #[test]
    fn substitutions_expand_to_fixpoint() {
        let mut substitutions = HashMap::new();
        substitutions.insert(5, SymbolicBool::new(false, vec![6, 7]));
        substitutions.insert(6, SymbolicBool::new(true, vec![8]));
        let mut expression = SymbolicBool::new(false, vec![5]);
        substitute_pending_symbols(&mut expression, &substitutions);
        assert_eq!(expression, SymbolicBool::new(true, vec![7, 8]));
    }

    #[test]
    fn program_dimensions_are_validated() {
        assert!(FactoredInstructionProgram::new(1, 2, Vec::new(), 2).is_err());
        assert!(FactoredInstructionProgram::new(4, 3, Vec::new(), 2).is_err());
        assert!(FactoredInstructionProgram::new(4, 2, Vec::new(), 3).is_ok());
    }
}

#[cfg(test)]
mod behavior_tests {
    //! Planner structure and sampled end-to-end invariant tests.

    use crate::test_support as common;

    use std::f64::consts::PI;

    use super::*;
    use crate::bits::packed_bit;
    use crate::factored::{
        FrameFactoredState, apply_pauli_measurement, apply_pauli_measurement_signed,
        apply_pauli_rotation, apply_pauli_symbolic,
    };
    use crate::frames::SymbolicPauliString;
    use crate::symbolic::SymbolicContext;
    use common::require_pending_record_conditions_are_causal;

    fn sampled(program: &FactoredInstructionProgram) -> Vec<Vec<u64>> {
        crate::sampler::batch::sample_measurements_batch(program, 64, 0, 17).expect("samples")
    }

    fn record(row: &[u64], id: usize) -> bool {
        packed_bit(row, id - 1)
    }

    // ==============================================================================
    // Measurement-relation rewriting in the program constructor
    // ==============================================================================

    /// A dense outcome expression is rewritten against the relation an earlier
    /// measurement established, collapsing eight terms to two.
    #[test]
    fn a_measurement_relation_reduces_a_dense_suffix_sign() {
        let dense_sign = SymbolicBool::new(false, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let record_condition = 9;
        let branch = 10;
        let later_record_condition = 11;
        let outcome = xor_bool(&dense_sign, &symbolic_bool(branch));

        let program = FactoredInstructionProgram::with_context(
            1,
            0,
            vec![
                IntroduceDormantMeasurementBranch {
                    branch,
                    outcome_plan: SymbolicBoolEvaluationPlan::new(&outcome),
                    outcome: outcome.clone(),
                    record: Some(1),
                    record_condition: Some(record_condition),
                    exp_val: None,
                }
                .into(),
                RecordMeasurement {
                    outcome_plan: SymbolicBoolEvaluationPlan::new(&dense_sign),
                    outcome: dense_sign.clone(),
                    record: Some(2),
                    record_condition: Some(later_record_condition),
                    exp_val: None,
                }
                .into(),
            ],
            0,
            SymbolicContext::with_next_condition(12),
            Vec::new(),
        )
        .expect("valid dimensions");

        let FactoredInstruction::RecordMeasurement(reduced) = &program.instructions[1] else {
            panic!("the second instruction is a record");
        };
        assert_eq!(
            reduced.outcome,
            xor_bool(&symbolic_bool(record_condition), &symbolic_bool(branch)),
            "the dense sign collapses to the record symbol xor the branch symbol"
        );
        assert_eq!(
            reduced.outcome_plan,
            SymbolicBoolEvaluationPlan::new(&reduced.outcome),
            "the evaluation plan is recompiled after the rewrite"
        );
    }

    /// The same relation is *not* applied when it would not shrink the expression.
    #[test]
    fn a_measurement_relation_keeps_an_already_sparse_record_sign() {
        let dense_sign = SymbolicBool::new(false, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let record_condition = 9;
        let branch = 10;
        let outcome = xor_bool(&dense_sign, &symbolic_bool(branch));
        let sparse_record = symbolic_bool(record_condition);

        let program = FactoredInstructionProgram::with_context(
            1,
            0,
            vec![
                IntroduceDormantMeasurementBranch {
                    branch,
                    outcome_plan: SymbolicBoolEvaluationPlan::new(&outcome),
                    outcome,
                    record: Some(1),
                    record_condition: Some(record_condition),
                    exp_val: None,
                }
                .into(),
                RecordMeasurement {
                    outcome_plan: SymbolicBoolEvaluationPlan::new(&sparse_record),
                    outcome: sparse_record.clone(),
                    record: Some(2),
                    record_condition: Some(11),
                    exp_val: None,
                }
                .into(),
            ],
            0,
            SymbolicContext::with_next_condition(12),
            Vec::new(),
        )
        .expect("valid dimensions");

        let FactoredInstruction::RecordMeasurement(reduced) = &program.instructions[1] else {
            panic!("the second instruction is a record");
        };
        assert_eq!(reduced.outcome, sparse_record);
    }

    // ==============================================================================
    // Dormant promotion
    // ==============================================================================

    /// Promotion picks the *highest* dormant qubit carrying an X, and rewrites the
    /// rest of the queue into the new coordinates.
    #[test]
    fn dormant_promotion_uses_the_highest_pivot_and_remaps_the_queue() {
        let mut pending = PendingFactoredState::new(4, 0).expect("k <= n");
        pending.pending_operations.push(
            PendingPauliRotation {
                kernel_angle: 0.25,
                pauli: SymbolicPauliString::new(&pauli_x(4, 0) * &pauli_x(4, 2)),
            }
            .into(),
        );
        pending.pending_operations.push(
            PendingPauliMeasurement {
                pauli: SymbolicPauliString::new(pauli_z(4, 2)),
                ..PendingPauliMeasurement::default()
            }
            .into(),
        );

        process_next_pending_operation(&mut pending).expect("the rotation plans");
        assert_eq!(pending.k, 1, "promotion creates one active qubit");
        let PendingOperation::PauliMeasurement(measurement) = &pending.pending_operations[0] else {
            panic!("the measurement is still queued");
        };
        assert_eq!(
            measurement.pauli.pauli,
            pauli_z(4, 0),
            "the queued measurement is rewritten into the promoted coordinate"
        );
    }

    // ==============================================================================
    // Symbolic minimization while planning
    // ==============================================================================

    /// A Pauli correction conditioned on `record_condition xor outcome` is a reset:
    /// the planner's substitution must recognise the second measurement's outcome as
    /// identically false rather than carrying nine symbols into the sampler.
    #[test]
    fn a_measurement_record_substitution_cancels_a_reset() {
        let mut state = FrameFactoredState::new(1, 0).expect("k <= n");
        let dense_sign = SymbolicBool::new(false, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        state.context.bump_next_condition_for(&dense_sign);
        let record_condition = state.context.fresh_condition();
        apply_pauli_measurement_signed(
            &mut state,
            &pauli_x(1, 0),
            &dense_sign,
            Some(1),
            Some(record_condition),
        );
        let correction = xor_bool(&symbolic_bool(record_condition), &dense_sign);
        apply_pauli_symbolic(&mut state, &pauli_z(1, 0), &correction).expect("no constant term");
        let second_condition = state.context.fresh_condition();
        apply_pauli_measurement_signed(
            &mut state,
            &pauli_x(1, 0),
            &SymbolicBool::default(),
            Some(2),
            Some(second_condition),
        );

        let program =
            plan_factored_updates(PendingFactoredState::from_frame_state(state)).expect("plans");
        assert!(program.instructions.len() >= 2);

        let FactoredInstruction::IntroduceDormantMeasurementBranch(first) =
            &program.instructions[0]
        else {
            panic!("the first instruction is a dormant branch measurement");
        };
        assert_eq!(
            first.outcome.conditions.len(),
            dense_sign.conditions.len() + 1,
            "the recorded outcome still carries the original symbolic sign plus its branch"
        );

        let mut checked = false;
        for instruction in &program.instructions[1..] {
            if let FactoredInstruction::RecordMeasurement(record) = instruction {
                assert!(
                    !record.outcome.constant && record.outcome.conditions.is_empty(),
                    "the substitution cancels the expanded branch expression"
                );
                checked = true;
            }
        }
        assert!(checked, "a second measurement is recorded");
    }

    // ==============================================================================
    // Active-width accounting
    // ==============================================================================

    /// A rotation about `Z0 X1` promotes the dormant qubit rather than rewriting the
    /// active basis, so the peak active width is 2.
    #[test]
    fn a_virtual_active_h_rewrite_promotes_a_dormant_qubit() {
        let mut state = FrameFactoredState::new(2, 1).expect("k <= n");
        apply_pauli_rotation(&mut state, &(&pauli_z(2, 0) * &pauli_x(2, 1)), PI / 2.0);
        apply_pauli_measurement(&mut state, &pauli_z(2, 1));
        let program =
            plan_factored_updates(PendingFactoredState::from_frame_state(state)).expect("plans");
        assert_eq!(program.max_k, 2);
        assert!(sampled(&program).iter().all(|row| record(row, 1)));
    }

    /// Measuring the same dormant Pauli twice reuses the tableau; the amplitude
    /// vector is never touched, so the peak active width stays at its initial 1.
    #[test]
    fn a_repeated_dormant_measurement_never_touches_the_amplitudes() {
        let mut state = FrameFactoredState::new(2, 1).expect("k <= n");
        let measured = &pauli_z(2, 0) * &pauli_x(2, 1);
        apply_pauli_measurement(&mut state, &measured);
        apply_pauli_measurement(&mut state, &measured);
        let program =
            plan_factored_updates(PendingFactoredState::from_frame_state(state)).expect("plans");
        assert_eq!(program.max_k, 1);
        assert!(
            sampled(&program)
                .iter()
                .all(|row| record(row, 1) == record(row, 2))
        );
    }

    #[test]
    fn a_dormant_measurement_sign_feeds_a_later_promotion() {
        let mut state = FrameFactoredState::new(1, 0).expect("k <= n");
        apply_pauli_measurement(&mut state, &pauli_x(1, 0));
        apply_pauli_rotation(&mut state, &pauli_z(1, 0), PI / 2.0);
        apply_pauli_measurement(&mut state, &pauli_x(1, 0));
        let program =
            plan_factored_updates(PendingFactoredState::from_frame_state(state)).expect("plans");
        assert_eq!(program.max_k, 1);
        assert!(
            sampled(&program)
                .iter()
                .all(|row| record(row, 1) != record(row, 2))
        );
    }

    // ==============================================================================
    // Planner contracts the sampler and the frontend depend on
    // ==============================================================================

    /// Detectors are re-anchored on these checkpoints, so there must be one before
    /// the first operation and one after every operation.
    #[test]
    fn planning_checkpoints_bracket_every_operation() {
        let mut state = FrameFactoredState::new(3, 0).expect("k <= n");
        apply_pauli_rotation(&mut state, &pauli_x(3, 0), 0.3);
        apply_pauli_measurement(&mut state, &pauli_z(3, 1));
        apply_pauli_measurement(&mut state, &pauli_x(3, 2));
        let mut pending = PendingFactoredState::from_frame_state(state);

        let operations = pending.pending_operations.len();
        let mut planned = 0;
        while has_pending_operations(&pending) {
            process_next_pending_operation(&mut pending).expect("plans");
            planned += 1;
            require_pending_record_conditions_are_causal(&pending, "mid-planning queue");
        }
        assert_eq!(planned, operations);
        assert_eq!(
            pending.pending_prefix_instruction_indices.len(),
            operations + 1,
            "one checkpoint before the first operation and one after each"
        );
        assert!(
            pending
                .pending_prefix_instruction_indices
                .windows(2)
                .all(|pair| pair[0] <= pair[1]),
            "checkpoints are nondecreasing instruction indices"
        );
        assert_eq!(
            *pending
                .pending_prefix_instruction_indices
                .last()
                .expect("at least one checkpoint"),
            pending.instructions.len() as i32
        );
    }

    /// `nsymbols` counts the 1-based condition ids, and every planned instruction's
    /// evaluation plan matches the expression next to it.
    #[test]
    fn planned_programs_are_internally_consistent() {
        let mut state = FrameFactoredState::new(3, 0).expect("k <= n");
        apply_pauli_rotation(&mut state, &(&pauli_x(3, 0) * &pauli_x(3, 2)), 0.4);
        apply_pauli_measurement(&mut state, &pauli_z(3, 0));
        apply_pauli_rotation(&mut state, &pauli_x(3, 1), 0.2);
        apply_pauli_measurement(&mut state, &pauli_x(3, 1));
        let program =
            plan_factored_updates(PendingFactoredState::from_frame_state(state)).expect("plans");

        assert_eq!(
            program.nsymbols as i32,
            (program.context.next_condition - 1).max(0)
        );
        assert_eq!(program.nrecords, 2);
        assert_eq!(program.ndetectors, 0);
        assert_eq!(program.nexpvals, 0);
        assert!(
            !program.use_active_components,
            "max_k is far below the gate"
        );

        for instruction in &program.instructions {
            if let Some(sign) = instruction.sign() {
                let FactoredInstruction::ApplyPrecomputedActivePauliRotation(rotation) =
                    instruction
                else {
                    continue;
                };
                assert_eq!(rotation.sign_plan, SymbolicBoolEvaluationPlan::new(sign));
            }
            if let Some(outcome) = instruction.outcome() {
                let plan = match instruction {
                    FactoredInstruction::RecordMeasurement(record) => &record.outcome_plan,
                    FactoredInstruction::MeasurePrecomputedActivePauli(measure) => {
                        &measure.outcome_plan
                    }
                    FactoredInstruction::IntroduceDormantMeasurementBranch(branch) => {
                        &branch.outcome_plan
                    }
                    FactoredInstruction::RecordDetector(detector) => &detector.outcome_plan,
                    _ => continue,
                };
                assert_eq!(*plan, SymbolicBoolEvaluationPlan::new(outcome));
            }
        }
    }
}
