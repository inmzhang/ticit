//! Symbolic booleans: XOR expressions over condition symbols that stand for
//! not-yet-sampled random bits.
//!
//! # Layout contract
//!
//! Condition ids are **1-based and dense**: symbol `c` occupies bit `c - 1` of
//! the per-shot symbol bitset, so `next_condition - 1` is the number of symbols
//! a plan must allocate. [`SymbolicBool::conditions`] is always strictly
//! ascending and duplicate-free — downstream code merge-joins and binary-searches
//! these lists, and compares them for *semantic* equality, all of which are only
//! valid in canonical form. Code that writes the field directly must restore the
//! invariant itself.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Not;

use crate::bits::bit_word_count;
use crate::bits::{check_probability, normalize_conditions, symbol_bit_mask, symbol_word_index};
use crate::errors::{Result, TicitError};

/// `constant XOR s_{c_1} XOR ... XOR s_{c_n}`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolicBool {
    pub constant: bool,
    pub conditions: Vec<i32>,
}

impl SymbolicBool {
    /// Builds an expression, normalizing `conditions` (sort, then drop
    /// even-multiplicity entries since `s ^ s == 0`).
    ///
    /// Panics on a non-positive condition id: ids are minted by
    /// [`SymbolicContext`], so a zero or negative one is a programming bug.
    pub fn new(constant: bool, conditions: Vec<i32>) -> Self {
        Self {
            constant,
            conditions: normalize_conditions(&conditions),
        }
    }

    /// Largest condition id in the expression, or 0 when it has none.
    pub fn max_condition(&self) -> i32 {
        self.conditions.last().copied().unwrap_or(0)
    }
}

impl From<bool> for SymbolicBool {
    fn from(constant: bool) -> Self {
        Self {
            constant,
            conditions: Vec::new(),
        }
    }
}

impl fmt::Display for SymbolicBool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        if self.constant {
            f.write_str("1")?;
            first = false;
        }
        for condition in &self.conditions {
            if !first {
                f.write_str(" xor ")?;
            }
            write!(f, "s{condition}")?;
            first = false;
        }
        if first {
            f.write_str("0")?;
        }
        Ok(())
    }
}

impl Not for SymbolicBool {
    type Output = SymbolicBool;

    /// Flips only the constant; the symbol set is untouched.
    fn not(mut self) -> SymbolicBool {
        self.constant = !self.constant;
        self
    }
}

impl Not for &SymbolicBool {
    type Output = SymbolicBool;

    fn not(self) -> SymbolicBool {
        self.clone().not()
    }
}

/// The bare symbol `s_condition`.
pub fn symbolic_bool(condition: i32) -> SymbolicBool {
    SymbolicBool::new(false, vec![condition])
}

/// XOR of two expressions: symmetric difference of the condition sets.
pub fn xor_bool(lhs: &SymbolicBool, rhs: &SymbolicBool) -> SymbolicBool {
    let mut conditions = Vec::with_capacity(lhs.conditions.len() + rhs.conditions.len());
    let (mut i, mut j) = (0, 0);
    while i < lhs.conditions.len() && j < rhs.conditions.len() {
        let (a, b) = (lhs.conditions[i], rhs.conditions[j]);
        match a.cmp(&b) {
            std::cmp::Ordering::Less => {
                conditions.push(a);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                conditions.push(b);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    conditions.extend_from_slice(&lhs.conditions[i..]);
    conditions.extend_from_slice(&rhs.conditions[j..]);
    SymbolicBool {
        constant: lhs.constant != rhs.constant,
        conditions,
    }
}

/// XOR of an expression with a literal.
pub fn xor_bool_constant(expr: &SymbolicBool, value: bool) -> SymbolicBool {
    SymbolicBool {
        constant: expr.constant != value,
        conditions: expr.conditions.clone(),
    }
}

/// Word-addressed form of a [`SymbolicBool`], precomputed once at planning time.
///
/// Evaluation is: XOR-parity of `value_words[word_indices[i]] & word_masks[i]`
/// over all `i`, then XOR `constant`. `word_indices` is ascending and deduped —
/// consecutive conditions falling in the same 64-bit word are coalesced into a
/// single mask — and samplers use `word_indices.last()` as the bounds check for
/// the symbol bitset.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolicBoolEvaluationPlan {
    pub constant: bool,
    pub conditions: Vec<i32>,
    pub word_indices: Vec<usize>,
    pub word_masks: Vec<u64>,
}

impl SymbolicBoolEvaluationPlan {
    pub fn new(expr: &SymbolicBool) -> Self {
        let mut plan = Self {
            constant: expr.constant,
            conditions: expr.conditions.clone(),
            word_indices: Vec::new(),
            word_masks: Vec::new(),
        };
        // Ascending condition ids give ascending word indices, so merging with
        // the previous entry is enough to keep `word_indices` deduped.
        for &condition in &plan.conditions {
            let word = symbol_word_index(condition);
            let mask = symbol_bit_mask(condition);
            if plan.word_indices.last() == Some(&word) {
                *plan
                    .word_masks
                    .last_mut()
                    .expect("word_masks and word_indices grow together") |= mask;
            } else {
                plan.word_indices.push(word);
                plan.word_masks.push(mask);
            }
        }
        plan
    }
}

/// A joint distribution over `nbits` fresh symbols.
///
/// `assignments[row]` is a packed bitset addressed by `packed_bit(row, i)`,
/// where bit `i` corresponds to `conditions[i]`; `probabilities[row]` is the
/// weight of that row and the weights sum to 1.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SymbolicCategoricalDistribution {
    pub nbits: usize,
    pub conditions: Vec<i32>,
    pub assignments: Vec<Vec<u64>>,
    pub probabilities: Vec<f64>,
}

/// Allocator and registry for condition symbols.
///
/// The C++ shares one of these through a `shared_ptr` held by several frames.
/// Here it is passed in explicitly as `&mut SymbolicContext` by the few core
/// operations that mint or observe symbols, which keeps the frames free of
/// interior mutability; the factored state owns the single instance.
#[derive(Clone, Debug)]
pub struct SymbolicContext {
    pub next_condition: i32,
    /// Ascending iteration order is a reproducibility contract: it fixes the
    /// order in which the sampler draws Bernoulli symbols.
    pub bernoulli_probabilities: BTreeMap<i32, f64>,
    pub categorical_distributions: Vec<SymbolicCategoricalDistribution>,
    // Only a construction-time guard against one condition joining two
    // categorical groups; nothing outside this module reads it.
    condition_to_categorical: HashMap<i32, usize>,
}

impl Default for SymbolicContext {
    fn default() -> Self {
        Self {
            next_condition: 1,
            bernoulli_probabilities: BTreeMap::new(),
            categorical_distributions: Vec::new(),
            condition_to_categorical: HashMap::new(),
        }
    }
}

impl SymbolicContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Starts allocation at `next_condition`. Panics if it is not positive.
    #[cfg(test)]
    pub fn with_next_condition(next_condition: i32) -> Self {
        assert!(next_condition > 0, "next condition id must be positive");
        Self {
            next_condition,
            ..Self::default()
        }
    }

    /// Ensures future fresh ids stay above `condition`.
    pub fn bump_next_condition(&mut self, condition: i32) {
        assert!(condition >= 0, "condition id must be nonnegative");
        self.next_condition = self.next_condition.max(condition + 1);
    }

    /// [`bump_next_condition`](Self::bump_next_condition) for every symbol an
    /// expression mentions.
    pub fn bump_next_condition_for(&mut self, expr: &SymbolicBool) {
        self.bump_next_condition(expr.max_condition());
    }

    pub fn fresh_condition(&mut self) -> i32 {
        let condition = self.next_condition;
        self.next_condition += 1;
        condition
    }

    pub fn fresh_bernoulli_condition(&mut self, probability: f64) -> Result<i32> {
        let probability = check_probability(probability)?;
        let condition = self.fresh_condition();
        if self.condition_to_categorical.contains_key(&condition) {
            return Err(TicitError::new(
                "condition already belongs to a categorical distribution",
            ));
        }
        self.bernoulli_probabilities.insert(condition, probability);
        Ok(condition)
    }

    pub fn fresh_bernoulli_bool(&mut self, probability: f64) -> Result<SymbolicBool> {
        Ok(symbolic_bool(self.fresh_bernoulli_condition(probability)?))
    }

    /// Mints `nbits` fresh symbols jointly distributed per `assignments` /
    /// `probabilities`.
    pub fn fresh_categorical_conditions(
        &mut self,
        nbits: usize,
        assignments: &[Vec<u64>],
        probabilities: &[f64],
    ) -> Result<Vec<i32>> {
        if assignments.is_empty() {
            return Err(TicitError::new(
                "categorical symbolic distribution needs at least one assignment",
            ));
        }
        if nbits == 0 || assignments.len() != probabilities.len() {
            return Err(TicitError::new("invalid categorical symbolic distribution"));
        }
        let nwords = bit_word_count(nbits);
        if assignments.iter().any(|row| row.len() != nwords) {
            return Err(TicitError::new("categorical assignment length mismatch"));
        }
        let mut total = 0.0;
        for &probability in probabilities {
            total += check_probability(probability)?;
        }
        if (total - 1.0).abs() > 1e-12 {
            return Err(TicitError::new(
                "categorical symbolic distribution probabilities must sum to 1",
            ));
        }

        let conditions: Vec<i32> = (0..nbits).map(|_| self.fresh_condition()).collect();
        let group = self.categorical_distributions.len();
        self.categorical_distributions
            .push(SymbolicCategoricalDistribution {
                nbits,
                conditions: conditions.clone(),
                assignments: assignments.to_vec(),
                probabilities: probabilities.to_vec(),
            });
        for &condition in &conditions {
            self.condition_to_categorical.insert(condition, group);
        }
        Ok(conditions)
    }

    pub fn fresh_categorical_bools(
        &mut self,
        nbits: usize,
        assignments: &[Vec<u64>],
        probabilities: &[f64],
    ) -> Result<Vec<SymbolicBool>> {
        Ok(self
            .fresh_categorical_conditions(nbits, assignments, probabilities)?
            .into_iter()
            .map(symbolic_bool)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{packed_bit, packed_bits};

    #[test]
    fn constructor_normalizes_conditions() {
        let expr = SymbolicBool::new(false, vec![66, 1, 66, 3, 3, 66]);
        assert_eq!(expr.conditions, vec![1, 66]);
        assert!(!expr.constant);
        assert!(SymbolicBool::new(true, vec![5, 5]).conditions.is_empty());
        assert_eq!(SymbolicBool::from(true), SymbolicBool::new(true, vec![]));
        assert_eq!(SymbolicBool::default(), SymbolicBool::from(false));
    }

    #[test]
    fn xor_is_symmetric_difference() {
        let lhs = SymbolicBool::new(false, vec![1, 66]);
        assert_eq!(xor_bool(&lhs, &symbolic_bool(66)), symbolic_bool(1));
        assert_eq!(
            xor_bool(&symbolic_bool(3), &symbolic_bool(3)),
            SymbolicBool::from(false)
        );
        assert_eq!(
            xor_bool(
                &SymbolicBool::new(false, vec![1, 3]),
                &SymbolicBool::new(false, vec![2, 3, 4])
            ),
            SymbolicBool::new(false, vec![1, 2, 4])
        );
        assert!(xor_bool(&SymbolicBool::from(true), &symbolic_bool(2)).constant);
        assert!(!xor_bool(&SymbolicBool::from(true), &SymbolicBool::new(true, vec![2])).constant);
    }

    #[test]
    fn xor_with_a_literal_keeps_the_symbols() {
        let expr = SymbolicBool::new(true, vec![4, 9]);
        let flipped = xor_bool_constant(&expr, true);
        assert!(!flipped.constant);
        assert_eq!(flipped.conditions, vec![4, 9]);
        assert_eq!(xor_bool_constant(&expr, false), expr);
    }

    #[test]
    fn not_flips_only_the_constant() {
        let expr = SymbolicBool::new(false, vec![2, 7]);
        let negated = !&expr;
        assert!(negated.constant);
        assert_eq!(negated.conditions, expr.conditions);
        assert_eq!(!negated, expr);
        assert_eq!(!SymbolicBool::from(false), SymbolicBool::from(true));
    }

    #[test]
    fn max_condition_reads_the_last_entry() {
        assert_eq!(SymbolicBool::default().max_condition(), 0);
        assert_eq!(SymbolicBool::from(true).max_condition(), 0);
        assert_eq!(SymbolicBool::new(false, vec![8, 3]).max_condition(), 8);
    }

    #[test]
    fn display_spells_out_the_xor() {
        assert_eq!(SymbolicBool::default().to_string(), "0");
        assert_eq!(SymbolicBool::from(true).to_string(), "1");
        assert_eq!(symbolic_bool(4).to_string(), "s4");
        assert_eq!(
            SymbolicBool::new(true, vec![9, 2]).to_string(),
            "1 xor s2 xor s9"
        );
    }

    #[test]
    #[should_panic(expected = "condition id must be positive")]
    fn condition_ids_must_be_positive() {
        symbolic_bool(0);
    }

    #[test]
    fn evaluation_plan_coalesces_conditions_sharing_a_word() {
        let expr = SymbolicBool::new(true, vec![129, 1, 65, 2, 64]);
        let plan = SymbolicBoolEvaluationPlan::new(&expr);
        assert!(plan.constant);
        assert_eq!(plan.conditions, vec![1, 2, 64, 65, 129]);
        assert_eq!(plan.word_indices, vec![0, 1, 2]);
        assert_eq!(plan.word_masks, vec![1 | (1 << 1) | (1 << 63), 1, 1]);
    }

    #[test]
    fn evaluation_plan_of_a_constant_has_no_words() {
        let plan = SymbolicBoolEvaluationPlan::new(&SymbolicBool::from(true));
        assert!(plan.constant);
        assert!(plan.conditions.is_empty());
        assert!(plan.word_indices.is_empty());
        assert!(plan.word_masks.is_empty());
        assert_eq!(
            plan,
            SymbolicBoolEvaluationPlan::new(&SymbolicBool::from(true))
        );
    }

    fn evaluate(plan: &SymbolicBoolEvaluationPlan, symbol_words: &[u64]) -> bool {
        let mut value = plan.constant;
        for (&word, &mask) in plan.word_indices.iter().zip(&plan.word_masks) {
            value ^= (symbol_words[word] & mask).count_ones() % 2 == 1;
        }
        value
    }

    #[test]
    fn evaluation_plan_agrees_with_direct_condition_lookup() {
        let expr = SymbolicBool::new(true, vec![1, 2, 64, 65, 130]);
        let plan = SymbolicBoolEvaluationPlan::new(&expr);
        let symbol_words = vec![0b0101u64, 1 << 1, 0];
        let expected = expr.conditions.iter().fold(expr.constant, |value, &c| {
            value ^ packed_bit(&symbol_words, (c - 1) as usize)
        });
        assert_eq!(evaluate(&plan, &symbol_words), expected);
        assert!(!expected);
    }

    #[test]
    fn conditions_are_allocated_from_one() {
        let mut context = SymbolicContext::new();
        assert_eq!(context.next_condition, 1);
        assert_eq!(context.fresh_condition(), 1);
        assert_eq!(context.fresh_condition(), 2);
        assert_eq!(context.next_condition, 3);

        let mut shifted = SymbolicContext::with_next_condition(12);
        assert_eq!(shifted.fresh_condition(), 12);
    }

    #[test]
    fn bump_lifts_allocation_above_an_existing_expression() {
        let mut context = SymbolicContext::new();
        context.bump_next_condition_for(&SymbolicBool::new(false, (1..=8).collect()));
        assert_eq!(context.fresh_condition(), 9);
        context.bump_next_condition(2);
        assert_eq!(context.fresh_condition(), 10);
        context.bump_next_condition(0);
        assert_eq!(context.next_condition, 11);
    }

    #[test]
    #[should_panic(expected = "next condition id must be positive")]
    fn context_start_must_be_positive() {
        SymbolicContext::with_next_condition(0);
    }

    #[test]
    #[should_panic(expected = "condition id must be nonnegative")]
    fn bump_rejects_negative_conditions() {
        SymbolicContext::new().bump_next_condition(-1);
    }

    #[test]
    fn fresh_bernoulli_bool_registers_its_probability() {
        let mut context = SymbolicContext::new();
        let first = context
            .fresh_bernoulli_bool(0.25)
            .expect("valid probability");
        let second = context
            .fresh_bernoulli_bool(0.5)
            .expect("valid probability");
        assert_eq!(first, symbolic_bool(1));
        assert_eq!(second, symbolic_bool(2));
        assert_eq!(context.bernoulli_probabilities.get(&1), Some(&0.25));
        assert_eq!(context.bernoulli_probabilities.get(&2), Some(&0.5));
        assert!(context.fresh_bernoulli_condition(1.5).is_err());
        assert!(context.fresh_bernoulli_condition(-0.1).is_err());
    }

    #[test]
    fn bernoulli_probabilities_iterate_in_condition_order() {
        let mut context = SymbolicContext::with_next_condition(60);
        for probability in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] {
            context
                .fresh_bernoulli_condition(probability)
                .expect("valid probability");
        }
        let ids: Vec<i32> = context.bernoulli_probabilities.keys().copied().collect();
        assert_eq!(ids, vec![60, 61, 62, 63, 64, 65, 66]);
    }

    fn two_bit_distribution() -> (Vec<Vec<u64>>, Vec<f64>) {
        let assignments = vec![
            packed_bits(&[false, false]),
            packed_bits(&[true, false]),
            packed_bits(&[false, true]),
            packed_bits(&[true, true]),
        ];
        (assignments, vec![0.7, 0.1, 0.1, 0.1])
    }

    #[test]
    fn fresh_categorical_bools_mints_one_symbol_per_bit() {
        let (assignments, probabilities) = two_bit_distribution();
        let mut context = SymbolicContext::new();
        let bools = context
            .fresh_categorical_bools(2, &assignments, &probabilities)
            .expect("valid distribution");
        assert_eq!(bools, vec![symbolic_bool(1), symbolic_bool(2)]);
        assert_eq!(context.next_condition, 3);
        assert!(context.bernoulli_probabilities.is_empty());

        let distribution = &context.categorical_distributions[0];
        assert_eq!(distribution.nbits, 2);
        assert_eq!(distribution.conditions, vec![1, 2]);
        assert_eq!(distribution.probabilities, probabilities);
        assert!(packed_bit(&distribution.assignments[1], 0));
        assert!(!packed_bit(&distribution.assignments[1], 1));
        assert!(!packed_bit(&distribution.assignments[2], 0));
        assert!(packed_bit(&distribution.assignments[2], 1));
    }

    #[test]
    fn categorical_distributions_accumulate_in_registration_order() {
        let (assignments, probabilities) = two_bit_distribution();
        let mut context = SymbolicContext::new();
        context
            .fresh_categorical_conditions(2, &assignments, &probabilities)
            .expect("valid distribution");
        let second = context
            .fresh_categorical_conditions(
                1,
                &[packed_bits(&[false]), packed_bits(&[true])],
                &[0.5, 0.5],
            )
            .expect("valid distribution");
        assert_eq!(second, vec![3]);
        assert_eq!(context.categorical_distributions.len(), 2);
        assert_eq!(context.categorical_distributions[1].conditions, vec![3]);
    }

    #[test]
    fn categorical_distributions_are_validated() {
        let (assignments, probabilities) = two_bit_distribution();
        let mut context = SymbolicContext::new();
        let message = |result: Result<Vec<i32>>| result.expect_err("invalid").to_string();

        assert_eq!(
            message(context.fresh_categorical_conditions(2, &[], &[])),
            "categorical symbolic distribution needs at least one assignment"
        );
        assert_eq!(
            message(context.fresh_categorical_conditions(0, &assignments, &probabilities)),
            "invalid categorical symbolic distribution"
        );
        assert_eq!(
            message(context.fresh_categorical_conditions(2, &assignments, &[1.0])),
            "invalid categorical symbolic distribution"
        );
        assert_eq!(
            message(context.fresh_categorical_conditions(
                2,
                &[packed_bits(&[false, false]), vec![0, 0]],
                &[0.5, 0.5]
            )),
            "categorical assignment length mismatch"
        );
        assert_eq!(
            message(context.fresh_categorical_conditions(2, &assignments, &[0.7, 0.1, 0.1, 0.2])),
            "categorical symbolic distribution probabilities must sum to 1"
        );
        assert_eq!(
            message(context.fresh_categorical_conditions(2, &assignments, &[0.7, 0.1, 0.1, 1.5])),
            "probability must be between 0 and 1"
        );
        assert!(context.categorical_distributions.is_empty());
    }
}
