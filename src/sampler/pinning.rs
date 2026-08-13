//! Pinned measurement parities: fixing what the *noiseless* circuit measures.
//!
//! # Why this is a branch-level problem
//!
//! A caller wants to say "the XOR of these measurement records is 1 in the
//! noiseless circuit" — the logical outcome that selects a compiled branch of
//! an adaptive program. A record's value is its intrinsic measurement branch
//! XOR a frame that, in the noiseless circuit, contains only exogenous noise
//! symbols (all zero). So the constraint expands into a linear equation over
//! the *branch* symbols, and that is where it must be enforced.
//!
//! Enforcing the recorded parity directly would be wrong for anything but the
//! reference sample: it would suppress exactly the noise-induced flips a
//! decoder is meant to correct, and the pinned parity would come out
//! error-free by construction.
//!
//! # Solving
//!
//! Every constraint becomes one row over branch symbols. Rows are reduced
//! against each other by Gaussian elimination over GF(2), each row taking as
//! its pivot the branch drawn *last* — so when the pivot instruction runs,
//! every other term in its row already holds a value and the pin is a plain
//! XOR of assigned symbols. A row that reduces to no branches at all is
//! deterministic: it either already holds, or the constraint is impossible and
//! compilation fails.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::errors::{Result, TicitError};
use crate::factored::{FactoredInstruction, FactoredInstructionProgram};
use crate::symbolic::{SymbolicBool, SymbolicBoolEvaluationPlan, symbolic_bool, xor_bool};

/// A parity the noiseless circuit must produce over a set of measurement
/// records.
///
/// Used through [`SamplerOptions::pin_measurements`], where the records are the
/// zero-based indices a sampling call returns.
///
/// [`SamplerOptions::pin_measurements`]: crate::SamplerOptions::pin_measurements
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MeasurementParity {
    /// Zero-based measurement records whose XOR is constrained. A single index
    /// pins one record; repeated indices cancel, as XOR implies.
    pub records: Vec<usize>,
    /// The value that XOR must take in the noiseless circuit.
    pub value: bool,
}

impl MeasurementParity {
    /// A constraint on the XOR of `records`.
    #[must_use]
    pub fn new(records: impl Into<Vec<usize>>, value: bool) -> Self {
        Self {
            records: records.into(),
            value,
        }
    }
}

/// One pinned measurement branch.
///
/// At `instruction`, the branch symbol takes the value of `plan` — an XOR over
/// branch symbols assigned by earlier instructions — instead of being drawn.
#[derive(Clone, Debug, Default)]
pub(crate) struct ForcedBranch {
    pub instruction: usize,
    pub plan: SymbolicBoolEvaluationPlan,
}

/// How a symbol gets its value during execution. Symbols that appear here are
/// exactly those an instruction assigns; every other symbol is exogenous noise
/// and is therefore zero in the noiseless circuit.
#[derive(Clone, Copy, Debug)]
enum SymbolSource {
    /// A measurement branch drawn at this instruction — a free variable.
    Branch,
    /// A record condition assigned from this instruction's outcome expression.
    Derived(usize),
}

/// Instruction-level index of what assigns each symbol and each record.
struct ProgramSymbols {
    /// Only measurement branches and record conditions land here; the table is
    /// sized by the measurement count, not by the (far larger) symbol count.
    sources: HashMap<i32, SymbolSource>,
    /// Instruction index that draws each branch symbol, used to order pivots.
    branch_instruction: HashMap<i32, usize>,
    /// Instruction index writing each one-based record.
    record_instruction: HashMap<i32, usize>,
    /// Noiseless expansion of a symbol into branch symbols, memoized.
    expansions: HashMap<i32, SymbolicBool>,
}

impl ProgramSymbols {
    fn new(program: &FactoredInstructionProgram) -> Self {
        let mut sources = HashMap::new();
        let mut branch_instruction = HashMap::new();
        let mut record_instruction = HashMap::new();
        for (index, instruction) in program.instructions.iter().enumerate() {
            let branch = match instruction {
                FactoredInstruction::MeasurePrecomputedActivePauli(inst) => Some(inst.branch),
                FactoredInstruction::IntroduceDormantMeasurementBranch(inst) => Some(inst.branch),
                _ => None,
            };
            // An expectation probe rides the measurement opcodes but samples
            // nothing, so its branch symbol is never assigned.
            let probes = instruction.exp_val().is_some();
            if let Some(branch) = branch
                && !probes
            {
                sources.insert(branch, SymbolSource::Branch);
                branch_instruction.insert(branch, index);
            }
            if let Some(condition) = instruction.record_condition()
                && !probes
            {
                sources
                    .entry(condition)
                    .or_insert(SymbolSource::Derived(index));
            }
            if let Some(record) = instruction.record()
                && !probes
            {
                record_instruction.insert(record, index);
            }
        }
        Self {
            sources,
            branch_instruction,
            record_instruction,
            expansions: HashMap::new(),
        }
    }

    /// The record's noiseless value as an XOR over branch symbols.
    fn record_expansion(
        &mut self,
        program: &FactoredInstructionProgram,
        record: usize,
    ) -> Result<SymbolicBool> {
        let one_based = i32::try_from(record + 1)
            .map_err(|_| TicitError::new("pinned measurement record index is out of range"))?;
        let Some(&instruction) = self.record_instruction.get(&one_based) else {
            return Err(TicitError::new(format!(
                "pinned measurement record {record} is not written by this circuit"
            )));
        };
        let outcome = program.instructions[instruction]
            .outcome()
            .ok_or_else(|| TicitError::new("measurement instruction has no outcome expression"))?
            .clone();
        self.expand(program, &outcome)
    }

    /// Expands `expr` with every exogenous symbol held at zero.
    ///
    /// Iterative rather than recursive: a record condition can depend on an
    /// arbitrarily long chain of earlier ones, and the chain length is set by
    /// the circuit, not by anything this code controls.
    fn expand(
        &mut self,
        program: &FactoredInstructionProgram,
        expr: &SymbolicBool,
    ) -> Result<SymbolicBool> {
        for &condition in &expr.conditions {
            self.expand_symbol(program, condition)?;
        }
        let mut out = SymbolicBool::from(expr.constant);
        for &condition in &expr.conditions {
            let expansion = self
                .expansions
                .get(&condition)
                .expect("every condition was just expanded");
            out = xor_bool(&out, expansion);
        }
        Ok(out)
    }

    fn expand_symbol(&mut self, program: &FactoredInstructionProgram, symbol: i32) -> Result<()> {
        let mut stack = vec![symbol];
        let mut in_progress: HashSet<i32> = HashSet::new();
        while let Some(&top) = stack.last() {
            if self.expansions.contains_key(&top) {
                in_progress.remove(&top);
                stack.pop();
                continue;
            }
            match self.sources.get(&top).copied() {
                // Exogenous noise is zero in the noiseless circuit.
                None => {
                    self.expansions.insert(top, SymbolicBool::from(false));
                }
                Some(SymbolSource::Branch) => {
                    self.expansions.insert(top, symbolic_bool(top));
                }
                Some(SymbolSource::Derived(instruction)) => {
                    in_progress.insert(top);
                    let outcome = program.instructions[instruction].outcome().ok_or_else(|| {
                        TicitError::new("record condition has no outcome expression")
                    })?;
                    let mut pending = 0usize;
                    for &condition in &outcome.conditions {
                        if self.expansions.contains_key(&condition) {
                            continue;
                        }
                        if in_progress.contains(&condition) {
                            return Err(TicitError::new("measurement record expression is cyclic"));
                        }
                        stack.push(condition);
                        pending += 1;
                    }
                    if pending > 0 {
                        continue;
                    }
                    let mut value = SymbolicBool::from(outcome.constant);
                    for &condition in &outcome.conditions {
                        let expansion = self
                            .expansions
                            .get(&condition)
                            .expect("all conditions resolved above");
                        value = xor_bool(&value, expansion);
                    }
                    self.expansions.insert(top, value);
                }
            }
            in_progress.remove(&top);
            stack.pop();
        }
        Ok(())
    }

    /// The row term drawn last, which is the only one safe to pin.
    fn last_drawn(&self, conditions: &[i32]) -> Option<(usize, i32)> {
        conditions
            .iter()
            .filter_map(|&symbol| {
                self.branch_instruction
                    .get(&symbol)
                    .map(|&instruction| (instruction, symbol))
            })
            .max()
    }
}

/// Compiles measurement-parity constraints into the branch pins that satisfy
/// them.
///
/// # Errors
///
/// Returns an error if a record is not written by the circuit, or if a
/// constraint's parity is already determined to the opposite value — the
/// caller asked for something the circuit cannot produce.
pub(crate) fn plan_pinned_measurements(
    program: &FactoredInstructionProgram,
    constraints: &[MeasurementParity],
) -> Result<Vec<ForcedBranch>> {
    if constraints.is_empty() {
        return Ok(Vec::new());
    }
    let mut symbols = ProgramSymbols::new(program);
    // Keyed by the pivot's instruction index, which is also the elimination
    // order: reducing a row always removes its highest-index term.
    let mut pivot_rows: BTreeMap<usize, (i32, SymbolicBool)> = BTreeMap::new();

    for constraint in constraints {
        // The row is `value XOR (noiseless parity)`, so a satisfied constraint
        // is the row evaluating to zero.
        let mut row = SymbolicBool::from(constraint.value);
        for &record in &constraint.records {
            let expansion = symbols.record_expansion(program, record)?;
            row = xor_bool(&row, &expansion);
        }
        loop {
            let Some((instruction, pivot)) = symbols.last_drawn(&row.conditions) else {
                if row.constant {
                    return Err(TicitError::new(format!(
                        "pinned measurement parity over {:?} is deterministic and cannot be {}",
                        constraint.records,
                        u8::from(constraint.value),
                    )));
                }
                break;
            };
            match pivot_rows.get(&instruction) {
                Some((_, existing)) => row = xor_bool(&row, existing),
                None => {
                    pivot_rows.insert(instruction, (pivot, row));
                    break;
                }
            }
        }
    }

    let mut forced = Vec::with_capacity(pivot_rows.len());
    for (instruction, (pivot, row)) in pivot_rows {
        // `row == 0` means `pivot == everything else in the row`.
        let assignment = xor_bool(&row, &symbolic_bool(pivot));
        forced.push(ForcedBranch {
            instruction,
            plan: SymbolicBoolEvaluationPlan::new(&assignment),
        });
    }
    Ok(forced)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{parse_ticit_text, plan_ticit_factored_program};

    fn planned(text: &str) -> FactoredInstructionProgram {
        let parsed = parse_ticit_text(text).expect("test circuit parses");
        plan_ticit_factored_program(&parsed).expect("test circuit plans")
    }

    #[test]
    fn a_free_coin_is_pinned_at_its_own_instruction() {
        let program = planned("H 0\nM 0\n");
        let forced = plan_pinned_measurements(&program, &[MeasurementParity::new([0], true)])
            .expect("a fair coin can be pinned");
        assert_eq!(forced.len(), 1);
        assert!(forced[0].plan.conditions.is_empty());
        assert!(forced[0].plan.constant);
    }

    #[test]
    fn a_deterministic_record_rejects_the_wrong_value() {
        let program = planned("M 0\n");
        let error = plan_pinned_measurements(&program, &[MeasurementParity::new([0], true)])
            .expect_err("|0> always measures 0");
        assert!(error.to_string().contains("deterministic"));
    }

    #[test]
    fn a_deterministic_record_accepts_the_right_value() {
        let program = planned("M 0\n");
        let forced = plan_pinned_measurements(&program, &[MeasurementParity::new([0], false)])
            .expect("|0> always measures 0");
        assert!(forced.is_empty());
    }

    #[test]
    fn a_parity_pins_only_its_last_free_coin() {
        let program = planned("H 0\nH 1\nM 0\nM 1\n");
        let forced = plan_pinned_measurements(&program, &[MeasurementParity::new([0, 1], true)])
            .expect("two fair coins can meet a parity");
        assert_eq!(forced.len(), 1, "only the last draw is pinned");
        assert_eq!(forced[0].plan.conditions.len(), 1, "it follows the first");
    }

    #[test]
    fn independent_constraints_take_distinct_pivots() {
        let program = planned("H 0\nH 1\nM 0\nM 1\n");
        let forced = plan_pinned_measurements(
            &program,
            &[
                MeasurementParity::new([0], true),
                MeasurementParity::new([0, 1], false),
            ],
        )
        .expect("independent constraints solve");
        assert_eq!(forced.len(), 2);
        assert_ne!(forced[0].instruction, forced[1].instruction);
    }

    #[test]
    fn contradictory_constraints_are_rejected() {
        let program = planned("H 0\nM 0\nM 0\n");
        let error = plan_pinned_measurements(
            &program,
            &[
                MeasurementParity::new([0], true),
                MeasurementParity::new([1], false),
            ],
        )
        .expect_err("the second measurement repeats the first");
        assert!(error.to_string().contains("deterministic"));
    }

    #[test]
    fn an_unwritten_record_is_rejected() {
        let program = planned("H 0\nM 0\n");
        let error = plan_pinned_measurements(&program, &[MeasurementParity::new([7], true)])
            .expect_err("record 7 does not exist");
        assert!(error.to_string().contains("not written"));
    }
}
