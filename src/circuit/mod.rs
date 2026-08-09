//! Parsing and compilation of `.ticit` circuits.

pub(crate) mod ir;
pub(crate) mod lowering;
pub(crate) mod parser;

use std::path::Path;

use self::ir::{CircuitDetector, CircuitObservableInclude, QuantumCircuit};
use self::lowering::{CircuitLoweringResult, lower_circuit_to_factored};
pub(crate) use self::parser::{parse_ticit_circuit_file, parse_ticit_circuit_text};
use crate::errors::{Result, TicitError};
use crate::factored::{
    FactoredInstruction, FactoredInstructionProgram, FrameFactoredState, PendingFactoredState,
    RecordDetector,
};
use crate::pending_optimizer::optimize_pending_operations;
use crate::planner::plan_factored_updates;
use crate::sampler::prepared::{Sampler, SamplerOptions};
use crate::symbolic::{SymbolicBool, SymbolicBoolEvaluationPlan, xor_bool};

/// A parsed `.ticit` circuit ready to compile for batch sampling.
#[derive(Clone, Debug)]
pub struct Circuit {
    pub(crate) state: FrameFactoredState,
    pub(crate) measurement_records: Vec<SymbolicBool>,
    pub(crate) detectors: Vec<CircuitDetector>,
    pub(crate) observables: Vec<CircuitObservableInclude>,
    pub(crate) expectation_values: usize,
}

impl Circuit {
    /// Parses a circuit from `.ticit` source text and lowers it into ticit's
    /// symbolic frame representation.
    ///
    /// # Errors
    ///
    /// Returns [`TicitError::Parse`] for malformed source and
    /// [`TicitError::Unsupported`] for a valid instruction ticit cannot execute.
    ///
    /// # Examples
    ///
    /// ```
    /// use ticit::Circuit;
    ///
    /// let circuit = Circuit::from_text("H 0\nM 0\nDETECTOR rec[-1]")?;
    /// assert_eq!(circuit.qubit_count(), 1);
    /// assert_eq!(circuit.measurement_record_count(), 1);
    /// assert_eq!(circuit.detector_count(), 1);
    /// # Ok::<(), ticit::TicitError>(())
    /// ```
    pub fn from_text(text: &str) -> Result<Self> {
        lowered_circuit(parse_ticit_circuit_text(text)?)
    }

    /// Parses and lowers a circuit from a UTF-8 `.ticit` file.
    ///
    /// # Errors
    ///
    /// Returns [`TicitError::Io`] if the file cannot be read, or the same parse
    /// and lowering errors as [`from_text`](Self::from_text).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        lowered_circuit(parse_ticit_circuit_file(path)?)
    }

    /// Compiles this circuit into a reusable batch sampler.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit cannot be planned, an option overflows
    /// an internal index, or the required active state is unsupported.
    pub fn compile(&self, options: SamplerOptions) -> Result<Sampler> {
        Sampler::new(self, options)
    }

    /// Number of qubits named by the circuit.
    #[must_use]
    pub fn qubit_count(&self) -> usize {
        self.state.n
    }

    /// Number of measurement results written to `rec`.
    #[must_use]
    pub fn measurement_record_count(&self) -> usize {
        self.measurement_records.len()
    }

    /// Number of `DETECTOR` and `DISCARD` declarations.
    #[must_use]
    pub fn detector_count(&self) -> usize {
        self.detectors.len()
    }

    /// Number of observable indices, including unused gaps before the largest
    /// `OBSERVABLE_INCLUDE` index.
    #[must_use]
    pub fn observable_count(&self) -> usize {
        self.observables
            .iter()
            .map(|observable| observable.index + 1)
            .max()
            .unwrap_or(0)
    }

    /// Number of expectation values produced by `EXP_VAL` instructions.
    #[must_use]
    pub fn expectation_value_count(&self) -> usize {
        self.expectation_values
    }

    /// Whether at least one detector is declared as a `DISCARD` check.
    #[must_use]
    pub fn has_detector_postselection(&self) -> bool {
        self.detectors.iter().any(|detector| detector.discard)
    }

    /// Whether every detector is declared as a `DISCARD` check.
    ///
    /// Returns `false` for a circuit with no detectors.
    #[must_use]
    pub fn all_detectors_postselected(&self) -> bool {
        !self.detectors.is_empty() && self.detectors.iter().all(|detector| detector.discard)
    }
}

/// Translates detector positions from instruction indices to pending-operation
/// prefixes through the lowering's prefix table.
fn detectors_with_lowered_positions(
    circuit: &QuantumCircuit,
    lowered: &CircuitLoweringResult,
) -> Result<Vec<CircuitDetector>> {
    let mut detectors = circuit.detectors.clone();
    for detector in &mut detectors {
        let counts = &lowered.instruction_pending_operation_counts;
        if detector.after_instruction >= counts.len() {
            return Err(TicitError::new("detector source position is out of range"));
        }
        detector.after_pending_operation = counts[detector.after_instruction];
    }
    Ok(detectors)
}

fn lowered_circuit(circuit: QuantumCircuit) -> Result<Circuit> {
    let lowered = lower_circuit_to_factored(&circuit)?;
    let detectors = detectors_with_lowered_positions(&circuit, &lowered)?;
    Ok(Circuit {
        state: lowered.state,
        measurement_records: lowered.measurement_records,
        detectors,
        observables: circuit.observables,
        expectation_values: circuit.nexpvals,
    })
}

// ==============================================================================
// Detector event splicing
// ==============================================================================

/// XOR of the referenced measurement-record outcomes.
fn detector_expression(
    detector: &CircuitDetector,
    measurement_records: &[SymbolicBool],
) -> Result<SymbolicBool> {
    let mut out = SymbolicBool::default();
    for &record in &detector.records {
        if record == 0 || record > measurement_records.len() {
            return Err(TicitError::new(
                "detector references an out-of-range measurement record",
            ));
        }
        out = xor_bool(&out, &measurement_records[record - 1]);
    }
    Ok(out)
}

/// Maps a pending-operation prefix to the instruction index it planned into.
fn instruction_checkpoint_for_pending_prefix(
    program: &FactoredInstructionProgram,
    pending_prefix: usize,
) -> Result<usize> {
    if pending_prefix == 0 && program.pending_prefix_instruction_indices.is_empty() {
        return Ok(0);
    }
    let Some(&checkpoint) = program
        .pending_prefix_instruction_indices
        .get(pending_prefix)
    else {
        return Err(TicitError::new(
            "detector pending-operation position is out of range",
        ));
    };
    if checkpoint < 0 || checkpoint as usize > program.instructions.len() {
        return Err(TicitError::new(
            "detector instruction checkpoint is out of range",
        ));
    }
    Ok(checkpoint as usize)
}

/// Splices one `RecordDetector` per detector into the instruction stream at
/// its resolved checkpoint, then rebuilds the program (rerunning the final
/// reduction pass over the enlarged stream).
fn insert_detector_events(
    program: FactoredInstructionProgram,
    detectors: &[CircuitDetector],
    measurement_records: &[SymbolicBool],
    postselection_mask: &[u8],
) -> Result<FactoredInstructionProgram> {
    if detectors.is_empty() {
        return Ok(program);
    }
    let mut events: Vec<Vec<RecordDetector>> = vec![Vec::new(); program.instructions.len() + 1];
    for (idx, detector) in detectors.iter().enumerate() {
        let outcome = detector_expression(detector, measurement_records)?;
        let instruction = RecordDetector {
            outcome_plan: SymbolicBoolEvaluationPlan::new(&outcome),
            outcome,
            records: detector.records.iter().map(|&r| r as i32).collect(),
            detector: (idx + 1) as i32,
            postselect: detector.discard
                || postselection_mask.get(idx).is_some_and(|&flag| flag != 0),
        };
        let checkpoint =
            instruction_checkpoint_for_pending_prefix(&program, detector.after_pending_operation)?;
        events[checkpoint].push(instruction);
    }

    let mut instructions: Vec<FactoredInstruction> =
        Vec::with_capacity(program.instructions.len() + detectors.len());
    let mut events = events.into_iter();
    let leading = events
        .next()
        .expect("events has instructions.len() + 1 entries");
    instructions.extend(leading.into_iter().map(FactoredInstruction::from));
    for (instruction, following) in program.instructions.into_iter().zip(events) {
        instructions.push(instruction);
        instructions.extend(following.into_iter().map(FactoredInstruction::from));
    }

    // The checkpoint list is not carried over: instruction indices just
    // shifted, so the old table is stale, and nothing downstream reads it.
    FactoredInstructionProgram::with_context(
        program.n,
        program.initial_k,
        instructions,
        program.max_k,
        program.context,
        Vec::new(),
    )
}

/// Optimizes, plans, and applies caller-selected detector postselection.
pub(crate) fn plan_circuit(
    parsed: &Circuit,
    postselection_mask: &[u8],
) -> Result<FactoredInstructionProgram> {
    let mut pending = PendingFactoredState::from_frame_state(parsed.state.clone());
    let detector_prefixes: Vec<usize> = parsed
        .detectors
        .iter()
        .map(|detector| detector.after_pending_operation)
        .collect();
    let optimization = optimize_pending_operations(&mut pending, &detector_prefixes)?;
    let mut detectors = parsed.detectors.clone();
    for detector in &mut detectors {
        let remapped = optimization
            .prefix_remap
            .get(detector.after_pending_operation)
            .copied()
            .unwrap_or(-1);
        if remapped < 0 {
            return Err(TicitError::new(
                "detector pending-operation prefix was not preserved by optimization",
            ));
        }
        detector.after_pending_operation = remapped as usize;
    }
    let program = plan_factored_updates(pending)?;
    insert_detector_events(
        program,
        &detectors,
        &parsed.measurement_records,
        postselection_mask,
    )
}

pub(crate) fn has_postselection(program: &FactoredInstructionProgram) -> bool {
    program.instructions.iter().any(|instruction| {
        matches!(instruction, FactoredInstruction::RecordDetector(detector) if detector.postselect)
    })
}

#[cfg(test)]
pub(crate) fn parse_ticit_text(text: &str) -> Result<Circuit> {
    Circuit::from_text(text)
}

#[cfg(test)]
pub(crate) fn parse_ticit_file(path: impl AsRef<Path>) -> Result<Circuit> {
    Circuit::from_file(path)
}

#[cfg(test)]
pub(crate) fn plan_ticit_factored_program(parsed: &Circuit) -> Result<FactoredInstructionProgram> {
    plan_circuit(parsed, &[])
}
