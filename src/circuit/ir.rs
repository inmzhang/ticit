//! Flattened circuit representation shared by parsing and lowering.
//!
//! `REPEAT` blocks are expanded, rotation angles use the kernel convention,
//! and correlated-error chains are collapsed before lowering begins. Record
//! and detector indices are 1-based throughout the internal pipeline.

use crate::pauli::PauliString;

/// Every operation the flattened IR can carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CircuitInstructionKind {
    #[default]
    Tick,
    H,
    HNegXy,
    HNegXz,
    HNegYz,
    HXy,
    HYz,
    CNegXyz,
    CNegZyx,
    CXNegYz,
    CXyNegZ,
    CXyz,
    CZNegYx,
    CZyNegX,
    CZyx,
    S,
    SDag,
    SqrtX,
    SqrtXDag,
    SqrtY,
    SqrtYDag,
    X,
    Y,
    Z,
    CX,
    CY,
    CZ,
    Swap,
    CxSwap,
    CzSwap,
    ISwap,
    ISwapDag,
    SqrtXx,
    SqrtXxDag,
    SqrtYy,
    SqrtYyDag,
    SqrtZz,
    SqrtZzDag,
    SwapCx,
    Xcx,
    Xcy,
    Xcz,
    Ycx,
    Ycy,
    Ycz,
    T,
    TDag,
    PauliRotation,
    MZ,
    MX,
    MY,
    Mrz,
    Mrx,
    Mry,
    RZ,
    RX,
    RY,
    Mpp,
    ExpVal,
    XError,
    YError,
    ZError,
    Depolarize1,
    Depolarize2,
    Depolarize3,
    PauliChannel1,
    PauliChannel2,
    PauliChannel3,
    PauliProductChannel,
    HeraldedErase,
    HeraldedPauliChannel1,
    MPad,
    FeedbackX,
    FeedbackY,
    FeedbackZ,
}

/// One target of a measurement-like instruction.
///
/// `inverted` is the `.ticit` `!` prefix: the recorded bit is flipped. For
/// [`MPad`](CircuitInstructionKind::MPad) the `qubit` field is not a qubit at
/// all but the literal pad value, which is why `MPAD` targets do not grow the
/// circuit's qubit count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CircuitMeasurementTarget {
    pub qubit: usize,
    pub inverted: bool,
}

/// A signed Pauli product, as measured by `MPP` or rotated by `R_PAULI`/`SPP`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CircuitPauliProduct {
    pub pauli: PauliString,
    pub inverted: bool,
}

/// A classically controlled Pauli: apply to `qubit` iff record `record` is 1.
///
/// `record` is 1-based, like every measurement-record reference in the IR.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CircuitFeedbackTarget {
    pub record: usize,
    pub qubit: usize,
}

/// One flattened instruction. Which fields are populated depends on `kind`;
/// the rest keep their defaults.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CircuitInstruction {
    pub kind: CircuitInstructionKind,
    /// Single-probability channels, and the measurement flip probability of
    /// measurement-like instructions.
    pub probability: f64,
    /// Rotation angle in the kernel convention `exp(-i * angle * P)`. Format
    /// half-turn arguments are converted by the frontend, not stored raw.
    pub kernel_angle: f64,
    pub qubits: Vec<usize>,
    pub measurement_targets: Vec<CircuitMeasurementTarget>,
    pub pauli_products: Vec<CircuitPauliProduct>,
    pub feedback_targets: Vec<CircuitFeedbackTarget>,
    /// Multi-probability channels (`PAULI_CHANNEL_*`, `HERALDED_PAULI_CHANNEL_1`)
    /// and the absolute per-alternative probabilities of a correlated-error group.
    pub probabilities: Vec<f64>,
    /// Index of this instruction's first expectation-value slot, for `EXP_VAL`.
    pub exp_val: Option<usize>,
    /// 1-based source line.
    pub line: usize,
}

impl CircuitInstruction {
    pub fn new(kind: CircuitInstructionKind, line: usize) -> Self {
        Self {
            kind,
            line,
            ..Self::default()
        }
    }
}

/// A `DETECTOR` annotation: the parity of a set of measurement records.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CircuitDetector {
    /// 1-based record indices whose XOR forms the detector outcome.
    pub records: Vec<usize>,
    /// Declared coordinates plus the accumulated `SHIFT_COORDS` offset.
    pub coords: Vec<f64>,
    pub line: usize,
    /// Number of instructions emitted before this detector, i.e. the point in
    /// the instruction stream at which its records are all available.
    pub after_instruction: usize,
    /// Same position expressed in lowered pending operations. Filled in by the
    /// lowering pass; zero until then.
    pub after_pending_operation: usize,
    /// Whether this detector came from a `DISCARD` declaration.
    pub discard: bool,
}

/// An `OBSERVABLE_INCLUDE` annotation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CircuitObservableInclude {
    pub index: usize,
    /// 1-based record indices. Pauli targets are accepted and dropped.
    pub records: Vec<usize>,
    pub line: usize,
}

/// A fully flattened circuit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QuantumCircuit {
    pub nqubits: usize,
    pub nrecords: usize,
    pub nexpvals: usize,
    pub instructions: Vec<CircuitInstruction>,
    pub detectors: Vec<CircuitDetector>,
    pub observables: Vec<CircuitObservableInclude>,
}

impl QuantumCircuit {
    /// Number of logical observables, defined as the largest declared index
    /// plus one — not the number of `OBSERVABLE_INCLUDE` instructions, several
    /// of which may contribute to the same observable.
    #[cfg(test)]
    pub fn num_observables(&self) -> usize {
        self.observables
            .iter()
            .map(|observable| observable.index + 1)
            .max()
            .unwrap_or(0)
    }
}
