//! Typed errors for the stabilizer-frame engine.

/// Errors returned by [`TableauSimulator`](crate::tableau_simulator::TableauSimulator) operations.
///
/// Every variant models a condition the caller can provoke and reasonably
/// handle; internal invariants (a decomposed net phase that is not real, a
/// partner label missing from the amplitude map) are enforced with
/// `debug_assert!` rather than surfaced here.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SimError {
    /// Pruning removed every amplitude and would produce a zero state.
    #[error("prune epsilon {epsilon} removed every amplitude")]
    EmptyStateAfterPruning {
        /// The pruning threshold that erased the state.
        epsilon: f64,
    },

    /// A `T` or measurement pushed the live-label count past the configured cap.
    ///
    /// Stabilizer rank grows by at most a factor of two per `T`, so this bounds
    /// the exponential blow-up of magic-heavy circuits.
    #[error("stabilizer rank {rank} exceeds cap {cap}")]
    RankOverflow {
        /// Live-label count that triggered the overflow.
        rank: usize,
        /// The live-label ceiling in force.
        cap: usize,
    },

    /// A forced measurement outcome has (numerically) zero probability, so the
    /// requested post-selection is unreachable from the current state.
    #[error("cannot post-select outcome {outcome} with probability {probability:e}")]
    PostselectImpossible {
        /// The requested (impossible) outcome.
        outcome: bool,
        /// Probability the engine computed for it.
        probability: f64,
    },

    /// A multi-qubit gate or Clifford support received one qubit more than once.
    #[error("gate received repeated qubit index {0}")]
    RepeatedQubit(usize),

    /// A controlled-Pauli operation requires commuting control and target axes.
    #[error("controlled-Pauli control and target must commute")]
    NonCommutingControlledPaulis,

    /// A controlled Pauli carries a sign or phase this API does not represent:
    /// conditioning on `−P` is a different operation, not a global phase.
    #[error("controlled-Pauli axes must be positive Hermitian Paulis")]
    InvalidControlledPauli,

    /// A measurement or rotation axis has an imaginary coefficient, so it is not
    /// an observable.
    #[error("measurement and rotation axes must be Hermitian Paulis")]
    NonHermitianPauli,

    /// A batched conditional Pauli named a measurement its own batch has not
    /// produced. Records are indexed from the start of the batch, so this is a
    /// malformed instruction stream rather than a state the run reached.
    #[error("conditional Pauli reads batch record {index}, which the batch has not produced")]
    MissingBatchRecord {
        /// The out-of-range record index.
        index: usize,
    },

    /// A read-only observable names a qubit outside the live register.
    #[error("observable qubit index {index} is outside a {num_qubits}-qubit register")]
    QubitIndexOutOfRange {
        /// Largest out-of-range support index.
        index: usize,
        /// Live simulator width.
        num_qubits: usize,
    },
}
