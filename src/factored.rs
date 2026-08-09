//! The factored simulator state and the instruction stream it plans into.
//!
//! # The two states
//!
//! [`FrameFactoredState`] is what a circuit is applied to. Cliffords are
//! absorbed into a tableau and never touch amplitudes; everything non-Clifford
//! (rotations, measurements, classical records) is *queued* as a
//! [`PendingOperation`] whose Pauli has already been pushed through both frames,
//! so the queue is expressed in the frames' coordinates rather than the
//! circuit's.
//!
//! [`PendingFactoredState`] is that queue handed to the planner
//! ([`crate::planner`]), which drains it into [`FactoredInstruction`]s. Only six
//! opcodes reach the sampler, and only three of them do quantum work.
//!
//! # Active and dormant qubits
//!
//! Qubits `[0, k)` are *active*: they have real amplitudes in a dense `2^k`
//! vector. Qubits `[k, n)` are *dormant*: still `|0>` up to the tableau, holding
//! only a classical bit. `k` moves during planning — a rotation that touches a
//! dormant qubit promotes it, and a measurement of an active Pauli retires one.
//!
//! # Equality
//!
//! Instruction equality ignores [`SymbolicBoolEvaluationPlan`]s because they
//! are derived from the symbolic expression next to them. Precomputed kernels
//! are the canonical runtime representation and are compared directly.

use crate::active::{
    PrecomputedActivePauliMeasurementKernel, PrecomputedActivePauliRotationKernel,
};
use crate::errors::{Result, TicitError};
use crate::frames::{
    ActivePauliFrame, CliffordFrame, ConditionalPauliString, DormantState, SymbolicPauliString,
    conjugate_by, preimage,
};
use crate::pauli::PauliString;
use crate::symbolic::{
    SymbolicBool, SymbolicBoolEvaluationPlan, SymbolicCategoricalDistribution, SymbolicContext,
    xor_bool,
};

// ==============================================================================
// Pending operations
// ==============================================================================

/// A queued `exp(-i * kernel_angle * P)`, where a true `sign` negates the angle.
#[derive(Clone, Debug, Default)]
pub struct PendingPauliRotation {
    pub kernel_angle: f64,
    pub pauli: SymbolicPauliString,
}

impl PartialEq for PendingPauliRotation {
    fn eq(&self, other: &Self) -> bool {
        self.kernel_angle == other.kernel_angle && self.pauli == other.pauli
    }
}

/// A queued Pauli measurement.
///
/// `record` is the 1-based measurement record it writes; `record_condition` is
/// the symbol later operations use to refer to its outcome. `exp_val`, when set,
/// turns this into a non-destructive expectation probe that writes an expectation
/// slot instead of sampling — it rides the measurement path so that neither the
/// planner nor the sampler needs a second instruction family.
#[derive(Clone, Debug, Default)]
pub struct PendingPauliMeasurement {
    pub pauli: SymbolicPauliString,
    pub record: Option<i32>,
    pub record_condition: Option<i32>,
    pub exp_val: Option<i32>,
}

impl PartialEq for PendingPauliMeasurement {
    fn eq(&self, other: &Self) -> bool {
        self.pauli == other.pauli
            && self.record == other.record
            && self.record_condition == other.record_condition
            && self.exp_val == other.exp_val
    }
}

/// A record written from a purely classical expression — no quantum action.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PendingClassicalRecord {
    pub outcome: SymbolicBool,
    pub record: Option<i32>,
    pub record_condition: Option<i32>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PendingOperation {
    PauliRotation(PendingPauliRotation),
    PauliMeasurement(PendingPauliMeasurement),
    ClassicalRecord(PendingClassicalRecord),
}

impl PendingOperation {
    /// The operation's Pauli body, or `None` for a classical record.
    ///
    /// The optimizer and the planner both branch on this rather than on the
    /// variant, since "does it have a quantum body" is the question they ask.
    pub fn pauli(&self) -> Option<&SymbolicPauliString> {
        match self {
            Self::PauliRotation(rotation) => Some(&rotation.pauli),
            Self::PauliMeasurement(measurement) => Some(&measurement.pauli),
            Self::ClassicalRecord(_) => None,
        }
    }

    pub fn pauli_mut(&mut self) -> Option<&mut SymbolicPauliString> {
        match self {
            Self::PauliRotation(rotation) => Some(&mut rotation.pauli),
            Self::PauliMeasurement(measurement) => Some(&mut measurement.pauli),
            Self::ClassicalRecord(_) => None,
        }
    }

    #[cfg(test)]
    pub fn record_condition(&self) -> Option<i32> {
        match self {
            Self::PauliRotation(_) => None,
            Self::PauliMeasurement(measurement) => measurement.record_condition,
            Self::ClassicalRecord(record) => record.record_condition,
        }
    }

    /// Largest condition id anywhere in the operation, for
    /// [`SymbolicContext::bump_next_condition`].
    pub fn max_condition(&self) -> i32 {
        match self {
            Self::PauliRotation(rotation) => rotation.pauli.sign.max_condition(),
            Self::PauliMeasurement(measurement) => measurement
                .pauli
                .sign
                .max_condition()
                .max(measurement.record_condition.unwrap_or(0)),
            Self::ClassicalRecord(record) => record
                .outcome
                .max_condition()
                .max(record.record_condition.unwrap_or(0)),
        }
    }
}

impl From<PendingPauliRotation> for PendingOperation {
    fn from(rotation: PendingPauliRotation) -> Self {
        Self::PauliRotation(rotation)
    }
}

impl From<PendingPauliMeasurement> for PendingOperation {
    fn from(measurement: PendingPauliMeasurement) -> Self {
        Self::PauliMeasurement(measurement)
    }
}

impl From<PendingClassicalRecord> for PendingOperation {
    fn from(record: PendingClassicalRecord) -> Self {
        Self::ClassicalRecord(record)
    }
}

// ==============================================================================
// Planned instructions
// ==============================================================================

/// `exp(-i * kernel_angle * P)` on the dense active vector.
#[derive(Clone, Debug, Default)]
pub struct ApplyPrecomputedActivePauliRotation {
    pub rotation_kernel: PrecomputedActivePauliRotationKernel,
    pub sign: SymbolicBool,
    pub sign_plan: SymbolicBoolEvaluationPlan,
}

impl PartialEq for ApplyPrecomputedActivePauliRotation {
    /// The evaluation plan is derived from the symbolic sign.
    fn eq(&self, other: &Self) -> bool {
        self.rotation_kernel == other.rotation_kernel && self.sign == other.sign
    }
}

/// Promotes a dormant qubit into the active vector, doubling it.
#[derive(Clone, Debug, Default)]
pub struct PromoteDormantRotation {
    pub kernel_angle: f64,
    pub sign: SymbolicBool,
    pub sign_plan: SymbolicBoolEvaluationPlan,
}

impl PartialEq for PromoteDormantRotation {
    fn eq(&self, other: &Self) -> bool {
        self.kernel_angle == other.kernel_angle && self.sign == other.sign
    }
}

/// Writes a record whose value is already determined by earlier symbols.
#[derive(Clone, Debug, Default)]
pub struct RecordMeasurement {
    pub outcome: SymbolicBool,
    pub record: Option<i32>,
    pub record_condition: Option<i32>,
    pub outcome_plan: SymbolicBoolEvaluationPlan,
    pub exp_val: Option<i32>,
}

impl PartialEq for RecordMeasurement {
    fn eq(&self, other: &Self) -> bool {
        self.outcome == other.outcome
            && self.record == other.record
            && self.record_condition == other.record_condition
            && self.exp_val == other.exp_val
    }
}

/// A detector: the XOR of some measurement records, or a symbolic expression.
///
/// The planner never emits these — the frontend injects them after planning,
/// anchored on [`PendingFactoredState::pending_prefix_instruction_indices`].
#[derive(Clone, Debug, Default)]
pub struct RecordDetector {
    pub outcome: SymbolicBool,
    pub records: Vec<i32>,
    pub detector: i32,
    pub outcome_plan: SymbolicBoolEvaluationPlan,
    pub postselect: bool,
}

impl PartialEq for RecordDetector {
    fn eq(&self, other: &Self) -> bool {
        self.outcome == other.outcome
            && self.records == other.records
            && self.detector == other.detector
            && self.postselect == other.postselect
    }
}

/// Born-rule samples an active Pauli, then projects and halves the vector.
#[derive(Clone, Debug, Default)]
pub struct MeasurePrecomputedActivePauli {
    pub kernel: PrecomputedActivePauliMeasurementKernel,
    pub branch: i32,
    pub outcome: SymbolicBool,
    pub record: Option<i32>,
    pub record_condition: Option<i32>,
    pub outcome_plan: SymbolicBoolEvaluationPlan,
    pub exp_val: Option<i32>,
}

impl PartialEq for MeasurePrecomputedActivePauli {
    fn eq(&self, other: &Self) -> bool {
        self.kernel == other.kernel
            && self.branch == other.branch
            && self.outcome == other.outcome
            && self.record == other.record
            && self.record_condition == other.record_condition
            && self.exp_val == other.exp_val
    }
}

/// A fair coin: measuring a dormant qubit in a basis it is unbiased in costs no
/// quantum work at all, only a fresh symbol.
#[derive(Clone, Debug, Default)]
pub struct IntroduceDormantMeasurementBranch {
    pub branch: i32,
    pub outcome: SymbolicBool,
    pub record: Option<i32>,
    pub record_condition: Option<i32>,
    pub outcome_plan: SymbolicBoolEvaluationPlan,
    pub exp_val: Option<i32>,
}

impl PartialEq for IntroduceDormantMeasurementBranch {
    fn eq(&self, other: &Self) -> bool {
        self.branch == other.branch
            && self.outcome == other.outcome
            && self.record == other.record
            && self.record_condition == other.record_condition
            && self.exp_val == other.exp_val
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FactoredInstruction {
    ApplyPrecomputedActivePauliRotation(ApplyPrecomputedActivePauliRotation),
    PromoteDormantRotation(PromoteDormantRotation),
    RecordMeasurement(RecordMeasurement),
    RecordDetector(RecordDetector),
    MeasurePrecomputedActivePauli(MeasurePrecomputedActivePauli),
    IntroduceDormantMeasurementBranch(IntroduceDormantMeasurementBranch),
}

impl FactoredInstruction {
    /// The 1-based measurement record this instruction writes, if any.
    pub fn record(&self) -> Option<i32> {
        match self {
            Self::ApplyPrecomputedActivePauliRotation(_)
            | Self::PromoteDormantRotation(_)
            | Self::RecordDetector(_) => None,
            Self::RecordMeasurement(instruction) => instruction.record,
            Self::MeasurePrecomputedActivePauli(instruction) => instruction.record,
            Self::IntroduceDormantMeasurementBranch(instruction) => instruction.record,
        }
    }

    pub fn record_condition(&self) -> Option<i32> {
        match self {
            Self::ApplyPrecomputedActivePauliRotation(_)
            | Self::PromoteDormantRotation(_)
            | Self::RecordDetector(_) => None,
            Self::RecordMeasurement(instruction) => instruction.record_condition,
            Self::MeasurePrecomputedActivePauli(instruction) => instruction.record_condition,
            Self::IntroduceDormantMeasurementBranch(instruction) => instruction.record_condition,
        }
    }

    pub fn exp_val(&self) -> Option<i32> {
        match self {
            Self::ApplyPrecomputedActivePauliRotation(_)
            | Self::PromoteDormantRotation(_)
            | Self::RecordDetector(_) => None,
            Self::RecordMeasurement(instruction) => instruction.exp_val,
            Self::MeasurePrecomputedActivePauli(instruction) => instruction.exp_val,
            Self::IntroduceDormantMeasurementBranch(instruction) => instruction.exp_val,
        }
    }

    pub fn detector(&self) -> Option<i32> {
        match self {
            Self::RecordDetector(instruction) => Some(instruction.detector),
            _ => None,
        }
    }

    /// The instruction's symbolic sign, for the two rotation opcodes.
    pub fn sign(&self) -> Option<&SymbolicBool> {
        match self {
            Self::ApplyPrecomputedActivePauliRotation(instruction) => Some(&instruction.sign),
            Self::PromoteDormantRotation(instruction) => Some(&instruction.sign),
            _ => None,
        }
    }

    pub fn sign_mut(&mut self) -> Option<&mut SymbolicBool> {
        match self {
            Self::ApplyPrecomputedActivePauliRotation(instruction) => Some(&mut instruction.sign),
            Self::PromoteDormantRotation(instruction) => Some(&mut instruction.sign),
            _ => None,
        }
    }

    /// The instruction's symbolic outcome, for the four recording opcodes.
    pub fn outcome(&self) -> Option<&SymbolicBool> {
        match self {
            Self::ApplyPrecomputedActivePauliRotation(_) | Self::PromoteDormantRotation(_) => None,
            Self::RecordMeasurement(instruction) => Some(&instruction.outcome),
            Self::RecordDetector(instruction) => Some(&instruction.outcome),
            Self::MeasurePrecomputedActivePauli(instruction) => Some(&instruction.outcome),
            Self::IntroduceDormantMeasurementBranch(instruction) => Some(&instruction.outcome),
        }
    }

    pub fn outcome_mut(&mut self) -> Option<&mut SymbolicBool> {
        match self {
            Self::ApplyPrecomputedActivePauliRotation(_) | Self::PromoteDormantRotation(_) => None,
            Self::RecordMeasurement(instruction) => Some(&mut instruction.outcome),
            Self::RecordDetector(instruction) => Some(&mut instruction.outcome),
            Self::MeasurePrecomputedActivePauli(instruction) => Some(&mut instruction.outcome),
            Self::IntroduceDormantMeasurementBranch(instruction) => Some(&mut instruction.outcome),
        }
    }

    /// Recompiles the evaluation plan after the expression next to it changed.
    pub fn refresh_plan(&mut self) {
        match self {
            Self::ApplyPrecomputedActivePauliRotation(instruction) => {
                instruction.sign_plan = SymbolicBoolEvaluationPlan::new(&instruction.sign);
            }
            Self::PromoteDormantRotation(instruction) => {
                instruction.sign_plan = SymbolicBoolEvaluationPlan::new(&instruction.sign);
            }
            Self::RecordMeasurement(instruction) => {
                instruction.outcome_plan = SymbolicBoolEvaluationPlan::new(&instruction.outcome);
            }
            Self::RecordDetector(instruction) => {
                instruction.outcome_plan = SymbolicBoolEvaluationPlan::new(&instruction.outcome);
            }
            Self::MeasurePrecomputedActivePauli(instruction) => {
                instruction.outcome_plan = SymbolicBoolEvaluationPlan::new(&instruction.outcome);
            }
            Self::IntroduceDormantMeasurementBranch(instruction) => {
                instruction.outcome_plan = SymbolicBoolEvaluationPlan::new(&instruction.outcome);
            }
        }
    }

    /// Largest condition id anywhere in the instruction, including the branch
    /// symbol the two sampling opcodes mint.
    pub fn max_condition(&self) -> i32 {
        let branch = match self {
            Self::MeasurePrecomputedActivePauli(instruction) => instruction.branch,
            Self::IntroduceDormantMeasurementBranch(instruction) => instruction.branch,
            _ => 0,
        };
        let expression = match (self.sign(), self.outcome()) {
            (Some(sign), _) => sign.max_condition(),
            (None, Some(outcome)) => outcome.max_condition(),
            (None, None) => 0,
        };
        branch
            .max(expression)
            .max(self.record_condition().unwrap_or(0))
    }
}

macro_rules! from_instruction {
    ($($variant:ident),* $(,)?) => {
        $(
            impl From<$variant> for FactoredInstruction {
                fn from(instruction: $variant) -> Self {
                    Self::$variant(instruction)
                }
            }
        )*
    };
}

from_instruction!(
    ApplyPrecomputedActivePauliRotation,
    PromoteDormantRotation,
    RecordMeasurement,
    RecordDetector,
    MeasurePrecomputedActivePauli,
    IntroduceDormantMeasurementBranch,
);

// ==============================================================================
// Exogenous sampling groups
// ==============================================================================

/// Bernoulli symbols that share a probability, so one geometric-gap walk can
/// serve all of them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BernoulliSampleGroup {
    pub probability: f64,
    pub conditions: Vec<i32>,
}

/// Categorical distributions that are almost always the all-false row, grouped
/// by identical parameters. `event_probabilities` are conditional on the event
/// happening at all, so the sampler can decide "did anything happen" first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RareCategoricalSampleGroup {
    pub event_probability: f64,
    pub nbits: usize,
    pub conditions: Vec<Vec<i32>>,
    pub assignments: Vec<Vec<u64>>,
    pub probabilities: Vec<f64>,
    pub event_rows: Vec<usize>,
    pub event_probabilities: Vec<f64>,
}

// ==============================================================================
// Frame-factored state
// ==============================================================================

/// A circuit's state as a Clifford tableau, a Pauli frame, dormant classical
/// bits, and a queue of everything that could not be absorbed into those.
#[derive(Clone, Debug)]
pub struct FrameFactoredState {
    pub n: usize,
    pub k: usize,
    pub clifford: CliffordFrame,
    pub active_frame: ActivePauliFrame,
    pub dormant: DormantState,
    pub context: SymbolicContext,
    pub pending_operations: Vec<PendingOperation>,
}

impl FrameFactoredState {
    pub fn new(n: usize, k: usize) -> Result<Self> {
        Self::with_context(n, k, SymbolicContext::new())
    }

    pub fn with_context(n: usize, k: usize, context: SymbolicContext) -> Result<Self> {
        if k > n {
            return Err(TicitError::new(
                "active qubit count exceeds total qubit count",
            ));
        }
        Ok(Self {
            n,
            k,
            clifford: CliffordFrame::new(n),
            // The Pauli frame spans all n qubits: corrections on dormant qubits
            // still have to be conjugated correctly when one is promoted.
            active_frame: ActivePauliFrame::new(n),
            dormant: DormantState::new(n - k),
            context,
            pending_operations: Vec::new(),
        })
    }
}

/// Pushes a Pauli through both frames, into the coordinates the queue speaks.
fn prepare_pending_pauli(state: &FrameFactoredState, pauli: &PauliString) -> SymbolicPauliString {
    assert_eq!(
        pauli.nqubits, state.n,
        "Pauli string dimension does not match frame-factored state"
    );
    let pre = preimage(&state.clifford, pauli);
    conjugate_by(&state.active_frame, &pre)
}

/// Absorbs a conditional Pauli correction into the Pauli frame.
pub fn apply_pauli_conditional(state: &mut FrameFactoredState, pauli: &ConditionalPauliString) {
    assert_eq!(
        pauli.pauli.nqubits, state.n,
        "Pauli string dimension does not match frame-factored state"
    );
    state.context.bump_next_condition(pauli.condition);
    let pre = preimage(&state.clifford, &pauli.pauli);
    if pre.has_nonidentity_body() {
        state
            .active_frame
            .add_pauli(&pre, pauli.condition, &mut state.context);
    }
}

pub fn apply_pauli(state: &mut FrameFactoredState, pauli: &PauliString, condition: i32) {
    apply_pauli_conditional(
        state,
        &ConditionalPauliString::new(pauli.clone(), condition),
    );
}

/// Applies a Pauli gated on a symbolic expression by decomposing it: one frame
/// term per symbol, plus an always-true term when the constant is set.
pub fn apply_pauli_symbolic(
    state: &mut FrameFactoredState,
    pauli: &PauliString,
    condition: &SymbolicBool,
) -> Result<()> {
    if condition.constant {
        // The frame can only gate on a symbol, so an unconditional correction
        // borrows a Bernoulli symbol that is true with probability 1.
        let always = state.context.fresh_bernoulli_condition(1.0)?;
        apply_pauli(state, pauli, always);
    }
    for &condition_id in &condition.conditions {
        apply_pauli(state, pauli, condition_id);
    }
    Ok(())
}

pub fn apply_pauli_rotation(
    state: &mut FrameFactoredState,
    pauli: &PauliString,
    kernel_angle: f64,
) -> PendingPauliRotation {
    let rotation = PendingPauliRotation {
        kernel_angle,
        pauli: prepare_pending_pauli(state, pauli),
    };
    state.pending_operations.push(rotation.clone().into());
    rotation
}

#[cfg(test)]
pub fn apply_pauli_measurement(
    state: &mut FrameFactoredState,
    pauli: &PauliString,
) -> PendingPauliMeasurement {
    let measurement = PendingPauliMeasurement {
        pauli: prepare_pending_pauli(state, pauli),
        ..PendingPauliMeasurement::default()
    };
    state.pending_operations.push(measurement.clone().into());
    measurement
}

/// [`apply_pauli_measurement`] with a known sign flip and explicit record slots.
pub fn apply_pauli_measurement_signed(
    state: &mut FrameFactoredState,
    pauli: &PauliString,
    sign: &SymbolicBool,
    record: Option<i32>,
    record_condition: Option<i32>,
) -> PendingPauliMeasurement {
    let prepared = prepare_pending_pauli(state, pauli);
    let measurement = PendingPauliMeasurement {
        pauli: SymbolicPauliString::with_sign(prepared.pauli, xor_bool(&prepared.sign, sign)),
        record,
        record_condition,
        exp_val: None,
    };
    state.pending_operations.push(measurement.clone().into());
    measurement
}

/// Queues a non-destructive expectation probe writing expectation slot
/// `exp_val`.
pub fn apply_pauli_expectation(
    state: &mut FrameFactoredState,
    pauli: &PauliString,
    exp_val: i32,
) -> Result<PendingPauliMeasurement> {
    if exp_val < 0 {
        return Err(TicitError::new(
            "expectation value index must be nonnegative",
        ));
    }
    let measurement = PendingPauliMeasurement {
        pauli: prepare_pending_pauli(state, pauli),
        record: None,
        record_condition: None,
        exp_val: Some(exp_val),
    };
    state.pending_operations.push(measurement.clone().into());
    Ok(measurement)
}

pub fn apply_classical_record(
    state: &mut FrameFactoredState,
    outcome: &SymbolicBool,
    record: Option<i32>,
    record_condition: Option<i32>,
) -> PendingClassicalRecord {
    state.context.bump_next_condition_for(outcome);
    let classical_record = PendingClassicalRecord {
        outcome: outcome.clone(),
        record,
        record_condition,
    };
    state
        .pending_operations
        .push(classical_record.clone().into());
    classical_record
}

// ==============================================================================
// Clifford gate forwarding
// ==============================================================================

// One-line forwards from the factored state to its Clifford frame, generated
// rather than written out: there are 46 of them and every one is identical.
// They are not re-exported at the crate root, since the names collide with the
// `crate::frames` originals; call them as `factored::left_h(&mut state, q)`.
macro_rules! forward_gate {
    ($($name:ident($($arg:ident),+)),* $(,)?) => {
        $(
            pub fn $name(state: &mut FrameFactoredState, $($arg: usize),+) {
                crate::frames::$name(&mut state.clifford, $($arg),+)
            }
        )*
    };
}

forward_gate!(
    left_h(q),
    left_h_nxy(q),
    left_h_nxz(q),
    left_h_nyz(q),
    left_h_xy(q),
    left_h_yz(q),
    left_c_nxyz(q),
    left_c_nzyx(q),
    left_c_xnyz(q),
    left_c_xynz(q),
    left_c_xyz(q),
    left_c_znyx(q),
    left_c_zynx(q),
    left_c_zyx(q),
    left_s(q),
    left_sdg(q),
    left_sqrt_x(q),
    left_sqrt_x_dag(q),
    left_sqrt_y(q),
    left_sqrt_y_dag(q),
    left_x(q),
    left_y(q),
    left_z(q),
    left_cx(control, target),
    left_cy(control, target),
    left_cz(a, b),
    left_swap(a, b),
    left_cxswap(a, b),
    left_czswap(a, b),
    left_iswap(a, b),
    left_iswap_dag(a, b),
    left_sqrt_xx(a, b),
    left_sqrt_xx_dag(a, b),
    left_sqrt_yy(a, b),
    left_sqrt_yy_dag(a, b),
    left_sqrt_zz(a, b),
    left_sqrt_zz_dag(a, b),
    left_swapcx(a, b),
    left_xcx(control, target),
    left_xcy(control, target),
    left_xcz(control, target),
    left_ycx(control, target),
    left_ycy(control, target),
    left_ycz(control, target),
);

// ==============================================================================
// Pending state handed to the planner
// ==============================================================================

/// The planner's working state: the operation queue plus everything the
/// symbolic-minimization machinery accumulates while draining it.
#[derive(Clone, Debug)]
pub struct PendingFactoredState {
    pub n: usize,
    pub initial_k: usize,
    pub k: usize,
    pub max_k: usize,
    pub dormant: DormantState,
    pub context: SymbolicContext,
    /// In expectation mode the planner cannot rewrite queued Paulis in place
    /// (an expectation probe must see the original coordinates), so basis
    /// changes compose into this frame and are applied lazily instead.
    pub pending_frame: CliffordFrame,
    pub pending_frame_active: bool,
    pub pending_operations: Vec<PendingOperation>,
    /// Expectation mode consumes the queue by advancing this instead of popping,
    /// because pushing signs forward needs the already-processed prefix intact.
    pub pending_operation_cursor: usize,
    /// Transposed occupancy bitset over queued Paulis: 64 operations per word,
    /// `[block * n + q]`. Built lazily, and only used in expectation mode where
    /// the linear scan would be quadratic.
    pub pending_x_operation_blocks: Vec<u64>,
    pub pending_z_operation_blocks: Vec<u64>,
    pub pending_operation_blocks_valid: bool,
    pub pending_relations: Vec<SymbolicBool>,
    pub pending_relation_words: Vec<Vec<usize>>,
    pub pending_relation_index: std::collections::HashMap<i32, Vec<usize>>,
    pub pending_substitutions: std::collections::HashMap<i32, SymbolicBool>,
    pub instructions: Vec<FactoredInstruction>,
    /// Instruction index before *and* after each processed operation. The
    /// frontend re-anchors detectors on these, so both checkpoints matter.
    pub pending_prefix_instruction_indices: Vec<i32>,
    pub pending_operations_optimized: bool,
    pub has_expectation: bool,
    pub next_record: i32,
}

impl PendingFactoredState {
    #[cfg(test)]
    pub fn new(n: usize, k: usize) -> Result<Self> {
        Self::with_context(n, k, SymbolicContext::new())
    }

    #[cfg(test)]
    pub fn with_context(n: usize, k: usize, context: SymbolicContext) -> Result<Self> {
        if k > n {
            return Err(TicitError::new(
                "active qubit count exceeds total qubit count",
            ));
        }
        Ok(Self {
            n,
            initial_k: k,
            k,
            max_k: k,
            dormant: DormantState::new(n - k),
            context,
            pending_frame: CliffordFrame::new(n),
            pending_frame_active: false,
            pending_operations: Vec::new(),
            pending_operation_cursor: 0,
            pending_x_operation_blocks: Vec::new(),
            pending_z_operation_blocks: Vec::new(),
            pending_operation_blocks_valid: false,
            pending_relations: Vec::new(),
            pending_relation_words: Vec::new(),
            pending_relation_index: std::collections::HashMap::new(),
            pending_substitutions: std::collections::HashMap::new(),
            instructions: Vec::new(),
            pending_prefix_instruction_indices: Vec::new(),
            pending_operations_optimized: false,
            has_expectation: false,
            next_record: 1,
        })
    }

    /// Lowers a frame-factored state: the queue is adopted as-is, and the scan
    /// picks up whether any expectation probe is present (which changes how the
    /// planner consumes the queue) and where record numbering resumes.
    pub fn from_frame_state(state: FrameFactoredState) -> Self {
        let FrameFactoredState {
            n,
            k,
            dormant,
            mut context,
            pending_operations,
            ..
        } = state;
        let dormant = DormantState::from_bits(dormant.bits, &mut context);
        let mut has_expectation = false;
        let mut next_record = 1;
        for operation in &pending_operations {
            context.bump_next_condition(operation.max_condition());
            if let PendingOperation::PauliMeasurement(measurement) = operation {
                if measurement.exp_val.is_some() {
                    has_expectation = true;
                }
                if let Some(record) = measurement.record {
                    next_record = next_record.max(record + 1);
                }
            }
        }
        Self {
            n,
            initial_k: k,
            k,
            max_k: k,
            dormant,
            context,
            pending_frame: CliffordFrame::new(n),
            pending_frame_active: false,
            pending_operations,
            pending_operation_cursor: 0,
            pending_x_operation_blocks: Vec::new(),
            pending_z_operation_blocks: Vec::new(),
            pending_operation_blocks_valid: false,
            pending_relations: Vec::new(),
            pending_relation_words: Vec::new(),
            pending_relation_index: std::collections::HashMap::new(),
            pending_substitutions: std::collections::HashMap::new(),
            instructions: Vec::new(),
            pending_prefix_instruction_indices: Vec::new(),
            pending_operations_optimized: false,
            has_expectation,
            next_record,
        }
    }
}

impl From<FrameFactoredState> for PendingFactoredState {
    fn from(state: FrameFactoredState) -> Self {
        Self::from_frame_state(state)
    }
}

/// What [`crate::pending_optimizer::optimize_pending_operations`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PendingOptimizationStats {
    pub input_operations: usize,
    pub output_operations: usize,
    pub fused_rotations: usize,
    pub cancelled_rotations: usize,
    pub measurement_left_swaps: usize,
    /// Where each requested prefix ended up. Entries are filled in for prefix 0,
    /// the end of the queue, and every preserved prefix; all others are -1.
    pub prefix_remap: Vec<i32>,
}

// ==============================================================================
// Planned program
// ==============================================================================

/// A fully planned circuit: a flat instruction stream plus the exogenous
/// sampling plan its symbols need.
#[derive(Clone, Debug, Default)]
pub struct FactoredInstructionProgram {
    pub n: usize,
    pub initial_k: usize,
    pub max_k: usize,
    pub instructions: Vec<FactoredInstruction>,
    pub pending_prefix_instruction_indices: Vec<i32>,
    pub context: SymbolicContext,
    pub nsymbols: usize,
    pub nrecords: usize,
    pub ndetectors: usize,
    pub nexpvals: usize,
    pub sampled_categorical_distributions: Vec<SymbolicCategoricalDistribution>,
    pub sampled_rare_categorical_groups: Vec<RareCategoricalSampleGroup>,
    pub sampled_bernoulli_conditions: Vec<i32>,
    pub sampled_bernoulli_probabilities: Vec<f64>,
    pub sampled_low_probability_bernoulli_groups: Vec<BernoulliSampleGroup>,
    /// Always safe to ignore; set only when the cost model predicts a real win.
    pub active_component_plan: Option<std::sync::Arc<crate::component_plan::ActiveComponentPlan>>,
    pub use_active_components: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active::ActivePauliAction;
    use crate::pauli::{pauli_x, pauli_z};
    use crate::symbolic::symbolic_bool;

    #[test]
    fn planned_instruction_stays_compact() {
        assert_eq!(std::mem::size_of::<FactoredInstruction>(), 256);
    }

    #[test]
    fn rotation_equality_ignores_the_derived_plan() {
        let pauli = pauli_x(2, 0);
        let action = ActivePauliAction::new(&pauli).expect("X is Hermitian");
        let sign = symbolic_bool(3);
        let rotation_kernel =
            PrecomputedActivePauliRotationKernel::new(&action, 0.5).expect("k < 62");
        let with_plan = ApplyPrecomputedActivePauliRotation {
            rotation_kernel,
            sign: sign.clone(),
            sign_plan: SymbolicBoolEvaluationPlan::new(&sign),
        };
        let without_plan = ApplyPrecomputedActivePauliRotation {
            rotation_kernel: PrecomputedActivePauliRotationKernel::new(&action, 0.5)
                .expect("k < 62"),
            sign,
            ..ApplyPrecomputedActivePauliRotation::default()
        };
        assert_eq!(with_plan, without_plan);
    }

    #[test]
    fn applying_a_pauli_to_a_frame_state_queues_a_frame_term() {
        let mut state = FrameFactoredState::new(2, 0).expect("k <= n");
        apply_pauli(&mut state, &pauli_x(2, 0), 4);
        assert_eq!(state.active_frame.terms.len(), 1);
        assert_eq!(state.active_frame.terms[0].condition, 4);
        // An identity preimage contributes nothing at all.
        apply_pauli(&mut state, &PauliString::new(2), 5);
        assert_eq!(state.active_frame.terms.len(), 1);
    }

    #[test]
    fn lowering_detects_expectation_probes_and_record_numbering() {
        let mut state = FrameFactoredState::new(1, 0).expect("k <= n");
        apply_pauli_measurement_signed(
            &mut state,
            &pauli_z(1, 0),
            &SymbolicBool::default(),
            Some(7),
            None,
        );
        apply_pauli_expectation(&mut state, &pauli_z(1, 0), 0).expect("nonnegative slot");
        let pending = PendingFactoredState::from_frame_state(state);
        assert!(pending.has_expectation);
        assert_eq!(pending.next_record, 8);
    }
}
