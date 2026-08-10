//! Lowering from circuit IR to a [`FrameFactoredState`].
//!
//! One pass absorbs Clifford gates into the frame, queues rotations and
//! measurements, and turns noise channels into symbolic random variables.

use crate::bits::packed_bits;
use crate::circuit::ir::{
    Circuit, CircuitInstruction, CircuitInstructionKind as Kind, CircuitMeasurementTarget,
    CircuitPauliProduct,
};
use crate::errors::{Result, TicitError};
use crate::factored::{
    self, FrameFactoredState, apply_classical_record, apply_pauli, apply_pauli_expectation,
    apply_pauli_measurement_signed, apply_pauli_rotation, apply_pauli_symbolic,
};
use crate::pauli::{PauliString, pauli_x, pauli_y, pauli_z};
use crate::symbolic::{SymbolicBool, SymbolicContext, symbolic_bool, xor_bool};

/// Everything the frontend needs from lowering one circuit.
#[derive(Clone, Debug)]
pub struct CircuitLoweringResult {
    pub state: FrameFactoredState,
    /// One symbolic outcome per measurement record, 1-based downstream.
    pub measurement_records: Vec<SymbolicBool>,
    /// Prefix table: entry `i` is the pending-operation count after the first
    /// `i` instructions. Length `ninstructions + 1`, seeded with 0. Detector
    /// positions are translated through this table.
    pub instruction_pending_operation_counts: Vec<usize>,
}

// ==============================================================================
// Small helpers shared across instruction kinds
// ==============================================================================

fn check_probability(probability: f64) -> Result<f64> {
    // The negated comparison rejects NaN.
    if !(0.0..=1.0).contains(&probability) {
        return Err(TicitError::new("probability must be between 0 and 1"));
    }
    Ok(probability)
}

/// Mints the next record: returns its 1-based index and outcome condition.
fn reserve_measurement_record(
    records: &mut Vec<SymbolicBool>,
    context: &mut SymbolicContext,
) -> (i32, i32) {
    let condition = context.fresh_condition();
    records.push(symbolic_bool(condition));
    (records.len() as i32, condition)
}

fn measurement_sign_with_readout_error(
    context: &mut SymbolicContext,
    inverted: bool,
    probability: f64,
) -> Result<SymbolicBool> {
    let mut sign = SymbolicBool::from(inverted);
    // Exact zero skips the flip symbol entirely; condition ids (and therefore
    // the downstream RNG draw order) depend on this, so no epsilon.
    if probability != 0.0 {
        sign = xor_bool(
            &sign,
            &context.fresh_bernoulli_bool(check_probability(probability)?)?,
        );
    }
    Ok(sign)
}

fn pauli_on_axis(n: usize, axis: char, q: usize) -> PauliString {
    match axis {
        'X' => pauli_x(n, q),
        'Y' => pauli_y(n, q),
        'Z' => pauli_z(n, q),
        _ => unreachable!("axis characters are minted internally"),
    }
}

fn feedback_pauli(state: &FrameFactoredState, kind: Kind, q: usize) -> Result<PauliString> {
    match kind {
        Kind::FeedbackX => Ok(pauli_x(state.n, q)),
        Kind::FeedbackY => Ok(pauli_y(state.n, q)),
        Kind::FeedbackZ => Ok(pauli_z(state.n, q)),
        _ => Err(TicitError::unsupported("unsupported feedback kind")),
    }
}

fn record_symbol(records: &[SymbolicBool], record: usize) -> Result<SymbolicBool> {
    if record == 0 || record > records.len() {
        return Err(TicitError::new("measurement record target out of range"));
    }
    Ok(records[record - 1].clone())
}

fn require_target_multiple(
    instruction: &CircuitInstruction,
    multiple: usize,
    message: &str,
) -> Result<()> {
    if multiple == 0 || !instruction.qubits.len().is_multiple_of(multiple) {
        return Err(TicitError::new(message));
    }
    Ok(())
}

fn require_distinct_targets(qubits: &[usize]) -> Result<()> {
    for i in 0..qubits.len() {
        for j in i + 1..qubits.len() {
            if qubits[i] == qubits[j] {
                return Err(TicitError::new(
                    "multi-qubit operation requires distinct qubits",
                ));
            }
        }
    }
    Ok(())
}

/// Axis code → (x, z) bits, in the channel encoding I=0, X=1, Y=2, Z=3.
fn pauli_axis_bits(axis: usize) -> (bool, bool) {
    match axis {
        0 => (false, false),
        1 => (true, false),
        2 => (true, true),
        3 => (false, true),
        _ => unreachable!("axis codes are two bits"),
    }
}

/// The `!`-inverted flag of a rotated/errored product becomes a phase flip.
fn maybe_inverted_pauli_product(product: &CircuitPauliProduct) -> PauliString {
    let mut pauli = product.pauli.clone();
    if product.inverted {
        pauli.phase_shift(2);
    }
    pauli
}

// ==============================================================================
// Noise channels — each becomes one categorical distribution
// ==============================================================================

fn apply_depolarize1(state: &mut FrameFactoredState, q: usize, probability: f64) -> Result<()> {
    let probability = check_probability(probability)?;
    let assignments = [
        packed_bits(&[false, false]),
        packed_bits(&[false, true]),
        packed_bits(&[true, false]),
        packed_bits(&[true, true]),
    ];
    let third = probability / 3.0;
    let bits = state.context.fresh_categorical_bools(
        2,
        &assignments,
        &[1.0 - probability, third, third, third],
    )?;
    apply_pauli_symbolic(state, &pauli_x(state.n, q), &bits[0])?;
    apply_pauli_symbolic(state, &pauli_z(state.n, q), &bits[1])
}

fn apply_depolarize2(
    state: &mut FrameFactoredState,
    q1: usize,
    q2: usize,
    probability: f64,
) -> Result<()> {
    let probability = check_probability(probability)?;
    let mut assignments = vec![packed_bits(&[false, false, false, false])];
    for x1 in [false, true] {
        for z1 in [false, true] {
            for x2 in [false, true] {
                for z2 in [false, true] {
                    if x1 || z1 || x2 || z2 {
                        assignments.push(packed_bits(&[x1, z1, x2, z2]));
                    }
                }
            }
        }
    }
    let mut probabilities = vec![probability / 15.0; 16];
    probabilities[0] = 1.0 - probability;
    let bits = state
        .context
        .fresh_categorical_bools(4, &assignments, &probabilities)?;
    apply_pauli_symbolic(state, &pauli_x(state.n, q1), &bits[0])?;
    apply_pauli_symbolic(state, &pauli_z(state.n, q1), &bits[1])?;
    apply_pauli_symbolic(state, &pauli_x(state.n, q2), &bits[2])?;
    apply_pauli_symbolic(state, &pauli_z(state.n, q2), &bits[3])
}

/// `PAULI_CHANNEL_n`: a `4^n`-way categorical over `2n` (x, z) bit pairs. Axis
/// codes are read most-significant-target-first, so target `i`'s bits sit at
/// positions `2i`/`2i+1` while its axis code is extracted from the high end of
/// the case number.
fn apply_pauli_channel(
    state: &mut FrameFactoredState,
    qubits: &[usize],
    probabilities: &[f64],
) -> Result<()> {
    require_distinct_targets(qubits)?;
    let n = qubits.len();
    let cases = 1usize << (2 * n);
    if probabilities.len() != cases - 1 {
        return Err(TicitError::new(
            "Pauli channel probability count does not match target arity",
        ));
    }
    let mut assignments = Vec::with_capacity(cases);
    assignments.push(packed_bits(&vec![false; 2 * n]));
    let mut total_error_probability = 0.0;
    for &probability in probabilities {
        total_error_probability += check_probability(probability)?;
    }
    if total_error_probability > 1.0 + 1e-12 {
        return Err(TicitError::new(
            "Pauli channel probabilities sum to more than 1",
        ));
    }
    let mut distribution = Vec::with_capacity(cases);
    distribution.push((1.0 - total_error_probability).max(0.0));
    for code in 1..cases {
        let mut assignment = vec![false; 2 * n];
        let mut remainder = code;
        for pos in (0..n).rev() {
            let (x, z) = pauli_axis_bits(remainder & 3);
            remainder >>= 2;
            assignment[2 * pos] = x;
            assignment[2 * pos + 1] = z;
        }
        assignments.push(packed_bits(&assignment));
        distribution.push(probabilities[code - 1]);
    }
    let bits = state
        .context
        .fresh_categorical_bools(2 * n, &assignments, &distribution)?;
    for (i, &q) in qubits.iter().enumerate() {
        apply_pauli_symbolic(state, &pauli_x(state.n, q), &bits[2 * i])?;
        apply_pauli_symbolic(state, &pauli_z(state.n, q), &bits[2 * i + 1])?;
    }
    Ok(())
}

fn apply_depolarize_n(
    state: &mut FrameFactoredState,
    qubits: &[usize],
    probability: f64,
) -> Result<()> {
    let probability = check_probability(probability)?;
    let cases = 1usize << (2 * qubits.len());
    let uniform = vec![probability / (cases - 1) as f64; cases - 1];
    apply_pauli_channel(state, qubits, &uniform)
}

/// A flushed `E`/`ELSE_CORRELATED_ERROR` chain: one alternative per product,
/// probabilities already made absolute by the parser.
fn apply_pauli_product_channel(
    state: &mut FrameFactoredState,
    products: &[CircuitPauliProduct],
    probabilities: &[f64],
) -> Result<()> {
    if products.len() != probabilities.len() {
        return Err(TicitError::new(
            "Pauli product channel probability count mismatch",
        ));
    }
    let mut assignments = Vec::with_capacity(products.len() + 1);
    assignments.push(packed_bits(&vec![false; products.len()]));
    let mut total = 0.0;
    for &probability in probabilities {
        total += check_probability(probability)?;
    }
    if total > 1.0 + 1e-12 {
        return Err(TicitError::new(
            "Pauli product channel probabilities sum to more than 1",
        ));
    }
    let mut distribution = Vec::with_capacity(products.len() + 1);
    distribution.push((1.0 - total).max(0.0));
    for i in 0..products.len() {
        let mut assignment = vec![false; products.len()];
        assignment[i] = true;
        assignments.push(packed_bits(&assignment));
        distribution.push(probabilities[i]);
    }
    let bits =
        state
            .context
            .fresh_categorical_bools(products.len(), &assignments, &distribution)?;
    for (i, product) in products.iter().enumerate() {
        apply_pauli_symbolic(state, &maybe_inverted_pauli_product(product), &bits[i])?;
    }
    Ok(())
}

/// Heralded channels: a 5-way categorical over 3 bits (herald, x, z) with the
/// assignment table {000, 100, 110, 111, 101} = {no event, herald+I, herald+X,
/// herald+Y, herald+Z}. The herald bit becomes a measurement record.
fn apply_heralded_channel(
    state: &mut FrameFactoredState,
    records: &mut Vec<SymbolicBool>,
    q: usize,
    probabilities: &[f64],
) -> Result<()> {
    if probabilities.len() != 4 {
        return Err(TicitError::new(
            "heralded Pauli channel expects four probabilities",
        ));
    }
    let mut total = 0.0;
    for &probability in probabilities {
        total += check_probability(probability)?;
    }
    if total > 1.0 + 1e-12 {
        return Err(TicitError::new(
            "heralded channel probabilities sum to more than 1",
        ));
    }
    let assignments = [
        packed_bits(&[false, false, false]),
        packed_bits(&[true, false, false]),
        packed_bits(&[true, true, false]),
        packed_bits(&[true, true, true]),
        packed_bits(&[true, false, true]),
    ];
    let bits = state.context.fresh_categorical_bools(
        3,
        &assignments,
        &[
            1.0 - total,
            probabilities[0],
            probabilities[1],
            probabilities[2],
            probabilities[3],
        ],
    )?;
    records.push(bits[0].clone());
    apply_classical_record(state, &bits[0], Some(records.len() as i32), None);
    apply_pauli_symbolic(state, &pauli_x(state.n, q), &bits[1])?;
    apply_pauli_symbolic(state, &pauli_z(state.n, q), &bits[2])
}

/// `MPAD`: the target is the literal pad value, optionally noisy and inverted.
fn apply_mpad(
    state: &mut FrameFactoredState,
    records: &mut Vec<SymbolicBool>,
    target: &CircuitMeasurementTarget,
    probability: f64,
) -> Result<()> {
    if target.qubit != 0 && target.qubit != 1 {
        return Err(TicitError::new("MPAD targets must be 0 or 1"));
    }
    let mut outcome = SymbolicBool::from((target.qubit != 0) != target.inverted);
    if probability != 0.0 {
        outcome = xor_bool(
            &outcome,
            &state
                .context
                .fresh_bernoulli_bool(check_probability(probability)?)?,
        );
    }
    records.push(outcome.clone());
    apply_classical_record(state, &outcome, Some(records.len() as i32), None);
    Ok(())
}

// ==============================================================================
// Measurements, resets, feedback
// ==============================================================================

fn apply_record_feedback(
    state: &mut FrameFactoredState,
    records: &[SymbolicBool],
    instruction: &CircuitInstruction,
) -> Result<()> {
    for target in &instruction.feedback_targets {
        let pauli = feedback_pauli(state, instruction.kind, target.qubit)?;
        let condition = record_symbol(records, target.record)?;
        apply_pauli_symbolic(state, &pauli, &condition)?;
    }
    Ok(())
}

/// Reset = unrecorded measurement plus a correction conditioned on its branch.
fn apply_reset(
    state: &mut FrameFactoredState,
    measurement_pauli: &PauliString,
    correction_pauli: &PauliString,
) {
    let condition = state.context.fresh_condition();
    apply_pauli_measurement_signed(
        state,
        measurement_pauli,
        &SymbolicBool::default(),
        None,
        Some(condition),
    );
    apply_pauli(state, correction_pauli, condition);
}

fn apply_measurement_reset(
    state: &mut FrameFactoredState,
    records: &mut Vec<SymbolicBool>,
    measurement_pauli: &PauliString,
    correction_pauli: &PauliString,
    target: &CircuitMeasurementTarget,
    probability: f64,
) -> Result<()> {
    let sign =
        measurement_sign_with_readout_error(&mut state.context, target.inverted, probability)?;
    let (record, record_condition) = reserve_measurement_record(records, &mut state.context);
    apply_pauli_measurement_signed(
        state,
        measurement_pauli,
        &sign,
        Some(record),
        Some(record_condition),
    );
    // The correction undoes the *physical* branch, which is the recorded bit
    // with the readout error taken back out.
    let correction_condition = xor_bool(&symbolic_bool(record_condition), &sign);
    apply_pauli_symbolic(state, correction_pauli, &correction_condition)
}

fn apply_measurement(
    state: &mut FrameFactoredState,
    records: &mut Vec<SymbolicBool>,
    axis: char,
    target: &CircuitMeasurementTarget,
    probability: f64,
) -> Result<()> {
    let sign =
        measurement_sign_with_readout_error(&mut state.context, target.inverted, probability)?;
    let (record, record_condition) = reserve_measurement_record(records, &mut state.context);
    apply_pauli_measurement_signed(
        state,
        &pauli_on_axis(state.n, axis, target.qubit),
        &sign,
        Some(record),
        Some(record_condition),
    );
    Ok(())
}

// ==============================================================================
// The instruction dispatch
// ==============================================================================

fn apply_single_qubit_clifford(state: &mut FrameFactoredState, kind: Kind, q: usize) {
    use factored as f;
    match kind {
        Kind::H => f::left_h(state, q),
        Kind::HNegXy => f::left_h_nxy(state, q),
        Kind::HNegXz => f::left_h_nxz(state, q),
        Kind::HNegYz => f::left_h_nyz(state, q),
        Kind::HXy => f::left_h_xy(state, q),
        Kind::HYz => f::left_h_yz(state, q),
        Kind::CNegXyz => f::left_c_nxyz(state, q),
        Kind::CNegZyx => f::left_c_nzyx(state, q),
        Kind::CXNegYz => f::left_c_xnyz(state, q),
        Kind::CXyNegZ => f::left_c_xynz(state, q),
        Kind::CXyz => f::left_c_xyz(state, q),
        Kind::CZNegYx => f::left_c_znyx(state, q),
        Kind::CZyNegX => f::left_c_zynx(state, q),
        Kind::CZyx => f::left_c_zyx(state, q),
        Kind::S => f::left_s(state, q),
        Kind::SDag => f::left_sdg(state, q),
        Kind::SqrtX => f::left_sqrt_x(state, q),
        Kind::SqrtXDag => f::left_sqrt_x_dag(state, q),
        Kind::SqrtY => f::left_sqrt_y(state, q),
        Kind::SqrtYDag => f::left_sqrt_y_dag(state, q),
        Kind::X => f::left_x(state, q),
        Kind::Y => f::left_y(state, q),
        Kind::Z => f::left_z(state, q),
        _ => unreachable!("dispatched only for single-qubit Clifford kinds"),
    }
}

fn apply_two_qubit_clifford(state: &mut FrameFactoredState, kind: Kind, a: usize, b: usize) {
    use factored as f;
    match kind {
        Kind::CX => f::left_cx(state, a, b),
        Kind::CY => f::left_cy(state, a, b),
        Kind::CZ => f::left_cz(state, a, b),
        Kind::Swap => f::left_swap(state, a, b),
        Kind::CxSwap => f::left_cxswap(state, a, b),
        Kind::CzSwap => f::left_czswap(state, a, b),
        Kind::ISwap => f::left_iswap(state, a, b),
        Kind::ISwapDag => f::left_iswap_dag(state, a, b),
        Kind::SqrtXx => f::left_sqrt_xx(state, a, b),
        Kind::SqrtXxDag => f::left_sqrt_xx_dag(state, a, b),
        Kind::SqrtYy => f::left_sqrt_yy(state, a, b),
        Kind::SqrtYyDag => f::left_sqrt_yy_dag(state, a, b),
        Kind::SqrtZz => f::left_sqrt_zz(state, a, b),
        Kind::SqrtZzDag => f::left_sqrt_zz_dag(state, a, b),
        Kind::SwapCx => f::left_swapcx(state, a, b),
        Kind::Xcx => f::left_xcx(state, a, b),
        Kind::Xcy => f::left_xcy(state, a, b),
        Kind::Xcz => f::left_xcz(state, a, b),
        Kind::Ycx => f::left_ycx(state, a, b),
        Kind::Ycy => f::left_ycy(state, a, b),
        Kind::Ycz => f::left_ycz(state, a, b),
        _ => unreachable!("dispatched only for two-qubit Clifford kinds"),
    }
}

struct Accumulator {
    state: FrameFactoredState,
    records: Vec<SymbolicBool>,
}

fn apply_instruction(acc: &mut Accumulator, instruction: &CircuitInstruction) -> Result<()> {
    let state = &mut acc.state;
    match instruction.kind {
        Kind::Tick => Ok(()),

        Kind::H
        | Kind::HNegXy
        | Kind::HNegXz
        | Kind::HNegYz
        | Kind::HXy
        | Kind::HYz
        | Kind::CNegXyz
        | Kind::CNegZyx
        | Kind::CXNegYz
        | Kind::CXyNegZ
        | Kind::CXyz
        | Kind::CZNegYx
        | Kind::CZyNegX
        | Kind::CZyx
        | Kind::S
        | Kind::SDag
        | Kind::SqrtX
        | Kind::SqrtXDag
        | Kind::SqrtY
        | Kind::SqrtYDag
        | Kind::X
        | Kind::Y
        | Kind::Z => {
            for &q in &instruction.qubits {
                apply_single_qubit_clifford(state, instruction.kind, q);
            }
            Ok(())
        }

        Kind::CX
        | Kind::CY
        | Kind::CZ
        | Kind::Swap
        | Kind::CxSwap
        | Kind::CzSwap
        | Kind::ISwap
        | Kind::ISwapDag
        | Kind::SqrtXx
        | Kind::SqrtXxDag
        | Kind::SqrtYy
        | Kind::SqrtYyDag
        | Kind::SqrtZz
        | Kind::SqrtZzDag
        | Kind::SwapCx
        | Kind::Xcx
        | Kind::Xcy
        | Kind::Xcz
        | Kind::Ycx
        | Kind::Ycy
        | Kind::Ycz => {
            require_target_multiple(instruction, 2, "two-qubit Clifford requires paired targets")?;
            for pair in instruction.qubits.chunks_exact(2) {
                apply_two_qubit_clifford(state, instruction.kind, pair[0], pair[1]);
            }
            Ok(())
        }

        Kind::T | Kind::TDag => {
            let kernel_angle = if instruction.kind == Kind::T {
                std::f64::consts::PI / 8.0
            } else {
                -std::f64::consts::PI / 8.0
            };
            for &q in &instruction.qubits {
                apply_pauli_rotation(state, &pauli_z(state.n, q), kernel_angle);
            }
            Ok(())
        }

        Kind::PauliRotation => {
            for product in &instruction.pauli_products {
                apply_pauli_rotation(
                    state,
                    &maybe_inverted_pauli_product(product),
                    instruction.kernel_angle,
                );
            }
            Ok(())
        }

        Kind::MZ | Kind::MX | Kind::MY => {
            let axis = match instruction.kind {
                Kind::MX => 'X',
                Kind::MY => 'Y',
                _ => 'Z',
            };
            for target in &instruction.measurement_targets {
                apply_measurement(
                    state,
                    &mut acc.records,
                    axis,
                    target,
                    instruction.probability,
                )?;
            }
            Ok(())
        }

        Kind::Mrz | Kind::Mrx | Kind::Mry => {
            for target in &instruction.measurement_targets {
                let (measurement, correction) = match instruction.kind {
                    Kind::Mrx => (
                        pauli_x(state.n, target.qubit),
                        pauli_z(state.n, target.qubit),
                    ),
                    Kind::Mry => (
                        pauli_y(state.n, target.qubit),
                        pauli_x(state.n, target.qubit),
                    ),
                    _ => (
                        pauli_z(state.n, target.qubit),
                        pauli_x(state.n, target.qubit),
                    ),
                };
                apply_measurement_reset(
                    state,
                    &mut acc.records,
                    &measurement,
                    &correction,
                    target,
                    instruction.probability,
                )?;
            }
            Ok(())
        }

        Kind::RZ | Kind::RX | Kind::RY => {
            for &q in &instruction.qubits {
                let (measurement, correction) = match instruction.kind {
                    Kind::RX => (pauli_x(state.n, q), pauli_z(state.n, q)),
                    Kind::RY => (pauli_y(state.n, q), pauli_x(state.n, q)),
                    _ => (pauli_z(state.n, q), pauli_x(state.n, q)),
                };
                apply_reset(state, &measurement, &correction);
            }
            Ok(())
        }

        Kind::Mpp => {
            for product in &instruction.pauli_products {
                // The `!` inversion flips the recorded bit through the sign;
                // the measured body keeps its parsed phase.
                let sign = measurement_sign_with_readout_error(
                    &mut state.context,
                    product.inverted,
                    instruction.probability,
                )?;
                let (record, record_condition) =
                    reserve_measurement_record(&mut acc.records, &mut state.context);
                apply_pauli_measurement_signed(
                    state,
                    &product.pauli,
                    &sign,
                    Some(record),
                    Some(record_condition),
                );
            }
            Ok(())
        }

        Kind::ExpVal => {
            let base = instruction
                .exp_val
                .ok_or_else(|| TicitError::new("EXP_VAL index range is invalid"))?;
            for (i, product) in instruction.pauli_products.iter().enumerate() {
                let pauli = maybe_inverted_pauli_product(product);
                apply_pauli_expectation(state, &pauli, (base + i) as i32)?;
            }
            Ok(())
        }

        Kind::XError | Kind::YError | Kind::ZError => {
            let probability = check_probability(instruction.probability)?;
            let axis = match instruction.kind {
                Kind::XError => 'X',
                Kind::YError => 'Y',
                _ => 'Z',
            };
            for &q in &instruction.qubits {
                let flip = state.context.fresh_bernoulli_bool(probability)?;
                apply_pauli_symbolic(state, &pauli_on_axis(state.n, axis, q), &flip)?;
            }
            Ok(())
        }

        Kind::Depolarize1 => {
            for &q in &instruction.qubits {
                apply_depolarize1(state, q, instruction.probability)?;
            }
            Ok(())
        }

        Kind::Depolarize2 => {
            require_target_multiple(instruction, 2, "DEPOLARIZE2 requires paired targets")?;
            for pair in instruction.qubits.chunks_exact(2) {
                apply_depolarize2(state, pair[0], pair[1], instruction.probability)?;
            }
            Ok(())
        }

        Kind::Depolarize3 => {
            require_target_multiple(instruction, 3, "DEPOLARIZE3 requires triples of targets")?;
            for triple in instruction.qubits.chunks_exact(3) {
                apply_depolarize_n(state, triple, instruction.probability)?;
            }
            Ok(())
        }

        Kind::PauliChannel1 => {
            for &q in &instruction.qubits {
                apply_pauli_channel(state, &[q], &instruction.probabilities)?;
            }
            Ok(())
        }

        Kind::PauliChannel2 => {
            require_target_multiple(instruction, 2, "PAULI_CHANNEL_2 requires paired targets")?;
            for pair in instruction.qubits.chunks_exact(2) {
                apply_pauli_channel(state, pair, &instruction.probabilities)?;
            }
            Ok(())
        }

        Kind::PauliChannel3 => {
            require_target_multiple(
                instruction,
                3,
                "PAULI_CHANNEL_3 requires triples of targets",
            )?;
            for triple in instruction.qubits.chunks_exact(3) {
                apply_pauli_channel(state, triple, &instruction.probabilities)?;
            }
            Ok(())
        }

        Kind::PauliProductChannel => apply_pauli_product_channel(
            state,
            &instruction.pauli_products,
            &instruction.probabilities,
        ),

        Kind::HeraldedErase => {
            let p = check_probability(instruction.probability)? / 4.0;
            for &q in &instruction.qubits {
                apply_heralded_channel(state, &mut acc.records, q, &[p, p, p, p])?;
            }
            Ok(())
        }

        Kind::HeraldedPauliChannel1 => {
            for &q in &instruction.qubits {
                apply_heralded_channel(state, &mut acc.records, q, &instruction.probabilities)?;
            }
            Ok(())
        }

        Kind::MPad => {
            for target in &instruction.measurement_targets {
                apply_mpad(state, &mut acc.records, target, instruction.probability)?;
            }
            Ok(())
        }

        Kind::FeedbackX | Kind::FeedbackY | Kind::FeedbackZ => {
            apply_record_feedback(state, &acc.records, instruction)
        }
    }
}

/// Lowers a parsed circuit into a frame-factored state plus record symbols.
pub fn lower_circuit_to_factored(circuit: &Circuit) -> Result<CircuitLoweringResult> {
    let mut acc = Accumulator {
        state: FrameFactoredState::new(circuit.nqubits, 0)?,
        records: Vec::new(),
    };
    let mut pending_counts = Vec::with_capacity(circuit.instructions.len() + 1);
    pending_counts.push(0);
    for instruction in &circuit.instructions {
        apply_instruction(&mut acc, instruction)?;
        pending_counts.push(acc.state.pending_operations.len());
    }
    if acc.records.len() != circuit.nrecords {
        return Err(TicitError::new("circuit measurement record count mismatch"));
    }
    Ok(CircuitLoweringResult {
        state: acc.state,
        measurement_records: acc.records,
        instruction_pending_operation_counts: pending_counts,
    })
}

#[cfg(test)]
mod tests {
    //! Lowering tests and numeric checks for every noise-channel categorical table.

    use crate::test_support as common;

    use super::*;
    use crate::circuit::{parse_ticit_circuit_text, parse_ticit_text, plan_ticit_factored_program};
    use crate::factored::{FactoredInstruction, PendingFactoredState, PendingOperation};
    use crate::planner::plan_factored_updates;
    use crate::symbolic::SymbolicCategoricalDistribution;

    fn parsed(text: &str) -> Circuit {
        parse_ticit_circuit_text(text).expect("test circuit parses")
    }

    /// Parses, plans, and samples a circuit; returns each shot's record bits.
    fn sampled_records(text: &str, shots: usize, seed: u64) -> Vec<Vec<u64>> {
        let parsed = parse_ticit_text(text).expect("test circuit parses");
        let program = plan_ticit_factored_program(&parsed).expect("test circuit plans");
        crate::sampler::batch::sample_measurements_batch(&program, shots, 0, seed)
            .expect("sampling succeeds")
    }

    fn record_bit(words: &[u64], record: usize) -> bool {
        words[(record - 1) >> 6] & (1 << ((record - 1) & 63)) != 0
    }

    fn lowered_distributions(text: &str) -> Vec<SymbolicCategoricalDistribution> {
        let lowered = lower_circuit_to_factored(&parsed(text)).expect("test circuit lowers");
        lowered.state.context.categorical_distributions.clone()
    }

    /// Packs assignment rows into their first word for compact comparisons; every
    /// channel here has at most 6 bits.
    fn assignment_words(distribution: &SymbolicCategoricalDistribution) -> Vec<u64> {
        distribution
            .assignments
            .iter()
            .map(|row| {
                assert_eq!(row.len(), 1, "assignment rows in these tests fit one word");
                row[0]
            })
            .collect()
    }

    fn approx_all(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "probability row count");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= 1e-12, "probability {a} != {e}");
        }
    }

    // ==============================================================================
    // Catalogue §4.1 — REPEAT flattening, prefix tables, detector positions
    // ==============================================================================

    #[test]
    fn repeat_lowering_produces_the_pinned_prefix_table() {
        let circuit = parsed("REPEAT 2 {\nM !0\n}\nDETECTOR rec[-1] rec[-2]\n");
        assert_eq!(circuit.nqubits, 1);
        assert_eq!(circuit.nrecords, 2);
        assert_eq!(circuit.instructions.len(), 2);
        assert_eq!(circuit.detectors[0].records, vec![2, 1]);
        assert_eq!(circuit.detectors[0].after_instruction, 2);

        let lowered = lower_circuit_to_factored(&circuit).expect("repeat fixture lowers");
        assert_eq!(lowered.measurement_records.len(), 2);
        assert_eq!(lowered.instruction_pending_operation_counts, vec![0, 1, 2]);

        let pending = PendingFactoredState::from_frame_state(lowered.state);
        let program = plan_factored_updates(pending).expect("repeat fixture plans");
        // One leading checkpoint plus one after each of the two pending operations.
        assert_eq!(program.pending_prefix_instruction_indices.len(), 3);
        for shot in sampled_records("REPEAT 2 {\nM !0\n}\nDETECTOR rec[-1] rec[-2]\n", 8, 5) {
            assert!(
                record_bit(&shot, 1) && record_bit(&shot, 2),
                "both M !0 records read 1"
            );
        }
    }

    #[test]
    fn a_clifford_advances_instructions_but_not_pending_operations() {
        let parsed = parse_ticit_text("M 0\nH 0\nDETECTOR rec[-1]\n").expect("fixture parses");
        assert_eq!(parsed.detectors[0].after_instruction, 2);
        let lowered = lower_circuit_to_factored(&parsed).expect("fixture lowers");
        assert_eq!(lowered.instruction_pending_operation_counts[2], 1);
    }

    // ==============================================================================
    // Catalogue §4.1/§4.2 — feedback lowering structure
    // ==============================================================================

    #[test]
    fn feedback_record_values_cover_xyz_paulis() {
        // Catalogue §4.1/§4.2 pins CX/CY. The MX case also closes its uncovered
        // record-controlled Z path: Z maps |+> to |->, so MX records one.
        for text in [
            "M !0\nCX rec[-1] 1\nM 1\n",
            "M !0\nCY rec[-1] 1\nM 1\n",
            "M !0\nH 1\nCZ rec[-1] 1\nMX 1\n",
        ] {
            for shot in sampled_records(text, 8, 3) {
                assert!(
                    record_bit(&shot, 1) && record_bit(&shot, 2),
                    "feedback flips qubit 1"
                );
            }
        }
    }

    #[test]
    fn feedback_lowers_to_a_record_conditioned_frame_term() {
        // Structural effect — one Pauli-frame term gated on the record's
        // outcome condition. (Record values pinned in the test above.)
        for (text, xbit, zbit) in [
            ("M !0\nCX rec[-1] 1\nM 1\n", true, false),
            ("M !0\nCY rec[-1] 1\nM 1\n", true, true),
            ("M !0\nCZ rec[-1] 1\nM 1\n", false, true),
        ] {
            let lowered =
                lower_circuit_to_factored(&parsed(text)).expect("feedback fixture lowers");
            let terms = &lowered.state.active_frame.terms;
            assert_eq!(
                terms.len(),
                1,
                "feedback contributes exactly one frame term"
            );
            assert_eq!(terms[0].pauli.xbit(1), xbit);
            assert_eq!(
                terms[0].pauli.zbit(1),
                zbit,
                "feedback applies the requested Pauli"
            );
            // The record symbol of `M !0` is its outcome condition: the inversion
            // lives in the measurement sign, not the record expression.
            let record_condition = lowered.measurement_records[0].conditions[0];
            assert_eq!(terms[0].condition, record_condition);
            assert_eq!(
                lowered.state.pending_operations.len(),
                2,
                "two measurements queued"
            );
        }
    }

    #[test]
    fn an_ordinary_cy_is_a_gate_not_feedback() {
        let lowered =
            lower_circuit_to_factored(&parsed("X 0\nCY 0 1\nM 1\n")).expect("gate fixture lowers");
        // The gate path touches only the Clifford frame: no Pauli-frame terms.
        // (The X gate is absorbed into the frame too, not queued.)
        assert!(lowered.state.active_frame.terms.is_empty());
        assert_eq!(lowered.state.pending_operations.len(), 1);
        for shot in sampled_records("X 0\nCY 0 1\nM 1\n", 8, 3) {
            assert!(record_bit(&shot, 1), "gate-path CY flips qubit 1");
        }
    }

    // ==============================================================================
    // Catalogue §4.3(a) — rotation lowering angles
    // ==============================================================================

    #[test]
    fn rotation_lowering_matches_the_pinned_angles() {
        let quarter = std::f64::consts::FRAC_PI_4;

        let angles = |text: &str| -> Vec<f64> {
            let lowered =
                lower_circuit_to_factored(&parsed(text)).expect("rotation fixture lowers");
            lowered
                .state
                .pending_operations
                .iter()
                .map(|op| match op {
                    PendingOperation::PauliRotation(rotation) => rotation.kernel_angle,
                    other => panic!("expected a rotation, found {other:?}"),
                })
                .collect()
        };

        let rx = angles("R_X(0.5) 0\n");
        assert_eq!(rx.len(), 1);
        assert!((rx[0] - quarter).abs() <= 1e-12);

        let rz = angles("R_Z(pi/pi) 0\n");
        assert!((rz[0] - std::f64::consts::FRAC_PI_2).abs() <= 1e-12);

        let rxx = angles("R_XX(0.25) 0 1\n");
        assert!((rxx[0] - std::f64::consts::PI / 8.0).abs() <= 1e-12);

        let rp = angles("R_PAULI(-0.5) X0*Z1\n");
        assert!((rp[0] + quarter).abs() <= 1e-12);

        // U3 expands to Z(lambda), Y(theta), Z(phi).
        let u3 = angles("U3(0.5,0.25,-0.5) 0\n");
        assert_eq!(u3.len(), 3);
        assert!((u3[0] + quarter).abs() <= 1e-12);
        assert!((u3[1] - quarter).abs() <= 1e-12);
        assert!((u3[2] - std::f64::consts::PI / 8.0).abs() <= 1e-12);

        // The queued body survives conjugation through a fresh frame untouched.
        let lowered = lower_circuit_to_factored(&parsed("R_X(0.5) 0\n")).expect("lowers");
        let PendingOperation::PauliRotation(rotation) = &lowered.state.pending_operations[0] else {
            panic!("expected a rotation");
        };
        assert!(rotation.pauli.pauli.same_body(&crate::pauli_x(1, 0)));
    }

    // ==============================================================================
    // Catalogue §4.3(b) — MPAD and record numbering
    // ==============================================================================

    #[test]
    fn mpad_and_pair_measurements_number_records_in_order() {
        let circuit = parsed("MXX !0 1\nMPAD 1 0\nOBSERVABLE_INCLUDE(0) rec[-1] X0\n");
        let lowered = lower_circuit_to_factored(&circuit).expect("MPAD fixture lowers");
        assert_eq!(lowered.measurement_records.len(), 3);
        assert_eq!(circuit.observables[0].records, vec![3]);

        let pad_only = parsed("MPAD 1\n");
        assert_eq!(pad_only.nqubits, 0);
        let lowered = lower_circuit_to_factored(&pad_only).expect("bare MPAD lowers");
        assert_eq!(lowered.measurement_records.len(), 1);
        // A constant pad has no noise symbol: its outcome is the literal 1.
        assert!(lowered.measurement_records[0].constant);
        assert!(lowered.measurement_records[0].conditions.is_empty());
    }

    // ==============================================================================
    // Catalogue §4.3(d) — correlated-error chains
    // ==============================================================================

    #[test]
    fn a_correlated_error_chain_lowers_to_one_categorical() {
        let circuit = parsed("E(0.25) X0\nELSE_CORRELATED_ERROR(0.5) Z0\nM 0\n");
        assert_eq!(circuit.instructions.len(), 2);
        approx_all(&circuit.instructions[0].probabilities, &[0.25, 0.375]);

        let distributions =
            lowered_distributions("E(0.25) X0\nELSE_CORRELATED_ERROR(0.5) Z0\nM 0\n");
        assert_eq!(distributions.len(), 1);
        let channel = &distributions[0];
        assert_eq!(channel.nbits, 2);
        assert_eq!(assignment_words(channel), vec![0b00, 0b01, 0b10]);
        approx_all(&channel.probabilities, &[0.375, 0.25, 0.375]);
    }

    // ==============================================================================
    // Noise-channel categorical tables — never asserted in the C++ suite
    // ==============================================================================

    #[test]
    fn depolarize1_is_a_four_way_categorical_over_xz_bits() {
        let distributions = lowered_distributions("DEPOLARIZE1(0.3) 0\n");
        assert_eq!(distributions.len(), 1);
        let channel = &distributions[0];
        assert_eq!(channel.nbits, 2);
        assert_eq!(channel.conditions.len(), 2);
        // Rows: I, Z (z bit only), X (x bit only), Y (both). Bit 0 is x, bit 1 is z.
        assert_eq!(assignment_words(channel), vec![0b00, 0b10, 0b01, 0b11]);
        approx_all(&channel.probabilities, &[0.7, 0.1, 0.1, 0.1]);
    }

    #[test]
    fn depolarize2_is_a_sixteen_way_categorical() {
        let distributions = lowered_distributions("DEPOLARIZE2(0.15) 0 1\n");
        let channel = &distributions[0];
        assert_eq!(channel.nbits, 4);
        assert_eq!(channel.assignments.len(), 16);
        // First row is the identity; the rest enumerate (x1, z1, x2, z2) in nested
        // loop order with bit 0 = x1 .. bit 3 = z2.
        let words = assignment_words(channel);
        assert_eq!(words[0], 0b0000);
        assert_eq!(words[1], 0b1000); // z2 only
        assert_eq!(words[2], 0b0100); // x2 only
        assert_eq!(words[15], 0b1111); // Y on both qubits
        let mut expected = vec![0.15 / 15.0; 16];
        expected[0] = 0.85;
        approx_all(&channel.probabilities, &expected);
    }

    #[test]
    fn pauli_channel_1_orders_rows_x_y_z() {
        let distributions = lowered_distributions("PAULI_CHANNEL_1(0.1,0.2,0.3) 0\n");
        let channel = &distributions[0];
        assert_eq!(channel.nbits, 2);
        // Axis codes 1, 2, 3 are X, Y, Z; Y sets both bits.
        assert_eq!(assignment_words(channel), vec![0b00, 0b01, 0b11, 0b10]);
        approx_all(&channel.probabilities, &[0.4, 0.1, 0.2, 0.3]);
    }

    #[test]
    fn pauli_channel_2_reads_axis_codes_most_significant_target_first() {
        // 15 probabilities; make row 1 (code 1 = I on q0, X on q1) and row 4
        // (code 4 = X on q0, I on q1) distinguishable.
        let mut probs = [0.0; 15];
        probs[0] = 0.01; // code 1
        probs[3] = 0.02; // code 4
        let text = format!(
            "PAULI_CHANNEL_2({}) 0 1\n",
            probs
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        let distributions = lowered_distributions(&text);
        let channel = &distributions[0];
        assert_eq!(channel.nbits, 4);
        // Bit layout: (x0, z0, x1, z1). Code 1 → X on target 1 → bit 2. Code 4 →
        // X on target 0 → bit 0.
        let words = assignment_words(channel);
        assert_eq!(words[1], 0b0100);
        assert_eq!(words[4], 0b0001);
        assert!((channel.probabilities[1] - 0.01).abs() <= 1e-12);
        assert!((channel.probabilities[4] - 0.02).abs() <= 1e-12);
        assert!((channel.probabilities[0] - 0.97).abs() <= 1e-12);
    }

    #[test]
    fn depolarize3_spreads_uniformly_over_63_cases() {
        let distributions = lowered_distributions("DEPOLARIZE3(0.63) 0 1 2\n");
        let channel = &distributions[0];
        assert_eq!(channel.nbits, 6);
        assert_eq!(channel.assignments.len(), 64);
        assert!((channel.probabilities[0] - 0.37).abs() <= 1e-12);
        for &p in &channel.probabilities[1..] {
            assert!((p - 0.01).abs() <= 1e-12);
        }
    }

    #[test]
    fn heralded_channels_use_the_pinned_assignment_table() {
        // HERALDED_ERASE(0.2) splits its probability into four equal quarters.
        let distributions = lowered_distributions("HERALDED_ERASE(0.2) 0\n");
        let channel = &distributions[0];
        assert_eq!(channel.nbits, 3);
        // {no herald, herald+I, herald+X, herald+Y, herald+Z}; bit 0 is the herald,
        // bit 1 the X flip, bit 2 the Z flip.
        assert_eq!(
            assignment_words(channel),
            vec![0b000, 0b001, 0b011, 0b111, 0b101]
        );
        approx_all(&channel.probabilities, &[0.8, 0.05, 0.05, 0.05, 0.05]);

        let distributions = lowered_distributions("HERALDED_PAULI_CHANNEL_1(0,0.1,0,0) 1\n");
        approx_all(&distributions[0].probabilities, &[0.9, 0.0, 0.1, 0.0, 0.0]);
    }

    #[test]
    fn heralded_channels_append_records_ahead_of_measurements() {
        let lowered = lower_circuit_to_factored(&parsed(
            "SPP X0*Z1\nPAULI_CHANNEL_1(0.1,0.2,0.3) 0\nDEPOLARIZE3(0.1) 0 1 2\n\
         HERALDED_ERASE(0.1) 0\nHERALDED_PAULI_CHANNEL_1(0,0.1,0,0) 1\nM 0 1\n",
        ))
        .expect("noise fixture lowers");
        // Two heralds then two measurements.
        assert_eq!(lowered.measurement_records.len(), 4);
        // The herald outcomes are the categorical herald bits, not fresh records.
        assert_eq!(lowered.measurement_records[0].conditions.len(), 1);
        assert_eq!(lowered.measurement_records[1].conditions.len(), 1);
    }

    // ==============================================================================
    // Planned frontend programs — detector splicing
    // ==============================================================================

    #[test]
    fn detector_events_are_spliced_at_their_source_positions() {
        let parsed = parse_ticit_text("M !0\nDETECTOR rec[-1]\nM 0\nDETECTOR rec[-1] rec[-2]\n")
            .expect("detector fixture parses");
        let program = plan_ticit_factored_program(&parsed).expect("detector fixture plans");
        assert_eq!(program.ndetectors, 2);
        assert_eq!(program.nrecords, 2);

        let detectors: Vec<_> = program
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                FactoredInstruction::RecordDetector(detector) => Some(detector),
                _ => None,
            })
            .collect();
        assert_eq!(detectors.len(), 2);
        assert_eq!(detectors[0].detector, 1);
        assert_eq!(detectors[0].records, vec![1]);
        assert_eq!(detectors[1].detector, 2);
        assert_eq!(detectors[1].records, vec![2, 1]);

        // The first detector must appear before the instruction that plans the
        // second measurement: source order is preserved through the splice.
        let first_detector_pos = program
            .instructions
            .iter()
            .position(|i| matches!(i, FactoredInstruction::RecordDetector(d) if d.detector == 1))
            .expect("first detector present");
        let second_detector_pos = program
            .instructions
            .iter()
            .position(|i| matches!(i, FactoredInstruction::RecordDetector(d) if d.detector == 2))
            .expect("second detector present");
        assert!(first_detector_pos < second_detector_pos);
        // Splicing rebuilds the program; the stale checkpoint table is dropped.
        assert!(program.pending_prefix_instruction_indices.is_empty());
    }

    #[test]
    fn a_detector_free_plan_keeps_its_checkpoints() {
        let parsed = parse_ticit_text("M 0\nM 0\n").expect("plain fixture parses");
        let program = plan_ticit_factored_program(&parsed).expect("plain fixture plans");
        assert_eq!(program.ndetectors, 0);
        assert_eq!(program.pending_prefix_instruction_indices.len(), 3);
    }

    // ==============================================================================
    // End-to-end: a real benchmark circuit plans successfully
    // ==============================================================================

    #[test]
    fn the_msc_d3_benchmark_circuit_lowers_and_plans() {
        let path = common::msc_d3_circuit();
        let parsed = crate::circuit::parse_ticit_file(path).expect("msc d3 parses");
        let nrecords = parsed.nrecords;
        let ndetectors = parsed.detectors.len();
        assert!(nrecords > 0 && ndetectors > 0);
        let program = plan_ticit_factored_program(&parsed).expect("msc d3 plans");
        assert_eq!(program.nrecords, nrecords);
        assert_eq!(program.ndetectors, ndetectors);
        assert!(program.max_k >= 1);
    }
}
