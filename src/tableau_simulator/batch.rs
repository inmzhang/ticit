//! A compact instruction set the engine can replay without re-translating it.
//!
//! The engine's procedural API is shaped for a caller that decides what to do
//! next from the outcome it just read. Its heaviest client does the opposite:
//! `bloc_compile`'s physical verification assembles one op stream per circuit
//! node and replays that same stream against a fresh simulator for every shot.
//! Everything between the IR and the frame — coordinate lookups and
//! [`PauliString`] construction — is then pure repetition.
//!
//! [`Instruction`] is that stream after translation: plain qubit indices, named
//! gates, and the two Paulis (a measurement observable, a multi-qubit rotation
//! axis) that genuinely cannot be reduced to an index. Building it costs the
//! translation once; [`TableauSimulator::apply_batch`] replays it with no allocation
//! beyond the outcome vector.
//!
//! # Why these gates and no others
//!
//! Both gate families are *named*, not Pauli-addressed, and both are realized
//! as short compositions of the engine's primitive frame updates:
//!
//! * A single-qubit Clifford ([`Gate1Q`]) is one or two primitives, instead of
//!   building a Clifford tableau and inverting it per application.
//! * A two-qubit `<A>C<B>` gate is `CZ` conjugated by the basis rotations that
//!   carry `Z` onto `A` and `B` — at most five primitives, none of which
//!   allocate, where the generic
//!   [`controlled_pauli`](TableauSimulator::controlled_pauli) path pays two Pauli
//!   preimages and four allocations.
//!
//! A multi-qubit *Pauli* has no index-addressed form, so
//! [`Instruction::Measure`] and [`Instruction::TPauli`] carry a
//! [`PauliString`] — built once when the batch is assembled, borrowed on every
//! replay. There is deliberately no multi-qubit Pauli *gate* variant: unlike a
//! rotation axis or a measured observable, a Pauli product is just its factors
//! applied in sequence, so [`Instruction::Pauli`] already spans it.

use crate::circuit::Circuit;
use crate::circuit::ir::{CircuitInstructionKind as Kind, CircuitPauliProduct};
use crate::random::{rand_float, sample_bernoulli};
use crate::{Pauli, PauliBasis, PauliString, neg};

use super::{MeasureResult, SimError, TOL, TableauSimulator};

// ==============================================================================
// Single-qubit Clifford gates
// ==============================================================================

/// A named single-qubit Clifford, identified by the signed Pauli images
/// `(X → …, Z → …)` of [`Gate1Q::images`].
///
/// The names and the tableaux are stim's. Variants that stim spells with
/// underscores and negation markers (`H_XY`, `C_NZYX`) are spelled here in
/// camel case with the same letters in the same order (`Hxy`, `Cnzyx`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Gate1Q {
    /// `X → +X`, `Z → −Z`.
    X,
    /// `X → −X`, `Z → −Z`.
    Y,
    /// `X → −X`, `Z → +Z`.
    Z,
    /// Hadamard: `X → +Z`, `Z → +X`.
    H,
    /// Phase gate `S = √Z`: `X → +Y`, `Z → +Z`.
    S,
    /// `S† = √Z†`: `X → −Y`, `Z → +Z`.
    SDag,
    /// `√X`: `X → +X`, `Z → −Y`.
    SqrtX,
    /// `√X†`: `X → +X`, `Z → +Y`.
    SqrtXDag,
    /// `√Y`: `X → −Z`, `Z → +X`.
    SqrtY,
    /// `√Y†`: `X → +Z`, `Z → −X`.
    SqrtYDag,
    /// `H_XY`: `X → +Y`, `Z → −Z`.
    Hxy,
    /// `H_YZ`: `X → −X`, `Z → +Y`.
    Hyz,
    /// `H_NXY`: `X → −Y`, `Z → −Z`.
    Hnxy,
    /// `H_NXZ`: `X → −Z`, `Z → −X`.
    Hnxz,
    /// `H_NYZ`: `X → −X`, `Z → −Y`.
    Hnyz,
    /// `C_XYZ`, the order-three cycle `X → Y → Z → X`.
    Cxyz,
    /// `C_ZYX`, the inverse cycle `X → Z → Y → X`.
    Czyx,
    /// `C_NXYZ`: `X → −Y`, `Z → −X`.
    Cnxyz,
    /// `C_XNYZ`: `X → −Y`, `Z → +X`.
    Cxnyz,
    /// `C_XYNZ`: `X → +Y`, `Z → −X`.
    Cxynz,
    /// `C_NZYX`: `X → −Z`, `Z → −Y`.
    Cnzyx,
    /// `C_ZNYX`: `X → +Z`, `Z → −Y`.
    Cznyx,
    /// `C_ZYNX`: `X → −Z`, `Z → +Y`.
    Czynx,
}

impl Gate1Q {
    /// The gate's signed Pauli tableau as `(X → image, Z → image)`, each image
    /// an axis and whether it is negated.
    ///
    /// This *is* the gate's definition: [`TableauSimulator::gate1`] realizes each
    /// variant as a composition of primitive frame updates, and the pairing of
    /// composition to tableau is what the unit tests pin.
    ///
    /// # Examples
    ///
    /// ```
    /// use ticit::PauliBasis;
    /// use ticit::Gate1Q;
    ///
    /// // The Hadamard exchanges X and Z without a sign.
    /// assert_eq!(
    ///     Gate1Q::H.images(),
    ///     ((PauliBasis::Z, false), (PauliBasis::X, false))
    /// );
    /// ```
    #[must_use]
    pub const fn images(self) -> ((PauliBasis, bool), (PauliBasis, bool)) {
        use PauliBasis::{X, Y, Z};
        match self {
            Gate1Q::X => ((X, false), (Z, true)),
            Gate1Q::Y => ((X, true), (Z, true)),
            Gate1Q::Z => ((X, true), (Z, false)),
            Gate1Q::H => ((Z, false), (X, false)),
            Gate1Q::S => ((Y, false), (Z, false)),
            Gate1Q::SDag => ((Y, true), (Z, false)),
            Gate1Q::SqrtX => ((X, false), (Y, true)),
            Gate1Q::SqrtXDag => ((X, false), (Y, false)),
            Gate1Q::SqrtY => ((Z, true), (X, false)),
            Gate1Q::SqrtYDag => ((Z, false), (X, true)),
            Gate1Q::Hxy => ((Y, false), (Z, true)),
            Gate1Q::Hyz => ((X, true), (Y, false)),
            Gate1Q::Hnxy => ((Y, true), (Z, true)),
            Gate1Q::Hnxz => ((Z, true), (X, true)),
            Gate1Q::Hnyz => ((X, true), (Y, true)),
            Gate1Q::Cxyz => ((Y, false), (X, false)),
            Gate1Q::Czyx => ((Z, false), (Y, false)),
            Gate1Q::Cnxyz => ((Y, true), (X, true)),
            Gate1Q::Cxnyz => ((Y, true), (X, false)),
            Gate1Q::Cxynz => ((Y, false), (X, true)),
            Gate1Q::Cnzyx => ((Z, true), (Y, true)),
            Gate1Q::Cznyx => ((Z, false), (Y, true)),
            Gate1Q::Czynx => ((Z, true), (Y, false)),
        }
    }
}

// ==============================================================================
// Instructions
// ==============================================================================

/// One step of a [`TableauSimulator::apply_batch`] program.
///
/// # Examples
///
/// ```
/// use ticit::{Gate1Q, Instruction};
/// use ticit::{Pauli, PauliBasis, PauliString, TableauSimulator};
///
/// // Prepare a Bell pair and read both halves out in the Z basis.
/// let bell = [
///     Instruction::Gate1 {
///         gate: Gate1Q::H,
///         qubit: 0,
///     },
///     Instruction::Gate2 {
///         control: PauliBasis::Z,
///         target: PauliBasis::X,
///         control_qubit: 0,
///         target_qubit: 1,
///     },
///     Instruction::Measure(PauliString::single(2, 0, Pauli::Z)),
///     Instruction::Measure(PauliString::single(2, 1, Pauli::Z)),
/// ];
///
/// let mut sim = TableauSimulator::with_seed(2, 7);
/// let outcome = sim.apply_batch(&bell)?;
/// assert_eq!(outcome.records.len(), 2);
/// assert_eq!(outcome.records[0].outcome, outcome.records[1].outcome);
/// # Ok::<(), ticit::SimError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    /// A named single-qubit Clifford.
    Gate1 {
        /// The gate to apply.
        gate: Gate1Q,
        /// Its operand.
        qubit: usize,
    },
    /// A two-qubit `<A>C<B>` gate: apply `target` on `target_qubit` when
    /// `control_qubit` is in the `−1` eigenstate of `control`. `CX` is
    /// `control: Z, target: X`; `CZ` is `Z, Z`.
    Gate2 {
        /// The control qubit's Pauli axis (`A`).
        control: PauliBasis,
        /// The target qubit's Pauli axis (`B`).
        target: PauliBasis,
        /// Qubit the control axis is read on.
        control_qubit: usize,
        /// Qubit the target axis acts on.
        target_qubit: usize,
    },
    /// A single-qubit Pauli.
    Pauli {
        /// Which Pauli.
        basis: PauliBasis,
        /// Its operand.
        qubit: usize,
    },
    /// `T`/`T†` about a single-qubit basis axis.
    T {
        /// The rotation axis.
        basis: PauliBasis,
        /// Its operand.
        qubit: usize,
        /// `true` selects `T†`.
        adjoint: bool,
    },
    /// `T`/`T†` about an arbitrary Pauli axis.
    TPauli {
        /// The rotation axis.
        axis: PauliString,
        /// `true` selects `T†`.
        adjoint: bool,
    },
    /// `exp(-i * kernel_angle * axis)` for an arbitrary real angle.
    PauliRotation {
        /// Hermitian Pauli rotation axis.
        axis: PauliString,
        /// Angle in radians under the kernel convention.
        kernel_angle: f64,
    },
    /// Measure a Pauli observable, appending its result to
    /// [`BatchOutcome::records`].
    Measure(PauliString),
    /// Measure a Pauli observable and independently flip the recorded bit.
    MeasureWithReadoutError {
        /// Hermitian observable to measure.
        observable: PauliString,
        /// Probability of flipping the classical record after projection.
        probability: f64,
    },
    /// Append a classical record, optionally flipped by Bernoulli noise.
    Record {
        /// Record value before noise.
        value: bool,
        /// Probability of flipping `value`.
        flip_probability: f64,
    },
    /// Sample at most one Pauli alternative and apply it.
    RandomPauli {
        /// Absolute probability of each corresponding alternative.
        probabilities: Vec<f64>,
        /// Pauli alternatives; remaining probability means no operation.
        alternatives: Vec<PauliString>,
        /// Whether to append a record indicating that an alternative fired.
        heralded: bool,
    },
    /// Read a Pauli expectation value without changing the state.
    Expectation(PauliString),
    /// Reset a qubit to the `+1` eigenstate of `basis`.
    Reset {
        /// The basis to reset into.
        basis: PauliBasis,
        /// Its operand.
        qubit: usize,
    },
    /// Apply a single-qubit Pauli if an earlier measurement *in this batch*
    /// came out `−1`.
    ///
    /// `control` indexes [`BatchOutcome::records`], so it counts measurements
    /// (not instructions) from the start of the batch. Corrections conditioned
    /// on anything outside the batch belong between batches, where the caller
    /// still owns the decision.
    ConditionalPauli {
        /// Which Pauli to apply.
        basis: PauliBasis,
        /// Its operand.
        qubit: usize,
        /// Index into [`BatchOutcome::records`] of the gating measurement.
        control: usize,
    },
}

/// What one [`TableauSimulator::apply_batch`] run produced.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchOutcome {
    /// Measurement, explicit-record, and herald records in batch order.
    pub records: Vec<MeasureResult>,
    /// One entry per [`Instruction::Expectation`], in batch order.
    pub expectation_values: Vec<f64>,
    /// Highest stabilizer rank observed after any instruction, or `0` for an
    /// empty batch. Callers that budget memory against the rank need the peak,
    /// which a post-batch [`TableauSimulator::rank`] read would miss — a Pauli
    /// rotation can grow the rank before a later measurement collapses it.
    pub max_rank: usize,
}

fn gate1_for(kind: Kind) -> Option<Gate1Q> {
    Some(match kind {
        Kind::H => Gate1Q::H,
        Kind::HNegXy => Gate1Q::Hnxy,
        Kind::HNegXz => Gate1Q::Hnxz,
        Kind::HNegYz => Gate1Q::Hnyz,
        Kind::HXy => Gate1Q::Hxy,
        Kind::HYz => Gate1Q::Hyz,
        Kind::CNegXyz => Gate1Q::Cnxyz,
        Kind::CNegZyx => Gate1Q::Cnzyx,
        Kind::CXNegYz => Gate1Q::Cxnyz,
        Kind::CXyNegZ => Gate1Q::Cxynz,
        Kind::CXyz => Gate1Q::Cxyz,
        Kind::CZNegYx => Gate1Q::Cznyx,
        Kind::CZyNegX => Gate1Q::Czynx,
        Kind::CZyx => Gate1Q::Czyx,
        Kind::S => Gate1Q::S,
        Kind::SDag => Gate1Q::SDag,
        Kind::SqrtX => Gate1Q::SqrtX,
        Kind::SqrtXDag => Gate1Q::SqrtXDag,
        Kind::SqrtY => Gate1Q::SqrtY,
        Kind::SqrtYDag => Gate1Q::SqrtYDag,
        Kind::X => Gate1Q::X,
        Kind::Y => Gate1Q::Y,
        Kind::Z => Gate1Q::Z,
        _ => return None,
    })
}

fn gate2_for(kind: Kind) -> Option<(PauliBasis, PauliBasis)> {
    use PauliBasis::{X, Y, Z};
    Some(match kind {
        Kind::CX => (Z, X),
        Kind::CY => (Z, Y),
        Kind::CZ => (Z, Z),
        Kind::Xcx => (X, X),
        Kind::Xcy => (X, Y),
        Kind::Xcz => (X, Z),
        Kind::Ycx => (Y, X),
        Kind::Ycy => (Y, Y),
        Kind::Ycz => (Y, Z),
        _ => return None,
    })
}

fn push_gate1(out: &mut Vec<Instruction>, gate: Gate1Q, qubit: usize) {
    out.push(Instruction::Gate1 { gate, qubit });
}

fn push_gate2(
    out: &mut Vec<Instruction>,
    control: PauliBasis,
    target: PauliBasis,
    control_qubit: usize,
    target_qubit: usize,
) {
    out.push(Instruction::Gate2 {
        control,
        target,
        control_qubit,
        target_qubit,
    });
}

fn push_cx(out: &mut Vec<Instruction>, control: usize, target: usize) {
    push_gate2(out, PauliBasis::Z, PauliBasis::X, control, target);
}

fn push_swap(out: &mut Vec<Instruction>, a: usize, b: usize) {
    push_cx(out, a, b);
    push_cx(out, b, a);
    push_cx(out, a, b);
}

fn push_sqrt_zz(out: &mut Vec<Instruction>, a: usize, b: usize, adjoint: bool) {
    push_cx(out, a, b);
    push_gate1(out, if adjoint { Gate1Q::SDag } else { Gate1Q::S }, b);
    push_cx(out, a, b);
}

fn push_sqrt_pair(
    out: &mut Vec<Instruction>,
    basis: PauliBasis,
    a: usize,
    b: usize,
    adjoint: bool,
) {
    let rotations = match basis {
        PauliBasis::X => Some((Gate1Q::H, Gate1Q::H)),
        PauliBasis::Y => Some((Gate1Q::SqrtX, Gate1Q::SqrtXDag)),
        PauliBasis::Z => None,
    };
    if let Some((before, after)) = rotations {
        push_gate1(out, before, a);
        push_gate1(out, before, b);
        push_sqrt_zz(out, a, b, adjoint);
        push_gate1(out, after, a);
        push_gate1(out, after, b);
    } else {
        push_sqrt_zz(out, a, b, adjoint);
    }
}

fn push_pair(out: &mut Vec<Instruction>, kind: Kind, a: usize, b: usize) {
    if let Some((control, target)) = gate2_for(kind) {
        push_gate2(out, control, target, a, b);
        return;
    }
    match kind {
        Kind::Swap => push_swap(out, a, b),
        Kind::CxSwap => {
            push_cx(out, a, b);
            push_swap(out, a, b);
        }
        Kind::CzSwap => {
            push_gate2(out, PauliBasis::Z, PauliBasis::Z, a, b);
            push_swap(out, a, b);
        }
        Kind::SwapCx => {
            push_swap(out, a, b);
            push_cx(out, a, b);
        }
        Kind::ISwap | Kind::ISwapDag => {
            push_gate2(out, PauliBasis::Z, PauliBasis::Z, a, b);
            let phase = if kind == Kind::ISwap {
                Gate1Q::S
            } else {
                Gate1Q::SDag
            };
            push_gate1(out, phase, a);
            push_gate1(out, phase, b);
            push_swap(out, a, b);
        }
        Kind::SqrtXx | Kind::SqrtXxDag => {
            push_sqrt_pair(out, PauliBasis::X, a, b, kind == Kind::SqrtXxDag);
        }
        Kind::SqrtYy | Kind::SqrtYyDag => {
            push_sqrt_pair(out, PauliBasis::Y, a, b, kind == Kind::SqrtYyDag);
        }
        Kind::SqrtZz | Kind::SqrtZzDag => {
            push_sqrt_pair(out, PauliBasis::Z, a, b, kind == Kind::SqrtZzDag);
        }
        _ => unreachable!("called only for two-qubit Clifford kinds"),
    }
}

fn product_pauli(product: &CircuitPauliProduct) -> PauliString {
    if product.inverted {
        neg(product.pauli.clone())
    } else {
        product.pauli.clone()
    }
}

fn pauli_from_code(nqubits: usize, qubits: &[usize], mut code: usize) -> PauliString {
    let mut pauli = PauliString::new(nqubits);
    for &qubit in qubits.iter().rev() {
        pauli.set(
            qubit,
            match code & 3 {
                0 => Pauli::I,
                1 => Pauli::X,
                2 => Pauli::Y,
                _ => Pauli::Z,
            },
        );
        code >>= 2;
    }
    pauli
}

fn pauli_channel_alternatives(nqubits: usize, qubits: &[usize]) -> Vec<PauliString> {
    (1..1usize << (2 * qubits.len()))
        .map(|code| pauli_from_code(nqubits, qubits, code))
        .collect()
}

fn push_random_pauli(
    out: &mut Vec<Instruction>,
    probabilities: Vec<f64>,
    alternatives: Vec<PauliString>,
    heralded: bool,
) {
    out.push(Instruction::RandomPauli {
        probabilities,
        alternatives,
        heralded,
    });
}

fn tableau_instructions(circuit: &Circuit) -> Vec<Instruction> {
    let mut out = Vec::with_capacity(circuit.instructions.len());
    for instruction in &circuit.instructions {
        let kind = instruction.kind;
        if let Some(gate) = gate1_for(kind) {
            for &qubit in &instruction.qubits {
                push_gate1(&mut out, gate, qubit);
            }
            continue;
        }

        match kind {
            Kind::Tick => {}
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
                for pair in instruction.qubits.chunks_exact(2) {
                    push_pair(&mut out, kind, pair[0], pair[1]);
                }
            }
            Kind::T | Kind::TDag => {
                for &qubit in &instruction.qubits {
                    out.push(Instruction::T {
                        basis: PauliBasis::Z,
                        qubit,
                        adjoint: kind == Kind::TDag,
                    });
                }
            }
            Kind::PauliRotation => {
                for product in &instruction.pauli_products {
                    out.push(Instruction::PauliRotation {
                        axis: product_pauli(product),
                        kernel_angle: instruction.kernel_angle,
                    });
                }
            }
            Kind::MZ | Kind::MX | Kind::MY | Kind::Mrz | Kind::Mrx | Kind::Mry => {
                let (basis, reset) = match kind {
                    Kind::MX | Kind::Mrx => (PauliBasis::X, kind == Kind::Mrx),
                    Kind::MY | Kind::Mry => (PauliBasis::Y, kind == Kind::Mry),
                    _ => (PauliBasis::Z, kind == Kind::Mrz),
                };
                for target in &instruction.measurement_targets {
                    let observable =
                        PauliString::single(circuit.nqubits, target.qubit, Pauli::from(basis));
                    out.push(Instruction::MeasureWithReadoutError {
                        observable: if target.inverted {
                            neg(observable)
                        } else {
                            observable
                        },
                        probability: instruction.probability,
                    });
                    if reset {
                        out.push(Instruction::Reset {
                            basis,
                            qubit: target.qubit,
                        });
                    }
                }
            }
            Kind::RZ | Kind::RX | Kind::RY => {
                let basis = match kind {
                    Kind::RX => PauliBasis::X,
                    Kind::RY => PauliBasis::Y,
                    _ => PauliBasis::Z,
                };
                for &qubit in &instruction.qubits {
                    out.push(Instruction::Reset { basis, qubit });
                }
            }
            Kind::Mpp => {
                out.extend(instruction.pauli_products.iter().map(|product| {
                    Instruction::MeasureWithReadoutError {
                        observable: product_pauli(product),
                        probability: instruction.probability,
                    }
                }));
            }
            Kind::ExpVal => {
                out.extend(
                    instruction
                        .pauli_products
                        .iter()
                        .map(|product| Instruction::Expectation(product_pauli(product))),
                );
            }
            Kind::XError | Kind::YError | Kind::ZError => {
                let pauli = match kind {
                    Kind::XError => Pauli::X,
                    Kind::YError => Pauli::Y,
                    _ => Pauli::Z,
                };
                for &qubit in &instruction.qubits {
                    push_random_pauli(
                        &mut out,
                        vec![instruction.probability],
                        vec![PauliString::single(circuit.nqubits, qubit, pauli)],
                        false,
                    );
                }
            }
            Kind::Depolarize1 | Kind::Depolarize2 | Kind::Depolarize3 => {
                let arity = match kind {
                    Kind::Depolarize1 => 1,
                    Kind::Depolarize2 => 2,
                    _ => 3,
                };
                for qubits in instruction.qubits.chunks_exact(arity) {
                    let alternatives = pauli_channel_alternatives(circuit.nqubits, qubits);
                    push_random_pauli(
                        &mut out,
                        vec![
                            instruction.probability / alternatives.len() as f64;
                            alternatives.len()
                        ],
                        alternatives,
                        false,
                    );
                }
            }
            Kind::PauliChannel1 | Kind::PauliChannel2 | Kind::PauliChannel3 => {
                let arity = match kind {
                    Kind::PauliChannel1 => 1,
                    Kind::PauliChannel2 => 2,
                    _ => 3,
                };
                for qubits in instruction.qubits.chunks_exact(arity) {
                    push_random_pauli(
                        &mut out,
                        instruction.probabilities.clone(),
                        pauli_channel_alternatives(circuit.nqubits, qubits),
                        false,
                    );
                }
            }
            Kind::PauliProductChannel => push_random_pauli(
                &mut out,
                instruction.probabilities.clone(),
                instruction
                    .pauli_products
                    .iter()
                    .map(product_pauli)
                    .collect(),
                false,
            ),
            Kind::HeraldedErase | Kind::HeraldedPauliChannel1 => {
                let probabilities = if kind == Kind::HeraldedErase {
                    vec![instruction.probability / 4.0; 4]
                } else {
                    instruction.probabilities.clone()
                };
                for &qubit in &instruction.qubits {
                    push_random_pauli(
                        &mut out,
                        probabilities.clone(),
                        (0..4)
                            .map(|code| pauli_from_code(circuit.nqubits, &[qubit], code))
                            .collect(),
                        true,
                    );
                }
            }
            Kind::MPad => {
                for target in &instruction.measurement_targets {
                    debug_assert!(target.qubit <= 1);
                    out.push(Instruction::Record {
                        value: (target.qubit != 0) != target.inverted,
                        flip_probability: instruction.probability,
                    });
                }
            }
            Kind::FeedbackX | Kind::FeedbackY | Kind::FeedbackZ => {
                let basis = match kind {
                    Kind::FeedbackX => PauliBasis::X,
                    Kind::FeedbackY => PauliBasis::Y,
                    _ => PauliBasis::Z,
                };
                for target in &instruction.feedback_targets {
                    debug_assert!(target.record > 0);
                    out.push(Instruction::ConditionalPauli {
                        basis,
                        qubit: target.qubit,
                        control: target.record - 1,
                    });
                }
            }
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
            | Kind::Z => unreachable!("single-qubit Cliffords were handled above"),
        }
    }
    out
}

impl Instruction {
    /// Translates a circuit once into instructions reusable by
    /// [`TableauSimulator::apply_batch`].
    #[must_use]
    pub fn from_circuit(circuit: &Circuit) -> Vec<Self> {
        tableau_instructions(circuit)
    }
}

impl TableauSimulator {
    /// Apply a named single-qubit Clifford.
    ///
    /// Realized from the engine's primitive frame updates rather than from
    /// [`Gate1Q::images`]: one or two `O(⌈n/64⌉)` row operations, no tableau
    /// and no allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use ticit::TableauSimulator;
    /// use ticit::Gate1Q;
    ///
    /// // C_XYZ cubes to the identity, so |+⟩ comes back to |+⟩.
    /// let mut sim = TableauSimulator::with_seed(1, 0);
    /// sim.h(0);
    /// for _ in 0..3 {
    ///     sim.gate1(Gate1Q::Cxyz, 0);
    /// }
    /// assert!((sim.peek_x(0)? - 1.0).abs() < 1e-9);
    /// # Ok::<(), ticit::SimError>(())
    /// ```
    pub fn gate1(&mut self, gate: Gate1Q, qubit: usize) {
        // Every composition is verified against `gate.images()` in the tests;
        // the period-three block below is the full 2×2×2 of {S, S†} against
        // {√X, √X†} in both orders, which is exactly the eight signed cycles.
        match gate {
            Gate1Q::X => self.x(qubit),
            Gate1Q::Y => self.y(qubit),
            Gate1Q::Z => self.z(qubit),
            Gate1Q::H => self.h(qubit),
            Gate1Q::S => self.s(qubit),
            Gate1Q::SDag => self.s_dag(qubit),
            Gate1Q::SqrtX => self.sqrt_x(qubit),
            Gate1Q::SqrtXDag => self.sqrt_x_dag(qubit),
            Gate1Q::SqrtY => self.sqrt_y(qubit),
            Gate1Q::SqrtYDag => self.sqrt_y_dag(qubit),
            Gate1Q::Hxy => {
                self.s_dag(qubit);
                self.x(qubit);
            }
            Gate1Q::Hyz => {
                self.sqrt_x(qubit);
                self.z(qubit);
            }
            Gate1Q::Hnxy => {
                self.s(qubit);
                self.x(qubit);
            }
            Gate1Q::Hnxz => {
                self.h(qubit);
                self.y(qubit);
            }
            Gate1Q::Hnyz => {
                self.sqrt_x(qubit);
                self.y(qubit);
            }
            Gate1Q::Cxyz => {
                self.sqrt_x(qubit);
                self.s(qubit);
            }
            Gate1Q::Cnxyz => {
                self.sqrt_x(qubit);
                self.s_dag(qubit);
            }
            Gate1Q::Cxnyz => {
                self.sqrt_x_dag(qubit);
                self.s_dag(qubit);
            }
            Gate1Q::Cxynz => {
                self.sqrt_x_dag(qubit);
                self.s(qubit);
            }
            Gate1Q::Czyx => {
                self.s_dag(qubit);
                self.sqrt_x_dag(qubit);
            }
            Gate1Q::Cnzyx => {
                self.s_dag(qubit);
                self.sqrt_x(qubit);
            }
            Gate1Q::Cznyx => {
                self.s(qubit);
                self.sqrt_x(qubit);
            }
            Gate1Q::Czynx => {
                self.s(qubit);
                self.sqrt_x_dag(qubit);
            }
        }
    }

    /// Apply the two-qubit gate `<A>C<B>`: `target` acts on `target_qubit`
    /// exactly when `control_qubit` sits in the `−1` eigenstate of `control`.
    ///
    /// Same operator as
    /// [`controlled_pauli`](Self::controlled_pauli) on the corresponding
    /// single-qubit axes, reached without building either Pauli. Writing
    /// `CZ = ½[(I + Z_c) ⊗ I + (I − Z_c) ⊗ Z_t]`, conjugating a qubit by any
    /// Clifford `G` replaces that qubit's `Z` with `G Z G†` and touches nothing
    /// else, so `<A>C<B> = (G_A ⊗ G_B) · CZ · (G_A ⊗ G_B)†` for any `G_A`,
    /// `G_B` carrying `Z` onto `A` and `B` — here `H` for `X` and `√X†` for
    /// `Y`. `CX`, `CZ` and `XCZ` skip the conjugation and hit the engine's
    /// primitives directly.
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit. (The
    /// [`controlled_pauli`](Self::controlled_pauli) path reports that case as
    /// [`SimError::NonCommutingControlledPaulis`] instead, since two distinct
    /// axes on one qubit anticommute.)
    pub fn gate2(
        &mut self,
        control: PauliBasis,
        target: PauliBasis,
        control_qubit: usize,
        target_qubit: usize,
    ) -> Result<(), SimError> {
        // Checked up front so the conjugation below cannot leave a half-applied
        // basis rotation behind: like every other engine entry point, a
        // rejected gate must not touch the state.
        if control_qubit == target_qubit {
            return Err(SimError::RepeatedQubit(control_qubit));
        }
        use PauliBasis::{X, Z};
        match (control, target) {
            (Z, Z) => self.cz(control_qubit, target_qubit),
            (Z, X) => self.cx(control_qubit, target_qubit),
            // `XCZ` is `CZ` with the Hadamard on the control, which is what a
            // `CX` aimed the other way already is.
            (X, Z) => self.cx(target_qubit, control_qubit),
            _ => {
                self.basis_to_z(control, control_qubit);
                self.basis_to_z(target, target_qubit);
                let applied = self.cz(control_qubit, target_qubit);
                self.z_to_basis(target, target_qubit);
                self.z_to_basis(control, control_qubit);
                applied
            }
        }
    }

    /// Apply `T`/`T†` about a single-qubit basis axis.
    ///
    /// Equivalent to [`t_pauli`](Self::t_pauli) on the corresponding
    /// single-qubit Pauli, but reached by conjugating the engine's `Z`-axis
    /// rotation (`T_B = G · T_Z · G†` for `G Z G† = B`), so no observable is
    /// built and the rotation decomposes a stored frame row rather than a Pauli
    /// product.
    ///
    /// # Errors
    /// Propagates [`t_pauli`](Self::t_pauli) errors.
    pub fn t_basis(
        &mut self,
        basis: PauliBasis,
        qubit: usize,
        adjoint: bool,
    ) -> Result<(), SimError> {
        self.basis_to_z(basis, qubit);
        let rotated = if adjoint {
            self.t_dag(qubit)
        } else {
            self.t(qubit)
        };
        // The frame rotation is unconditional: `t` only fails after leaving the
        // amplitude map untouched, and the caller's register must come back in
        // its own basis either way.
        self.z_to_basis(basis, qubit);
        rotated
    }

    fn sample_bernoulli(&mut self, probability: f64) -> Result<bool, SimError> {
        sample_bernoulli(&mut self.core.rng, probability)
            .map_err(|_| SimError::InvalidProbability(probability))
    }

    fn sample_alternative(
        &mut self,
        probabilities: &[f64],
    ) -> Result<(Option<usize>, f64), SimError> {
        let mut total = 0.0;
        for &probability in probabilities {
            if !(0.0..=1.0).contains(&probability) {
                return Err(SimError::InvalidProbability(probability));
            }
            total += probability;
        }
        if total > 1.0 + 1e-12 {
            return Err(SimError::InvalidProbabilityDistribution);
        }
        if probabilities.is_empty() || total == 0.0 {
            return Ok((None, total));
        }
        if probabilities.len() == 1 {
            return Ok((self.sample_bernoulli(probabilities[0])?.then_some(0), total));
        }

        let sample = rand_float(&mut self.core.rng);
        let mut cumulative = 0.0;
        for (index, &probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if sample < cumulative {
                return Ok((Some(index), total.min(1.0)));
            }
        }
        Ok((None, total.min(1.0)))
    }

    fn append_record(
        &mut self,
        outcome: &mut BatchOutcome,
        value: bool,
        flip_probability: f64,
    ) -> Result<(), SimError> {
        let recorded = value ^ self.sample_bernoulli(flip_probability)?;
        outcome.records.push(MeasureResult {
            outcome: recorded,
            probability: if recorded == value {
                1.0 - flip_probability
            } else {
                flip_probability
            },
            deterministic: flip_probability <= TOL || 1.0 - flip_probability <= TOL,
        });
        Ok(())
    }

    fn measure_with_readout_error(
        &mut self,
        observable: &PauliString,
        probability: f64,
    ) -> Result<MeasureResult, SimError> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(SimError::InvalidProbability(probability));
        }
        let raw = self.measure_observable(observable)?;
        let outcome = raw.outcome ^ self.sample_bernoulli(probability)?;
        let raw_true = if raw.outcome {
            raw.probability
        } else {
            1.0 - raw.probability
        };
        let recorded_true = raw_true * (1.0 - probability) + (1.0 - raw_true) * probability;
        let recorded_probability = if outcome {
            recorded_true
        } else {
            1.0 - recorded_true
        };
        Ok(MeasureResult {
            outcome,
            probability: recorded_probability,
            deterministic: recorded_probability <= TOL || 1.0 - recorded_probability <= TOL,
        })
    }

    fn apply_random_pauli(
        &mut self,
        outcome: &mut BatchOutcome,
        probabilities: &[f64],
        alternatives: &[PauliString],
        heralded: bool,
    ) -> Result<(), SimError> {
        if probabilities.len() != alternatives.len() {
            return Err(SimError::InvalidProbabilityDistribution);
        }
        let (selected, total) = self.sample_alternative(probabilities)?;
        if let Some(index) = selected {
            self.pauli(&alternatives[index]);
        }
        if heralded {
            let fired = selected.is_some();
            outcome.records.push(MeasureResult {
                outcome: fired,
                probability: if fired { total } else { 1.0 - total },
                deterministic: total <= TOL || 1.0 - total <= TOL,
            });
        }
        Ok(())
    }

    /// Apply a circuit, returning its measurement records.
    ///
    /// The circuit is translated to [`Instruction`]s and replayed through
    /// [`apply_batch`](Self::apply_batch). Detector and observable annotations
    /// do not add entries to the returned outcome.
    ///
    /// Every executable parser instruction is supported, including arbitrary
    /// Pauli rotations, noise channels, heralded records, `MPAD`, and `EXP_VAL`.
    ///
    /// # Errors
    ///
    /// Returns the relevant execution error. As with
    /// [`apply_batch`](Self::apply_batch), a failure leaves the operations
    /// before the failing instruction applied.
    ///
    /// # Examples
    ///
    /// ```
    /// use ticit::{Circuit, TableauSimulator};
    ///
    /// let circuit = Circuit::from_text("H 0\nCX 0 1\nM 0 1")?;
    /// let mut sim = TableauSimulator::with_seed(0, 7);
    /// let outcome = sim.apply_circuit(&circuit)?;
    /// assert_eq!(outcome.records[0].outcome, outcome.records[1].outcome);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn apply_circuit(&mut self, circuit: &Circuit) -> Result<BatchOutcome, SimError> {
        let instructions = tableau_instructions(circuit);
        self.ensure_qubits(circuit.nqubits);
        self.apply_batch(&instructions)
    }

    /// Run `instructions` in order, collecting every measurement.
    ///
    /// This is exactly the loop a caller would write by hand over the same
    /// operations — including the failure behaviour. An error aborts the batch
    /// where it happened, so the simulator reflects the instructions *before*
    /// the failure and the outcomes of those measurements are lost with the
    /// discarded [`BatchOutcome`].
    ///
    /// # Errors
    /// Whatever the underlying operation returns, plus
    /// [`SimError::MissingBatchRecord`] if an
    /// [`Instruction::ConditionalPauli`] names a measurement the batch has not
    /// reached yet.
    ///
    /// # Examples
    ///
    /// ```
    /// use ticit::{Gate1Q, Instruction};
    /// use ticit::{Pauli, PauliBasis, PauliString, TableauSimulator};
    ///
    /// // Measure a |+⟩ and undo the random outcome with a feedforward Z.
    /// let program = [
    ///     Instruction::Gate1 { gate: Gate1Q::H, qubit: 0 },
    ///     Instruction::Measure(PauliString::single(1, 0, Pauli::X)),
    ///     Instruction::ConditionalPauli { basis: PauliBasis::Z, qubit: 0, control: 0 },
    /// ];
    ///
    /// let mut sim = TableauSimulator::with_seed(1, 3);
    /// sim.apply_batch(&program)?;
    /// assert!((sim.peek_x(0)? - 1.0).abs() < 1e-9);
    /// # Ok::<(), ticit::SimError>(())
    /// ```
    pub fn apply_batch(&mut self, instructions: &[Instruction]) -> Result<BatchOutcome, SimError> {
        let mut outcome = BatchOutcome::default();
        for instruction in instructions {
            match instruction {
                Instruction::Gate1 { gate, qubit } => self.gate1(*gate, *qubit),
                Instruction::Gate2 {
                    control,
                    target,
                    control_qubit,
                    target_qubit,
                } => self.gate2(*control, *target, *control_qubit, *target_qubit)?,
                Instruction::Pauli { basis, qubit } => self.basis_pauli(*basis, *qubit),
                Instruction::T {
                    basis,
                    qubit,
                    adjoint,
                } => self.t_basis(*basis, *qubit, *adjoint)?,
                Instruction::TPauli { axis, adjoint } => self.t_pauli(axis, *adjoint)?,
                Instruction::PauliRotation { axis, kernel_angle } => {
                    self.pauli_rotation(axis, *kernel_angle)?;
                }
                Instruction::Measure(observable) => {
                    outcome.records.push(self.measure_observable(observable)?);
                }
                Instruction::MeasureWithReadoutError {
                    observable,
                    probability,
                } => outcome
                    .records
                    .push(self.measure_with_readout_error(observable, *probability)?),
                Instruction::Record {
                    value,
                    flip_probability,
                } => self.append_record(&mut outcome, *value, *flip_probability)?,
                Instruction::RandomPauli {
                    probabilities,
                    alternatives,
                    heralded,
                } => {
                    self.apply_random_pauli(&mut outcome, probabilities, alternatives, *heralded)?
                }
                Instruction::Expectation(observable) => outcome
                    .expectation_values
                    .push(self.peek_observable_expectation(observable)?),
                Instruction::Reset { basis, qubit } => self.reset_basis(*basis, *qubit)?,
                Instruction::ConditionalPauli {
                    basis,
                    qubit,
                    control,
                } => {
                    let gate = outcome
                        .records
                        .get(*control)
                        .ok_or(SimError::MissingBatchRecord { index: *control })?;
                    if gate.outcome {
                        self.basis_pauli(*basis, *qubit);
                    }
                }
            }
            // Only rotations, measurement and reset can move the rank, but
            // reading it is a length load — cheaper than deciding whether to.
            outcome.max_rank = outcome.max_rank.max(self.rank());
        }
        Ok(outcome)
    }

    /// Conjugate `qubit` so that `basis` becomes `Z`: the `G†` of `G Z G† =
    /// basis`. Shared by [`gate2`](Self::gate2) and
    /// [`t_basis`](Self::t_basis), which both realize a `Z`-axis operation
    /// about another axis.
    fn basis_to_z(&mut self, basis: PauliBasis, qubit: usize) {
        match basis {
            PauliBasis::X => self.h(qubit),
            PauliBasis::Y => self.sqrt_x(qubit),
            PauliBasis::Z => {}
        }
    }

    /// Undo [`basis_to_z`](Self::basis_to_z).
    fn z_to_basis(&mut self, basis: PauliBasis, qubit: usize) {
        match basis {
            PauliBasis::X => self.h(qubit),
            PauliBasis::Y => self.sqrt_x_dag(qubit),
            PauliBasis::Z => {}
        }
    }

    fn basis_pauli(&mut self, basis: PauliBasis, qubit: usize) {
        match basis {
            PauliBasis::X => self.x(qubit),
            PauliBasis::Y => self.y(qubit),
            PauliBasis::Z => self.z(qubit),
        }
    }

    fn reset_basis(&mut self, basis: PauliBasis, qubit: usize) -> Result<(), SimError> {
        match basis {
            PauliBasis::X => self.reset_x(qubit),
            PauliBasis::Y => self.reset_y(qubit),
            PauliBasis::Z => self.reset_z(qubit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    use crate::Pauli;
    use num_complex::Complex64;
    use paulimer::{Clifford, CliffordUnitary, DensePauli, Pauli as PaulimerPauli, PauliMutable};

    /// Every [`Gate1Q`], for the tableau sweep.
    const ALL_GATE1: [Gate1Q; 23] = [
        Gate1Q::X,
        Gate1Q::Y,
        Gate1Q::Z,
        Gate1Q::H,
        Gate1Q::S,
        Gate1Q::SDag,
        Gate1Q::SqrtX,
        Gate1Q::SqrtXDag,
        Gate1Q::SqrtY,
        Gate1Q::SqrtYDag,
        Gate1Q::Hxy,
        Gate1Q::Hyz,
        Gate1Q::Hnxy,
        Gate1Q::Hnxz,
        Gate1Q::Hnyz,
        Gate1Q::Cxyz,
        Gate1Q::Czyx,
        Gate1Q::Cnxyz,
        Gate1Q::Cxnyz,
        Gate1Q::Cxynz,
        Gate1Q::Cnzyx,
        Gate1Q::Cznyx,
        Gate1Q::Czynx,
    ];

    const BASES: [PauliBasis; 3] = [PauliBasis::X, PauliBasis::Y, PauliBasis::Z];

    /// A generic three-qubit state: entangled, magic on two qubits, and no
    /// Pauli eigenstate anywhere. A gate that differs from its reference by a
    /// Pauli or a sign is visible here; one that differs only by a global phase
    /// is not, which is exactly the equivalence the frame tracks.
    fn scrambled() -> TableauSimulator {
        let mut sim = TableauSimulator::with_seed(3, 0xC0FF_EE01);
        sim.h(0);
        sim.t(0).expect("magic injection stays under the rank cap");
        sim.h(1);
        sim.s(1);
        sim.cx(0, 1).expect("distinct operands");
        sim.sqrt_x(2);
        sim.t(2).expect("magic injection stays under the rank cap");
        sim.cz(1, 2).expect("distinct operands");
        sim
    }

    /// `|⟨a|b⟩|`, the global-phase-blind state comparison.
    fn overlap(a: &TableauSimulator, b: &TableauSimulator) -> f64 {
        let (left, right) = (a.state_vector(), b.state_vector());
        assert_eq!(
            left.len(),
            right.len(),
            "compared registers differ in width"
        );
        let inner: Complex64 = left
            .iter()
            .zip(&right)
            .map(|(x, y)| x.conj() * y)
            .sum::<Complex64>();
        inner.norm()
    }

    fn assert_same_state(actual: &TableauSimulator, expected: &TableauSimulator, what: &str) {
        let fidelity = overlap(actual, expected);
        assert!(
            (fidelity - 1.0).abs() < 1e-9,
            "{what}: states differ (overlap {fidelity})"
        );
    }

    /// The one-qubit signed Pauli `±{X,Y,Z}` as a dense Pauli.
    fn signed_image(axis: PauliBasis, negated: bool) -> DensePauli {
        let mut image = match axis {
            PauliBasis::X => <DensePauli as PaulimerPauli>::x(0, 1),
            PauliBasis::Y => <DensePauli as PaulimerPauli>::y(0, 1),
            PauliBasis::Z => <DensePauli as PaulimerPauli>::z(0, 1),
        };
        if negated {
            image.add_assign_phase_exp(2); // ×(−1)
        }
        image
    }

    /// The reference realization of a [`Gate1Q`]: its tableau, turned into a
    /// [`CliffordUnitary`] the same way `bloc_compile`'s stim-pinned gate table
    /// does it (`from_preimages` yields the inverse, so invert).
    fn reference_clifford(gate: Gate1Q) -> CliffordUnitary {
        let ((x_axis, x_neg), (z_axis, z_neg)) = gate.images();
        CliffordUnitary::from_preimages(&[signed_image(x_axis, x_neg), signed_image(z_axis, z_neg)])
            .inverse()
    }

    fn single_qubit_pauli(basis: PauliBasis, qubit: usize, n: usize) -> PauliString {
        PauliString::single(n, qubit, basis.into())
    }

    /// Each [`Gate1Q`] composition must reproduce the tableau its own
    /// [`Gate1Q::images`] advertises — the property the whole fast path rests
    /// on, since the compositions were derived by hand from those tableaux.
    #[test]
    fn gate1_compositions_match_their_tableaux() {
        for gate in ALL_GATE1 {
            for qubit in 0..3 {
                let mut fast = scrambled();
                fast.gate1(gate, qubit);

                let mut reference = scrambled();
                reference.apply_clifford(&reference_clifford(gate), &[qubit]);

                assert_same_state(&fast, &reference, &format!("{gate:?} on qubit {qubit}"));
            }
        }
    }

    /// The `CZ`-conjugation fast path must reproduce
    /// [`TableauSimulator::controlled_pauli`] on the same axes, for all nine
    /// `<A>C<B>` combinations and in both operand orders (the `X`/`Z` special
    /// cases are asymmetric).
    #[test]
    fn gate2_matches_controlled_pauli() {
        for control in BASES {
            for target in BASES {
                for (c, t) in [(0usize, 1usize), (1, 0), (0, 2)] {
                    let mut fast = scrambled();
                    fast.gate2(control, target, c, t)
                        .expect("distinct operands");

                    let mut reference = scrambled();
                    let n = reference.num_qubits();
                    reference
                        .controlled_pauli(
                            &single_qubit_pauli(control, c, n),
                            &single_qubit_pauli(target, t, n),
                        )
                        .expect("single-qubit axes on distinct qubits commute");

                    assert_same_state(
                        &fast,
                        &reference,
                        &format!("{control:?}C{target:?} on ({c},{t})"),
                    );
                }
            }
        }
    }

    /// A rejected two-qubit gate must not leave a basis rotation behind.
    #[test]
    fn gate2_rejects_repeated_operands_without_mutating() {
        let mut sim = scrambled();
        let before = sim.state_vector();
        for control in BASES {
            for target in BASES {
                assert_eq!(
                    sim.gate2(control, target, 1, 1),
                    Err(SimError::RepeatedQubit(1))
                );
            }
        }
        assert_eq!(sim.state_vector(), before);
    }

    /// `t_basis` conjugates the engine's `Z` rotation; it must agree with
    /// `t_pauli` on the matching axis, in both rotation directions.
    #[test]
    fn t_basis_matches_t_pauli() {
        for basis in BASES {
            for adjoint in [false, true] {
                let mut fast = scrambled();
                fast.t_basis(basis, 1, adjoint)
                    .expect("a single-qubit axis rotation stays under the cap");

                let mut reference = scrambled();
                let n = reference.num_qubits();
                reference
                    .t_pauli(&single_qubit_pauli(basis, 1, n), adjoint)
                    .expect("a single-qubit axis rotation stays under the cap");

                assert_same_state(&fast, &reference, &format!("T_{basis:?} adjoint={adjoint}"));
            }
        }
    }

    /// A multi-qubit rotation axis is the one operation the instruction set
    /// cannot reduce to single-qubit steps, which is why `TPauli` carries a
    /// whole Pauli. It must agree with the engine's own entry point.
    #[test]
    fn t_pauli_instruction_matches_the_engine() {
        let axis = PauliString::from_terms(3, [(0, Pauli::Z), (2, Pauli::Z)]);

        for adjoint in [false, true] {
            let mut batched = scrambled();
            batched
                .apply_batch(&[Instruction::TPauli {
                    axis: axis.clone(),
                    adjoint,
                }])
                .expect("a two-qubit axis rotation stays under the cap");

            let mut manual = scrambled();
            manual
                .t_pauli(&axis, adjoint)
                .expect("a two-qubit axis rotation stays under the cap");

            assert_same_state(&batched, &manual, &format!("T_Z0Z2 adjoint={adjoint}"));
        }
    }

    /// A batch must land the same state, the same outcomes and the same RNG
    /// position as the equivalent procedural sequence run from the same seed.
    #[test]
    fn batch_matches_the_equivalent_procedural_run() {
        let program = [
            Instruction::Gate1 {
                gate: Gate1Q::H,
                qubit: 0,
            },
            Instruction::T {
                basis: PauliBasis::Z,
                qubit: 0,
                adjoint: false,
            },
            Instruction::Gate2 {
                control: PauliBasis::Z,
                target: PauliBasis::X,
                control_qubit: 0,
                target_qubit: 1,
            },
            Instruction::Gate1 {
                gate: Gate1Q::Cxyz,
                qubit: 2,
            },
            Instruction::Measure(PauliString::single(3, 1, Pauli::Z)),
            Instruction::Pauli {
                basis: PauliBasis::X,
                qubit: 2,
            },
            Instruction::Reset {
                basis: PauliBasis::X,
                qubit: 2,
            },
            Instruction::Measure(PauliString::single(3, 2, Pauli::Z)),
        ];

        let mut batched = TableauSimulator::with_seed(3, 99);
        let outcome = batched.apply_batch(&program).expect("valid program");

        let mut manual = TableauSimulator::with_seed(3, 99);
        manual.h(0);
        manual.t(0).expect("under the cap");
        manual.cx(0, 1).expect("distinct operands");
        manual.gate1(Gate1Q::Cxyz, 2);
        let first = manual.measure(1).expect("within the rank cap");
        manual.x(2);
        manual
            .reset_x(2)
            .expect("reset is a measure plus a frame Z");
        let second = manual.measure(2).expect("within the rank cap");

        assert_eq!(outcome.records, vec![first, second]);
        assert_eq!(
            outcome.max_rank, 2,
            "the T doubles the rank and nothing collapses it"
        );
        assert_same_state(&batched, &manual, "batch vs procedural");
    }

    /// A conditional Pauli reads its own batch's records, so a feedforward
    /// correction inside one batch must undo the measurement it follows.
    #[test]
    fn conditional_pauli_reads_its_own_batch() {
        // Measure Z on |+⟩, then flip back to |+⟩ if the outcome was −1: the
        // state is |0⟩ or |1⟩ before the correction and always |0⟩ after.
        let program = [
            Instruction::Gate1 {
                gate: Gate1Q::H,
                qubit: 0,
            },
            Instruction::Measure(PauliString::single(1, 0, Pauli::Z)),
            Instruction::ConditionalPauli {
                basis: PauliBasis::X,
                qubit: 0,
                control: 0,
            },
        ];
        for seed in 0..8 {
            let mut sim = TableauSimulator::with_seed(1, seed);
            sim.apply_batch(&program).expect("valid program");
            let z = sim.peek_z(0).expect("qubit 0 is live");
            assert!((z - 1.0).abs() < 1e-9, "seed {seed}: correction left |1⟩");
        }
    }

    #[test]
    fn conditional_pauli_rejects_an_unreached_record() {
        let program = [Instruction::ConditionalPauli {
            basis: PauliBasis::X,
            qubit: 0,
            control: 0,
        }];
        let mut sim = TableauSimulator::with_seed(1, 0);
        assert_eq!(
            sim.apply_batch(&program),
            Err(SimError::MissingBatchRecord { index: 0 })
        );
    }

    /// An empty batch is a no-op and reports no peak, so a caller folding
    /// `max_rank` into a running maximum is unaffected by it.
    #[test]
    fn empty_batch_reports_no_peak() {
        let mut sim = TableauSimulator::with_seed(2, 0);
        let outcome = sim.apply_batch(&[]).expect("empty program");
        assert_eq!(outcome.max_rank, 0);
        assert!(outcome.records.is_empty());
    }

    /// The peak is the maximum *over the batch*, not the rank at the end: the
    /// reset below collapses the magic the `T` injected.
    #[test]
    fn max_rank_reports_the_peak_not_the_final_rank() {
        let program = [
            Instruction::Gate1 {
                gate: Gate1Q::H,
                qubit: 0,
            },
            Instruction::T {
                basis: PauliBasis::Z,
                qubit: 0,
                adjoint: false,
            },
            Instruction::Reset {
                basis: PauliBasis::Z,
                qubit: 0,
            },
        ];
        let mut sim = TableauSimulator::with_seed(1, 5);
        let outcome = sim.apply_batch(&program).expect("valid program");
        assert_eq!(outcome.max_rank, 2);
        assert_eq!(sim.rank(), 1);
    }

    #[test]
    fn apply_circuit_runs_rotations_noise_records_and_expectations() {
        let text = "\
H 0
R_Z(0.2) 0
EXP_VAL X0
M(1) 0
MPAD 1
CX rec[-1] 1
M 1
X_ERROR(1) 2
M 2
PAULI_CHANNEL_1(1,0,0) 3
M 3
HERALDED_PAULI_CHANNEL_1(0,1,0,0) 4
M 4
HERALDED_ERASE(0) 5
E(1) X6
M 6
MR(1) 7
M 7
DEPOLARIZE3(0) 8 9 10
MPP Z8*Z9*Z10
";
        let mut sim = TableauSimulator::with_seed(0, 9);
        let circuit = Circuit::from_text(text).expect("full circuit parses");
        let outcome = sim
            .apply_batch(&Instruction::from_circuit(&circuit))
            .expect("full circuit executes");

        assert!((outcome.expectation_values[0] - (0.2 * PI).cos()).abs() < 1e-9);
        let records: Vec<bool> = outcome
            .records
            .iter()
            .map(|record| record.outcome)
            .collect();
        assert_eq!(
            &records[1..],
            &[
                true, true, true, true, true, true, false, true, true, false, false
            ]
        );
        assert_eq!(sim.num_qubits(), 11);
    }

    #[test]
    fn apply_circuit_two_qubit_cliffords_match_the_sampler_frame() {
        type Gate = fn(&mut crate::frames::CliffordFrame, usize, usize);
        let cases: [(&str, Gate); 22] = [
            ("CX", crate::frames::left_cx),
            ("CY", crate::frames::left_cy),
            ("CZ", crate::frames::left_cz),
            ("SWAP", crate::frames::left_swap),
            ("CXSWAP", crate::frames::left_cxswap),
            ("CZSWAP", crate::frames::left_czswap),
            ("ISWAP", crate::frames::left_iswap),
            ("ISWAP_DAG", crate::frames::left_iswap_dag),
            ("SQRT_XX", crate::frames::left_sqrt_xx),
            ("SQRT_XX_DAG", crate::frames::left_sqrt_xx_dag),
            ("SQRT_YY", crate::frames::left_sqrt_yy),
            ("SQRT_YY_DAG", crate::frames::left_sqrt_yy_dag),
            ("SQRT_ZZ", crate::frames::left_sqrt_zz),
            ("SQRT_ZZ_DAG", crate::frames::left_sqrt_zz_dag),
            ("SWAPCX", crate::frames::left_swapcx),
            ("XCX", crate::frames::left_xcx),
            ("XCY", crate::frames::left_xcy),
            ("XCZ", crate::frames::left_xcz),
            ("YCX", crate::frames::left_ycx),
            ("YCY", crate::frames::left_ycy),
            ("YCZ", crate::frames::left_ycz),
            ("ZCY", crate::frames::left_cy),
        ];

        let row_pauli = |row: super::super::frame::RowPauli| {
            let mut pauli = PauliString::new(2);
            for qubit in 0..2 {
                let bit = 1 << qubit;
                pauli.set(
                    qubit,
                    match (row.x[0] & bit != 0, row.z[0] & bit != 0) {
                        (false, false) => Pauli::I,
                        (true, false) => Pauli::X,
                        (false, true) => Pauli::Z,
                        (true, true) => Pauli::Y,
                    },
                );
            }
            pauli.set_phase(row.phase.into());
            pauli
        };

        for (name, gate) in cases {
            let mut sim = TableauSimulator::with_seed(2, 0);
            let circuit = Circuit::from_text(&format!("{name} 0 1")).expect("circuit parses");
            sim.apply_circuit(&circuit)
                .expect("Clifford circuit executes");
            let mut expected = crate::frames::CliffordFrame::new(2);
            gate(&mut expected, 0, 1);
            for qubit in 0..2 {
                assert_eq!(
                    row_pauli(sim.core.r.preimage_x(qubit)),
                    crate::frames::preimage(&expected, &PauliString::single(2, qubit, Pauli::X),),
                    "{name} X{qubit} preimage",
                );
                assert_eq!(
                    row_pauli(sim.core.r.preimage_z(qubit)),
                    crate::frames::preimage(&expected, &PauliString::single(2, qubit, Pauli::Z),),
                    "{name} Z{qubit} preimage",
                );
            }
        }
    }

    /// A failure aborts the batch in place: the instructions before it have
    /// been applied and the ones after have not.
    #[test]
    fn a_failed_instruction_aborts_the_batch_in_place() {
        let program = [
            Instruction::Gate1 {
                gate: Gate1Q::H,
                qubit: 0,
            },
            Instruction::Gate2 {
                control: PauliBasis::Z,
                target: PauliBasis::X,
                control_qubit: 0,
                target_qubit: 0,
            },
            Instruction::Gate1 {
                gate: Gate1Q::X,
                qubit: 0,
            },
        ];
        let mut sim = TableauSimulator::with_seed(1, 0);
        assert_eq!(
            sim.apply_batch(&program),
            Err(SimError::RepeatedQubit(0)),
            "the batch must surface the operand error"
        );

        let mut prefix = TableauSimulator::with_seed(1, 0);
        prefix.h(0);
        assert_same_state(&sim, &prefix, "aborted batch keeps its prefix");
    }
}
