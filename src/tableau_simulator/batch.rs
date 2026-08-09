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

use crate::{PauliBasis, PauliString};

use super::{MeasureResult, SimError, TableauSimulator};

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
    /// Measure a Pauli observable, appending its result to
    /// [`BatchOutcome::records`].
    Measure(PauliString),
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
    /// One entry per [`Instruction::Measure`], in batch order.
    pub records: Vec<MeasureResult>,
    /// Highest stabilizer rank observed after any instruction, or `0` for an
    /// empty batch. Callers that budget memory against the rank need the peak,
    /// which a post-batch [`TableauSimulator::rank`] read would miss — a `T` grows the
    /// rank before a post-selection collapses it back.
    pub max_rank: usize,
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
                Instruction::Measure(observable) => {
                    outcome.records.push(self.measure_observable(observable)?);
                }
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
            // Only `T`, measurement and reset can move the rank, but reading it
            // is a length load — cheaper than deciding whether to read it.
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
