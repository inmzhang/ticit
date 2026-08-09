//! Differential tests for the simulator's `Frame` tableau against `paulimer`.
//!
//! `Frame` replicates `paulimer::CliffordUnitary`'s sign conventions exactly —
//! the simulator's phases are pinned by cross-validation, so "equivalent up to
//! a global phase" is not good enough. Every test here runs the same operation
//! on both representations and compares all `2n` preimages, bits and phases.
//!
//! `Frame` is private to `ticit::tableau_simulator`, so this file compiles the
//! module directly instead of widening the crate's public API. That works
//! because `frame.rs` names no type from the rest of `bloc_utils` — its Pauli
//! operands arrive as `PauliWords`, a borrowed pair of word masks — and its
//! `paulimer` interop sits behind `cfg(test)`, which an integration-test target
//! enables. Keep it that way: `paulimer` is a dev-dependency now, so anything
//! of it that leaked into the library would stop the crate building at all.

#[path = "../src/tableau_simulator/frame.rs"]
#[allow(dead_code)]
mod frame;

use binar::Bitwise;
use frame::{Frame, PauliWords, RowPauli};
use paulimer::{
    Clifford, CliffordMutable, CliffordUnitary, DensePauli, Pauli, PauliObservable,
    PositionedPauliObservable, SparsePauli, commutes_with,
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use ticit::{Pauli as BlocPauli, PauliString};

/// The frame's view of a `PauliString`, as `ticit::tableau_simulator` builds it.
fn pauli_words(pauli: &PauliString) -> PauliWords<'_> {
    PauliWords {
        x: pauli.x_words(),
        z: pauli.z_words(),
    }
}

/// The `SparsePauli` a `PauliString` names, for feeding the oracle.
fn sparse(pauli: &PauliString) -> SparsePauli {
    let terms: Vec<PositionedPauliObservable> = (0..pauli.nqubits)
        .filter_map(|qubit| {
            let observable = match pauli.get(qubit) {
                BlocPauli::I => return None,
                BlocPauli::X => PauliObservable::PlusX,
                BlocPauli::Y => PauliObservable::PlusY,
                BlocPauli::Z => PauliObservable::PlusZ,
            };
            Some(PositionedPauliObservable {
                qubit_id: qubit,
                observable,
            })
        })
        .collect();
    SparsePauli::from(terms.as_slice())
}

/// The unsigned `PauliString` a `SparsePauli` names, dropping its phase.
///
/// The operands `Frame` still takes carry no sign — `left_pauli` ignores one by
/// conjugation, and `preimage_into` derives its own from the `Y` sites — so the
/// random generators below stay on `paulimer`'s side, where they also feed the
/// oracle, and cross over here.
fn unsigned(pauli: &SparsePauli, n: usize) -> PauliString {
    let mut out = PauliString::new(n);
    for qubit in 0..n {
        out.set(
            qubit,
            match (pauli.x_bits().index(qubit), pauli.z_bits().index(qubit)) {
                (false, false) => BlocPauli::I,
                (true, false) => BlocPauli::X,
                (false, true) => BlocPauli::Z,
                (true, true) => BlocPauli::Y,
            },
        );
    }
    out
}

/// Qubit counts spanning the word-boundary cases: sub-word, exactly one word,
/// one bit past it, two words plus one, and a physical-circuit width.
const SIZES: [usize; 6] = [1, 3, 64, 65, 129, 300];

// ==============================================================================
// Comparison helpers
// ==============================================================================

/// Word-wise equality that tolerates different padding widths — `binar` rounds
/// bit vectors up to 512-bit blocks, and the two sides need not agree on how
/// many trailing zero words they carry.
fn words_eq(left: &[u64], right: &[u64]) -> bool {
    (0..left.len().max(right.len()))
        .all(|i| left.get(i).copied().unwrap_or(0) == right.get(i).copied().unwrap_or(0))
}

fn assert_row_eq(got: &RowPauli, want: &DensePauli, context: &str) {
    assert_pauli_eq(&got.to_dense(), want, context);
}

fn assert_pauli_eq(got: &DensePauli, want: &DensePauli, context: &str) {
    assert_eq!(
        got.xz_phase_exponent(),
        want.xz_phase_exponent(),
        "{context}: phase exponent\n  got {got:#}\n want {want:#}"
    );
    assert!(
        words_eq(got.x_bits().as_words(), want.x_bits().as_words()),
        "{context}: x bits\n  got {got:#}\n want {want:#}"
    );
    assert!(
        words_eq(got.z_bits().as_words(), want.z_bits().as_words()),
        "{context}: z bits\n  got {got:#}\n want {want:#}"
    );
}

fn assert_same_tableau(frame: &Frame, clifford: &CliffordUnitary, context: &str) {
    assert_eq!(
        frame.num_qubits(),
        clifford.num_qubits(),
        "{context}: qubit count"
    );
    for qubit in 0..clifford.num_qubits() {
        assert_row_eq(
            &frame.preimage_x(qubit),
            &clifford.preimage_x(qubit),
            &format!("{context}: preimage_x({qubit})"),
        );
        assert_row_eq(
            &frame.preimage_z(qubit),
            &clifford.preimage_z(qubit),
            &format!("{context}: preimage_z({qubit})"),
        );
    }
}

// ==============================================================================
// Random operand generation
// ==============================================================================

/// A random Hermitian Pauli of weight `1..=max_weight`, with random signs so
/// the phase-carrying paths are exercised too.
fn random_pauli(rng: &mut SmallRng, n: usize, max_weight: usize) -> SparsePauli {
    let weight = rng.random_range(1..=max_weight.min(n));
    let mut terms: Vec<PositionedPauliObservable> = Vec::with_capacity(weight);
    while terms.len() < weight {
        let qubit_id = rng.random_range(0..n);
        if terms.iter().any(|term| term.qubit_id == qubit_id) {
            continue;
        }
        let observable = match rng.random_range(0..6) {
            0 => PauliObservable::PlusX,
            1 => PauliObservable::PlusY,
            2 => PauliObservable::PlusZ,
            3 => PauliObservable::MinusX,
            4 => PauliObservable::MinusY,
            _ => PauliObservable::MinusZ,
        };
        terms.push(PositionedPauliObservable {
            qubit_id,
            observable,
        });
    }
    SparsePauli::from(terms.as_slice())
}

/// A random *unsigned* Pauli of weight `1..=max_weight`.
///
/// Where the operand's sign changes the answer — a controlled Pauli's does,
/// since it decides which eigenstate triggers the target — both sides have to
/// be fed the same thing, and `PauliWords` has no sign to feed. So the pair is
/// generated here and crossed to `paulimer` with [`sparse`].
fn random_pauli_string(rng: &mut SmallRng, n: usize, max_weight: usize) -> PauliString {
    let weight = rng.random_range(1..=max_weight.min(n));
    let mut paulis = PauliString::new(n);
    let mut placed = 0;
    while placed < weight {
        let qubit = rng.random_range(0..n);
        if paulis.get(qubit) != BlocPauli::I {
            continue;
        }
        paulis.set(
            qubit,
            match rng.random_range(0..3) {
                0 => BlocPauli::X,
                1 => BlocPauli::Y,
                _ => BlocPauli::Z,
            },
        );
        placed += 1;
    }
    paulis
}

/// A commuting pair, as `left_mul_controlled_pauli` requires (it
/// `debug_assert`s commutation, which fires in test builds).
fn random_commuting_pair(
    rng: &mut SmallRng,
    n: usize,
    max_weight: usize,
) -> (PauliString, PauliString) {
    let control = random_pauli_string(rng, n, max_weight);
    loop {
        let target = random_pauli_string(rng, n, max_weight);
        if control.commutes_with(&target) {
            debug_assert!(commutes_with(&sparse(&control), &sparse(&target)));
            return (control, target);
        }
    }
}

fn distinct_qubits(rng: &mut SmallRng, n: usize, count: usize) -> Vec<usize> {
    let mut picked: Vec<usize> = Vec::with_capacity(count);
    while picked.len() < count {
        let qubit = rng.random_range(0..n);
        if !picked.contains(&qubit) {
            picked.push(qubit);
        }
    }
    picked
}

/// Apply one random operation to both representations, returning a description
/// for failure messages.
fn apply_random_op(
    rng: &mut SmallRng,
    frame: &mut Frame,
    clifford: &mut CliffordUnitary,
    n: usize,
) -> String {
    // Two-qubit gates need two qubits; on a single-qubit frame the sampler
    // falls through to the one-qubit half of the table.
    let choices = if n >= 2 { 17 } else { 11 };
    let qubit = rng.random_range(0..n);
    match rng.random_range(0..choices) {
        0 => {
            frame.left_h(qubit);
            clifford.left_mul_hadamard(qubit);
            format!("h({qubit})")
        }
        1 => {
            frame.left_s(qubit);
            clifford.left_mul_root_z(qubit);
            format!("s({qubit})")
        }
        2 => {
            frame.left_s_dag(qubit);
            clifford.left_mul_root_z_inverse(qubit);
            format!("s_dag({qubit})")
        }
        3 => {
            frame.left_sqrt_x(qubit);
            clifford.left_mul_root_x(qubit);
            format!("sqrt_x({qubit})")
        }
        4 => {
            frame.left_sqrt_x_dag(qubit);
            clifford.left_mul_root_x_inverse(qubit);
            format!("sqrt_x_dag({qubit})")
        }
        5 => {
            frame.left_sqrt_y(qubit);
            clifford.left_mul_root_y(qubit);
            format!("sqrt_y({qubit})")
        }
        6 => {
            frame.left_sqrt_y_dag(qubit);
            clifford.left_mul_root_y_inverse(qubit);
            format!("sqrt_y_dag({qubit})")
        }
        7 => {
            frame.left_x(qubit);
            clifford.left_mul_x(qubit);
            format!("x({qubit})")
        }
        8 => {
            frame.left_y(qubit);
            clifford.left_mul_y(qubit);
            format!("y({qubit})")
        }
        9 => {
            frame.left_z(qubit);
            clifford.left_mul_z(qubit);
            format!("z({qubit})")
        }
        10 => {
            let pauli = random_pauli(rng, n, 4);
            frame.left_pauli(pauli_words(&unsigned(&pauli, n)));
            clifford.left_mul_pauli(&pauli);
            format!("pauli({pauli:#})")
        }
        11 => {
            let pair = distinct_qubits(rng, n, 2);
            frame.left_cx(pair[0], pair[1]);
            clifford.left_mul_cx(pair[0], pair[1]);
            format!("cx({}, {})", pair[0], pair[1])
        }
        12 => {
            let pair = distinct_qubits(rng, n, 2);
            frame.left_cz(pair[0], pair[1]);
            clifford.left_mul_cz(pair[0], pair[1]);
            format!("cz({}, {})", pair[0], pair[1])
        }
        13 => {
            let pair = distinct_qubits(rng, n, 2);
            frame.left_swap(pair[0], pair[1]);
            clifford.left_mul_swap(pair[0], pair[1]);
            format!("swap({}, {})", pair[0], pair[1])
        }
        14 => {
            let (control, target) = random_commuting_pair(rng, n, 3);
            frame.left_controlled_pauli(pauli_words(&control), pauli_words(&target));
            clifford.left_mul_controlled_pauli(&sparse(&control), &sparse(&target));
            format!("controlled_pauli({control}, {target})")
        }
        15 => {
            let support = distinct_qubits(rng, n, 1);
            let gate = CliffordUnitary::random(1, rng);
            frame.left_clifford(&gate, &support);
            clifford.left_mul_clifford(&gate, &support);
            format!("clifford_1q({support:?})")
        }
        _ => {
            let support = distinct_qubits(rng, n, 2);
            let gate = CliffordUnitary::random(2, rng);
            frame.left_clifford(&gate, &support);
            clifford.left_mul_clifford(&gate, &support);
            format!("clifford_2q({support:?})")
        }
    }
}

// ==============================================================================
// Tests
// ==============================================================================

#[test]
fn gate_sequences_match_paulimer() {
    for (seed, &n) in SIZES.iter().enumerate() {
        let mut rng = SmallRng::seed_from_u64(0xB10C + seed as u64);
        let mut frame = Frame::identity(n);
        let mut clifford = CliffordUnitary::identity(n);
        let steps = if n > 128 { 30 } else { 120 };
        for step in 0..steps {
            let op = apply_random_op(&mut rng, &mut frame, &mut clifford, n);
            assert_same_tableau(&frame, &clifford, &format!("n={n} step={step} after {op}"));
        }
    }
}

type FrameGate = fn(&mut Frame, usize);
type CliffordGate = fn(&mut CliffordUnitary, usize);

/// The single-qubit table is the one place the `√Y` decompositions and the
/// phase-only gates can hide sign errors, so sweep every gate exhaustively
/// against a random frame rather than trusting the random walk to reach them.
#[test]
fn every_single_qubit_gate_matches_paulimer() {
    let n = 3;
    let gates: [(&str, FrameGate, CliffordGate); 10] = [
        ("h", Frame::left_h, CliffordUnitary::left_mul_hadamard),
        ("s", Frame::left_s, CliffordUnitary::left_mul_root_z),
        (
            "s_dag",
            Frame::left_s_dag,
            CliffordUnitary::left_mul_root_z_inverse,
        ),
        (
            "sqrt_x",
            Frame::left_sqrt_x,
            CliffordUnitary::left_mul_root_x,
        ),
        (
            "sqrt_x_dag",
            Frame::left_sqrt_x_dag,
            CliffordUnitary::left_mul_root_x_inverse,
        ),
        (
            "sqrt_y",
            Frame::left_sqrt_y,
            CliffordUnitary::left_mul_root_y,
        ),
        (
            "sqrt_y_dag",
            Frame::left_sqrt_y_dag,
            CliffordUnitary::left_mul_root_y_inverse,
        ),
        ("x", Frame::left_x, CliffordUnitary::left_mul_x),
        ("y", Frame::left_y, CliffordUnitary::left_mul_y),
        ("z", Frame::left_z, CliffordUnitary::left_mul_z),
    ];
    let mut rng = SmallRng::seed_from_u64(7);
    let base = CliffordUnitary::random(n, &mut rng);
    for (name, on_frame, on_clifford) in gates {
        for qubit in 0..n {
            let mut frame = Frame::from_clifford_unitary(&base);
            let mut clifford = base.clone();
            on_frame(&mut frame, qubit);
            on_clifford(&mut clifford, qubit);
            assert_same_tableau(&frame, &clifford, &format!("{name}({qubit})"));
        }
    }
}

#[test]
fn preimage_into_matches_paulimer() {
    for (seed, &n) in SIZES.iter().enumerate() {
        let mut rng = SmallRng::seed_from_u64(0x9E11 + seed as u64);
        let clifford = CliffordUnitary::random(n, &mut rng);
        let frame = Frame::from_clifford_unitary(&clifford);
        let words = frame.words();
        let mut x_mask = vec![0u64; words];
        let mut z_mask = vec![0u64; words];
        for trial in 0..20 {
            // Unsigned on both sides: `PauliWords` carries no sign, so the
            // phase compared here is exactly the `i^{#Y}` the frame derives.
            let pauli = random_pauli_string(&mut rng, n, 5.min(n));
            let phase = frame.preimage_into(pauli_words(&pauli), &mut x_mask, &mut z_mask);
            let want = clifford.preimage(&sparse(&pauli));
            let context = format!("n={n} trial={trial} pauli={pauli:#}");
            assert_eq!(phase, want.xz_phase_exponent(), "{context}: phase");
            assert!(
                words_eq(&x_mask, want.x_bits().as_words()),
                "{context}: x bits"
            );
            assert!(
                words_eq(&z_mask, want.z_bits().as_words()),
                "{context}: z bits"
            );
        }
    }
}

/// The identity that makes the measurement rewrite legal:
/// `exp(iπ/4·P)·R = R·exp(iπ/4·G)` with `G = R†PR`.
#[test]
fn right_pauli_exp_matches_left_mul_pauli_exp() {
    for (seed, &n) in SIZES.iter().enumerate() {
        let mut rng = SmallRng::seed_from_u64(0x5EED + seed as u64);
        let mut clifford = CliffordUnitary::random(n, &mut rng);
        let mut frame = Frame::from_clifford_unitary(&clifford);
        let words = frame.words();
        for trial in 0..10 {
            let pauli = random_pauli(&mut rng, n, 4.min(n));
            let g = clifford.preimage(&pauli);
            frame.right_pauli_exp(
                &g.x_bits().as_words()[..words],
                &g.z_bits().as_words()[..words],
                g.xz_phase_exponent(),
            );
            clifford.left_mul_pauli_exp(&pauli);
            assert_same_tableau(
                &frame,
                &clifford,
                &format!("n={n} trial={trial} pauli_exp({pauli:#})"),
            );
        }
    }
}

/// The other half of the rewrite: `S_p·R = R·Z_p` with `S_p = R Z_p R†`.
#[test]
fn right_pauli_z_matches_left_mul_image_z() {
    for (seed, &n) in SIZES.iter().enumerate() {
        let mut rng = SmallRng::seed_from_u64(0xF00D + seed as u64);
        let mut clifford = CliffordUnitary::random(n, &mut rng);
        let mut frame = Frame::from_clifford_unitary(&clifford);
        for trial in 0..10 {
            let pivot = rng.random_range(0..n);
            let stabilizer = clifford.image_z(pivot);
            frame.right_pauli_z(pivot);
            clifford.left_mul_pauli(&stabilizer);
            assert_same_tableau(
                &frame,
                &clifford,
                &format!("n={n} trial={trial} z({pivot})"),
            );
        }
    }
}

#[test]
fn images_match_paulimer() {
    for (seed, &n) in SIZES.iter().enumerate() {
        let mut rng = SmallRng::seed_from_u64(0x1A9E + seed as u64);
        let clifford = CliffordUnitary::random(n, &mut rng);
        let frame = Frame::from_clifford_unitary(&clifford);
        for qubit in 0..n {
            assert_row_eq(
                &frame.image_x(qubit),
                &clifford.image_x(qubit),
                &format!("n={n} image_x({qubit})"),
            );
            assert_row_eq(
                &frame.image_z(qubit),
                &clifford.image_z(qubit),
                &format!("n={n} image_z({qubit})"),
            );
        }
    }
}

#[test]
fn resize_matches_paulimer() {
    for (seed, &(from, to)) in [(3usize, 7usize), (60, 70), (64, 128), (65, 65), (129, 300)]
        .iter()
        .enumerate()
    {
        let mut rng = SmallRng::seed_from_u64(0xC0DE + seed as u64);
        let mut clifford = CliffordUnitary::random(from, &mut rng);
        let mut frame = Frame::from_clifford_unitary(&clifford);

        frame.resize(to);
        clifford.resize(to);
        assert_same_tableau(&frame, &clifford, &format!("grow {from}->{to}"));

        // Growing then shrinking back is a round trip: the appended qubits are
        // still untouched, which is exactly what `shrink_clifford` demands.
        frame.resize(from);
        clifford.resize(from);
        assert_same_tableau(&frame, &clifford, &format!("shrink {to}->{from}"));

        // The frame stays usable at the narrower width.
        let mut rerun = frame.clone();
        let op = apply_random_op(&mut rng, &mut rerun, &mut clifford, from);
        assert_same_tableau(&rerun, &clifford, &format!("after shrink, {op}"));
    }
}

#[test]
fn clifford_unitary_round_trip_preserves_the_tableau() {
    for (seed, &n) in SIZES.iter().enumerate() {
        let mut rng = SmallRng::seed_from_u64(0xA11CE + seed as u64);
        let clifford = CliffordUnitary::random(n, &mut rng);
        let frame = Frame::from_clifford_unitary(&clifford);
        let rebuilt = frame.to_clifford_unitary();
        assert_same_tableau(&frame, &rebuilt, &format!("n={n} round trip"));
        assert_eq!(
            frame,
            Frame::from_clifford_unitary(&rebuilt),
            "n={n}: frame round trip"
        );
    }
}

/// A long mixed-gate soak at the widest register we compile. Slow in debug
/// builds (the comparison is `O(n²)` per step), so it is opt-in.
#[test]
#[ignore = "slow soak: run with --ignored"]
fn long_gate_sequence_soak() {
    let n = 300;
    let mut rng = SmallRng::seed_from_u64(0xDEADBEEF);
    let mut frame = Frame::identity(n);
    let mut clifford = CliffordUnitary::identity(n);
    for step in 0..500 {
        let op = apply_random_op(&mut rng, &mut frame, &mut clifford, n);
        assert_same_tableau(&frame, &clifford, &format!("step={step} after {op}"));
    }
}
