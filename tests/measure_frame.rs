//! Cross-validation of the frame updates the simulator performs on the paths
//! that have no tableau equivalent left to differ from.
//!
//! The random-measurement branch rewrites the frame as
//! `R ← R·Z_p^s·exp(iπ/4·G)` with `G = −i·Q·Z_p` assembled from the frame
//! decomposition alone (see `TableauSimulator::measure_random`). Nothing about that
//! derivation is checkable against a tableau library op-for-op, so it is pinned here
//! against dense linear algebra instead: the post-measurement state must be the
//! projector applied to the pre-measurement state, exactly, up to global phase.
//!
//! That comparison is sensitive to every part of the derivation. Getting `G`
//! wrong by a Pauli — which is what applying `Z_p` on the wrong side of the
//! rotation does, since `Z_p` anticommutes with `G` — leaves the amplitude map
//! correct and the frame off by `iG`, so the state comes out as `G|ψ⟩`: an
//! invisible error in the outcome statistics and a fatal one in the state.

use std::f64::consts::FRAC_1_SQRT_2;

use num_complex::Complex64;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use ticit::{Pauli, PauliString, SimError, TableauSimulator};

// ==============================================================================
// Dense reference
// ==============================================================================

/// Apply the Hermitian Pauli `p` to a state vector, one site at a time.
///
/// Deliberately *not* the `i^{#Y}·X^a·Z^b` normal form the frame decomposes
/// into: each site's 2×2 matrix is applied directly, `Y = [[0, −i], [i, 0]]`
/// included. That makes this an independent check of the normal form's phase —
/// the engine agrees with it only if every `Y` site's factor of `i` is
/// accounted for.
fn apply_pauli(v: &[Complex64], p: &PauliString) -> Vec<Complex64> {
    const I: Complex64 = Complex64::new(0.0, 1.0);
    let mut out = v.to_vec();
    for site in 0..p.nqubits {
        let pauli = p.get(site);
        if pauli == Pauli::I {
            continue;
        }
        let bit = 1usize << site;
        let mut next = vec![Complex64::new(0.0, 0.0); out.len()];
        for (y, &value) in out.iter().enumerate() {
            let set = y & bit != 0;
            let (target, factor) = match pauli {
                Pauli::X => (y ^ bit, Complex64::new(1.0, 0.0)),
                Pauli::Z => (y, Complex64::new(if set { -1.0 } else { 1.0 }, 0.0)),
                Pauli::Y => (y ^ bit, if set { -I } else { I }),
                Pauli::I => unreachable!("skipped above"),
            };
            next[target] += factor * value;
        }
        out = next;
    }
    out
}

/// `Π_s^P |v⟩ = ½(I + (−1)^s P)|v⟩`, normalized. `None` when the outcome has no
/// weight in `v` (an eigenstate measured against its other eigenvalue).
fn project(v: &[Complex64], p: &PauliString, outcome: bool) -> Option<Vec<Complex64>> {
    let sign = if outcome { -1.0 } else { 1.0 };
    let applied = apply_pauli(v, p);
    let mut out: Vec<Complex64> = v
        .iter()
        .zip(applied)
        .map(|(&x, y)| 0.5 * (x + sign * y))
        .collect();
    let norm_sqr: f64 = out.iter().map(num_complex::Complex::norm_sqr).sum();
    if norm_sqr < 1e-12 {
        return None;
    }
    let scale = norm_sqr.sqrt().recip();
    for value in &mut out {
        *value *= scale;
    }
    Some(out)
}

/// `|⟨left|right⟩|`, which is `1` exactly when the two normalized states agree
/// up to a global phase — the only freedom `state_vector` leaves open.
fn overlap(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| a.conj() * b)
        .sum::<Complex64>()
        .norm()
}

fn assert_same_state(got: &[Complex64], want: &[Complex64], context: &str) {
    let fidelity = overlap(got, want);
    assert!(
        (fidelity - 1.0).abs() < 1e-9,
        "{context}: states differ (|<want|got>| = {fidelity})"
    );
}

// ==============================================================================
// Random circuits and observables
// ==============================================================================

/// A seeded Clifford+T state with a handful of magic terms.
fn random_state(n: usize, seed: u64, cliffords: usize, magic: usize) -> TableauSimulator {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sim = TableauSimulator::with_seed(n, seed);
    for _ in 0..magic {
        let q = rng.random_range(0..n);
        sim.h(q);
        sim.t(q).expect("a handful of T gates stays under the cap");
    }
    // A one-qubit register has no room for the two-qubit half of the table.
    let choices = if n >= 2 { 5 } else { 3 };
    for _ in 0..cliffords {
        let q = rng.random_range(0..n);
        match rng.random_range(0..choices) {
            0 => sim.h(q),
            1 => sim.s(q),
            2 => sim.sqrt_x(q),
            3 => {
                let other = (q + 1 + rng.random_range(0..n - 1)) % n;
                sim.cx(q, other).expect("operands are distinct");
            }
            _ => {
                let other = (q + 1 + rng.random_range(0..n - 1)) % n;
                sim.cz(q, other).expect("operands are distinct");
            }
        }
    }
    sim
}

/// A random Pauli string of weight `1..=max_weight`.
///
/// Unsigned, because a `PauliString` has no sign to carry: negating an
/// observable only swaps which outcome bit the engine reports, and leaves every
/// frame update identical. `Y` is drawn as often as `X` and `Z`, so the weights
/// reached here span all four residues of the normal-form phase `i^{#Y}`.
fn random_pauli(rng: &mut SmallRng, n: usize, max_weight: usize) -> PauliString {
    let weight = rng.random_range(1..=max_weight.min(n));
    let mut paulis = PauliString::new(n);
    let mut placed = 0;
    while placed < weight {
        let site = rng.random_range(0..n);
        if paulis.get(site) != Pauli::I {
            continue;
        }
        paulis.set(
            site,
            match rng.random_range(0..3) {
                0 => Pauli::X,
                1 => Pauli::Y,
                _ => Pauli::Z,
            },
        );
        placed += 1;
    }
    paulis
}

// ==============================================================================
// Tests
// ==============================================================================

/// The anchor: every measurement, on either outcome, must reproduce the dense
/// projection of the pre-measurement state.
#[test]
fn measurement_matches_the_dense_projection() {
    for n in [1usize, 2, 3, 4] {
        for seed in 0..12u64 {
            let base = random_state(n, 0x51D0 + seed, 12, n.min(2));
            let mut rng = SmallRng::seed_from_u64(0xBEEF + seed);
            let before = base.state_vector();
            for trial in 0..4 {
                let observable = random_pauli(&mut rng, n, n);
                for outcome in [false, true] {
                    let Some(want) = project(&before, &observable, outcome) else {
                        continue; // Unreachable; `postselect_observable` rejects it.
                    };
                    let mut sim = base.clone();
                    sim.postselect_observable(&observable, outcome)
                        .expect("an outcome with weight is post-selectable");
                    assert_same_state(
                        &sim.state_vector(),
                        &want,
                        &format!("n={n} seed={seed} trial={trial} {observable:#} -> {outcome}"),
                    );
                }
            }
        }
    }
}

/// Chained measurements, so the rewritten frame update has to stay correct on a
/// frame it produced itself rather than only on a freshly built one.
#[test]
fn chained_measurements_match_the_dense_projections() {
    for n in [2usize, 3, 4] {
        let mut rng = SmallRng::seed_from_u64(0xC4A1 + n as u64);
        let mut sim = random_state(n, 0xC4A1 + n as u64, 20, n.min(3));
        for step in 0..12 {
            let observable = random_pauli(&mut rng, n, n);
            let before = sim.state_vector();
            let outcome = rng.random_bool(0.5);
            let Some(want) = project(&before, &observable, outcome) else {
                continue;
            };
            sim.postselect_observable(&observable, outcome)
                .expect("an outcome with weight is post-selectable");
            assert_same_state(
                &sim.state_vector(),
                &want,
                &format!("n={n} step={step} {observable:#} -> {outcome}"),
            );
        }
    }
}

/// A measured observable is left an eigenstate of itself — the cheap invariant,
/// checked at a width where the pivot lands in the second label word so the
/// `b ⊕ e_p` mask is exercised past the 64-bit boundary.
#[test]
fn a_measured_observable_becomes_deterministic() {
    let n = 130;
    for outcome in [false, true] {
        let mut sim = TableauSimulator::with_seed(n, 5);
        sim.h(70);
        sim.t(70).expect("one T stays under the cap");
        sim.cx(70, 3).expect("operands are distinct");
        sim.h(3);

        let observable = PauliString::single(n, 70, Pauli::Z);
        sim.postselect_observable(&observable, outcome)
            .expect("Z on a magic qubit has weight on both outcomes");

        let sign = if outcome { -1.0 } else { 1.0 };
        let expectation = sim.peek_z(70).expect("qubit 70 is live");
        assert!(
            (expectation - sign).abs() < 1e-12,
            "outcome={outcome}: <Z> = {expectation}, want {sign}"
        );
        let repeat = sim.measure_observable(&observable).expect("within the cap");
        assert_eq!(repeat.outcome, outcome, "the repeat must agree");
        assert!(repeat.deterministic, "the repeat must be forced");
        assert!((repeat.probability - 1.0).abs() < 1e-12);
    }
}

/// `T`, `T†` and the resets decompose their basis axis through the frame's row
/// fast path instead of building a `PauliString`; the two must agree exactly.
/// `Y` is the one that carries an extra factor of `i` (`Y = i·X·Z`), so the
/// `reset_y` case is what pins that phase.
#[test]
fn basis_axis_paths_match_the_pauli_string_paths() {
    let n = 5;
    let seed = 0xAA55;
    let build = || random_state(n, seed, 15, 2);

    for q in 0..n {
        let mut fast = build();
        let mut slow = build();
        fast.t(q).expect("T stays under the cap");
        slow.t_pauli(&PauliString::single(n, q, Pauli::Z), false)
            .expect("T stays under the cap");
        assert_same_state(
            &fast.state_vector(),
            &slow.state_vector(),
            &format!("t({q})"),
        );

        let mut fast = build();
        let mut slow = build();
        fast.t_dag(q).expect("T† stays under the cap");
        slow.t_pauli(&PauliString::single(n, q, Pauli::Z), true)
            .expect("T† stays under the cap");
        assert_same_state(
            &fast.state_vector(),
            &slow.state_vector(),
            &format!("t_dag({q})"),
        );

        // The resets sample, so both sides must draw from identically seeded
        // RNGs — `build()` gives that, and the draw counts match.
        type Reset = fn(&mut TableauSimulator, usize) -> Result<(), SimError>;
        for (axis, reset) in [
            ("z", TableauSimulator::reset_z as Reset),
            ("x", TableauSimulator::reset_x),
            ("y", TableauSimulator::reset_y),
        ] {
            let mut fast = build();
            let mut slow = build();
            reset(&mut fast, q).expect("reset is a measurement plus a correction");
            let observable = PauliString::single(
                n,
                q,
                match axis {
                    "z" => Pauli::Z,
                    "x" => Pauli::X,
                    _ => Pauli::Y,
                },
            );
            if slow
                .measure_observable(&observable)
                .expect("within the rank cap")
                .outcome
            {
                match axis {
                    "z" => slow.x(q),
                    _ => slow.z(q),
                }
            }
            assert_same_state(
                &fast.state_vector(),
                &slow.state_vector(),
                &format!("reset_{axis}({q})"),
            );
        }
    }
}

/// The measurement branch must preserve norm: the fused projection and frame
/// compression are two unitaries around a projector, and `finalize`'s
/// renormalization would happily hide an error in either.
#[test]
fn the_compression_preserves_the_state_norm() {
    let n = 3;
    let mut rng = SmallRng::seed_from_u64(0x2E57);
    for seed in 0..8u64 {
        let mut sim = random_state(n, 0x2E57 + seed, 10, 2);
        for _ in 0..6 {
            let observable = random_pauli(&mut rng, n, n);
            sim.measure_observable(&observable)
                .expect("within the rank cap");
            let norm_sqr: f64 = sim
                .state_vector()
                .iter()
                .map(num_complex::Complex::norm_sqr)
                .sum();
            assert!(
                (norm_sqr - 1.0).abs() < 1e-9,
                "state norm drifted to {norm_sqr}"
            );
        }
    }
}

/// `|+⟩` measured in `Z`: the smallest case where the amplitude map and the
/// frame both move, with a hand-checkable answer.
#[test]
fn a_single_qubit_random_measurement_lands_on_the_basis_state() {
    for outcome in [false, true] {
        let mut sim = TableauSimulator::with_seed(1, 0);
        sim.h(0);
        let result = sim
            .postselect_z(0, outcome)
            .expect("Z on |+> is an even coin");
        assert!((result.probability - 0.5).abs() < 1e-12);

        let want = if outcome {
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
        } else {
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
        };
        assert_same_state(&sim.state_vector(), &want, &format!("Z|+> -> {outcome}"));
        // And the register is genuinely collapsed, not merely phase-adjusted.
        let x_expectation = sim.peek_x(0).expect("qubit 0 is live");
        assert!(x_expectation.abs() < 1e-12, "<X> = {x_expectation}");
    }
}

/// Sampled outcomes must follow the Born rule the dense state predicts — the
/// probability side of the branch, which the projection tests hold fixed.
#[test]
fn sampled_outcomes_follow_the_born_rule() {
    let n = 2;
    let base = {
        let mut sim = TableauSimulator::with_seed(n, 11);
        sim.h(0);
        sim.t(0).expect("one T stays under the cap");
        sim.h(0);
        sim
    };
    // ⟨Z_0⟩ on `h·t·h|0⟩` is cos(π/4) = 1/√2, so p(+1) = (1 + 1/√2)/2.
    let expectation = base.peek_z(0).expect("qubit 0 is live");
    assert!((expectation - FRAC_1_SQRT_2).abs() < 1e-12, "{expectation}");

    let shots: u32 = 20_000;
    let mut minus = 0u32;
    for seed in 0..shots {
        let mut sim = base.clone();
        sim.reseed_rng(u64::from(seed));
        if sim.measure(0).expect("within the rank cap").outcome {
            minus += 1;
        }
    }
    let observed = f64::from(minus) / f64::from(shots);
    let expected = (1.0 - FRAC_1_SQRT_2) / 2.0;
    // Three sigma at 20k shots is ~0.01.
    assert!(
        (observed - expected).abs() < 0.01,
        "sampled {observed}, expected {expected}"
    );
}
