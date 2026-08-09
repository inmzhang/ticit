//! Criterion benchmarks for [`ticit::TableauSimulator`].
//!
//! The simulator has two cost centres that scale independently, and the groups
//! below are arranged to separate them:
//!
//! * The **Clifford frame** `R`. Every Clifford gate is a tableau update of
//!   width `n` and never touches the amplitude map, so `clifford-frame` scales
//!   with the register and is flat in the rank.
//! * The **amplitude map** `|χ⟩`. `T`, measurement and the expectation reads
//!   sweep all live terms, so those groups scale with the rank and are
//!   near-flat in `n`.
//!
//! Within the amplitude-map groups the decisive question is whether the
//! observable's frame decomposition `Q = ζ·X^a Z^b` has `a = 0`. That single
//! bit picks between a cheap in-place pass and an expensive one that rehashes
//! every term into a shifted coset, so each such group benchmarks both, and
//! asserts during setup that it is actually on the branch it claims.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use ticit::{Pauli, PauliString, TableauSimulator};

#[path = "support/workload.rs"]
mod workload;

use workload::{
    MixedShape, Rng, SEED, clifford_stream, magic_qubits_for, magic_state, mixed_verify,
};

/// Register width for the rank-scaling groups. Held fixed so those benchmarks
/// vary in exactly one dimension; `n`-scaling is `clifford-frame`'s job.
const N: usize = 128;

// ==============================================================================
// Clifford frame — tableau throughput, independent of rank
// ==============================================================================

/// Gates per iteration. Reported as `Throughput::Elements` so the headline
/// number is gates/second, which stays comparable across register widths.
const CLIFFORD_GATES: usize = 10_000;

fn bench_clifford_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("clifford-frame");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(CLIFFORD_GATES as u64));

    for n in [64usize, 128, 256, 512] {
        group.bench_function(format!("n{n}"), |b| {
            b.iter_batched(
                || (TableauSimulator::with_seed(n, SEED), Rng::new(SEED)),
                |(mut sim, mut rng)| {
                    clifford_stream(&mut sim, &mut rng, n, CLIFFORD_GATES);
                    black_box(sim)
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

// ==============================================================================
// T gate — amplitude-map doubling
// ==============================================================================

/// One `T` on a qubit already in superposition, at three starting ranks.
///
/// Putting the target through `h` first is what makes this the worst case:
/// `Z_q` then pulls back to `X_q`, so `a != 0` and every live term is xor-ed
/// into a fresh coset key. The rank doubles, and the setup below asserts it
/// does rather than trusting the reasoning.
fn bench_t_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("t-gate");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(3));

    for rank in [256usize, 1024, 4096] {
        group.bench_function(format!("n{N}-rank{rank}"), |b| {
            let magic = magic_qubits_for(rank);
            // The first qubit above the magic block: fresh, so `h` alone puts
            // it in superposition without changing the rank.
            let target = magic as usize;
            let base = {
                let mut sim = magic_state(N, magic, SEED);
                sim.h(target);
                sim
            };
            assert_eq!(base.rank(), rank, "ladder must land on the requested rank");

            let mut probe = base.clone();
            probe
                .t(target)
                .expect("T on a superposed qubit stays under the cap");
            assert_eq!(
                probe.rank(),
                2 * rank,
                "this must be the rank-doubling path"
            );

            b.iter_batched(
                || base.clone(),
                |mut sim| {
                    sim.t(target).expect("T stays under the rank cap");
                    black_box(sim)
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

/// One `T` on a frame-diagonal axis — the cheap path.
///
/// The target here is a fresh `|0⟩` with no `h`, so `Z_q` pulls back to `Z_q`,
/// `a = 0`, and the coset shift `c ⊕ a` is the identity. Every term still costs
/// two hash-map probes, but they land on the key they came from, so the rank
/// holds instead of doubling. The gap against `t-gate` at the same rank is the
/// price of growing the map.
fn bench_t_gate_diagonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("t-gate-diagonal");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(3));

    let rank = 1024usize;
    group.bench_function(format!("n{N}-rank{rank}"), |b| {
        let magic = magic_qubits_for(rank);
        let target = magic as usize;
        let base = magic_state(N, magic, SEED);
        assert_eq!(base.rank(), rank);

        let mut probe = base.clone();
        probe.t(target).expect("diagonal T stays under the cap");
        assert_eq!(probe.rank(), rank, "diagonal T must not grow the map");

        b.iter_batched(
            || base.clone(),
            |mut sim| {
                sim.t(target).expect("T stays under the rank cap");
                black_box(sim)
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

// ==============================================================================
// Measurement — the two projection branches
// ==============================================================================

/// `Z` on an untouched `|0⟩` qubit: `a = 0`, so `measure_impl` takes the
/// frame-deterministic branch — one expectation sweep plus an eigenvalue-class
/// filter, with `R` left alone.
///
/// The `deterministic` flag on the result is the observable proof of the
/// branch: an `a != 0` observable on this state would sample at probability
/// `1/2` instead.
fn bench_measure_deterministic(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure-deterministic");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(3));

    let rank = 1024usize;
    group.bench_function(format!("n{N}-rank{rank}"), |b| {
        let magic = magic_qubits_for(rank);
        let target = magic as usize;
        let base = magic_state(N, magic, SEED);

        let outcome = base
            .clone()
            .measure(target)
            .expect("Z on a product-state qubit is measurable");
        assert!(outcome.deterministic, "setup must hit the a = 0 branch");
        assert!(
            (outcome.probability - 1.0).abs() < 1e-9,
            "|0> yields +1 surely"
        );

        b.iter_batched(
            || base.clone(),
            |mut sim| {
                let result = sim
                    .measure(black_box(target))
                    .expect("measurement succeeds");
                black_box((sim, result))
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

/// `Z` on a qubit freshly put through `h`: `Z_q` pulls back to `X_q`, so
/// `a != 0` and `measure_impl` takes the random branch — a projection pass and
/// a frame-compression pass over the map, each transiently doubling it, then a
/// `left_mul_pauli_exp` on `R`. Roughly four times the term traffic of the
/// deterministic branch.
fn bench_measure_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure-random");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(3));

    let rank = 1024usize;
    group.bench_function(format!("n{N}-rank{rank}"), |b| {
        let magic = magic_qubits_for(rank);
        let target = magic as usize;
        let base = {
            let mut sim = magic_state(N, magic, SEED);
            sim.h(target);
            sim
        };

        let outcome = base
            .clone()
            .measure(target)
            .expect("Z on a |+> qubit is measurable");
        assert!(!outcome.deterministic, "setup must hit the a != 0 branch");
        assert!(
            (outcome.probability - 0.5).abs() < 1e-9,
            "Z on |+> is an even coin"
        );

        b.iter_batched(
            || base.clone(),
            |mut sim| {
                let result = sim
                    .measure(black_box(target))
                    .expect("measurement succeeds");
                black_box((sim, result))
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

/// The same random branch at a rank where the *frame* is the cost.
///
/// `measure-random` above sits at rank 1024, where sweeping a thousand
/// amplitude terms buries the tableau work. Physical verification circuits
/// spend nearly all of their measurements at rank 1–4, so this variant pins the
/// rank at 4 and sweeps `n` instead. What is left to measure is the frame side:
/// the observable's preimage, the compression Pauli `G`, and the `2n`-row
/// right-multiplication that applies it.
fn bench_measure_random_lowrank(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure-random-lowrank");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(3));

    for n in [128usize, 256] {
        group.bench_function(format!("n{n}"), |b| {
            let magic = 2u32;
            let target = magic as usize;
            let base = {
                let mut sim = magic_state(n, magic, SEED);
                sim.h(target);
                sim
            };
            assert_eq!(base.rank(), 4, "the map must be small enough to ignore");

            let outcome = base
                .clone()
                .measure(target)
                .expect("Z on a |+> qubit is measurable");
            assert!(!outcome.deterministic, "setup must hit the a != 0 branch");

            b.iter_batched(
                || base.clone(),
                |mut sim| {
                    let result = sim
                        .measure(black_box(target))
                        .expect("measurement succeeds");
                    black_box((sim, result))
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

// ==============================================================================
// Expectation — non-collapsing readout
// ==============================================================================

/// `peek_observable_expectation` takes `&self`, so unlike measurement it needs
/// no per-iteration clone and the timings are free of setup noise.
///
/// The two variants split on the same `a` bit as measurement. The diagonal case
/// is a single pass accumulating `|x|²`; the off-diagonal case must look up the
/// coset partner `c ⊕ a` for every term, turning a linear scan into a scan plus
/// a hash probe each. The asserted values pin the branches: only the
/// off-diagonal pairing path can return `0` for `Z` on `|+⟩`.
fn bench_expectation(c: &mut Criterion) {
    let mut group = c.benchmark_group("expectation");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(3));

    let rank = 4096usize;
    let magic = magic_qubits_for(rank);
    // `superposed` carries an `h` (off-diagonal readout); `idle` stays in |0>
    // (diagonal readout). Both live in the same simulator.
    let superposed = magic as usize;
    let idle = superposed + 1;

    let sim = {
        let mut sim = magic_state(N, magic, SEED);
        sim.h(superposed);
        sim
    };
    assert_eq!(sim.rank(), rank);

    let diagonal = PauliString::single(N, idle, Pauli::Z);
    let off_diagonal = PauliString::single(N, superposed, Pauli::Z);
    assert!(
        (sim.peek_observable_expectation(&diagonal)
            .expect("in range")
            - 1.0)
            .abs()
            < 1e-9,
        "idle qubit is |0>"
    );
    assert!(
        sim.peek_observable_expectation(&off_diagonal)
            .expect("in range")
            .abs()
            < 1e-9,
        "superposed qubit is |+>, reachable only via the coset-pairing path"
    );

    group.bench_function(format!("n{N}-rank{rank}/diagonal"), |b| {
        b.iter(|| black_box(sim.peek_observable_expectation(black_box(&diagonal))));
    });
    group.bench_function(format!("n{N}-rank{rank}/off-diagonal"), |b| {
        b.iter(|| black_box(sim.peek_observable_expectation(black_box(&off_diagonal))));
    });

    group.finish();
}

// ==============================================================================
// CCZ — seven chained pi/8 rotations
// ==============================================================================

/// `ccz` on three qubits in superposition. It clones the simulator up front for
/// transactional rollback and then runs seven `t_pauli` calls on one-, two- and
/// three-qubit `Z` axes, so this measures the compound cost, including that
/// defensive clone.
fn bench_ccz(c: &mut Criterion) {
    let mut group = c.benchmark_group("ccz");
    let n = 64usize;

    group.bench_function(format!("n{n}"), |b| {
        let base = {
            let mut sim = TableauSimulator::with_seed(n, SEED);
            for q in 0..3 {
                sim.h(q);
            }
            sim
        };

        b.iter_batched(
            || base.clone(),
            |mut sim| {
                sim.ccz(0, 1, 2)
                    .expect("CCZ on three qubits stays under the cap");
                black_box(sim)
            },
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

// ==============================================================================
// Mixed workload — end-to-end verification shape
// ==============================================================================

/// The composite the other groups decompose: Clifford evolution, magic
/// injection, readout and reset, repeated. This is the number to watch for
/// whole-simulator regressions; the microbenchmarks say which part moved.
fn bench_mixed_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed-verify");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for n in [128usize, 256] {
        group.bench_function(format!("n{n}"), |b| {
            let shape = MixedShape::bench(n);
            b.iter(|| black_box(mixed_verify(black_box(&shape), SEED)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_clifford_frame,
    bench_t_gate,
    bench_t_gate_diagonal,
    bench_measure_deterministic,
    bench_measure_random,
    bench_measure_random_lowrank,
    bench_expectation,
    bench_ccz,
    bench_mixed_verify,
);
criterion_main!(benches);
