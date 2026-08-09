//! Criterion benchmarks for the batched instruction path.
//!
//! The point of comparison is not "batching versus not batching" — a batch is
//! a plain loop — but what the instruction set lets that loop skip. The
//! `vs-loop` group pits an [`Instruction`] stream against the same two-qubit
//! gates expressed the way a Pauli-addressed front end had to express them:
//! `controlled_pauli` on two freshly built [`PauliString`]s. Both sides drive
//! the same engine over the same gate sequence, so the ratio is exactly the
//! translation and allocation overhead the instruction set removes.
//!
//! There is no matching single-qubit arm: its comparison target was
//! `apply_clifford` on a tableau built per application, and that entry point is
//! gone from the public API (nothing in production had a `CliffordUnitary` to
//! hand). What is left, `1q-clifford/batch`, is a throughput baseline.
//!
//! `verify-replay` is the whole-workload number: an instruction stream whose
//! mix mirrors `bloc_compile`'s `d = 3` physical verification (58% two-qubit
//! gates, 20% measurements, 20% resets, 2% `T`), replayed across shots the way
//! that verifier replays a prepared node stream.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ticit::{Gate1Q, Instruction};
use ticit::{Pauli, PauliBasis, PauliString, TableauSimulator};

#[path = "support/workload.rs"]
mod workload;

use workload::{Rng, SEED};

/// Register width, between `bloc_compile`'s `d = 3` (64) and `d = 7` (207)
/// physical verification runs.
const N: usize = 128;

/// Stream length for the microbenchmarks. Long enough that the per-iteration
/// setup disappears, short enough to stay inside Criterion's budget.
const OPS: usize = 5000;

// ==============================================================================
// Shared stream generators
// ==============================================================================

/// One two-qubit gate as its `<A>C<B>` axes and operands.
type TwoQubitGate = (PauliBasis, PauliBasis, usize, usize);

/// A layered `CX`/`CZ`/`CY` stream over `N` qubits, in equal parts. `CX` and
/// `CZ` hit the engine's primitives directly; `CY` pays the basis conjugation,
/// so mixing all three keeps the fast path from flattering itself.
fn two_qubit_stream(count: usize) -> Vec<TwoQubitGate> {
    let mut rng = Rng::new(SEED);
    (0..count)
        .map(|i| {
            let control_qubit = rng.below(N);
            let target_qubit = rng.below_except(N, control_qubit);
            let target = match i % 3 {
                0 => PauliBasis::X, // CX
                1 => PauliBasis::Z, // CZ
                _ => PauliBasis::Y, // CY
            };
            (PauliBasis::Z, target, control_qubit, target_qubit)
        })
        .collect()
}

/// A single-qubit basis Pauli, the form a Pauli-addressed front end must build
/// for every operand of every gate.
fn basis_pauli(basis: PauliBasis, qubit: usize) -> PauliString {
    PauliString::single(N, qubit, basis.into())
}

// ==============================================================================
// Instruction stream versus the Pauli-addressed loop
// ==============================================================================

fn bench_two_qubit(c: &mut Criterion) {
    let gates = two_qubit_stream(OPS);
    let instructions: Vec<Instruction> = gates
        .iter()
        .map(
            |&(control, target, control_qubit, target_qubit)| Instruction::Gate2 {
                control,
                target,
                control_qubit,
                target_qubit,
            },
        )
        .collect();

    let mut group = c.benchmark_group("batch-vs-loop");
    group.throughput(Throughput::Elements(OPS as u64));

    // Clifford gates only touch the frame, and a frame row update costs the
    // same whatever the frame holds, so one simulator can carry every sample.
    group.bench_function("2q-layer/batch", |b| {
        let mut sim = TableauSimulator::with_seed(N, SEED);
        b.iter(|| {
            black_box(
                sim.apply_batch(black_box(&instructions))
                    .expect("distinct operands, no measurements"),
            );
        });
    });

    group.bench_function("2q-layer/controlled-pauli", |b| {
        let mut sim = TableauSimulator::with_seed(N, SEED);
        b.iter(|| {
            for &(control, target, control_qubit, target_qubit) in black_box(&gates) {
                sim.controlled_pauli(
                    &basis_pauli(control, control_qubit),
                    &basis_pauli(target, target_qubit),
                )
                .expect("single-qubit axes on distinct qubits commute");
            }
        });
    });

    group.finish();
}

fn bench_single_qubit(c: &mut Criterion) {
    // The period-three gates are the expensive end of the old path (two
    // preimages and an inversion for what is two frame row updates) and the
    // ones surface-code circuits actually emit alongside H and S.
    let cycle = [Gate1Q::H, Gate1Q::S, Gate1Q::Cxyz, Gate1Q::Hyz];
    let mut rng = Rng::new(SEED);
    let gates: Vec<(Gate1Q, usize)> = (0..OPS)
        .map(|i| (cycle[i % cycle.len()], rng.below(N)))
        .collect();
    let instructions: Vec<Instruction> = gates
        .iter()
        .map(|&(gate, qubit)| Instruction::Gate1 { gate, qubit })
        .collect();

    let mut group = c.benchmark_group("batch-vs-loop");
    group.throughput(Throughput::Elements(OPS as u64));

    group.bench_function("1q-clifford/batch", |b| {
        let mut sim = TableauSimulator::with_seed(N, SEED);
        b.iter(|| {
            black_box(
                sim.apply_batch(black_box(&instructions))
                    .expect("single-qubit Cliffords cannot fail"),
            );
        });
    });

    // The procedural counterpart: the same gates, dispatched one call at a
    // time. What separates the two is only the instruction decode, so the gap
    // is the ceiling on what batching can ever save on Clifford-only streams.
    group.bench_function("1q-clifford/procedural", |b| {
        let mut sim = TableauSimulator::with_seed(N, SEED);
        b.iter(|| {
            for &(gate, qubit) in black_box(&gates) {
                sim.gate1(gate, qubit);
            }
        });
    });

    group.finish();
}

// ==============================================================================
// Prepared-stream replay
// ==============================================================================

/// Instructions per round of [`verify_stream`], split to the measured `d = 3`
/// mix: 58 two-qubit gates, 20 measurements, 20 resets, 2 `T`s.
const ROUND: usize = 100;

/// A stream shaped like one prepared `bloc_compile` node stream.
///
/// Each round injects its two `T`s immediately before the resets that retire
/// them, on the same qubits. That is what keeps the stabilizer rank cycling
/// instead of drifting: a reset undoes a *freshly* injected `T`'s doubling,
/// but not one the intervening Clifford layer has already spread across the
/// frame (see `benches/support/workload.rs` for the measurements behind that).
fn verify_stream(rounds: usize) -> Vec<Instruction> {
    let mut rng = Rng::new(SEED ^ 0x00B1_0C00);
    let mut stream = Vec::with_capacity(rounds * ROUND);
    let mut magic_cursor = 0usize;

    for _ in 0..rounds {
        for i in 0..58 {
            let control_qubit = rng.below(N);
            let target_qubit = rng.below_except(N, control_qubit);
            let target = match i % 3 {
                0 => PauliBasis::X,
                1 => PauliBasis::Z,
                _ => PauliBasis::Y,
            };
            stream.push(Instruction::Gate2 {
                control: PauliBasis::Z,
                target,
                control_qubit,
                target_qubit,
            });
        }
        for _ in 0..20 {
            stream.push(Instruction::Measure(PauliString::single(
                N,
                rng.below(N),
                Pauli::Z,
            )));
        }
        for _ in 0..18 {
            stream.push(Instruction::Reset {
                basis: PauliBasis::Z,
                qubit: rng.below(N),
            });
        }
        for _ in 0..2 {
            let qubit = magic_cursor % N;
            magic_cursor += 1;
            stream.push(Instruction::T {
                basis: PauliBasis::Z,
                qubit,
                adjoint: false,
            });
            stream.push(Instruction::Reset {
                basis: PauliBasis::Z,
                qubit,
            });
        }
    }
    stream
}

fn bench_verify_replay(c: &mut Criterion) {
    const ROUNDS: usize = 20;
    const SHOTS: u64 = 4;
    let stream = verify_stream(ROUNDS);

    let mut group = c.benchmark_group("batch-verify-replay");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(SHOTS * (ROUNDS * ROUND) as u64));

    group.bench_function(format!("n{N}"), |b| {
        b.iter(|| {
            // A shot is a fresh state replaying the same prepared stream —
            // the shape `bloc_compile::execute` runs, and the reason the
            // translation is worth hoisting out of the loop.
            for shot in 0..SHOTS {
                let mut sim = TableauSimulator::with_seed(N, SEED ^ shot);
                black_box(
                    sim.apply_batch(black_box(&stream))
                        .expect("the stream retires its own magic"),
                );
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_two_qubit,
    bench_single_qubit,
    bench_verify_replay,
);
criterion_main!(benches);
