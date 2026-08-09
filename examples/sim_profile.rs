//! Long-running `TableauSimulator` workload shaped for `perf record`.
//!
//! Criterion's harness interleaves warm-up, sampling and analysis, so a profile
//! taken across `cargo bench` is diluted by the harness itself. This example is
//! a single flat run of the same `mixed_verify` workload the bench times, sized
//! to a handful of seconds so `perf` collects enough samples for the simulator's
//! own call tree to dominate.
//!
//! ```text
//! cargo build --release --example sim_profile
//! perf record -F 999 -g -- target/release/examples/sim_profile
//! perf report --stdio --percent-limit 1
//! ```
//!
//! Build with `CARGO_PROFILE_RELEASE_DEBUG=true` (or `--profile profiling`) to
//! get symbols — the workspace `release` profile strips them.

use std::time::Instant;

#[path = "../benches/support/workload.rs"]
mod workload;

use workload::{MixedShape, SEED, mixed_verify};

/// Register width and round count chosen for a ~5–10 s run. The bench shape
/// costs ~1.6 ms per round at `n = 128`, so this lands near 6 s while keeping
/// the per-round rank budget identical to the benchmark's.
const NUM_QUBITS: usize = 128;
const ROUNDS: usize = 4_000;

fn main() {
    let shape = MixedShape {
        rounds: ROUNDS,
        ..MixedShape::bench(NUM_QUBITS)
    };

    let started = Instant::now();
    let stats = mixed_verify(&shape, SEED);
    let elapsed = started.elapsed();

    let rounds = shape.rounds as f64;
    println!("qubits             {NUM_QUBITS}");
    println!("rounds             {ROUNDS}");
    println!("elapsed            {:.3} s", elapsed.as_secs_f64());
    println!(
        "per round          {:.3} ms",
        elapsed.as_secs_f64() * 1e3 / rounds
    );
    println!("peak rank          {}", stats.peak_rank);
    println!("mean readout rank  {:.1}", stats.mean_readout_rank);
    println!("final rank         {}", stats.final_rank);
    println!("minus outcomes     {}", stats.minus_outcomes);
}
