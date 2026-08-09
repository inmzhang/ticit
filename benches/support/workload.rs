//! Synthetic simulator workloads shared by the Criterion bench
//! (`benches/simulator.rs`) and the `perf` example (`examples/sim_profile.rs`).
//!
//! Both include this file with `#[path]` rather than importing it, because a
//! bench and an example are separate crates and `bloc_utils` must not grow a
//! public test-support surface just to feed its own benchmarks. Living in
//! `benches/support/` (a subdirectory) also keeps cargo's bench auto-discovery,
//! which only scans `benches/*.rs`, from treating it as a target of its own.
//!
//! Sharing matters for a specific reason: the profile is only a valid guide to
//! the bench if the two run the *same* instruction stream. Every generator here
//! is driven by an explicit seed and a local PRNG, so a given seed produces a
//! byte-identical gate sequence in both binaries and across dependency bumps.

#![allow(dead_code)] // Each consumer uses a subset of the generators.

use ticit::TableauSimulator;

/// Shared across the bench and the profile so both drive identical streams.
pub const SEED: u64 = 0x5EED_0000_0000_0001;

// ==============================================================================
// Deterministic randomness
// ==============================================================================

/// SplitMix64. Deliberately *not* `rand::SmallRng`: the workloads must stay
/// bit-identical across `rand` upgrades, or a dependency bump silently
/// invalidates every saved Criterion baseline.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish index in `0..bound`. The modulo bias is irrelevant for
    /// picking qubits and is worth the reproducibility of a two-line generator.
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// An index in `0..bound` other than `avoid`; the two-qubit gates reject a
    /// repeated operand with an error rather than acting as a no-op.
    pub fn below_except(&mut self, bound: usize, avoid: usize) -> usize {
        debug_assert!(bound >= 2, "no distinct partner exists on one qubit");
        let q = self.below(bound - 1);
        if q >= avoid { q + 1 } else { q }
    }
}

// ==============================================================================
// Clifford-frame stream
// ==============================================================================

/// Apply `count` pseudorandom Clifford gates over `n` qubits.
///
/// These touch only the frame `R`, never the amplitude map, so this isolates
/// the tableau update from everything the sparse-vector side costs.
/// The single/two-qubit mix is roughly 1:1, which is what the physical
/// verification circuits look like.
pub fn clifford_stream(sim: &mut TableauSimulator, rng: &mut Rng, n: usize, count: usize) {
    for _ in 0..count {
        let q = rng.below(n);
        match rng.below(6) {
            0 => sim.h(q),
            1 => sim.s(q),
            2 => sim.sqrt_x(q),
            3 => {
                let target = rng.below_except(n, q);
                sim.cx(q, target)
                    .expect("operands are distinct by construction");
            }
            4 => {
                let target = rng.below_except(n, q);
                sim.cz(q, target)
                    .expect("operands are distinct by construction");
            }
            _ => {
                let target = rng.below_except(n, q);
                sim.swap(q, target);
            }
        }
    }
}

// ==============================================================================
// Rank ladders
// ==============================================================================

/// A simulator on `n` qubits whose amplitude map holds exactly `2^magic_qubits`
/// terms, built by `h; t` on that many distinct low-index qubits.
///
/// Each pair doubles the rank exactly: after `h`, `Z_q` pulls back to `X_q`, so
/// the `T` shifts every label into a fresh coset, and with all-nonzero
/// `cos(π/8)`/`sin(π/8)` weights on distinct qubits nothing cancels into the
/// pruning threshold.
///
/// Qubits `magic_qubits..n` are left in `|0⟩`, which is what the "diagonal"
/// (frame-`a = 0`) benchmark variants measure against.
pub fn magic_state(n: usize, magic_qubits: u32, seed: u64) -> TableauSimulator {
    let mut sim = TableauSimulator::with_seed(n, seed);
    for q in 0..magic_qubits as usize {
        sim.h(q);
        sim.t(q)
            .expect("h;t on distinct fresh qubits stays under the rank cap");
    }
    debug_assert_eq!(sim.rank(), 1usize << magic_qubits);
    sim
}

/// `log2` of the rank a `magic_state` should be built at to reach `rank`.
pub fn magic_qubits_for(rank: usize) -> u32 {
    debug_assert!(rank.is_power_of_two(), "rank ladder is built by doubling");
    rank.trailing_zeros()
}

// ==============================================================================
// Mixed physical-verification workload
// ==============================================================================

/// Shape of [`mixed_verify`]. The per-round counts mirror the rhythm of a
/// physical-circuit verification run: a long Clifford stretch, a burst of magic
/// injection, a batch of detector-style readouts, then resets that return the
/// injected qubits to `|0⟩`.
pub struct MixedShape {
    pub n: usize,
    pub rounds: usize,
    pub cliffords_per_round: usize,
    /// `h; t` pairs per round. Each doubles the rank, so the round's resets
    /// must retire the same number or the map grows without bound.
    pub magic_per_round: usize,
    pub measurements_per_round: usize,
}

impl MixedShape {
    /// The bench-sized shape: ~80 ms per iteration at both `n = 128` and
    /// `n = 256`, comfortably inside Criterion's per-iteration budget.
    ///
    /// `magic_per_round = 10` is the tuned knob. It sets the post-injection
    /// ceiling at `2^10 = 1024` terms and a mean readout rank near 400, which
    /// is high enough that the amplitude-map loops — not the Clifford frame —
    /// dominate the measurement. Raising it to 11 doubles the cost past the
    /// budget; dropping to 8 leaves the readouts running at ~145 terms, where
    /// hash-map overhead masks the arithmetic.
    pub fn bench(n: usize) -> Self {
        MixedShape {
            n,
            rounds: 50,
            cliffords_per_round: 200,
            magic_per_round: 10,
            measurements_per_round: 20,
        }
    }
}

/// Outcome of a [`mixed_verify`] run — returned so callers can `black_box` it
/// and so the profiling example can report where the rank actually sat.
pub struct MixedStats {
    /// Live amplitude terms at the end (after the last round's resets, so this
    /// is the floor of the cycle, not a measure of the work done).
    pub final_rank: usize,
    /// Largest rank seen just after a round's magic injection — the ceiling of
    /// the cycle, and the number that reveals a slowly diverging workload.
    pub peak_rank: usize,
    /// Mean rank at the moment each readout was issued. This is the honest
    /// summary of how much amplitude-map work the run performed: the round's
    /// rank swings between 1 and `peak_rank`, and the readouts are what pay
    /// for it.
    pub mean_readout_rank: f64,
    /// Count of `−1` measurement outcomes, purely to keep the loop honest.
    pub minus_outcomes: u64,
}

/// A synthetic stand-in for `bloc_compile::execute`'s physical verification
/// loop: Clifford evolution, magic injection, readout, reset — repeated.
///
/// # Rank budget
///
/// Keeping this loop stationary took some care, because the obvious knobs do
/// not behave the way the phrase "measurements collapse the state" suggests.
/// Measured behaviour on this simulator:
///
/// * Each `h; t` injection multiplies the rank by exactly 2.
/// * A `reset_z` on a *freshly* injected qubit divides it by 2 again: `Z_q`
///   still pulls back close to `X_q`, so the projection undoes that qubit's
///   own doubling.
/// * A `reset_z` on a qubit that has since been through a few hundred Clifford
///   gates does **not**. The frame has spread that qubit's magic across the
///   register and the projection no longer targets it, so a FIFO "retire the
///   oldest" pipeline diverges into `RankOverflow` within a few rounds.
/// * The random single-qubit `Z` readouts are close to rank-*neutral*. They
///   never grow the compressed rank, but they do not reliably shrink it
///   either, so they cannot be relied on to pay for the injections.
///
/// Hence the shape below: every round injects `magic_per_round` qubits and
/// retires **those same qubits** before the round ends. The rank therefore
/// cycles between 1 and `2^magic_per_round` instead of drifting, and a
/// benchmark of it measures the simulator rather than its own divergence.
pub fn mixed_verify(shape: &MixedShape, seed: u64) -> MixedStats {
    let mut sim = TableauSimulator::with_seed(shape.n, seed);
    let mut rng = Rng::new(seed ^ 0x5DEE_CE66_D1D3_7B00);
    let mut peak_rank = sim.rank();
    let mut minus_outcomes = 0u64;
    let mut readout_rank_total = 0u64;
    let mut readout_count = 0u64;

    // Injection targets rotate through the register so the workload is not
    // confined to one corner of the frame.
    let mut magic_cursor = 0usize;

    for _ in 0..shape.rounds {
        clifford_stream(&mut sim, &mut rng, shape.n, shape.cliffords_per_round);

        // Magic injection: `h; t` on a block of distinct qubits.
        let magic: Vec<usize> = (0..shape.magic_per_round)
            .map(|i| (magic_cursor + i) % shape.n)
            .collect();
        magic_cursor = (magic_cursor + shape.magic_per_round) % shape.n;
        for &q in &magic {
            sim.h(q);
            sim.t(q).expect("injection stays under the rank cap");
        }
        peak_rank = peak_rank.max(sim.rank());

        // Readout: single-qubit `Z` on a scrambled frame, so these land on the
        // off-diagonal (`a != 0`) measurement path most of the time.
        for _ in 0..shape.measurements_per_round {
            readout_rank_total += sim.rank() as u64;
            readout_count += 1;
            let q = rng.below(shape.n);
            let outcome = sim
                .measure(q)
                .expect("projection never grows the compressed rank");
            minus_outcomes += u64::from(outcome.outcome);
        }

        // Retire the injected magic, restoring the round's rank budget.
        for &q in &magic {
            sim.reset_z(q)
                .expect("reset is a measurement plus a frame X");
        }
    }

    MixedStats {
        final_rank: sim.rank(),
        peak_rank,
        mean_readout_rank: readout_rank_total as f64 / readout_count.max(1) as f64,
        minus_outcomes,
    }
}
