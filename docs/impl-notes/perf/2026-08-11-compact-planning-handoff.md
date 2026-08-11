# Compact planning handoff — 2026-08-11

## Question and baseline

Can the noisy-CCZ preprocessing peak be reduced without giving back compile or
sampling performance? The baseline is ticit `6bf96e0`; the retained source is
that commit plus the two-file working-tree diff described below. Both were
built with stable Rust, `--release`, and `-Ctarget-cpu=native` and pinned to CPU
10 on the i5-14600KF.

Each row uses one prepared, single-thread sampler and three timed calls with
seeds 0, 1, and 2. Peak RSS is the highest `/proc/PID/status` `VmRSS` sampled
every 5 ms over the complete process. Counts matched exactly before/after.

## Why moving the state was not enough

The old frontend cloned the entire lowered `FrameFactoredState` before
planning. A direct move nearly halved d07 RSS, but made compile roughly 19%
slower and sampling 3–4% slower in same-binary tests. The clone was accidentally
providing a compact allocation layout and quarantining lowering allocations.
Cloning context or pending operations separately, changing queue shape, and
cloning only the final program all failed either the compile or sampling gate.

Source tracing then found redundant storage in `ActivePauliFrame`: every term's
full Pauli body remained owned even though production conjugation reads its
condition id from the term and its body from the existing transposed bitset.

## Retained design

The handoff has three measured size regimes, without a circuit-name check:

1. Below 100,000 active-frame terms, retain the old clone. It is cheap and gives
   small circuits their best sampling layout.
2. At 100,000 terms, move the retained queue/context into the planner. Hold the
   old active-frame allocations through planning, but release its large
   transpose immediately after pending-queue optimization.
3. At 200,000 terms, clear redundant Pauli bodies in place. The existing term
   vector and condition ids remain, so the frame's struct and small-circuit
   allocation layout do not change. The transpose remains the authoritative
   body representation.

The full matrix has clean gaps around those thresholds: the largest relevant
small case has 28,000 terms, noisy d05 has 174,066, and noisy d07 has 464,046.
A transition test crosses 200,000 terms and verifies conjugation still returns
all condition ids.

## CCZ results

### Noisy circuits

| Circuit | Compile before | Compile after | Sampling before | Sampling after | RSS before | RSS after | RSS change |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d05_p1e-3` | 2.326 s | 2.273 s | 123,773/s | 125,291/s | 320,424 KiB | 235,872 KiB | **−26.4%** |
| `d07_p1e-3` | 10.053 s | 9.875 s | 40,131/s | 39,500/s | 1,182,760 KiB | 500,604 KiB | **−57.7%** |
| `d09_p1e-3` | 35.206 s | 34.550 s | 17,683/s | 17,459/s | 3,321,436 KiB | 1,097,600 KiB | **−67.0%** |
| `d11_p1e-3` | 114.102 s | 106.785 s | 8,654/s | 8,951/s | 7,926,356 KiB | 2,502,152 KiB | **−68.4%** |

Current SymFT `686051a` one-shot peaks measured with the same CPU/process
protocol are 247,972 KiB (d05), 717,148 KiB (d07), and 1,645,340 KiB (d09).
The retained ticit path is respectively 4.9%, 30.2%, and 33.3% lower. SymFT d11
was not rerun in this campaign; its older documented peak is 5.74 GB.

### Noiseless control

Noiseless circuits stay on the old clone path.

| Circuit | Compile before | Compile after | Sampling before | Sampling after | RSS before | RSS after |
|---|---:|---:|---:|---:|---:|---:|
| `d05_p0` | 0.665 s | 0.667 s | 288,701/s | 294,548/s | 39,580 KiB | 39,972 KiB |
| `d07_p0` | 2.495 s | 2.489 s | 129,445/s | 131,653/s | 121,992 KiB | 124,972 KiB |
| `d09_p0` | 9.116 s | 9.038 s | 59,146/s | 59,966/s | 318,492 KiB | 319,868 KiB |
| `d11_p0` | 30.935 s | 30.627 s | 31,524/s | 31,905/s | 733,564 KiB | 733,520 KiB |

## Broad validation

All 11 SOFT benchmark circuits stayed on the small clone path. Every SOFT and
CCZ sampling row was within 2% of its baseline or faster after longer reruns of
the two initially noisy rows. Examples: `msc_d3` 6.032M → 6.061M shots/s,
`msc_d5` 257.4k → 255.9k, and `coherent_d5_r5` 9.507 → 9.509. The final
`cargo nextest run` passes 256 tests with 1 skipped.
