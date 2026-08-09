# msc_d3 scalar profile and thread scaling — 2026-08-08

Host: Intel Core i5-14600KF, release build with debug symbols. Runs were pinned
to distinct physical P-cores (`0,2,4,6,8,10`), detector postselection was off,
and the differential validator was stopped during every measurement.

## Why this became the scalar target

At one thread, ticit measured 2.17 Mshot/s on `msc_d3`, versus 3.53 Mshot/s for
native SymFT. This 38% deficit was materially larger than the 12% local gap on
CCZ `d05_p0`; noisy `d05_p1e-3` was already faster overall in ticit because its
presampling phase is faster.

Matched `perf stat -r 3` runs used 5,000,000 shots:

| Counter | ticit | SymFT |
|---|---:|---:|
| Core cycles | 12.17 B | 7.33 B |
| Instructions | 69.76 B | 33.13 B |
| IPC | 5.73 | 4.52 |
| Branch-miss rate | 0.16% | 0.35% |
| Cache-miss rate | 0.95% | 1.28% |

The problem is excess work, not stalls. In ticit, 55.0% of sampled cycles were
in uniform imaginary pair rotations. Native SymFT spent 45.9% in the matching
rotation path, whose small-state inline kernels contain explicit AVX2/FMA.
ticit's next-largest symbols were active measurement (10.8%), diagonal
projection (7.6%), dormant promotion (6.4%), and rotation-run dispatch (5.7%).

The rotation deficit is deferred to the required SIMD phase. Reassociating or
fusing the scalar oracle would discard its bit-exact evaluation-order contract.

## Scoped-thread implementation

The batch sampler now owns one independent state per requested worker and uses
`std::thread::scope` to process statically strided chunks. Active workers are
`min(requested_threads, max(1, chunk_count))`; one worker takes the original
inline path without spawning. Chunk-indexed seeds make counts independent of
worker assignment. No dependency or unsafe code was added.

`msc_d3`, 5,000,000 shots per repeat, three repeats:

| Physical cores | ticit Mshot/s | ticit speedup | SymFT Mshot/s | SymFT speedup |
|---:|---:|---:|---:|---:|
| 1 | 2.174 | 1.00x | 3.525 | 1.00x |
| 2 | 4.329 | 1.99x | 7.023 | 1.99x |
| 4 | 8.530 | 3.92x | 13.874 | 3.94x |
| 6 | 11.871 | 5.46x | 19.799 | 5.62x |

ticit returned identical counts at every thread width. The dedicated test also
checks one-vs-three-worker equality through postselection and verifies that a
one-chunk request caps itself to one active worker.

## Six-core breadth check

| Circuit | ticit shots/s | SymFT shots/s | ticit relative |
|---|---:|---:|---:|
| SOFT `msc_d3` | 11,871,131 | 19,799,000 | -40.0% |
| CCZ `d05_p0` | 1,234,355 | 1,348,570 | -8.5% |
| CCZ `d05_p1e-3` | 238,428 | 222,229 | +7.3% |
| CCZ `d07_p0` | 430,110 | 437,550 | -1.7% |

`d07_p1e-3` preprocessing took about 95 seconds locally while sampling 10,000
shots took 0.149 seconds, so it remains final coverage rather than a tight-loop
benchmark. CCZ d09/d11 are likewise excluded from fast iteration by policy.

## Reference server confirmation

On the dual EPYC 9254 server, the run was confined to the 24 physical cores of
socket 0 (CPUs 0–23). The same 5,000,000-shot, three-repeat workload measured:

| Physical cores | ticit Mshot/s | ticit speedup | SymFT Mshot/s | SymFT speedup |
|---:|---:|---:|---:|---:|
| 1 | 1.434 | 1.00x | 2.424 | 1.00x |
| 2 | 2.319 | 1.62x | 4.850 | 2.00x |
| 4 | 5.894 | 4.11x | 9.733 | 4.02x |
| 8 | 7.419 | 5.17x | 10.187 | 4.20x |
| 16 | 17.390 | 12.12x | 27.925 | 11.52x |
| 24 | 24.186 | 16.86x | 35.659 | 14.71x |

The shared host produced noisy 2/8-core points for both engines, but the end
result is clear: ticit's gap narrows from 40.8% at one core to 32.2% at 24 cores.
Counts stayed identical across ticit widths. The exact server checkout passes
`cargo test --all-targets`, including the threaded-count test.
