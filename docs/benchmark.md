# Benchmarks

## CPU: ticit vs SymFT vs Clifft

Remeasured on 2026-08-11 at ticit `78ca59b`, SymFT `686051a`, and Clifft
`b2a501d` (`0.7.1.dev34+gb2a501ddb`). The machine is an Intel Core i5-14600KF;
every tool was a current local source build pinned to CPU 10 with one sampler
thread. ticit used a release build with `-Ctarget-cpu=native`; SymFT used its
native `-O3` C++ kernels.

Throughput is the median of three timed repeats after preparation. The Clifft
`surface_d7_r7` value is the median of a five-repeat targeted rerun after one
full-matrix repeat was interrupted by a transient slowdown. Preparation is a
single timed parse + plan/lower + reference trajectory + one-shot warm-up; the
CCZ circuits have no detector or observable outputs, so their empty reference
vectors take the no-op path. Full per-repeat counts and timings are retained in
the [SOFT report](impl-notes/perf/2026-08-11-normalized-cpu-soft.md),
[CCZ report](impl-notes/perf/2026-08-11-normalized-cpu-ccz.md), and
[Clifft rerun](impl-notes/perf/2026-08-11-normalized-cpu-surface-d7-clifft-rerun.md)
(with adjacent JSON files).

### Sampling throughput (shots/s, single core)

| Circuit | ticit | SymFT | Clifft | ticit / SymFT |
|---|---:|---:|---:|---:|
| `msc_d3` (postselected) | **5.90 M** | 4.27 M | 1.00 M | 1.38x |
| `msc_d5` (postselected) | 237 k | **280 k** | 80.8 k | 0.848x |
| `msc_d7` | 106 | **108** | 52.0 | 0.980x |
| `msc_proxy_d7` | 94.1 k | **105 k** | 25.0 k | 0.894x |
| `coherent_d3_r1` | **4.93 M** | 4.49 M | 1.87 M | 1.10x |
| `coherent_d3_r3` | 655 k | **679 k** | 254 k | 0.965x |
| `coherent_d5_r1` | 38.7 k | **41.5 k** | 16.9 k | 0.932x |
| `coherent_d5_r5` | 14.4 | **15.3** | 0.773 | 0.942x |
| `distillation` | **3.46 M** | 2.95 M | 122 k | 1.17x |
| `surface_d7_r7` | **7.79 M** | 4.54 M | 156 k | 1.71x |
| `surface_d9_r9` | **3.53 M** | 2.05 M | 73.0 k | 1.72x |
| CCZ `d05_p0` | 288 k | **298 k** | 4.13 k | 0.966x |
| CCZ `d05_p1e-3` | **116 k** | 58.7 k | 3.39 k | 1.98x |
| CCZ `d07_p0` | **130 k** | 127 k | 1.82 k | 1.02x |
| CCZ `d07_p1e-3` | **39.0 k** | 19.5 k | 1.25 k | 1.99x |
| CCZ `d09_p0` | 59.1 k | **60.2 k** | 775 | 0.981x |
| CCZ `d09_p1e-3` | **16.3 k** | 9.22 k | 421 | 1.77x |
| CCZ `d11_p0` | 31.8 k | **33.7 k** | 349 | 0.942x |
| CCZ `d11_p1e-3` | **8.58 k** | 4.77 k | 191 | 1.80x |

### Circuit preparation time (seconds)

| Circuit | ticit | SymFT | Clifft |
|---|---:|---:|---:|
| `msc_d3` | 0.00590 | 0.00377 | 0.0285 |
| `msc_d5` | 0.0609 | 0.0378 | 0.0168 |
| `msc_d7` | 0.459 | 0.224 | 0.103 |
| `msc_proxy_d7` | 0.261 | 0.115 | 0.0469 |
| `coherent_d3_r1` | 0.00108 | 0.000927 | 0.000785 |
| `coherent_d3_r3` | 0.00237 | 0.00209 | 0.00159 |
| `coherent_d5_r1` | 0.00446 | 0.00418 | 0.00290 |
| `coherent_d5_r5` | 0.164 | 0.157 | 1.38 |
| `distillation` | 0.00543 | 0.00286 | 0.00335 |
| `surface_d7_r7` | 0.0225 | 0.0196 | 0.0115 |
| `surface_d9_r9` | 0.0531 | 0.0560 | 0.0268 |
| CCZ `d05_p0` | 0.728 | 0.0680 | 0.306 |
| CCZ `d05_p1e-3` | 2.46 | 6.05 | 2.77 |
| CCZ `d07_p0` | 2.73 | 0.257 | 1.66 |
| CCZ `d07_p1e-3` | 10.4 | 43.4 | 20.8 |
| CCZ `d09_p0` | 9.59 | 0.805 | 8.28 |
| CCZ `d09_p1e-3` | 36.1 | 218 | 93.3 |
| CCZ `d11_p0` | 32.0 | 2.52 | 32.9 |
| CCZ `d11_p1e-3` | 121 | 765 | 323 |

### Historical peak memory (max RSS, compile + sampling)

Peak RSS was not remeasured: reference normalization adds only one detector and
observable vector and does not change the retained plan layout. These values
remain the 2026-08-08 campaign at ticit `ce3eb52`, SymFT `bd77739`, and Clifft
`b165657`; they should not be read as memory measurements of the current
throughput builds.

| Circuit | ticit | SymFT | Clifft* |
|---|---:|---:|---:|
| `msc_d3` | 17.8 MB | 17.8 MB | 47.9 MB |
| `msc_d5` | 17.6 MB | 17.6 MB | 50.2 MB |
| `pure_surface_d9` | 29.3 MB | 29.0 MB | 56.8 MB |
| CCZ `d05_p0` | 39.0 MB | 41.6 MB | 74.2 MB |
| CCZ `d05_p1e-3` | 348 MB | 303 MB | 313 MB |
| CCZ `d07_p0` | 113 MB | 135 MB | 138 MB |
| CCZ `d07_p1e-3` | 1.17 GB | 1.00 GB | 1.18 GB |
| CCZ `d09_p0` | 282 MB | 342 MB | 297 MB |
| CCZ `d09_p1e-3` | 3.30 GB | 2.56 GB | 3.56 GB |
| CCZ `d11_p0` | 641 MB | 808 MB | 636 MB |
| CCZ `d11_p1e-3` | 7.90 GB | 5.74 GB | 8.97 GB |

*Clifft's column includes the Python interpreter's ~35-40 MB baseline.
Peak RSS is planning-dominated on the noisy CCZ circuits for all three
tools and flat in total shots (chunked streaming), so these are full-run
peaks.

### Notes on fairness

- Every row uses reference-normalized detector and observable bits. All three
  exact simulators receive the same SymFT reference trajectory; this avoids
  backend-local RNG/compiler ordering choosing different valid noiseless
  branches. Reference preparation is outside sampling throughput but included
  in preparation time.
- `msc_d3` and `msc_d5` use an all-detector postselection mask. Every other
  circuit uses an empty mask. All tools report aggregate attempted, discarded,
  accepted, and observable-0 counts; no tool materializes per-shot records in
  the timed region.
- The full matrices have 57/57 successful tool/circuit rows. Shot accounting is
  exact, non-postselected rows discard zero shots, and cross-tool discard and
  observable-0 rates agree within sampling noise. In particular,
  `distillation` is now compared on observable 0 under one shared reference
  convention instead of mixing raw and normalized parity.
- The CCZ fixtures contain `EXP_VAL` operations but no detector or observable
  annotations. Their discard and logical-error counters are therefore empty;
  the table compares circuit execution, not full syndrome output work.
- The full-run Clifft `surface_d7_r7` rates were 156k, 149k, and 114k shots/s.
  A quiet five-repeat rerun produced 154k-156k with a 1.2% range, confirming the
  final repeat was a transient system slowdown.
- ticit is faster than SymFT on 10 of 19 circuits. The largest current CPU wins
  are 1.71-1.72x on the pure-Clifford surface circuits and 1.77-1.99x on noisy
  CCZ; noiseless CCZ and the larger MSC cases remain near parity or favor
  SymFT.

## GPU: ticit vs SymFT

Remeasured with ticit `339d96e` and SymFT `925078b` on RTX 4090 D cards on
2026-08-10. Rates are sampling-only attempted shots/s after per-circuit tuning;
parsing, planning, RNG setup, and ticit's one-time cuTile JIT are reported
separately. These rates predate the 2026-08-11 normalization work and were not
rerun in the CPU refresh. Current GPU benchmark preparation computes the
reference trajectory separately on CPU and passes it to the GPU sampler. H200
results are omitted from this refresh.

### SOFT circuits (shots/s)

| Circuit | ticit | SymFT | Ratio |
|---|---:|---:|---:|
| `msc_d3` | 404 M | 65.8 M | 6.15x |
| `coherent_d3_r1` | 840 M | 117 M | 7.16x |
| `coherent_d3_r3` | 6.72 M | 23.5 M | 0.286x |
| `distillation` | 16.7 M | 18.0 M | 0.927x |
| `msc_d5` | 7.67 M | 4.96 M | 1.55x |
| `pure_d7` | 4.93 M | 4.40 M | 1.12x |
| `pure_d9` | 2.48 M | 2.41 M | 1.03x |
| `coherent_d5_r1` | 1.33 M | 1.28 M | 1.04x |
| `coherent_d5_r5` | 91.2 | 54.0 | 1.69x |
| **Geometric mean** |  |  | **1.49x** |

ticit wins seven of the nine retained SOFT circuits. The unverified
`msc_proxy_d7` and `MSC_d7` fixtures were removed and are not benchmarked.

### CCZ non-tels circuits (shots/s, 65,536-shot batch)

| Circuit | ticit | SymFT | Ratio |
|---|---:|---:|---:|
| `d05_p0` | 4.57 M | 187 k | 24.5x |
| `d05_p1e-3` | 3.79 M | 23.3 k | 163x |
| `d07_p0` | 3.80 M | 46.8 k | 81.1x |
| `d07_p1e-3` | 3.03 M | 1.73 k | 1,750x |
| `d09_p0` | 2.79 M | 12.0 k | 232x |
| `d09_p1e-3` | 2.14 M | 10.3 k | 207x |
| `d11_p0` | 2.07 M | 5.31 k | 390x |
| `d11_p1e-3` | 1.23 M | 4.75 k | 260x |
| **Geometric mean** |  |  | **202x** |

Only `msc_d3` and `msc_d5` use all-detector postselection masks; every other
row uses an empty mask and counts observable 0. Before benchmarking, all 17
circuits passed a three-way Ticit CPU / Ticit GPU / SymFT GPU statistical gate,
and every retained CCZ expectation channel passed a Bonferroni-corrected Ticit
CPU/GPU comparison. The old Ticit SOFT rates were not reproduced after the
correctness fixes—for example, `pure_d9` is 2.48 M rather than 8.91 M shots/s.
Full
correctness counts, unrounded rates, tuning sweeps, fixed-cost timings, and raw
log locations are in
[`2026-08-10-gpu-correctness-and-comparison.md`](impl-notes/perf/2026-08-10-gpu-correctness-and-comparison.md).

These tables are maintained by hand: when a change moves any number
meaningfully, the tables are re-measured and updated in the same commit
series (see `AGENTS.md`). Historical measurement details live in
`docs/impl-notes/perf/`; the source-mapped SymFT backport guide is
[`2026-08-09-symft-cpu-backports.md`](impl-notes/perf/2026-08-09-symft-cpu-backports.md),
and the Clifft-inspired retained/rejected rotation work is in
[`2026-08-09-clifft-symbolic-core.md`](impl-notes/perf/2026-08-09-clifft-symbolic-core.md).
