# Benchmarks

## CPU: ticit vs SymFT vs Clifft

The ticit sampling rows were rechecked at commit `800c87c` (2026-08-09) on an
Intel Core i5-14600KF, pinned to one P-core, one sampler thread; the original
three-way campaign and the preparation/RSS tables were measured at `ce3eb52`
(2026-08-08). ticit built with
`-Ctarget-cpu=native`; SymFT is the native C++ reference build
(`-O3`, `-march=native` kernels, AVX-512); Clifft is a dev build
(`0.0.1.dev95`) driven through its Python API (its sampling core is
native). Unless noted below, throughput numbers are medians of three
interleaved runs on a quiet machine; compile and memory are single runs that
repeat within ~3%.

### Sampling throughput (shots/s, single core)

| Circuit | ticit | SymFT | Clifft | ticit / SymFT |
|---|---:|---:|---:|---:|
| `msc_d3` | **5.19 M** | 3.53 M | 865 k | 1.47x |
| `msc_d3` postselected | **5.93 M** | 4.38 M | 1.13 M | 1.35x |
| `msc_d5` | 66.8 k | 67.2 k | 22.1 k | 0.99x |
| `msc_d5` postselected | 248 k | 249 k | 87.4 k | 0.99x |
| `distillation` | **3.48 M** | 2.35 M | 135 k | 1.48x |
| `pure_surface_d9` | **3.58 M** | 2.19 M | 80.5 k | 1.63x |
| CCZ `d05_p0` | **287 k** | 254 k | 4.40 k | 1.13x |
| CCZ `d05_p1e-3` | **121 k** | 59.3 k | 3.59 k | 2.04x |
| CCZ `d07_p0` | **129 k** | 110 k | 1.94 k | 1.18x |
| CCZ `d07_p1e-3` | **36.3 k** | 18.0 k | 677 | 2.02x |
| CCZ `d09_p0` | **55.9 k** | 54.9 k | 796 | 1.02x |
| CCZ `d09_p1e-3` | **16.9 k** | 8.78 k | 420 | 1.92x |
| CCZ `d11_p0` | 30.1 k | 31.3 k | 358 | 0.96x |
| CCZ `d11_p1e-3` | **8.91 k** | 4.65 k | 199 | 1.92x |

### Circuit compilation time (parse + plan + prepare, seconds)

| Circuit | ticit | SymFT | Clifft |
|---|---:|---:|---:|
| `msc_d3` | 0.005 | 0.02 | 0.001 |
| `msc_d5` | 0.05 | 0.08 | 0.003 |
| `pure_surface_d9` | 0.05 | 0.07 | 0.015 |
| CCZ `d05_p0` | 0.79 | 0.70 | 0.43 |
| CCZ `d05_p1e-3` | **2.7** | 8.2 | 3.0 |
| CCZ `d07_p0` | 2.9 | 2.8 | 2.5 |
| CCZ `d07_p1e-3` | **11.2** | 53.7 | 22.0 |
| CCZ `d09_p0` | 10.3 | 8.9 | 11.6 |
| CCZ `d09_p1e-3` | **38.5** | 257 | 99.4 |
| CCZ `d11_p0` | 32.4 | 29.6 | 43.3 |
| CCZ `d11_p1e-3` | **115** | 885 | 338 |

### Peak memory (max RSS, compile + sampling)

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

- The ticit benchmark driver uses `sample_counts_with_seed`, matching the
  aggregate-counter contract used by the SymFT and Clifft survivor paths. The
  public `sample`/`sample_with_seed` methods additionally materialize Clifft-
  shaped per-shot records and are not represented by these throughput rows.
- Circuits: `msc_d3_inject_cultivate_p1e-3`,
  `msc_d5_inject_cultivate_p1e-3`, `distillation`, and
  `pure_surface_d9_r9_p1e-3` from `testdata/circuits/soft/`; CCZ non-tels
  circuits from `testdata/circuits/ccz/` (`.clifft`, Stim format) at distance
  5/7/9/11 with p=0 and p=1e-3.
- Postselected discard counts agree across tools where semantics align
  (31.4% on `msc_d3` for ticit and Clifft alike); Clifft reports logical
  errors over all shots when not postselecting, ticit/SymFT over accepted
  shots. Throughput is unaffected.
- Clifft solves a different problem shape on the CCZ circuits (survivor
  sampling with full syndrome tracking), which is where its throughput
  gap is largest; treat those rows as cross-tool context, not a
  like-for-like kernel comparison.
- The distillation row uses raw parity and no detector postselection. ticit and
  SymFT produced identical aggregate counters over 60 million shots; Clifft's
  dev95 survivor API reported every attempted shot as passed, so its throughput
  is cross-tool context rather than identical output-contract work. The circuit
  has five observables; logical-error counts are deliberately not compared.
- ticit's compile-time lead on noisy circuits comes from hash-map interning
  and a candidate-bounded parent search in expression-plan preparation;
  its memory premium there (~15% at d05, growing to +29-38% at d09/d11
  p=1e-3) is unattributed in detail — the plan interner's owned keys are
  the leading suspect (see `docs/impl-notes/perf/`).
- Where ticit and SymFT are at parity (`msc_d5`, `d09_p0`, `d11_p0`), the
  workload is dominated by large-dimension active-state kernels that
  neither tool has specialized; ticit's biggest wins are the dim-16
  register-resident rotation runs (msc_d3) and the sparse presample
  representation (noisy CCZ, `pure_surface_d9`).
- The msc_d3/pure_surface_d9/d05/d07 throughput rows are medians of three
  interleaved runs; the msc_d5/d09/d11 rows are single processes with
  in-process sample repeats (their compile times make interleaved
  external repeats impractical — SymFT's d11_p1e-3 compile alone is
  ~15 minutes).
- The distillation ticit value is the mean of two interleaved current-build
  runs; SymFT and Clifft are three in-process repeats.

## GPU: ticit vs SymFT

Remeasured with ticit `339d96e` and SymFT `925078b` on RTX 4090 D cards on
2026-08-10. Rates are sampling-only attempted shots/s after per-circuit tuning;
parsing, planning, RNG setup, and ticit's one-time cuTile JIT are reported
separately. H200 results are omitted from this refresh.

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

Only `msc_d3` and `msc_d5` postselect detectors; every other row disables
detector postselection and counts observable 0. Before benchmarking, all 17
circuits passed a three-way Ticit CPU / Ticit GPU / SymFT GPU statistical
gate, and every retained CCZ expectation channel passed a Bonferroni-corrected
Ticit CPU/GPU comparison. The old Ticit SOFT rates were not reproduced after
the correctness fixes—for example, `pure_d9` is 2.48 M rather than 8.91 M
shots/s. Full
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
