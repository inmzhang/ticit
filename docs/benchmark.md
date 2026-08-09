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

Measured with ticit GPU code at `c6d0db1` and SymFT at `bd77739` on
2026-08-09; the two wide-state rows were remeasured in paired runs after the
generic bit-mask improvement at `bf46366`. Rates are public batch-sampler
throughput (`sample_s`) on one RTX 4090 or H200; parsing, planning, RNG setup,
and ticit's cuTile JIT are reported separately rather than charged to sampling. Each
sampler used its strongest measured memory-feasible launch configuration.

### SOFT circuits (shots/s)

| Circuit | 4090 ticit | 4090 SymFT | Ratio | H200 ticit | H200 SymFT | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `msc_d3` | 434 M | 72.4 M | 5.99x | 375 M | 110 M | 3.42x |
| `coherent_d3_r1` | 603 M | 132 M | 4.56x | 759 M | 190 M | 3.99x |
| `coherent_d3_r3` | 10.1 M | 21.0 M | 0.48x | 9.80 M | 21.3 M | 0.46x |
| `distillation` | 16.7 M | 13.9 M | 1.20x | 15.9 M | 15.7 M | 1.01x |
| `msc_d5` | 6.89 M | 4.24 M | 1.62x | 6.66 M | 6.42 M | 1.04x |
| `msc_proxy_d7` | 6.15 M | 3.15 M | 1.95x | 5.78 M | 4.37 M | 1.32x |
| `pure_d7` | 12.4 M | 7.01 M | 1.77x | 11.0 M | 8.04 M | 1.36x |
| `pure_d9` | 8.91 M | 4.39 M | 2.03x | 7.83 M | 5.21 M | 1.50x |
| `coherent_d5_r1` | 1.36 M | 1.49 M | 0.91x | 790 k | 2.19 M | 0.36x |
| `coherent_d5_r5` | 274 | 192 | 1.43x | 577 | 676 | 0.85x |
| `MSC_d7` sparse | 36.1 k | 31.2 k | 1.16x | 77.4 k | 61.9 k | 1.25x |
| **Geometric mean** |  |  | **1.66x** |  |  | **1.19x** |

ticit wins 9 of 11 circuits on the RTX 4090 and 8 of 11 on the H200.

### CCZ non-tels circuits (shots/s, 65,536-shot batch)

| Circuit | 4090 ticit | 4090 SymFT | Ratio | H200 ticit | H200 SymFT | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `d05_p0` | 4.52 M | 184 k | 24.6x | 4.39 M | 290 k | 15.1x |
| `d05_p1e-3` | 4.26 M | 17.8 k | 240x | 4.16 M | 36.1 k | 115x |
| `d07_p0` | 3.76 M | 47.6 k | 78.9x | 3.63 M | 79.8 k | 45.5x |
| `d07_p1e-3` | 3.34 M | 2.01 k | 1,663x | 3.13 M | 5.38 k | 582x |
| `d09_p0` | 2.70 M | 12.6 k | 214x | 2.93 M | 25.5 k | 115x |
| `d09_p1e-3` | 2.09 M | 10.2 k | 205x | 2.25 M | 631 | 3,568x |
| `d11_p0` | 2.07 M | 5.26 k | 394x | 2.46 M | 8.11 k | 303x |
| `d11_p1e-3` | 1.34 M | 5.16 k | 260x | 1.48 M | 7.55 k | 196x |
| **Geometric mean** |  |  | **208x** |  |  | **180x** |

The comparison uses raw detector parity and observable 0 for both tools,
avoiding SOFT benchmark issues #8 and #9. Independent RNG streams make exact
counters differ, but ticit's GPU results pass CPU differential checks. cuTile's
one-time kernel JIT costs about 9--79 seconds by circuit; SymFT is AOT-compiled,
so latency-sensitive one-off jobs should include that separately. Full settings,
unrounded rates, and compilation caveats are in
[`2026-08-09-gpu-comparison.md`](impl-notes/perf/2026-08-09-gpu-comparison.md).

These tables are maintained by hand: when a change moves any number
meaningfully, the tables are re-measured and updated in the same commit
series (see `AGENTS.md`). Historical measurement details live in
`docs/impl-notes/perf/`; the source-mapped SymFT backport guide is
[`2026-08-09-symft-cpu-backports.md`](impl-notes/perf/2026-08-09-symft-cpu-backports.md),
and the Clifft-inspired retained/rejected rotation work is in
[`2026-08-09-clifft-symbolic-core.md`](impl-notes/perf/2026-08-09-clifft-symbolic-core.md).
