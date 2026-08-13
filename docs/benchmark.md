# Benchmarks

## Current CCZ direct-T fixtures (2026-08-13)

These are the current CCZ benchmark numbers after regenerating the four noisy
portable fixtures with direct logical-T preparation and remapped `DETECTOR`
instructions. The CPU runs use ticit `0ca4db0`, SymFT `686051a`, and Clifft
`0.6.0` on an Intel Xeon Platinum 8260 (CPU 10 for ticit/SymFT and CPU 12 for
Clifft), one sampler thread, and three aggregate-only repeats. The timed
contract is attempted shots/s with `keep_records=false`, no detector
postselection, and reference-normalized detector/observable parity. Each
source marker `E(0.125)` is replaced by `E(0)` in the temporary benchmark copy;
the checked-in marker remains available for Bloc's source-probability rewrite.

### CPU sampling throughput (shots/s)

| Circuit | ticit | SymFT | Clifft |
|---|---:|---:|---:|
| `d05_p1e-3` | **42.6 k** | 28.8 k | 1.79 k |
| `d07_p1e-3` | **15.0 k** | 10.4 k | 728 |
| `d09_p1e-3` | **5.66 k** | 4.20 k | 332 |
| `d11_p1e-3` | **2.90 k** | 2.39 k | 132 |

All rows had zero detector discards. The logical-error counter is also zero,
as expected because these portable fixtures have no `OBSERVABLE_INCLUDE`
records; their 28 logical `EXP_VAL` probes remain present. Clifft's d11 noisy
row required 678 s of preparation and reached only 132 shots/s after three
repeats, so preparation and sampling must remain separate when using this
table.

### Ticit GPU sampling throughput (shots/s)

The GPU run uses ticit `0ca4db0` on an NVIDIA GeForce RTX 4090 D (driver
580.95.05), one GPU, and the same aggregate-only contract. Geometry-specific
pilots selected chunk sizes 65,536 for d5–d9 and 262,144 for d11. The measured
mean is over three repeats.

| Circuit | Ticit CUDA |
|---|---:|
| `d05_p1e-3` | 118 k |
| `d07_p1e-3` | 23.5 k |
| `d09_p1e-3` | 5.05 k |
| `d11_p1e-3` | unsupported: GPU exogenous mask plan exceeds i32 indexing |

SymFT GPU was intentionally not rerun and is not reported here. Ticit's d11
noisy CUDA planner currently rejects the fixture before sampling with an
i32-indexing limit; this is a capability limitation, not a measured zero rate.

## Historical CPU matrix (pre-direct-T fixtures)

The non-CCZ rows are the 2026-08-11 matrix at ticit `78ca59b`, SymFT
`686051a`, and Clifft `b2a501d` (`0.7.1.dev34+gb2a501ddb`). The CCZ rows were
remeasured overnight on 2026-08-11/12 at ticit `dd631a1`, the same SymFT
commit, and the same Clifft package after retaining only the 28 logical
`EXP_VAL` probes. The machine is an Intel Core i5-14600KF; every tool was
pinned to CPU 10 with one sampler thread. Ticit used a fresh 0.2.2 release
extension built with `-Ctarget-cpu=native` (artifact SHA-256
`681531e3af0d18e911606aa505ece7630eccead8200e3dd963d483152a7fa164`).

CCZ throughput is the arithmetic mean of three 10-second repeats after one
preparation. Preparation is timed parse + plan/lower + reference trajectory +
one-shot warm-up. Raw repeats and metadata are in the
[logical-probe CCZ report](impl-notes/perf/2026-08-11-logical-expval-cpu-ccz.md)
and its adjacent JSON file. The unchanged rows remain documented in the
[SOFT report](impl-notes/perf/2026-08-11-normalized-cpu-soft.md) and
[Clifft rerun](impl-notes/perf/2026-08-11-normalized-cpu-surface-d7-clifft-rerun.md).

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
| CCZ `d05_p1e-3` | **114 k** | 59.1 k | 3.42 k | 1.94x |
| CCZ `d07_p1e-3` | **30.5 k** | 19.5 k | 1.28 k | 1.56x |
| CCZ `d09_p1e-3` | **12.5 k** | 9.16 k | 406 | 1.36x |
| CCZ `d11_p1e-3` | **6.74 k** | 4.89 k | 190 | 1.38x |

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
| CCZ `d05_p1e-3` | 2.35 | 6.04 | 2.78 |
| CCZ `d07_p1e-3` | 10.2 | 44.1 | 20.8 |
| CCZ `d09_p1e-3` | 35.6 | 223 | 94.8 |
| CCZ `d11_p1e-3` | 108 | 788 | 327 |

### Peak memory (max RSS, compile + sampling)

All CCZ cells were remeasured from isolated processes against the current
28-probe fixtures. Each process performed preparation plus one configured
sample call; the three tools for one circuit used separate logical CPUs during
this memory-only pass. The final d=11 trio stayed below physical-memory
capacity and used no swap.

| Circuit | ticit | SymFT | Clifft* |
|---|---:|---:|---:|
| `msc_d3` | 17.8 MB | 17.8 MB | 47.9 MB |
| `msc_d5` | 17.6 MB | 17.6 MB | 50.2 MB |
| `pure_surface_d9` | 29.3 MB | 29.0 MB | 56.8 MB |
| CCZ `d05_p1e-3` | **265 MiB** | 290 MiB | 306 MiB |
| CCZ `d07_p1e-3` | **523 MiB** | 787 MiB | 1.14 GiB |
| CCZ `d09_p1e-3` | **1.07 GiB** | 1.67 GiB | 3.41 GiB |
| CCZ `d11_p1e-3` | **2.42 GiB** | 3.21 GiB | 8.54 GiB |

*Clifft's column includes the Python interpreter's ~35-40 MB baseline.
Peak RSS is planning-dominated on the noisy CCZ circuits for all three
tools and flat in total shots (chunked streaming), so these are full-run
peaks.

### Notes on fairness

The table above is a historical aggregate-counter run. Current `xtask bench`
uses `keep_records=true` by default, so its rates must not be mixed with these
rows; rerun all three tools together when refreshing the table.

- Every row uses reference-normalized detector and observable bits. All three
  exact simulators receive the same SymFT reference trajectory; this avoids
  backend-local RNG/compiler ordering choosing different valid noiseless
  branches. Reference preparation is outside sampling throughput but included
  in preparation time.
- `msc_d3` and `msc_d5` use an all-detector postselection mask. Every other
  circuit uses an empty mask. These historical runs report aggregate attempted,
  discarded, accepted, and observable-0 counts; no tool materialized per-shot
  records in that timed region. The current default is the record contract
  described above and requires a fresh three-way rerun.
- The retained matrix has 45/45 successful tool/circuit rows. Shot accounting is
  exact, non-postselected rows discard zero shots, and cross-tool discard and
  observable-0 rates agree within sampling noise. In particular,
  `distillation` is now compared on observable 0 under one shared reference
  convention instead of mixing raw and normalized parity.
- The CCZ fixtures used for this historical run contained 28 logical `EXP_VAL`
  operations but no detector or observable annotations. The current fixtures
  use direct logical-T preparation, inline the generated detector annotations,
  and have no `OBSERVABLE_INCLUDE` records. Do not compare their rates with this
  table.
- The full-run Clifft `surface_d7_r7` rates were 156k, 149k, and 114k shots/s.
  A quiet five-repeat rerun produced 154k-156k with a 1.2% range, confirming the
  final repeat was a transient system slowdown.
- ticit wins the four retained historical noisy CCZ rows by 1.36-1.94x.

## Historical GPU benchmark status

The standalone GPU benchmark harness has been removed. We are not publishing
GPU throughput numbers until a replacement benchmark is intentionally designed
and reviewed around one ticit-only workload contract.

The sampler contract remains explicit and tested: user-facing CPU/GPU sampling
keeps measurement, detector, observable, and `EXP_VAL` records by default;
`--count-only` and the `sample_circuit_counts*` APIs are aggregate-only opt-ins.
Both modes execute all `DETECTOR` and `OBSERVABLE_INCLUDE` instructions. The
count-only path omits raw record materialization and host record copies, so its
rate must never be presented as a record-producing rate. When a new benchmark
is added, it must print `keep_records`, record counts, plan instruction counts,
and the separate setup/execute/copy timings for every run.

Historical cross-project GPU measurements remain under `docs/impl-notes/perf/`
for archaeology only; they are not current benchmark results.
