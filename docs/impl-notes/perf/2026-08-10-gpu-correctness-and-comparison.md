# GPU correctness and throughput comparison — 2026-08-10

Ticit `339d96e` was compared with SymFT `925078b` on NVIDIA GeForce RTX
4090 D cards allocated through Slurm. SymFT is upstream `main` at `b6911a1`
plus the detector-postselection correctness fix at `89f3e19` and benchmark
metadata fix at `925078b`. H200 was intentionally omitted from this refresh.

The matrix has the nine retained SOFT circuits and all eight CCZ circuits.
`msc_proxy_d7` and `MSC_d7` were removed from Ticit's test corpus at
`f0b25cb` and are not benchmarked. Detector postselection is enabled only for
`msc_d3` and `msc_d5`; every other row uses `--no-postselect-detectors` and
observable 0.

## Correctness issues found and fixed

- Ticit GPU lowering now preserves each `RecordDetector.postselect` bit instead
  of treating detector records alike, and the regression is pinned by a plan
  test. Asynchronous record copies now retain their source buffers until the
  copy completes. The exact-k validation path also conditions binary
  categorical source channels, not only ordinary Bernoulli channels.
- SymFT's CUDA kernels previously processed `RecordDetector` instructions even
  when `postselect_detectors` was false, which reported detector hits as
  discards and suppressed their logical-error contribution. Commit `89f3e19`
  skips that work in both scalar and persistent kernels and adds a fired-
  detector regression test.

## Correctness gate

Each circuit was sampled independently with Ticit CPU, Ticit GPU, and SymFT
GPU. Entries are `discarded / logical_errors`; logical errors are counted over
accepted shots. The clean matrix used runtime code identical to `339d96e`
(the intervening commits only changed tests) and SymFT's corrected CUDA path.

| Circuit | Shots | Ticit CPU | Ticit GPU | SymFT GPU |
|---|---:|---:|---:|---:|
| `msc_d3` | 1,000,000 | 313,210 / 3 | 313,134 / 0 | 312,866 / 0 |
| `coherent_d3_r1` | 1,000,000 | 0 / 6,060 | 0 / 5,852 | 0 / 6,075 |
| `coherent_d3_r3` | 262,144 | 0 / 1,598 | 0 / 1,536 | 0 / 1,531 |
| `distillation` | 1,000,000 | 0 / 458,182 | 0 / 457,253 | 0 / 457,752 |
| `msc_d5` | 262,144 | 224,301 / 0 | 224,326 / 0 | 224,407 / 0 |
| `pure_d7` | 1,000,000 | 0 / 105,499 | 0 / 104,820 | 0 / 104,802 |
| `pure_d9` | 1,000,000 | 0 / 159,580 | 0 / 159,535 | 0 / 159,050 |
| `coherent_d5_r1` | 131,072 | 0 / 1,293 | 0 / 1,263 | 0 / 1,375 |
| `coherent_d5_r5` | 256 | 0 / 1 | 0 / 5 | 0 / 2 |
| `d05_p0` | 65,536 | 0 / 0 | 0 / 0 | 0 / 0 |
| `d05_p1e-3` | 65,536 | 0 / 0 | 0 / 0 | 0 / 0 |
| `d07_p0` | 65,536 | 0 / 0 | 0 / 0 | 0 / 0 |
| `d07_p1e-3` | 65,536 | 0 / 0 | 0 / 0 | 0 / 0 |
| `d09_p0` | 65,536 | 0 / 0 | 0 / 0 | 0 / 0 |
| `d09_p1e-3` | 65,536 | 0 / 0 | 0 / 0 | 0 / 0 |
| `d11_p0` | 65,536 | 0 / 0 | 0 / 0 | 0 / 0 |
| `d11_p1e-3` | 65,536 | 0 / 0 | 0 / 0 | 0 / 0 |

The populated three-way logical-error tests have no significant backend
difference (smallest Pearson omnibus p-value 0.0738). Sparse-row pairwise
exact tests are also insignificant (p at least 0.219), and the two MSC
discard tests have p-values 0.859 and 0.909. The zero-event CCZ rows establish
that none of the implementations emits spurious errors at the benchmark
exposure; they do not estimate the much rarer physical logical-error rate.

To make the zero-event CCZ result less vacuous, a second Ticit CPU/GPU check
retained every expectation value and compared each channel's sample mean using
independent RNG streams. `max z` is the largest standardized CPU/GPU
difference; the next column is the circuit-specific two-sided 5% Bonferroni
threshold. No channel exceeds its threshold, and no deterministic channel
differs.

| Circuit | Shots/backend | Expectation channels | Max z | Threshold |
|---|---:|---:|---:|---:|
| `d05_p0` | 4,096 | 100 | 2.874 | 3.481 |
| `d05_p1e-3` | 4,096 | 100 | 2.960 | 3.481 |
| `d07_p0` | 2,048 | 172 | 2.283 | 3.623 |
| `d07_p1e-3` | 2,048 | 172 | 3.253 | 3.623 |
| `d09_p0` | 1,024 | 268 | 2.657 | 3.737 |
| `d09_p1e-3` | 1,024 | 268 | 3.308 | 3.737 |
| `d11_p0` | 512 | 388 | 2.668 | 3.829 |
| `d11_p1e-3` | 512 | 388 | 3.313 | 3.829 |

The code gate was 265 tests plus 13 doctests passed, one ignored, under
`cargo test --features gpu`. A deterministic GPU smoke test and d05 exact-k
record exports for k=0 and k=1 also completed. SymFT's five-test CUDA CTest
gate passed on the 4090 before the matrix run.

## Measurement method

- Rates are attempted shots divided by sampling-only time. Parsing, planning,
  RNG setup, and Ticit's one-time cuTile JIT are excluded and remain visible
  in the raw logs.
- Ticit rates are the median of three fresh-process `sample_s` measurements.
  SymFT rates are its average across three in-process sampling repeats.
- Each case was tuned on RTX 4090 D for Ticit chunk size and for SymFT CUDA
  mode, threads per block, and shots per launch. Final jobs used distinct
  Slurm-assigned devices; `CUDA_VISIBLE_DEVICES` mappings were checked while
  the jobs were live.
- Driver: 580.95.05. SymFT used the CUDA 12.6 module; Ticit used its CUDA 13.3
  cuTile redistributable environment.
- Raw logs and retained runner scripts are under
  `/dssg/home/zhangyiming/workspace/gpu-comparison-2026-08-10/` on
  `gpucluster`.

## Tuned settings

| Circuit | Shots | Ticit chunk | SymFT mode | Threads/block | Shots/launch |
|---|---:|---:|---|---:|---:|
| `msc_d3` | 10,000,000 | 1,048,576 | presample | 32 | 4,194,304 |
| `coherent_d3_r1` | 10,000,000 | 1,048,576 | presample | 128 | 4,194,304 |
| `coherent_d3_r3` | 4,194,304 | 1,048,576 | GPU | 32 | 4,194,304 |
| `distillation` | 4,194,304 | 2,097,152 | presample | 32 | 4,194,304 |
| `msc_d5` | 1,048,576 | 1,048,576 | presample | 128 | 1,048,576 |
| `pure_d7` | 5,000,000 | 4,194,304 | presample | 128 | 4,194,304 |
| `pure_d9` | 5,000,000 | 262,144 | presample | 128 | 4,194,304 |
| `coherent_d5_r1` | 1,048,576 | 131,072 | GPU | 128 | 1,048,576 |
| `coherent_d5_r5` | 1,024 | 64 | GPU | 128 | 256 |
| `d05_p0` | 65,536 | 65,536 | GPU | 32 | 65,536 |
| `d05_p1e-3` | 65,536 | 32,768 | presample | 128 | 65,536 |
| `d07_p0` | 65,536 | 65,536 | GPU | 32 | 65,536 |
| `d07_p1e-3` | 65,536 | 65,536 | GPU | 32 | 65,536 |
| `d09_p0` | 65,536 | 65,536 | GPU | 32 | 65,536 |
| `d09_p1e-3` | 65,536 | 65,536 | GPU | 32 | 65,536 |
| `d11_p0` | 65,536 | 65,536 | GPU | 32 | 65,536 |
| `d11_p1e-3` | 65,536 | 65,536 | GPU | 32 | 65,536 |

The oversized 4,194,304-shot Ticit `pure_d9` candidate exhausted VRAM. SymFT
presample was rejected by its shared-memory guard for noisy d07/d09 and was
stopped as decisively noncompetitive for noisy d11 after more than seven
minutes, versus 24 seconds for the baseline GPU-exogenous sample.

## SOFT throughput (shots/s)

| Circuit | Ticit | SymFT | Ratio |
|---|---:|---:|---:|
| `msc_d3` | 404,296,279 | 65,784,800 | 6.146x |
| `coherent_d3_r1` | 840,294,685 | 117,406,000 | 7.157x |
| `coherent_d3_r3` | 6,722,519 | 23,523,200 | 0.286x |
| `distillation` | 16,684,094 | 17,991,700 | 0.927x |
| `msc_d5` | 7,666,943 | 4,955,690 | 1.547x |
| `pure_d7` | 4,932,287 | 4,401,790 | 1.121x |
| `pure_d9` | 2,475,549 | 2,405,850 | 1.029x |
| `coherent_d5_r1` | 1,329,293 | 1,281,670 | 1.037x |
| `coherent_d5_r5` | 91.177 | 54.013 | 1.688x |
| **Geometric mean** |  |  | **1.491x** |

Ticit wins seven of the nine retained SOFT rows. The corrected output path and
postselection policy materially reduce several old headline rates: for
example, `pure_d9` is 2.48 M rather than 8.91 M shots/s, and
`coherent_d5_r5` is 91.2 rather than 274 shots/s. This rerun changes both
implementations and the detector policy, so those differences should not be
attributed to one patch in isolation.

## CCZ throughput (shots/s, 65,536-shot batch)

| Circuit | Ticit | SymFT | Ratio |
|---|---:|---:|---:|
| `d05_p0` | 4,568,311 | 186,627 | 24.478x |
| `d05_p1e-3` | 3,794,830 | 23,266.6 | 163.102x |
| `d07_p0` | 3,796,457 | 46,838.8 | 81.054x |
| `d07_p1e-3` | 3,029,576 | 1,731.61 | 1,749.57x |
| `d09_p0` | 2,788,894 | 12,038.8 | 231.659x |
| `d09_p1e-3` | 2,140,110 | 10,324.6 | 207.283x |
| `d11_p0` | 2,068,917 | 5,308.27 | 389.754x |
| `d11_p1e-3` | 1,234,195 | 4,750.17 | 259.821x |
| **Geometric mean** |  |  | **201.826x** |

The CCZ ratios remain large for sampling-only throughput. They do not imply
lower one-off latency: both tools pay planning costs, Ticit additionally pays
cuTile JIT, and SymFT's slowest sampling rows take seconds per batch.
