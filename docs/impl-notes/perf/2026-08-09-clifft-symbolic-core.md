# Clifft symbolic-core patterns applied to ticit — 2026-08-09

Current ticit commit: `800c87c`. Baselines: `5e66de8` for the new AVX-512
dim-16 path and `ed2e79e` for the later dim-32 fusion. Local measurements use
one pinned P-core of an Intel Core i5-14600KF with `-Ctarget-cpu=native`;
server measurements use core 23 of `riling`'s AMD EPYC 9254 and its AVX-512
backend. PGO was not used.

## What transferred from Clifft

Clifft's recently merged symbolic sampling stack was built in explicit layers:
semantic plan foundation (#249), HIR rotation/measurement lowering (#250),
direct-Pauli state kernels (#256), a scalar executor (#258), forced record
replay (#261), an experimental frontend (#263), noisy-syndrome lifecycle
(#266), and centralized plan IDs (#268). The useful pattern is the ordering:
make the semantic plan and scalar oracle stable before adding shape-specific
execution kernels.

The two still-open follow-ups make the performance rule more concrete:

- [#283](https://github.com/unitaryfoundation/clifft/pull/283) fuses only
  constant-sign runs of at least three rotations whose X masks span at most
  two GF(2) directions. It precomposes a bounded U1/U2/U4 table and replaces
  repeated coefficient sweeps with one. Wider ranks, dynamic signs, and large
  selector spaces stay on the direct path. Its reported QV speedup is about
  3x, while non-QV guards stay neutral.
- [#284](https://github.com/unitaryfoundation/clifft/pull/284) adds a separate
  AVX-512 high-pivot U4 kernel and prepares lane-expanded sidecars only when
  AVX-512 is actually selected. The scalar descriptor remains the oracle and
  fallback; unsupported ISAs pay neither the sidecar memory nor ZMM code.

The transferable optimization is therefore not “port the U4 framework.” It is:

1. census real runs and prove how many coefficient/state visits can disappear;
2. pre-resolve shot-invariant signs, selectors, and coefficients once;
3. keep a qualifying shot's state in registers or make one chunk-resident
   sweep across the whole run;
4. dispatch narrowly and leave every other shape on the existing oracle;
5. retain only representative end-to-end wins with exact/tolerance tests.

ticit's current circuits do not justify Clifft's general GF(2)/U4 machinery:
the hot msc_d3 runs are dim-16 uniform-imaginary pair rotations, while
distillation has one five-step dim-32 diagonal run. The shortest useful
specializations target those two shapes directly.

## 1. AVX-512 dim-16 register runs — retained

Commit `ed2e79e` adds AVX-512 versions of the existing dim-16
register-resident rotation run and promotion-prefix run. AVX-512 is tried
before AVX2; unsupported shapes and ISAs retain the old path. Exact tests cover
all 15 four-qubit X masks, mixed sign masks, and every promotion start
dimension. The tests pass while `backend_name()` reports `avx512` on `riling`.

`msc_d3`, 30 million shots, ABBA on `riling`:

| Mode | Baseline rate | AVX-512 rate | Rate change | Execution change |
|---|---:|---:|---:|---:|
| raw | 3.875 Mshot/s | 3.946 Mshot/s | +1.84% | -4.88% |
| detector-postselected | 4.426 Mshot/s | 4.714 Mshot/s | +6.50% | -4.22% |

Aggregate counters matched in every pair. This is retained because the work
removed is structural (eight YMM state vectors become four ZMM vectors), and the
postselected gain is well outside the server's run-to-run noise.

## 2. msc_d5 X-basis fusion — rejected

A deliberately narrow prototype transformed a ten-step X-basis run once at
each boundary and applied its diagonal steps chunk-outer. Its direct numerical
kernel test agreed to `1e-11`, but the representative circuit result failed
both acceptance gates:

| `msc_d5`, 500k × 3 | Baseline | Prototype |
|---|---:|---:|
| sample time | 7.575 s | 24.611 s |
| throughput | 66,009 shot/s | 20,316 shot/s |
| execution time | 7.482 s | 24.516 s |
| discarded / 1.5M | 1,283,958 | 1,283,949 |

Throughput regressed 69.2%, and nine shots crossed a rounding-sensitive branch
boundary. The basis transforms and scalar work cost more than the eliminated
passes. The prototype was fully reverted; no generic X-basis fusion framework
was kept.

## 3. Five-step dim-32 diagonal fusion — retained

The baseline distillation profile (20 million shots, software `cpu-clock:u`
because hardware counters are restricted on this host) put 25.43% of samples
in `rotate_contiguous_active`. The circuit contains exactly five consecutive
constant-angle diagonal rotations at dim 32. Five independent coefficient
and state sweeps therefore accounted for almost all of that hotspot.

Commits `d4b2c1f` and `800c87c` add one fixed-shape kernel:

- parity sign bits for all 32 bases are resolved once into each five-step
  descriptor;
- AVX2 processes four amplitudes and AVX-512 eight amplitudes at a time;
- a chunk is loaded once, all five complex scalings run in registers, and the
  chunk is stored once;
- arithmetic uses separate multiply/add/subtract operations in the original
  source order, not FMA;
- the whole dim-32 executor is outlined and cold, so dim-16 and unrelated
  circuits retain their previous hot runner layout.

The exact unit test compares all 32 runtime sign masks against five sequential
rotations. `perf annotate` confirms the fixed loop is unrolled into YMM
`vmulpd`/`vaddpd`/`vsubpd`, one load/store pair per chunk, and `vzeroupper` at
the boundary. The same exact test passes through the AVX-512 dispatcher on
`riling`.

Local 20-million-shot ABBA, current outlined build:

| Metric | Baseline mean | Fused mean | Change |
|---|---:|---:|---:|
| throughput | 2.841 Mshot/s | 3.479 Mshot/s | **+22.43%** |
| total sample time | 7.039 s | 5.749 s | **-18.32%** |
| execution time | 5.508 s | 4.222 s | **-23.35%** |

The post-change profile puts the fused rotation kernel at 9.63% instead of the
old rotation path's 25.43%, a 62% reduction in sampled share. All four A/B
runs produced exactly 18,405,262 discarded, 1,594,738 accepted, and 672,704
logical errors.

Current AVX-512 ABBA on `riling`:

| Metric | Baseline mean | Fused mean | Change |
|---|---:|---:|---:|
| throughput | 2.441 Mshot/s | 2.818 Mshot/s | **+15.46%** |
| total sample time | 8.194 s | 7.097 s | **-13.39%** |
| execution time | 6.350 s | 5.276 s | **-16.92%** |

The server counters are identical to the local A/B counters for the same seed.

## Broad regression and correctness gates

The current build was checked against the `ed2e79e` native baseline on every
CCZ circuit. Aggregate counters matched exactly throughout:

| Circuit | Shots | Throughput change |
|---|---:|---:|
| `d05_p0` | 500k | +2.15% |
| `d05_p1e-3` | 200k | -0.69% |
| `d07_p0` | 300k | +1.26% |
| `d07_p1e-3` | 100k | -0.46% |
| `d09_p0` | 100k | +1.88% |
| `d09_p1e-3` | 50k | +0.22% |
| `d11_p0` | 50k | +1.40% |
| `d11_p1e-3` | 20k | +0.01% |

The non-target SOFT sweep ranged from -0.93% to +1.20%. `msc_d3` recovered
from a code-layout shift after the dim-32 executor was outlined (+0.8% in the
final 30-million-shot ABBA). `msc_d5`, whose instruction stream never
qualifies for the new kernel, was flat-to-positive under the README's
in-process-repeat method (+1.0% and +0.1% in two optimized processes versus
baseline). Every compared aggregate count matched.

Correctness gates on the retained source:

- `cargo nextest run`: 255/255 passed;
- Clippy with warnings denied: clean;
- C++ scalar-oracle differential matrix: 61/61 single-shot and 19/19 batch
  cases matched exactly (all SOFT circuits and CCZ d05/d07/d09/d11 at both
  noise levels);
- local AVX2 and remote AVX-512 five-step sign-mask tests: exact.

The full single-shot wrapper reached 60/61 before its one-hour process limit;
the remaining `d11_p1e-3 --expectations` pair was resumed directly and `cmp`
matched. The independent batch run completed normally with 19 matches and no
failures.
