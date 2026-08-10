# Tableau simulator vs sampler on MSC d3/d5

Measured on 2026-08-10 at ticit commit `ace00f3479f1`.

## Result

On one pinned P-core, the prepared CPU sampler is **34.94x faster** than
`TableauSimulator` on MSC d3 and **20.09x faster** on MSC d5.

The comparison uses matched ideal versions of
`testdata/circuits/soft/msc_d{3,5}_inject_cultivate_p1e-3.stim`. Stochastic
noise instructions are removed and measurement-flip probabilities are set to
zero because `TableauSimulator` has no stochastic-noise instruction. Clifford,
T/T-dagger, reset, measurement, MPP, and classical-feedforward operations are
unchanged.

| Circuit | Tableau shots | Tableau median | Tableau shots/s | Sampler shots | Sampler median | Sampler shots/s | Sampler / tableau |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSC d3 | 2,000,000 | 11.378 s | 175,773 | 30,000,000 | 4.884 s | 6,142,069 | **34.94x** |
| MSC d5 | 30,000 | 9.841 s | 3,048 | 1,000,000 | 16.328 s | 61,244 | **20.09x** |

Each value is the median of five sampling-only runs. Parsing, circuit lowering,
sampler preparation, instruction translation, and warm-up are outside the
timed region.

For context, the original noisy p=1e-3 circuits reach these sampler rates:

| Circuit | Shots | Median | Sampler shots/s |
|---|---:|---:|---:|
| MSC d3 | 30,000,000 | 5.688 s | 5,274,408 |
| MSC d5 | 1,000,000 | 16.196 s | 61,744 |

There is no noisy-tableau ratio: adding a benchmark-only noise implementation
would not measure a mode provided by `TableauSimulator`.

## Fastest single-core modes used

- `TableauSimulator`: each circuit was translated once to `Instruction`s and
  replayed with `apply_batch` on a fresh simulator per shot.
- `Sampler`: each circuit was compiled once and sampled with
  `sample_counts_with_seed`, the fastest aggregate-counter API. Default
  adaptive batch and chunk sizes and the runtime-selected SIMD backend were
  used with `SamplerOptions::threads = 1`.

| Circuit | Qubits | Replay instructions | Measurements/shot | Peak tableau rank |
|---|---:|---:|---:|---:|
| MSC d3 | 15 | 158 | 21 | 16 |
| MSC d5 | 42 | 815 | 112 | 1024 |

The output contracts differ because the fastest modes were requested: the
tableau path materializes every `MeasureResult`, while the sampler returns
aggregate counters. These are end-to-end product gaps, not per-gate kernel
comparisons.

## Environment

- CPU: Intel Core i5-14600KF; process pinned to P-core logical CPU 0
- OS: Linux x86-64
- Compiler: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Build: Cargo `--release` with `RUSTFLAGS="-C target-cpu=native"`
- Affinity: `taskset -c 0`
- Seed: fixed per shot/run; parsing and compilation were reused

## Interpretation

The sampler's bit-packed factored execution is substantially faster on both
circuits. Moving from d3 to d5 reduces tableau throughput by 57.66x and sampler
throughput by 100.29x, narrowing the sampler's lead from 34.94x to 20.09x.
The tableau peak rank rises from 16 to 1024 over the same change.
