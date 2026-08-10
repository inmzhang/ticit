# Tableau simulator vs sampler on MSC d3/d5

Measured on 2026-08-10 from ticit commit `8879ee29e37` with the tableau
circuit-replay changes described here applied.

## Result

On one pinned P-core, the prepared CPU sampler is **46.55x faster** than
`TableauSimulator` on MSC d3 and **27.80x faster** on MSC d5.

Both engines run the same original noisy
`testdata/circuits/soft/msc_d{3,5}_inject_cultivate_p1e-3.stim` inputs. The
tableau replay path now handles their Pauli channels and measurement-readout
noise directly; no instructions or probabilities are rewritten.

| Circuit | Tableau shots | Tableau median | Tableau shots/s | Sampler shots | Sampler median | Sampler shots/s | Sampler / tableau |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSC d3 | 700,000 | 6.310 s | 110,935 | 30,000,000 | 5.810 s | 5,163,658 | **46.55x** |
| MSC d5 | 25,000 | 10.000 s | 2,500 | 1,000,000 | 14.391 s | 69,487 | **27.80x** |

Each value is the median of five sampling-only runs. Parsing, circuit lowering,
sampler preparation, instruction translation, and warm-up are outside the
timed region.

## Fastest single-core modes used

- `TableauSimulator`: `Instruction::from_circuit` translated each circuit once,
  then `apply_batch` replayed it on a fresh simulator per shot.
- `Sampler`: each circuit was compiled once and sampled with
  `sample_counts_with_seed`, the fastest aggregate-counter API. Default
  adaptive batch and chunk sizes and the runtime-selected SIMD backend were
  used with `SamplerOptions::threads = 1`.

| Circuit | Qubits | Replay instructions | Measurements/shot | Peak tableau rank |
|---|---:|---:|---:|---:|
| MSC d3 | 15 | 662 | 21 | 16 |
| MSC d5 | 42 | 4,286 | 112 | 1,024 |

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
circuits. Moving from d3 to d5 reduces tableau throughput by 44.38x and sampler
throughput by 74.31x, narrowing the sampler's lead from 46.55x to 27.80x. The
tableau peak rank rises from 16 to 1,024 over the same change.
