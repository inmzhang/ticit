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

## PPVM 85-qubit MSD through Python

Measured on 2026-08-16 from ticit commit `b79196ac31e` and PPVM commit
`661fc66fffe1`, using the circuit from PPVM's
[magic-state-distillation example](https://queracomputing.github.io/ppvm/examples/msd/).
Ticit is **5.20x faster** than PPVM with the same scalar Python gate calls and
**2.07x faster** than PPVM's fused Python path.

| Python path | Median ms/shot | Shots/s | Speedup vs PPVM scalar |
|---|---:|---:|---:|
| PPVM scalar | 0.281 | 3,563.4 | 1.00x |
| PPVM fused | 0.112 | 8,932.6 | 2.51x |
| ticit scalar | 0.054 | 18,514.5 | **5.20x** |

Each full shot constructs the 85-qubit state, applies the five T gates and all
Clifford gates, then materializes all 85 measurements. The scalar rows execute
the same gate sequence one Python call at a time. The fused PPVM row uses target
lists, `cz_block`, and `measure_many`; ticit does not currently expose equivalent
bulk procedural methods to Python. PPVM starts each shot by forking an empty
tableau, while ticit constructs a fresh simulator.

The values are medians of seven interleaved 1,000-shot runs on P-core 0 of the
same i5-14600KF, using Python 3.12.7 and release wheels built with
`RUSTFLAGS="-C target-cpu=native"`. Imports, warm-up, and validation are outside
the timed regions. Before timing, both scalar states and PPVM's fused state had
32 live terms and matched on 275 Pauli expectations, including 20 nonzero
weight-four checks.

PPVM's separate `GeneralizedTableauSum` result is intentionally excluded: that
part of the example adds depolarizing noise, builds a mixed state once, and then
times sampling only. It is not the same full-circuit-per-shot contract as either
procedural tableau path.

Re-run with `ticit_py/benchmarks/ppvm_msd.py`; it needs only the two projects'
existing Python dependencies.
