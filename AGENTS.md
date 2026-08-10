# ticit — Rust port of SOFT/SymFT

## Goal

Port the SymFT reference implementation at `SOFT_ROOT` (exact C++/Python simulator
for noisy, adaptive Clifford-dominated quantum circuits, Stim-style input)
to pure idiomatic Rust, then out-perform the raw C++ version through
repeated profile → optimize cycles.

## Phasing (strict order)

1. **CPU port first.** Single-core, then multithreaded, then SIMD.
   Do not start GPU work until the Rust CPU version beats C++ on the
   benchmarks in `SOFT/benchmark/`.
   The CPU public API, CLI, and benchmarks expose batch sampling only;
   single-shot sampling remains an internal differential-validation hook.
2. **GPU port second.** CUDA backend, only after CPU goal is met.

## Reference source layout (C++)

- `SOFT/cpp/src/core/` — pauli, frames, symbolic Clifford–Pauli algebra
- `SOFT/cpp/src/circuit/` — circuit repr + lowering
- `SOFT/cpp/src/factored/` — planner, factored state, pending optimizer
- `SOFT/cpp/src/frontend/` — Stim parser, prepared sampler
- `SOFT/cpp/src/sampler/` — CPU sampling (single-core + multithreaded)
- `SOFT/cpp/src/simd/` — runtime-dispatched SIMD kernels
- `SOFT/cpp/src/cuda/` — CUDA backend
- `SOFT/benchmark/` — benchmark circuits, configs, methodology (ground truth
  for perf comparison)

## Benchmarking

Benchmark over a wide variety of circuits, never a narrow subset:

- All circuits in `SOFT/benchmark/circuit/`.
- All circuits in `testdata/circuits/ccz/`
  (`d05/d07/d09/d11` × `p0/p1e-3`, `.clifft` format).
- Fast profile/optimization loops may use CCZ d05/d07 and skip d09/d11;
  final validation and benchmark reporting still cover the full matrix.

Comparison baselines:

- **symft (SOFT):** `SOFT_ROOT`
- **clifft:** `CLIFFT_ROOT`
- CPU benchmarks report all three: ticit (Rust), symft, clifft.
- GPU benchmarks compare ticit vs symft (CUDA backend).

Known issues in SOFT's benchmark harness — do not reproduce them:

- [SOFT#8](https://github.com/haoliri0/SOFT/issues/8): harness mixes output
  conventions — Stim/Clifft paths reference-normalize detectors/observables
  against a noiseless reference, Tsim/SymFT use raw parity. Affects circuits
  with nonzero noiseless reference (`coherent_d3_r3`, `coherent_d5_r5` —
  postselection rejects different shots, throughput not comparable;
  `distillation`, `msc_d7` — logical classification differs). Use one
  convention for all tools (e.g. run clifft in raw-parity mode: omit
  `expected_detectors`/`expected_observables`).
- [SOFT#9](https://github.com/haoliri0/SOFT/issues/9): inconsistent
  logical-error metric — Stim/Tsim/SymFT count observable 0, Clifft path
  reports `logical_errors` (any observable set). Matters for `distillation`
  (5 observables). Use `observable_ones[0]` everywhere. Also note: Stim
  materializes full bit-packed detector output while Clifft/SymFT return
  aggregate counters — different output-contract work, caveat it in results.

## `tableau_simulator`

`TableauSimulator` is a second, independent product of this crate: a procedural
stim-`TableauSimulator`-style Clifford+T engine (Clifford frame + sparse
amplitude map over destabilizer-coset labels), used by the `bloc` workspace as
its verification engine. It shares the crate's `PauliString` but **not** its
frame — `tableau_simulator::frame` is its own flat, const-generic-width tableau,
which measured 1.7–3.9x faster than routing it through `frames::CliffordFrame`
on Clifford streams, T-driven rank growth, and measurement.

- Keep its public surface stim-named; the Clifford+T extension (`t`, `t_pauli`,
  `ccz`, `rank`, `apply_batch`) keeps bloc's spelling.
- `tableau_simulator::batch` is the replay path: consumers that run one op
  sequence per shot build an `Instruction` stream once instead of rebuilding
  `PauliString`s per shot.
- Unit tests beside `tableau_simulator` cover rank, measurement, and replay;
  `src/tableau_simulator/frame/differential_tests.rs` pins sign conventions
  against `paulimer`, a dev-dependency that must never reach the built library.

## Toolchain choices

- Stable Rust only.
- CLI argument parsing: `clap` with derive macros, behind the default `cli`
  feature so library consumers (`default-features = false`) do not link it.
- SIMD: `fearless_simd` (or comparable stable-toolchain SIMD crate).
- GPU: `cuda-oxide` or `cutile-rs` (NVlabs).
- CPU profiling: `perf` (see linux-perf skill).
- Tests: `cargo nextest run`; property tests vs reference with `quickcheck`;
  snapshots with `insta`.

## Machines

- **Local machine:** fast iteration and `perf` profiling are allowed.
- **CPU server:** `ssh riling` — benchmarking + perf profiling.
- **GPU server:** `ssh gpucluster` — multiple RTX 4090 + H200 via **slurm**.
- Sync local ↔ remote via git remotes (push/pull directly between repos),
  not rsync/scp.

## Agent roles

- **Claude Code:** the main agent uses Fable 5 as orchestrator, designer, and
  planner; Opus 5 subagents (`model: opus`) with high thinking effort write
  implementation code.
- **Codex:** the main agent uses its selected model for both coordination and
  implementation; spawning implementation subagents is not required.

## Working rules

- Correctness first: validate against C++ outputs / benchmark circuits
  before optimizing.
- Profile before optimizing; keep numbers from `perf` runs in the repo
  (e.g. `docs/impl-notes/perf/`) so regressions are visible.
- `docs/benchmark.md` carries the three-way benchmark tables (ticit / symft / clifft:
  throughput, compile time, peak RSS) stamped with the commit they were
  measured at. Whenever retained work moves any of those numbers meaningfully,
  re-measure and update the benchmark tables in the same session — stale
  results misrepresent the project to newcomers.
- In addition to algorithms and SIMD, test obvious release-build and hot-data
  layout improvements from the [Rust Performance Book](https://nnethercote.github.io/perf-book/title-page.html),
  retaining them only when representative benchmarks show a broad win.
- Do not use PGO for retained optimizations or benchmark baselines: crates.io
  packages installed with `cargo install` cannot ship the trained profile.
  Wins must come from source or ordinary Cargo/rustc build settings available
  to those users.
- Idiomatic Rust: `unsafe` is allowed only when profiling shows a measurable
  performance win. Keep the safe–unsafe boundary narrow, document and test its
  safety invariants, audit it carefully, and remove unsafe code that buys no
  measured improvement.
