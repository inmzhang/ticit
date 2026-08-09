# Differential validation: ticit vs C++ symft

The historical results below used temporary C++ and Rust dump commands to
compare packed records and expectation values shot by shot. Those commands
have since been removed; their raw-record hooks remain crate-private and are
exercised by unit tests.

## The oracle must be the sequential-scalar C++ build

Comparing against the stock benchmark build (`build/`,
`SYMFT_CPP_NATIVE=ON`) diverged on `msc_d3` at shot 2. Root-causing that
produced a chain of findings:

1. Not a draw-order bug: both engines consumed SplitMix64 identically on
   every minimal probe (Bernoulli, rare-group, categorical, branch).
2. Prefix-bisection localized the divergence to a six-product `MPP`
   line; instruction-level stepping then showed the first amplitude
   difference at a non-diagonal measurement *projection* — one ulp, in
   one amplitude (`b3`), real part only.
3. The cause is the oracle's **runtime SIMD dispatch**: even with
   `SYMFT_CPP_NATIVE=OFF` and `-O1 -ffp-contract=off
   -fno-tree-vectorize`, the AVX2/AVX512 kernel TUs are still compiled
   with `-mavx2`/`-mavx512f` and `dispatch_table()` picks them at
   runtime, so nondiagonal measure/project run FMA-fused arithmetic.
   Only `-DSYMFT_CPP_ENABLE_AVX2=OFF -DSYMFT_CPP_ENABLE_AVX512=OFF`
   yields the scalar table. That build (`SOFT/build-scalar/`) matches
   ticit bit-for-bit at the divergence site.
4. Why one ulp cascades: a deterministic measurement's probability that
   is mathematically 0 can compute to exactly `0.0` on one arithmetic
   and `~1e-33` on another. `sample_bernoulli` consumes **no draw** at
   `p <= 0` and one draw otherwise, so the RNG streams desynchronize
   and every subsequent shot differs. This is not ticit-specific: the
   C++ native build and the C++ strict build diverge from **each
   other** the same way (observed at shot 2 of the same trace). The
   fast build is not a bit-exactness target for anyone, including
   itself across compilers.

Consequently the historical validation pinned the oracle to `build-scalar`
(`SYMFT_CPP_NATIVE=OFF`, AVX TUs off, `-O1
-ffp-contract=off -fno-tree-vectorize`): sequential source-order
IEEE semantics, which is exactly what ticit's scalar kernels implement.
Statistical agreement with the fast build follows from bit-exact
agreement with the scalar build, because the draw-boundary events are
measure-zero in distribution (they only reorder rounding noise around
p ∈ {0, 1} decisions).

When ticit later grows SIMD kernels, the same reasoning applies in
reverse: ticit's own SIMD path will be validated statistically plus
scalar-vs-SIMD kernel-level, not by cross-engine bit-exactness.

## Matrix

Oracle: `build-scalar`. Shots 256 (ccz d09: 64, d11: 16 — planning
there is minutes per run; bit-exactness at 16 shots proves the same
property). Seeds 1, 7, 20260808 (d11: seed 1 only). ccz files also
compared with `--expectations` (raw f64 bits).

Fresh exhaustive rerun on 2026-08-09 at sampling-core commit `0626249`:
**80/80 matched, 0 failed** (61 single-shot plus 19 standalone batch cases).

Current scalar single-shot result: **61/61 cases match bit-exactly**.

| Suite | Exact matches | Remaining |
|---|---:|---:|
| 11 SOFT `.stim` circuits × 3 seeds | 33 | 0 |
| ccz d05/d07/d09 records + expectations | 24 | 0 |
| ccz d11_p0 records + expectations | 2 | 0 |
| ccz d11_p1e-3 records + expectations | 2 | 0 |

The standalone batch matrix is **19/19**: 128 record shots for all 11 SOFT
`.stim` circuits, 64 record+expectation shots for ccz d05/d07/d09, and 16 for
ccz d11. Every packed record word and raw expectation bit pattern matches.

After the one-/two-word residual-XOR fast paths were added, the affected batch
shapes were rechecked against saved C++ oracle output: `d05_p0` and
`d05_p1e-3` (16 record+expectation shots), plus `msc_d3` and
`pure_surface_d7` (128 record shots), all remain bit-exact.

The full Rust gate is 255/255 under `cargo nextest run`, with clean Clippy.
After compacting the planned-instruction layout, the scalar d05 batch output
was also rechecked byte-for-byte against the saved C++ oracle.
The BMI2 postselection path also matched native SymFT's aggregate result over
30 million `msc_d3` shots exactly: 9,400,620 discarded, 20,599,380 accepted,
and 24 logical errors.
The same source also passes `cargo test --all-targets` on `riling` using the
vendored `testdata/circuits/` corpus; this cross-host run caught and closed the
original hard-coded-path portability bug.
The server gate was rerun successfully after the residual-XOR fast paths.

## Planning cross-checks observed along the way

- `msc_d3` prefix (189 lines): identical `max_k`, instruction count,
  `nsymbols` (1227), exogenous sampling plan (2 rare categorical
  groups of 396 and 81 sets, one 41-condition low-probability group),
  packed exogenous bit-planes, and presampled-expression blocks
  (58 expressions) — all bit-identical before the kernel-level
  root cause was found.
- `d09_p1e-3.clifft`: identical plan metadata (`max_k` 8, 84 615
  instructions, 1 054 366 symbols); planning takes ticit 37 s vs C++
  43 s in these debug-oracle builds (not a benchmark).

## The python "single == batch" contract, resolved

`python/tests/test_measurement_sampling.py::test_single_and_batch_deterministic_samples_match`
asserts `circuit.sample(batch=False)` equals `sample(batch=True)` at the
same seed. The binding routes these to exactly the standalone helpers
(`_native.cpp:404-406`: `sample_measurements` vs
`sample_measurements_batch`) — the pair that diverges on noisy circuits
by design (different per-block seeding schemes). The contract holds
because the tested circuit is `X 0 / M 0`, which consumes **zero** RNG
draws on either backend; it is a vacuous-stream contract, not a
cross-backend stream-equality guarantee. Verified empirically against
the C++ (scalar oracle build):

| circuit | single vs batch, seed 5 |
|---|---|
| `X 0 / M 0` (deterministic) | EQUAL |
| `X_ERROR(0.3) 0 / M 0` | DIFFERENT |
| `H 0 / T 0 / M 0` | DIFFERENT |

Port implication: ticit's batch path must reproduce single-shot records
only for programs that consume no randomness; on noisy circuits the
two backends are separate reproducibility domains (each must be
seed-stable against itself and bit-exact against its C++ counterpart).

## Deferred tests wired after bit-exactness

- `circuit::lowering` unit tests: the five sampled-record-value pins
  (REPEAT both-records-1, CX/CY feedback, gate-path CY).
- `sampler::single_shot` unit tests: sqrt-gate directions (all-shots
  deterministic outcomes), bit-exact `SQRT_X ≡ H S H` /
  `SQRT_X_DAG ≡ H S_DAG H` record equality at seed 37, and the
  t-gate statistical window (200 shots seed 7, ones ∈ (10, 50)).
