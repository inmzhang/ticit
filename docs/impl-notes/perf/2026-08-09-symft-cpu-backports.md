# SymFT CPU optimization backports — 2026-08-09

This is a source-level backport guide from ticit commit `0626249` to SymFT
commit `bd77739` (`feat/exp-val`). The percentages below are isolated ticit
measurements against the immediately preceding ticit revision; they identify
where the algorithm paid off, but are not promises of the same C++ uplift.
Current end-to-end ticit/SymFT numbers are in `docs/benchmark.md`.

## Recommended order

| Order | Change | Main target | Risk |
|---:|---|---|---|
| 1 | Index expression-plan construction | noisy-circuit compile time | low |
| 2 | Hoist geometric constants and skip dead clears | presampling/expression blocks | low |
| 3 | Store rare noise hits sparsely and fuse row evaluation | noisy-circuit sampling | medium |
| 4 | Vectorize diagonal measurement | small active states | medium |
| 5 | Keep qualifying dim-16 rotation runs in registers | msc-style circuits | high, narrow |
| 6 | Remove redundant planned-instruction fields | instruction-heavy circuits | low |
| 7 | Prove promotion halves disjoint | broad, small win | low |

Do the compile-time change first: it shortens every later noisy-circuit
profile loop without changing sampling. Retain each sampling change only
after the full benchmark matrix shows a broad win.

## 1. Index expression-plan construction

Current SymFT bottlenecks in `cpp/src/sampler/presampled_expression.cpp`:

- `intern_exogenous_partial` linearly compares every earlier partial.
- `prepare_block_expression_parent_deltas` scans every earlier expression and
  materializes a symmetric-difference vector for every candidate.

ticit commit `014d978` preserves the exact first-encounter indices and greedy
parent choice while replacing that work with:

1. Hash-based interning of `(constant, exogenous_conditions)`.
2. An inverted index from each condition to earlier expression indices.
3. Candidate search over expressions sharing at least one condition, plus the
   earlier `(true, [])` expression for constant-true children. No other parent
   can beat the root cost.
4. A bounded symmetric-difference count that stops at the current best cost;
   only the winning delta is materialized.

For C++, avoid copying every condition vector into an `unordered_map` key.
Use a hash-to-index-bucket map and collision-check against the vectors already
owned by `block_expressions`. This should keep encounter order and avoid ticit's
likely noisy-circuit RSS penalty from owned hash keys. Visit candidate indices
in ascending order and retain the strict `< best_cost` rule so ties remain
byte-identical to the current full scan.

Add a deterministic test that runs both parent selectors over generated sorted
partials and compares parent index, constant delta, and condition delta. ticit's
equivalent test is `candidate_parent_search_matches_full_scan`.

Measured evidence:

- Isolated ticit end-to-end preparation: CCZ `d05_p1e-3` 17.1 s -> 4.7 s;
  `d07_p1e-3` 131.7 s -> 13.2 s.
- Current ticit versus SymFT compile time: d05 2.7 s vs 8.2 s, d07 11.2 s
  vs 53.7 s, d09 38.5 s vs 257 s, d11 115 s vs 885 s.

## 2. Remove cheap presampling overhead

### Hoist geometric-gap denominators

`cpp/src/sampler/random.hpp::sample_geometric_gap` recomputes
`log1p(-probability)` for every hit. Add a checked helper that computes the
denominator once per group/row, then pass it through each skip loop in
`exogenous_presample.cpp`, `batch_symbols.cpp`, and
`single_shot_sampler.cpp`. Keep the division and RNG draw order unchanged.
ticit commit `be240f7` reduced fixed-workload `pure_surface_d9` cycles by 2.7%.

### Do not clear dead symbol rows in expression mode

`cpp/src/sampler/prepared_sampler.cpp` currently passes
`!options_.postselect_detectors` to `reset_batch_executor`, so the normal
prepared expression path clears the full `nsymbols * batch_words` value table
for every block. Those exogenous values live in the presampled expression
block, while residual branch rows are assigned by whole-row writes before
their assigned bit becomes visible. The assigned mask must still be cleared.

Pass `false` for both prepared expression paths. Keep the default clear for
the inline-exogenous batch APIs, whose single-bit samplers require zeroed
rows. ticit commit `cce4bee` improved `pure_surface_d9` by 8.9% and
`d05_p1e-3` by 6.0%, with small-table msc cases flat.

## 3. Sparse skip-family noise and fused expression rows

`cpp/src/sampler/exogenous_presample.cpp` currently allocates and zeros
`program.nsymbols * shot_words` dense words, including rare categorical and
low-probability Bernoulli rows with only a few hits. For `d05_p1e-3`, this is a
roughly 49 MB table per chunk; 158,060 rare plus 15,371 low-probability
conditions are sparse candidates.

Backport ticit commits `7c52eed` and `5642511` into
`cpp/src/sampler/exogenous.hpp`, `exogenous_presample.cpp`, and
`presampled_expression.cpp`:

- Mark skip-family conditions in a bitset prepared once.
- Record draw-order `(condition - 1, shot)` hits in reusable scratch.
- Counting-sort the hits into per-condition CSR offsets and shot indices.
- Apply sparse hits directly to an expression destination; XOR duplicate hits
  rather than treating them as OR.
- Keep dense condition rows on the existing word-XOR path.
- Build each destination row by copy/fill, constant mask, then dense-row XORs
  and sparse-hit XORs while it remains cache-resident.

Choose the representation before drawing so the RNG stream stays identical.
The measured ticit policy is intentionally simple:

- Tables at or below 1 MiB stay entirely dense.
- A skip group goes dense when expected hits exceed
  `condition_count * max(shot_words / 2, 1)`.
- Otherwise it uses the sparse CSR path.

Measured isolated sparse-representation changes: local `d05_p1e-3` +48.3%,
`d07_p0` +5.3%, `d05_p0` +3.3%; EPYC `d05_p1e-3` +112.4%. The fused row
kernel separately removed 3.9% of instructions and 0.9% of cycles on a fixed
`pure_surface_d9` workload.

This backport needs an exact RNG/output test for dense versus sparse sinks,
including duplicate-hit XOR semantics and a partial final shot word.

## 4. Vectorize diagonal measurement

SymFT's `diagonal_probability_contiguous` and
`project_diagonal_contiguous` in `cpp/src/sampler/contiguous_active.cpp` are
scalar gather loops even though the nondiagonal pair already dispatches to
SIMD.

ticit commit `8e42de3` uses two equivalent formulations:

- Probability: scan the full state contiguously, mask lanes where
  `parity(basis & zmask)` selects the requested branch, and accumulate norms.
- Projection for `pivot >= vector_width_log2`: load the contiguous pivot-clear
  and pivot-set candidates, select by parity mask, scale, and store compactly.

Add AVX2 and AVX-512 entries beside the existing nondiagonal kernels in
`cpp/src/simd/simd.hpp`; retain the scalar gather for unsupported shapes and
the bit-exact oracle. Probability reassociates floating-point additions, so
test it by tolerance; projection should remain element-wise bit-identical.

ticit measured +6.0% on `msc_d3` and +4.2% on postselected `msc_d3`; CCZ was
within noise.

## 5. Register-resident dim-16 rotation runs

SymFT already detects up to 32 consecutive active rotations in
`cpp/src/sampler/batch_runtime.cpp::execute_shot_major_rotation_run`, but each
step calls `rotate_contiguous_active`, reloading and storing the same shot's
state. Do not replace that scheduler. Add one specialized execution token for
the measured hot shape:

- dimension 16;
- every rotation has `uniform_imag_pairs`;
- run length greater than one.

For AVX2, hold real and imaginary state in four YMM registers each, apply the
whole run through in-register XOR permutations and FMAs, then store once.
Prepare both sign variants per step once per run, gather each shot's signs into
one integer mask, and decode the mask inside the kernel. Keep the existing
per-rotation path as the fallback for every other shape.

After the base kernel is proven, absorb immediately preceding dormant
promotion prefixes when they take dimension 2/4/8 to 16. msc_d3's actual
rotation runs have lengths 4, 6, 6, and 7; the first three are fed by `P3`,
`P1`, and `P1` promotion prefixes. All 23 rotations qualify. Pure-surface and
CCZ d05 p0 contain no qualifying runs, so a generic framework would add
complexity without coverage.

Relevant ticit commits are `9b230a7`, `728fa02`, and `ce3eb52`; the code lives
in `src/contiguous.rs` and `src/sampler/batch/runtime.rs`. The initial
register kernel improved msc_d3 3.19 -> 4.19 Mshot/s (+31.5%). Moving sign
decode into it and fusing promotions produced further measured gains; current
ticit is 5.19 Mshot/s versus SymFT 3.53 Mshot/s on the same host.

Require exact per-amplitude tests against sequential scalar rotations for all
15 nonzero four-qubit X masks, both signs, alternating sign masks, and each
promotion start dimension before running circuit-level validation.

## 6. Shrink planned instructions

`cpp/src/factored/factored.hpp` retains data already encoded in runtime
kernels:

- `ApplyPrecomputedActivePauliRotation`: redundant `pauli`, `action`, and
  `kernel_angle` beside `rotation_kernel`.
- `MeasurePrecomputedActivePauli`: redundant `pauli` beside `kernel`.

Remove those fields after changing planner/component users to read the
precomputed kernel. Add a `static_assert` on `sizeof(FactoredInstruction)` so
the layout cannot silently grow again. ticit commit `8bc67e5` cut its variant
from 368 to 256 bytes and improved `d07_p0` by 4.0-4.6%, with msc/d05 flat.

## 7. Prove promotion ranges disjoint

`promote_contiguous_active` reads/writes `[0, dim)` and writes
`[dim, 2 * dim)` through the same pointers. Express those four ranges as
disjoint to the compiler (narrow `restrict`-qualified helper or a dedicated
SIMD kernel) and verify that loop-versioning/alias checks disappear in the
assembly. ticit commit `9c346fa` used split slices and measured roughly +1-3%
across msc/d05/d07. Drop this if C++ assembly or representative benchmarks do
not move.

## Already present or not worth repeating

- SymFT already has AVX2/AVX-512 uniform pair rotations and nondiagonal
  probability/projection; ticit originally ported those algorithms from C++.
- SymFT already has the shot-major rotation-run scheduler, one/two-word
  residual XOR fast paths, BMI2 postselection compaction, and worker threads.
- A precomputed small diagonal source table and a hoisted lane-permutation
  dispatch both regressed part of ticit's representative suite and were removed.
- Do not use PGO for a retained baseline; the project requires ordinary
  source/build settings available to installed users.

## Acceptance gate

For each retained backport:

1. Preserve scalar draw order and exact output across all SOFT and CCZ
   differential cases; SIMD probability kernels may use the existing numeric
   tolerance oracle.
2. Benchmark every SOFT circuit and CCZ d05/d07/d09/d11 at p0 and p1e-3, not
   only the workload that motivated the change.
3. Use raw detector/observable parity consistently and report observable 0,
   avoiding the known SOFT harness convention mismatches.
4. Record compile time, throughput, and peak RSS. In particular, reject an
   interner that recovers time by reproducing ticit's noisy-circuit memory
   premium.
