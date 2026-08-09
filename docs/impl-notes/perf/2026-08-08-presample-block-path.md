# Presample-side wins: sparse noise, expression rows, dead clears — 2026-08-08

Profiling host: Intel Core i5-14600KF, P-core 0, native build with debug
symbols, one sampler thread, validator stopped. The fast loop adds
`pure_surface_d9` because a DWARF call-graph profile showed its cycles were
dominated by the presample side, which earlier passes had never touched:
39.5% in `xor_condition_row`, 21.2% in packed exogenous resampling, and 18.9%
in libc memset/memcpy.

## Retained: slice-based `xor_condition_row`

The old loop indexed both rows word by word. LLVM emitted a guarded vector
path plus a scalar fallback, and 66% of the function's samples sat in the
scalar fallback's one-word XOR. Reslicing both rows once up front (safe
`[start..start + shot_words]` slices, then a `zip` loop) removes the
per-iteration bounds checks and the panic-ordering constraint, so the loop
compiles to clean unrolled AVX2 XOR.

Matched ABBA suite runs (baseline versus slice fix):

| Circuit | Baseline | Slice fix | Change |
|---|---:|---:|---:|
| SOFT `msc_d3` | 2.977 Mshot/s | 3.178 Mshot/s | **+6.8%** |
| SOFT `msc_d3` postselected | 3.669 Mshot/s | 3.920 Mshot/s | **+6.8%** |
| SOFT `pure_surface_d9` | 2.293 Mshot/s | 2.696 Mshot/s | **+17.6%** |
| CCZ `d05_p0` | 282.2k shots/s | 284.9k shots/s | +0.9% |
| CCZ `d05_p1e-3` | 63.9k shots/s | 70.2k shots/s | **+9.8%** |
| CCZ `d07_p0` | 122.7k shots/s | 123.8k shots/s | +0.9% |

The word-XOR order within a row is unchanged, so the scalar oracle contract
is unaffected; the full batch differential validation matrix (19 cases,
including CCZ d09/d11) stayed bit-identical to the C++ oracle.

## Retained: drop the dead per-block symbol-value clear

`reset_batch_executor` zeroed the whole `value_words` table
(`nsymbols x batch_words`) for every non-postselected block. A symbol census
showed that table is almost entirely dead in expression mode:
`pure_surface_d9` has 15,990 symbols of which 14,948 are exogenous — their
bits live in the presampled expression block, never in the runtime value
table — and `d05_p1e-3` has 191,134 symbols of which 173,439 are exogenous
(a 6 MB memset per 256-shot block).

The postselecting caller already skipped the clear ("values are dead until
reassigned"), and the same argument holds for the plain expression caller:
branch symbols are assigned only through whole-row writes
(`assign_batch_symbol`), and the single-bit OR samplers that need pre-zeroed
rows run only on the inline-exogenous path, whose callers still clear. Both
expression-mode callers now pass `clear_symbol_values = false`; the assigned
mask is still cleared unconditionally, so stale rows stay unreachable.

Matched ABBA suite runs (slice fix versus slice fix + no clear):

| Circuit | Slice fix | + no clear | Change |
|---|---:|---:|---:|
| SOFT `pure_surface_d9` | 2.890 Mshot/s | 3.146 Mshot/s | **+8.9%** |
| CCZ `d05_p1e-3` | 71.1k shots/s | 75.4k shots/s | **+6.0%** |
| CCZ `d07_p0` | 124.4k shots/s | 126.2k shots/s | +1.5% |
| CCZ `d05_p0` | 288.3k shots/s | 288.7k shots/s | +0.1% |
| SOFT `msc_d3` | 3.190 Mshot/s | 3.182 Mshot/s | -0.3% (noise) |
| SOFT `msc_d3` postselected | 3.938 Mshot/s | 3.917 Mshot/s | -0.5% (noise) |

`msc_d3` is flat because its 1,227-symbol value table is too small for the
clear to matter. The full 252-test gate and the 19-case batch differential
validation both pass after the change.

## Retained: fused block-expression evaluation

Block expressions were evaluated one condition-row XOR pass at a time. Each
row is now written in a single pass folding the parent copy, constant mask,
and condition XORs. Outlining the kernel (`#[inline(never)]`) was required:
the inlined form cost postselected `msc_d3` about 4% through code layout
alone. Fixed-workload d9 counters: -3.9% instructions, -0.9% cycles.

## Retained: plan-preparation complexity fix

`prepare_presampled_expression_plan_from_words` was quadratic in both
interning and greedy parent choice, and dominated planning on large-noise
circuits (61% of a d05_p1e-3 run). Interning now uses a hash map (indices
still in encounter order), and the parent search visits only candidates that
can beat the root cost: earlier expressions sharing a condition, plus the
`(true, [])` partial for constant-true children. A unit test pins the
selection against the retained quadratic reference; plan statistics are
byte-identical on the suite. End-to-end: d05_p1e-3 17.1s -> 4.7s,
d07_p1e-3 131.7s -> 13.2s. This also makes the p1e-3 CCZ circuits usable in
fast local iteration loops.

## Retained: geometric-gap log1p hoist

Every geometric skip paid a libm `log1p` per hit for a denominator constant
across the loop. All ten skip loops now hoist it; the division is kept so
gaps stay bit-identical. d9 fixed-workload cycles: -2.7%.

## Retained: sparse skip-family noise representation

The packed exogenous table stored every condition as a dense bit-plane even
though skip-family sampling (rare categorical + low-probability Bernoulli
groups) sets only ~2 bits per 2048-shot row at p=1e-3. Census: d9 has 13,266
of 14,948 exogenous conditions in rare groups; d05_p1e-3 has 158,060 (plus
15,371 low-group) of 173,439 — a 49 MB table re-zeroed and scattered into
every chunk, then re-read row-by-row during block evaluation.

Skip-family hits are now collected as `(condition, shot)` pairs in draw
order and counting-sorted into a per-condition CSR; block evaluation applies
them as single-bit XORs instead of full-row XORs, and the table memset
shrinks to the dense families' rows. Three deterministic representation
choices keep dense behavior where dense wins (decided *before* sampling, so
the drawn bits are provably unchanged — the RNG stream is consumed
identically either way):

- Tables at or under 1 MB (`SPARSE_MIN_TABLE_BYTES`) stay fully dense with
  one bulk clear: they are cache-resident, and per-chunk per-row bookkeeping
  at msc_d3's ~1,900 chunks/s costs more than it saves (measured +1.0% to
  +2.4% postselected msc_d3 regressions for three sparse-always variants).
- A group whose expected draw count exceeds `conditions * shot_words / 2`
  is scattered densely — per-bit application loses to one vector row XOR
  beyond about half a row of hits.
- Closure-based emit sinks were rejected: two explicit dense/sparse
  function variants measured about 1.5% better on msc_d3 postselected.

Matched ABBA suite runs (log1p hoist versus sparse representation):

| Circuit | Before | Sparse | Change |
|---|---:|---:|---:|
| CCZ `d05_p1e-3` | 81.3k shots/s | 120.6k shots/s | **+48.3%** |
| CCZ `d07_p0` | 126.0k shots/s | 132.6k shots/s | **+5.3%** |
| CCZ `d05_p0` | 286.8k shots/s | 296.2k shots/s | **+3.3%** |
| SOFT `pure_surface_d9` | 3.512 Mshot/s | 3.558 Mshot/s | +1.3% |
| SOFT `msc_d3` | 3.201 Mshot/s | 3.222 Mshot/s | +0.6% |
| SOFT `msc_d3` postselected | 3.942 Mshot/s | 3.932 Mshot/s | -0.3% (noise) |

The p0 circuits gain because their 17,695/44,185 residual-symbol rows are no
longer memset every chunk (their 8 exogenous conditions are all dense).

## Rejected: planner reduction clone elimination

Returning `Option` from `reduce_by_relation_once` (no clone or equality
compare on rejected rewrites) measured only -0.8% on a planning-heavy run,
inside noise — and the resulting code layout cost the *sampling* suite
broadly: d07_p0 -10%, d05_p1e-3 -8%, msc about -1.5%, all stable across
interleaved reruns despite the planner never running during sampling. The
commit was dropped; the allocation savings did not survive contact with
instruction layout.

## Session result

Whole-session ABBA, 3c0a9a1 baseline versus final HEAD, medians of three:

| Circuit | 3c0a9a1 | Final | Change |
|---|---:|---:|---:|
| SOFT `msc_d3` | 2.957 Mshot/s | 3.195 Mshot/s | **+8.1%** |
| SOFT `msc_d3` postselected | 3.672 Mshot/s | 3.910 Mshot/s | **+6.5%** |
| SOFT `pure_surface_d9` | 2.325 Mshot/s | 3.519 Mshot/s | **+51.3%** |
| CCZ `d05_p0` | 283.5k shots/s | 295.9k shots/s | **+4.4%** |
| CCZ `d05_p1e-3` | 64.3k shots/s | 118.7k shots/s | **+84.6%** |
| CCZ `d07_p0` | 122.8k shots/s | 131.9k shots/s | **+7.4%** |

End-to-end preprocessing additionally fell from 17.1s to 4.7s (d05_p1e-3)
and 131.7s to 13.2s (d07_p1e-3).

The dual EPYC 9254 reference server (CPU 0, native builds of both
revisions, medians of three interleaved passes) confirms the sweep — its
larger tables and slower memory amplify the sparse-representation win:

| Circuit | 3c0a9a1 | Final | Change |
|---|---:|---:|---:|
| SOFT `msc_d3` | 2.111 Mshot/s | 2.153 Mshot/s | +2.0% |
| SOFT `msc_d3` postselected | 2.554 Mshot/s | 2.652 Mshot/s | **+3.8%** |
| SOFT `pure_surface_d9` | 1.692 Mshot/s | 2.997 Mshot/s | **+77.1%** |
| CCZ `d05_p0` | 211.9k shots/s | 214.3k shots/s | +1.1% |
| CCZ `d05_p1e-3` | 44.4k shots/s | 94.3k shots/s | **+112.4%** |
| CCZ `d07_p0` | 95.4k shots/s | 97.7k shots/s | +2.4% |

## Bit-exactness evidence for the expression path

The C++ differential script's `--batch` mode exercises the *inline* batch
path, not the production expressions path, so this session also compared
full count output (shots, discarded, accepted, logical errors) between the
3c0a9a1 baseline binary and each retained change across 12 configurations:
seven SOFT circuits and three CCZ circuits, postselected and not, threaded,
and a partial-final-chunk shot count. All counts stayed identical at every
step. The single-shot validation mode does drive the packed presampler and
expression block against the C++ oracle, and the full single+batch matrix
passes after the sparse change.
