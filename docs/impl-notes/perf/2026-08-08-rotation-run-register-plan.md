# Next target: register-resident shot-major rotation runs

Status: implemented (`perf(rotation): register-resident dim-16 rotation
runs`). Local ABBA: msc_d3 +31.5% (3.19 -> 4.19 Mshot/s), postselected
+29.2% (3.89 -> 5.03 Mshot/s), past native SymFT's 4.31 Mshot/s on this
host; rotation-free circuits within noise. EPYC server: postselected
msc_d3 2.20 -> ~2.9-3.0 Mshot/s (+31-37%) through the same AVX2 token.
A dedicated AVX-512 variant was added later in `ed2e79e`; current server
results and the subsequent dim-32 fusion are recorded in
[`2026-08-09-clifft-symbolic-core.md`](2026-08-09-clifft-symbolic-core.md).

## Evidence

After the presample session, local `msc_d3` (6M shots, P-core 0) puts 45.9%
of self cycles in `rotate_contiguous_active` and 8.9% in its caller
`execute_shot_major_rotation_run`; SymFT's remaining local lead is ~9%
(3.195 vs ~3.53 Mshot/s unpostselected).

`perf annotate` shows the kernel is not FLOP-bound: the hottest instructions
are the pair store (`vmovupd` 7.3%), the four-lane permutation dispatch
(`jmp *%rsi` 5.3%), loads, and loop control — no FMA in the top 25. The
shot-major runner already keeps each shot's state L1-resident across a run
(shot-outer / rotation-inner, `SHOT_MAJOR_ROTATION_RUN_LIMIT = 32`), so the
remaining cost is the per-rotation load → permute → FMA → store chain
through L1, including store-to-load forwarding between consecutive
rotations of the same run.

## Design

For `k <= 3` (dim <= 8), keep the whole shot state in registers across the
run: `[f64x4; 2]` re + `[f64x4; 2]` im (4 ymm state registers, leaving 12
for coefficients/permutations), apply every rotation of the run in
registers, store once per shot. Expected effect: removes ~2 load/store
pairs per rotation per shot plus the forwarding stalls; per-rotation work
drops to permute + FMA.

Per rotation class inside the register loop:

- `uniform_imag_pairs`: in-register `permute4x64` (mask-dependent, the
  existing `permute_xor4!` cases) + one FMA pair; cross-128-bit pairs for
  xmask bit 2 pair via `vperm2f128`-equivalent shuffles on the two state
  vectors.
- diagonal: per-basis coefficient vectors; precompute both sign variants
  per rotation as `[f64x4; 2]` pairs at run start (or plan time).
- general pairs: scalar fallback — flush registers to memory, run the
  existing path, reload (or exclude such rotations from register runs at
  run-detection time).

Sign per (rotation, shot) is already materialized in
`rotation_run_sign_words`; in-register it selects a negated `q` broadcast.

Measured (static instruction-stream census):

- msc_d3: 23 rotations, all `k = 4` (dim 16), all `uniform_imag_pairs`,
  in runs of length 4, 6, 6, 7. No diagonal or general rotations in runs.
- pure_surface_d9 and d05_p0: zero planned rotation instructions — the
  change cannot touch their performance, so the blast radius is msc-shaped
  circuits only.

dim 16 kills the naive 8-ymm tile, but a vector-pair schedule fits AVX2:
state = 4 re + 4 im ymm registers; for `xmask < 4` each vector pairs with
itself (in-vector `permute4x64`, 12 registers live); for `xmask >= 4`
vectors pair as `b ^ (xmask >> 2)` with lane permute `xmask & 3` (4 temps,
14 registers live). Coefficients `c` and the sign-selected `q` broadcast
once per rotation. AVX-512 gets the roomy variant (4 + 4 zmm).

## Risks

This kernel has eaten five layout-fragile micro-optimizations (see the
rejected-experiments notes): any change here must be ABBA'd across the full
suite, and `#[inline(never)]` boundaries should be considered from the
start. The register-run variant should be a separate dispatch alongside the
existing per-rotation path, selected only for qualifying runs, so the
fallback shape stays identical.

## Follow-up: diagonal measurement pair vectorized

With rotations register-resident, postselected msc_d3's next scalar
consumers were `diagonal_probability_contiguous` (11.7%) and
`project_diagonal_contiguous` (8.7%), both `diagonal_source` gathers.
`diagonal_source` selects exactly the bases with
`parity(b & zmask) == diagonal_phase_bit ^ branch`, so probability became a
parity-masked AVX2 norm over the full state (tolerance-class reassociation,
scalar oracle retained under `TICIT_SIMD=scalar`; a 5M-shot postselected run
moved by one discarded shot with identical logical errors) and projection
became two contiguous loads plus a parity blend per output block for
`pivot >= 2` — bit-identical per element, pinned by an exact unit test over
every four-qubit Z mask.

ABBA: msc_d3 +6.0% (4.12 -> 4.36 Mshot/s), postselected +4.2%
(5.00 -> 5.21 Mshot/s); CCZ within the usual band (d05_p0 +0.6% cycles).
Local session total: msc_d3 2.96 -> 4.36 Mshot/s (+47%), postselected
3.67 -> 5.21 Mshot/s (+42%), about 21% past native SymFT postselected.

## Follow-up: sign decoding moved into the register kernel

`rotate_register_run_shot` was rebuilding a resolved step array per shot
(10.2% of postselected cycles). The run table now carries both sign
variants and is built once per run; each shot gathers its rotation sign
bits into one u32 the kernel decodes while broadcasting. Bit-identical
(counts unchanged; the exact-equality test covers both sign selections and
an alternating mask). Interleaved runs: postselected msc_d3 +7.6%
(5.20 -> 5.59 Mshot/s), unpostselected +9.8% (4.36 -> 4.79 Mshot/s);
rotation-free circuits 0 to +1.8%.

Post-change postselected profile: rotation kernel+driver 33.6%, diagonal
measurement pair 19.7%, branch-measurement glue 10.4% (hot instructions:
per-shot invnorm divide, RNG, branch compare — largely irreducible per-shot
scalar work). Local standing versus native SymFT on msc_d3 postselected:
5.59 vs 4.31 Mshot/s, ticit ahead by about 30%.

## Follow-up: promotion prefixes fused into the register run

An instruction-stream census of msc_d3 showed every rotation run is fed by
a dormant-promotion prefix: `P3 R4` (dims 2 -> 16, then four rotations) and
`P1 R6` twice (dim 8 -> 16, then six). Each promotion was a separate
whole-batch pass (`promote_contiguous_active` 6.0% + its driver 3.0% of
postselected cycles) re-walking every shot's state through L1. The run
executor now absorbs the prefix: one shot's pre-promotion state (2/4/8
amplitudes) is loaded once, promoted in registers (element-wise
`new_re = -q * old_im`, `new_im = q * old_re`, `old *= c` — exactly the
scalar promotion's products, so bit-identical), rotated, and stored once at
dim 16. Promotion sign bits ride in the same per-shot mask word as the
rotation signs. An exact-equality test covers all three start dims, every
promotion sign combination, and mixed rotation masks; the full-count matrix
is unchanged.

Quiet-machine interleaved medians (3 rounds): msc_d3 +8.7%
(4.78 -> 5.19 Mshot/s), postselected +5.9% (5.60 -> 5.93 Mshot/s, cycles
-5.3%).

Collateral worth recording: d05_p0 -1.9% and d07_p0 -1.5% (rate), stable
across five builds. This is code layout, not the new code running — a
promotion census shows every d05_p0 promotion precedes a measurement-branch
introduction, never a rotation, so the fusion never engages there; the
profile deltas smear across functions in *unchanged* files
(`xor_symbolic_bool_batch_into` +0.6pp). `#[inline(never)]`, `#[cold]`
(`.text.unlikely`), and source-order moves were all tried; the smear
persisted through every arrangement, which puts it inside this codebase's
demonstrated +/-2% layout lottery on those circuits (the rejected planner
experiment swung d07_p0 by -10% from dead code). Accepted: the msc gain is
algorithmic and durable, the CCZ p0 cost is layout noise that any future
commit can flip.

Local standing versus native SymFT on msc_d3 postselected: 5.93 vs
4.31 Mshot/s, ticit ahead by about 38%.
