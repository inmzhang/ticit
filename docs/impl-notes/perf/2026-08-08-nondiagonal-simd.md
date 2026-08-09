# Nondiagonal measurement SIMD — 2026-08-08

Profiles after pair-rotation SIMD still put 7–14% of sampling cycles in
nondiagonal probability and projection. Native SymFT has explicit versions of
the same kernels, so ticit now runtime-dispatches the matching four-lane AVX2/FMA
and eight-lane AVX-512 algorithms.

## Boundary and correctness

- Fearless SIMD tokens prove the complete target-feature sets. Every vector
  load and store goes through a bounds-checked `f64x4` or `f64x8` slice; ticit
  remains `#![forbid(unsafe_code)]`.
- Pivots below the vector width use the scalar path. `TICIT_SIMD=scalar` keeps
  the differential validator bit-exact.
- The independent probability/projection oracle covers both branches and all
  eight low-mask shapes. It passes on real AVX2 and AVX-512 machines.
- SIMD FMA and horizontal reduction intentionally change last-bit rounding.
  Oracle comparisons use tolerance; the exact cross-language harness forces
  scalar arithmetic.
- LLVM emits `vzeroupper` on normal returns and bounds-panic edges.
- Gates: 250/250 under local `cargo nextest run`, clean Clippy locally and on
  the server, and `cargo test --all-targets` passes on the server.

## Local AVX2 result

Intel Core i5-14600KF P-core 0. Each result compares saved release binaries
immediately before and after these kernels. Detector postselection was off.

| Circuit | Before | After | Change |
|---|---:|---:|---:|
| CCZ `d05_p0` | 224,565 shots/s | 255,781 shots/s | **+13.9%** |
| CCZ `d05_p1e-3` | 61,001 shots/s | 65,683 shots/s | **+7.7%** |
| CCZ `d07_p0` | 106,762 shots/s | 112,378 shots/s | **+5.3%** |
| SOFT `msc_d3` | 2.705 Mshot/s | 2.809 Mshot/s | **+3.8%** |

CCZ d09/d11 were intentionally excluded from this fast iteration loop.

## EPYC AVX-512 result

Matched ABBA runs on CPU 0 used the same binary with `TICIT_SIMD=scalar` versus
normal AVX-512 dispatch:

| Circuit | Scalar | AVX-512 | Change | Native SymFT | ticit vs SymFT |
|---|---:|---:|---:|---:|---:|
| CCZ `d05_p0` | 168,813 shots/s | 195,730 shots/s | **+15.9%** | 189,131 shots/s | **+3.5%** |
| CCZ `d07_p0` | 82,036 shots/s | 88,361 shots/s | **+7.7%** | 83,777 shots/s | **+5.5%** |

The shared host varied materially on `msc_d3`: one-core ABBA blocks averaged
1.581 Mshot/s scalar and 1.849 Mshot/s with all SIMD enabled (+16.9%); 24-core
SIMD blocks ranged from 30.1 to 33.4 Mshot/s while native SymFT ranged from
38.1 to 41.5 Mshot/s. Treat those as ranges, not a stable ranking.

## Rejected follow-up

Replacing release-time pair-kernel shape checks with debug assertions reduced
`msc_d3` cycles by 3.1%, but increased long-run `d07_p0` cycles by 0.66% with
unchanged instructions, consistent with an unfavorable layout shift. The
experiment was removed rather than trading breadth for a narrow win.

## Contiguous-promotion follow-up

The next profile put 9.15% self time in dormant-state promotion. Splitting the
old and new amplitude halves before iterating proves them disjoint to LLVM and
removes its runtime alias checks; no explicit SIMD or unsafe code was needed.

Matched local binaries produced identical aggregate results. ABBA runs on
`msc_d3` averaged 2.838 Mshot/s versus 2.803 Mshot/s (+1.23%). Follow-up runs
improved `d05_p0` from 251,919 to 259,479 shots/s (+3.0%) and `d07_p0` from
109,549 to 110,857 shots/s (+1.2%). As permitted for fast iteration, CCZ
d09/d11 were left to final validation.

A second shared fast path skips parity work when the diagonal Pauli has no Z
bits besides its pivot. Matched runs improved `msc_d3` by about 0.7%, `d05_p0`
by 2.0%, and `d07_p0` by 2.4%. Two broader general-mask experiments were
removed: a runtime x86-v3/POPCNT wrapper cost d07 about 1.3%, while storing the
parity mask in packed coordinates repeatedly cost d07 1–2% despite a narrow
`msc_d3` gain.

Hoisting shot-major slice construction outside a fused rotation run removed
2.39% of `msc_d3` instructions and 0.78% of core cycles. It was also removed:
matched d05 and d07 throughput each fell about 0.4–0.5%, another repeatable
code-layout trade rather than a broad win.
