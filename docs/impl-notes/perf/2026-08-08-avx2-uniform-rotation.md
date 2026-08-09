# Runtime SIMD uniform pair rotations — 2026-08-08

The scalar `msc_d3` profile put 55.0% of ticit's cycles in uniform imaginary
pair rotations. Native SymFT uses AVX2/FMA for the same small-state path, so
this was the first SIMD target after multithreading was complete.

## Boundary and correctness

- `fearless_simd` 0.6 detects the CPU once and supplies the AVX2/FMA feature
  token.
- The kernel loads and stores through bounds-checked `f64x4` slice vectors;
  ticit retains `#![forbid(unsafe_code)]`. Fearless SIMD owns the small audited
  target-feature boundary.
- One test compares all 15 nonzero four-qubit X masks (every pair-bit and
  four-lane XOR permutation) against the scalar kernel within `1e-14`.
- `TICIT_SIMD=scalar` preserves the bit-exact scalar oracle; the differential
  harness sets it explicitly. Normal execution reports `simd_backend avx2`.
- Full gate: 250/250 under `cargo nextest run`; clean Clippy.

## Local result

Intel Core i5-14600KF P-core 0, release with debug symbols, validator stopped:

| Circuit | Before | After | Change |
|---|---:|---:|---:|
| SOFT `msc_d3` | 2,174,323 shots/s | 2,726,218 shots/s | **+25.4%** |
| CCZ `d05_p0` | 223,948 shots/s | 226,298 shots/s | +1.0% |
| CCZ `d05_p1e-3` | 60,793 shots/s | 63,838 shots/s | +5.0% |

Matched 5,000,000-shot `perf stat -r 3` runs on `msc_d3`:

| Counter | Scalar | AVX2 | Change |
|---|---:|---:|---:|
| Core cycles | 12.17 B | 9.46 B | -22.3% |
| Instructions | 69.76 B | 50.58 B | -27.5% |
| Branches | 9.49 B | 7.82 B | -17.6% |

The post-change call graph attributes 33.0% of cycles to the AVX2 kernel and
13.3% to its caller/dispatch path. The next scalar consumers are active
measurement (11.2%), diagonal projection (9.8%), dormant promotion (8.6%),
non-diagonal projection (4.7%), and non-diagonal probability (3.6%). Native
SymFT remains faster on local `msc_d3` at 3.525 Mshot/s; the gap fell from
38.3% to 22.7%.

## Reference server AVX-512 result

The dual EPYC 9254 server exposes Fearless SIMD's Ice Lake-level AVX-512
token. The first AVX2-only revision was not useful there: a same-binary check
put `msc_d3` about 6.2% below scalar and `d05_p0` within 0.4% of scalar. The
retained revision dispatches a true eight-lane kernel before the AVX2
fallback. The 15-mask scalar oracle passes while executing that AVX-512 path,
and LLVM emits `vzeroupper` before its return and panic edges.

Matched ABBA runs used the same release binary and pinned CPUs. Each
one-core block sampled 5,000,000 shots five times; each 24-core block sampled
50,000,000 shots three times:

| `msc_d3` | Scalar | AVX-512 | Change |
|---|---:|---:|---:|
| CPU 0 | 1.175 Mshot/s | 1.525 Mshot/s | **+29.8%** |
| CPUs 0–23, 24 workers | 25.12 Mshot/s | 33.20 Mshot/s | **+32.2%** |

Native SymFT measured 38.34 Mshot/s on the same 24-core workload, so ticit's
remaining gap there is 13.4%. Server `d05_p0` was unchanged (126,945 scalar
versus 126,844 AVX-512 shots/s); `d05_p1e-3` was likewise within noise
(38,110 versus 37,884 shots/s). Those circuits spend little of their total
time in this kernel, so no circuit-specific dispatch was added.
