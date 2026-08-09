# SIMD Porting Spec — `SOFT/cpp/src/simd/`

(Explorer report. Derived from complete read of the seven simd/ files plus data-contract call sites: `sampler/contiguous_active.cpp`, `sampler/active_kernels.hpp`, `sampler/batch_active.cpp`, `sampler/active_state.cpp`, `cpp/CMakeLists.txt`.)

## 0. Orientation — two independent kernel tables

| | `symft::simd::KernelTable` | `symft::batch_interleaved::KernelTable` |
|---|---|---|
| Header | `simd/simd.hpp:11-64` | `simd/batch_interleaved.hpp:14-213` |
| Members | 9 fn pointers + `const char* name` | 17 fn pointers, no name |
| Backends | scalar / avx2 / avx512 runtime-dispatched | single impl, compile-time AVX2 fast paths |
| Layout | contiguous split-complex `re[basis]`, `im[basis]` | basis-major interleaved `re[basis*leading_shots + shot]` |
| Accessor | `dispatch_table()`, `scalar_table()`, `dispatch_name()` | `table()` (`batch_interleaved.cpp:1045`) |

**Third near-duplicate copy** of the AVX2 kernels as `inline` fns in `sampler/active_kernels.hpp:165-703` (`rotate_uniform_imag_pairs_soa_inline`, `rotate_real_pair_flip_soa_inline`, `rotate_general_pairs_soa_inline`) — compiled into caller under `-march=native`; **hot single-shot path uses these below a size threshold**. The simd.hpp table is NOT the only implementation.

## 1. `KernelTable` members (`simd/simd.hpp:11-64`)

### Group A — legacy/dead AoS + reduction kernels

1. `mul_assign(Complex* alpha, const Complex* coeff, double c, size_t n)` — `alpha[i] *= (c + coeff[i])`, AoS.
2. `norm_sum(const Complex* alpha, const size_t* indices, size_t n)` — `Σ |alpha[indices[i]]|²`, gather list.
3. `mul_assign_soa(double* re, double* im, const Complex* coeff, double c, size_t n)` — split-complex state, interleaved coeff.
4. `norm_sum_soa(const double* re, const double* im, const size_t* indices, size_t n)`.

**All four dead in production.** No caller outside table initializers + tests. Live equivalents: `detail::mul_assign_soa_inline` (`active_kernels.hpp:128`), `detail::norm_sum_soa_inline` (`:153`). Port last or never.

### Group B — measurement probability (nondiagonal)

5. `measure_nondiagonal_probability_soa(const double* re, const double* im, size_t dim, u64 xmask, u64 zmask, unsigned pivot, Complex coefficient1_even, bool branch) -> double` — Σ over `dim/2` output states of |amp|² after projecting onto `branch` eigenspace. Read-only. Called `contiguous_active.cpp:117` with `dim = out_dim << 1`.

### Group C — projection

6. `project_nondiagonal_soa(..., double* out_re, double* out_im, ..., double invnorm)` — writes `dim/2` normalized post-measurement amplitudes. Called `contiguous_active.cpp:153`; caller `copy_n`s scratch back (`:165-166`).

### Group D — pair rotations (`exp(-iφP)`, non-diagonal P)

7. `rotate_uniform_imag_pairs_soa(re, im, dim, xmask, pair_bit, c, q)` — 2×2 on every `(i, i^xmask)` pair, single scalar imaginary coeff `iq` (case zmask == 0).
8. `rotate_real_pair_flip_soa(re, im, dim, xmask, pair_bit, const double* phase_signs, c, base_coeff)` — real per-pair `q = phase_signs[pair]*base_coeff`, antisymmetric.
9. `rotate_general_pairs_soa(re, im, dim, xmask, pair_bit, const Complex* left_coeff, const Complex* right_coeff, c)` — independent complex per-pair coeffs.

**Only (7) reachable from production through the table, and only when `pair_count >= kSimdPairRotationThreshold = 16384`** (`active_internal.hpp:13`, `contiguous_active.cpp:34-52`); below → header-inline copy. (8),(9) have **no production caller via table** — general case is scalar loop `contiguous_active.cpp:56-76`.

## 2. Dispatch mechanism

`simd_dispatch.cpp` (46 lines). `__builtin_cpu_supports` (GCC/Clang), no env override, no ifunc. Order: avx512f&&avx512dq → avx512_table (if SYMFT_COMPILED_AVX512); avx2&&fma → avx2_table; else scalar. Plain aggregate struct of fn pointers, one `const KernelTable` per backend TU (`simd_scalar.cpp:215-226`, `simd_avx2.cpp:969-980`, `simd_avx512.cpp:815-826`). Function-local static reference = thread-safe once-init (`:37-40`). `dispatch_name()` → exact strings `"scalar"`/`"avx2"`/`"avx512"`, surfaced via `active_simd_backend()` (`active_state.cpp:207-209`) and `active_batch_backend()` (`batch_runtime.cpp:2097`).

Build gating (`cpp/CMakeLists.txt:88-105`): AVX2 TU `-mavx2 -mfma` + `SYMFT_COMPILED_AVX2=1`; AVX512 TU `-mavx512f -mavx512dq -mfma` + `SYMFT_COMPILED_AVX512=1`.

⚠️ **Build hazard, don't replicate**: `SYMFT_CPP_NATIVE` (ON) adds `-march=native` to *every* TU incl. simd_scalar.cpp — (a) "scalar" reference compiled with FMA+autovectorization, not strict-IEEE; (b) non-intrinsic loops inside simd_avx2.cpp may codegen AVX-512 on a 512 host → faults on AVX2-only CPU. `batch_interleaved.cpp` fast paths gated on `__AVX2__ && __FMA__` compile-time — vanish silently on non-native builds.

## 3. Data-layout assumptions

### 3.1 Contiguous split-complex (groups B/C/D)

- `re[]`/`im[]` independent double arrays by basis. `coeff*` are interleaved `Complex*`, reinterpret-cast at entry.
- **Alignment: none required** — all `loadu`/`storeu`. Only aligned intrinsics on stack temporaries. Backing = `std::vector<double>`.
- **`dim` always power of two** = 2^k, k < 63.
- **`pair_bit`/`pivot` = HIGHEST set bit of xmask** (`active_state.cpp:64,72-76`). Load-bearing: `selector = 1<<pair_bit` is top X bit, `lower_mask = xmask & (selector-1)` only below-pivot bits, `i ^ xmask` for low-half i lands in high half of same `2*selector` block. Entire blocked/permute addressing depends on this.
- Vector-width multiples: AVX2 partner-permute loops `offset += 4` **no tail** (legal since `selector >= 4` under `pair_bit >= 2`); AVX512 `+= 8` under `pair_bit >= 3`. `pair_bit == 1` fast paths loop `basis + width <= dim` no tail (power-of-two dim).
- Aliasing: rotates strictly in-place (all four pair values read before writes); measure read-only; project has distinct out scratch (in-place would be safe but unused — model as disjoint `&mut [f64]`).
- Tails only in pure-scalar fallback loops and `selector==1 && xmask==1` AVX2 paths.

### 3.2 Reachability map

`pair_bit == 0` ⟹ xmask == 1 ⟹ case A. `pair_bit == 1` ⟹ xmask ∈ {2,3} ⟹ case B. `pair_bit >= 2` ⟹ AVX2 case C (partner-permute); AVX512 case C needs `pair_bit >= 3`.

⟹ AVX2 gather/scatter + `xor_contiguous_run` fallback block **unreachable for dim >= 4**. AVX512 `pair_bit == 2` falls to that block where vector sub-loops do zero iterations → pure scalar. `avx512_measure/project` bail to scalar for `pivot < 3` vs AVX2 `pivot < 2` — **AVX-512 machine slower than AVX2 at pivot == 2. Do not reproduce**: single 4-wide fallback under the 8-wide path.

## 4. Scalar reference semantics (`simd_scalar.cpp`) — normative

```
insert_zero_bit(packed, bit):
    low_mask = (1 << bit) - 1
    return (packed & low_mask) | ((packed & ~low_mask) << 1)
```

`inv_sqrt2 = 0.707106781186547524400844362104849039` (`:75,105`); `batch_interleaved.cpp:15` uses shorter literal — same f64.

### 4.5 `scalar_measure_nondiagonal_probability_soa` (`:66-91`)
```
out_dim = dim >> 1;  probability = 0
for idx in 0..out_dim:
    source0 = insert_zero_bit(idx, pivot)
    source1 = source0 ^ xmask
    odd     = popcount(source0 & zmask) & 1 != 0
    direction = (branch != odd) ? -1.0 : +1.0
    c1r = direction * coefficient1_even.re; c1i = direction * coefficient1_even.im
    ar = inv_sqrt2*re[source0] + c1r*re[source1] - c1i*im[source1]
    ai = inv_sqrt2*im[source0] + c1r*im[source1] + c1i*re[source1]
    probability += ar*ar + ai*ai
```
Left-to-right, five separately-rounded ops (no FMA). `coefficient1_even = conj(even_phase) * inv_sqrt2` (`active_state.cpp:94`).

### 4.6 `scalar_project_nondiagonal_soa` (`:93-120`)
Same body; writes `out_re[idx] = ar*invnorm; out_im[idx] = ai*invnorm`. Compacts 2^k → 2^(k-1).

### 4.7 `scalar_rotate_uniform_imag_pairs_soa` (`:122-147`)
```
selector = 1 << pair_bit;  step = selector << 1
for block step_by step: for offset in 0..selector:
    i0 = block + offset;  i1 = i0 ^ xmask
    (read all four first)
    re[i0] = c*r0 - q*im1;  im[i0] = c*im0 + q*r1
    re[i1] = c*r1 - q*im0;  im[i1] = c*im1 + q*r0
```
2×2 unitary [[c, iq],[iq, c]].

### 4.8 `scalar_rotate_real_pair_flip_soa` (`:149-177`)
Same walk + linear `pair_idx` into `phase_signs` (block/offset order = insert_zero_bit order).
```
q = phase_signs[pair_idx++] * base_coeff
re[i0] = c*r0 - q*r1;  im[i0] = c*im0 - q*im1
re[i1] = c*r1 + q*r0;  im[i1] = c*im1 + q*im0
```

### 4.9 `scalar_rotate_general_pairs_soa` (`:179-213`)
```
lr,li = left[2p], left[2p+1];  rr,ri = right[2p], right[2p+1]
re[i0] = c*r0  + rr*r1  - ri*im1;  im[i0] = c*im0 + rr*im1 + ri*r1
re[i1] = c*r1  + lr*r0  - li*im0;  im[i1] = c*im1 + lr*im0 + li*r0
```

Also: `scalar_mul_assign` uses `std::complex::operator*=` → C99 Annex-G NaN/Inf recovery (num_complex is naive-only; observable only for non-finite).

## 5. AVX2 deltas

- `norm_sum`/`norm_sum_soa` NOT specialized (scalar loop + pragma).
- measure/project: `pivot < 2` → scalar_table; else dispatch on `xmask & 3` into 4 template instantiations.
- **Central trick — partner permute** (`avx2_*_partner_permute_soa<LaneMask>`, `:102-214`): split `lower_mask` into `chunk_mask = lower_mask & ~3` (applied to address) + `LaneMask = lower_mask & 3` (register permute). `i1 = block + selector + (offset ^ chunk_mask)` one contiguous 4-wide load; corrected by `permute_lanes_xor2_const<LaneMask>` (involution, reapplied before store): 0→identity, 1→`permute_pd 0b0101`, 2→`permute4x64 0x4e`, 3→`permute4x64 0x1b`. **Replaces gather/scatter — the single most important idea to port.**
- Small-pair_bit fast paths: `selector==1&&xmask==1` in-register `permute_pd` swap (scalar tail `basis += 2`); `pair_bit==1` `0x4e`/`0x1b` (no tail). Sign-patterned coefficient vectors fold antisymmetry (e.g. `set_pd(q1,-q1,q0,-q0)`), single fmadd covers both pair members.
- Parity vectorized: `nondiagonal_coefficient_sign_mask4` (`:254-276`) — 6-step XOR-fold (shifts 32,16,8,4,2,1), bit0, XOR branch, `slli 63` → sign mask applied by `xor_pd` (flips 0.0 → -0.0; inert here).
- ⚠️ **FMA order-of-operations divergence**: `ar = fnmadd(c1i, im1, fmadd(c1r, r1, mul(c0, r0)))` — products never rounded vs scalar's 5 roundings. Genuine numerical divergence. Rotate kernels also reassociate the 3-term general case.
- Horizontal: `(p0+p2) + (p1+p3)` via 128-add + hadd; differs from scalar sequential ~1e-16 rel. Test tolerance 1e-12 (`symft_tests.cpp:1128`).
- Scatter emulated via aligned stack buffer (AVX2 has none); gather/scatter live only in unreachable fallback — **skip in port**.
- No masked ops.

## 6. AVX512 deltas

- Regressions: `mul_assign`/`mul_assign_soa` plain scalar (AVX2 vectorizes both).
- Higher thresholds: measure/project scalar for `pivot < 3`; rotates need `pair_bit >= 3`. `pivot/pair_bit == 2` slower than AVX2 (§3.2).
- `permute_lanes_xor3(lanes, mask)` (`:241-268`): runtime switch over `mask & 7`, eight `_mm512_permutexvar_pd` index vectors (`j → j ^ mask`). Not templated.
- Real gather AND scatter: `_mm512_i64gather_pd(vindex, base, 8)` (arg order reversed vs AVX2!), `_mm512_i64scatter_pd`.
- Coeff de-interleave via strided gathers (`complex_indices8`, `load_complex_re8/im8`) — slower than AVX2's unpack+permute; **port the shuffle version at both widths**.
- Horizontal: `_mm512_reduce_add_pd` (tree) — third distinct summation order.
- No mask registers used, no vpconflict, no compress/expand. No AVX512-only kernel.

## 7. `batch_interleaved` (basis-major)

Layout: `active[basis * leading_shots + shot]`, split-complex. `leading_shots = active_pitch` (padded to multiple of 4, `padded_batch_active_pitch`, `batch_internal.hpp:57-63`; capacity ≤ 2 exempt). Live lanes = `active_shots ≤ leading_shots`.

AVX2 helpers **pitch2 only** (`leading_shots == 2 && active_shots == 2`), compile-time `__AVX2__ && __FMA__` gate: `[b0s0, b0s1, b1s0, b1s1]` lanes; `maybe_swap_pitch2_basis_rows` = `permute4x64 0x4e` when `high0 > high1`.

**Live/dead census (17 members): only 4 live** — `rotate_uniform_imag_pairs` (`batch_active.cpp:205`, pair_count < `kXmaskRotationPairThreshold = 64` && pitch ≠ 2), `rotate_uniform_imag_xmask` (`:217`), `rotate_uniform_imag_xmask_const` (`:187`, uniform sign), `promote_first_dormant_rotation` (`:369`). **13 of 17 dead — do not port.**

Distinctive techniques worth porting:
- **64-shot word blocking on sign/branch bits**: per word `bits = sign_bits[word] & live`; 3-way: `bits == 0` (all-minus), `bits == live` (all-plus), else per-lane select. Batch branch-free predication — big win.
- Branchless `select_double_bits(f, t, mask)` via bit_cast: `(~mask & f) ^ (mask & t)`, `mask = 0 - bit` → Rust `f64::from_bits`.
- `low_bits_mask(nbits)` explicit `<< 64` UB guard → Rust `checked_shl`/branch.

Numerical notes: pitch2 measure accumulates imaginary term first (opposite of scalar); pitch2 project fuses where scalar doesn't.

## 8. Numerical-reproducibility hazards (ranked)

1. **Reduction order in measure_nondiagonal**: scalar sequential vs AVX2 4-acc `(p0+p2)+(p1+p3)` vs AVX512 8-acc tree — three different answers. Tolerance 1e-12 abs. Rust port: pick one canonical order per width or accept tolerance.
2. **FMA contraction/reassociation**: AVX2/512 fuse explicitly; Rust `a*b+c` never auto-contracts, so Rust scalar = strict IEEE ≠ C++ scalar built `-march=native`. **Establish which reference you match before writing tests.**
3. **`-march=native` on scalar TU**: C++ "scalar" backend not strict-IEEE; golden values host-dependent.
4. Sign by XOR vs multiply: equivalent incl. -0.0; note if replaced with branch.
5. Two inv_sqrt2 literals — same f64; don't retype.
6. std::complex NaN/Inf recovery in scalar_mul_assign only.
7. No rsqrt/rcp approximations; no transcendentals in kernels (sin/cos evaluated once in kernel ctors: `active_state.cpp:55-56`, `batch_active.cpp:349-350`).
8. `std::clamp(probability, 0.0, 1.0)` applied by CALLERS (`contiguous_active.cpp:108,126`), not kernels — preserve boundary.

## 9. Recommended Rust mapping

| Group | Recommendation |
|---|---|
| A (legacy) | plain scalar loops, or skip |
| B/C (measure/project nondiagonal) | portable-SIMD width-generic f64x4/f64x8; chunk_mask + lane-XOR-permute via swizzle; XOR-fold parity portable. **Always table-dispatched, highest perf weight** |
| D `rotate_uniform_imag_pairs` | portable SIMD; **collapse the inline/table duplicate copies and delete the 16384 threshold** (no TU boundary in Rust) |
| D flip/general | scalar first; the open-coded scalar general-pair loop in `contiguous_active.cpp:56-76` is the real optimization target |
| E batch live 4 | autovectorizable scalar loops over shot axis (unit-stride, ≥64 lanes); only pitch2 needs hand-SIMD (niche) |
| F batch dead 13 | do not port |
| G dispatch | `OnceLock<&'static Backend>` + `is_x86_feature_detected!`; **enum dispatch at outer call boundary** (lets LLVM specialize whole loop, fn-pointer table prevents). Preserve exact strings "scalar"/"avx2"/"avx512" |

`core::arch` warranted only for: runtime-mask lane swizzle (or match over const-generic swizzles), maybe `_mm256_permute4x64_pd` if match not hoisted. Skip gathers/scatters entirely.

Perf weight ranking (call structure): 1. `measure_nondiagonal_probability_soa` 2. `project_nondiagonal_soa` (+ caller's copy_n — double-buffer instead in Rust) 3. `rotate_uniform_imag_pairs_soa` + inline twin 4. **open-coded scalar general-pair rotation (`contiguous_active.cpp:56-76`) — largest un-taken optimization in C++, most promising place to beat it** 5. batch `rotate_uniform_imag_xmask{,_const}`.

Structural advice: fix AVX512 threshold regression (one path parameterized by LANES ∈ {4,8}, scalar fallback at `selector < LANES`); encode `pair_bit = highest_set_bit(xmask)` invariant in a newtype; keep the 64-shot word 3-way branch; quickcheck property: every backend agrees with scalar to 1e-12 over random `(k, xmask, zmask, pivot, branch, state)` (C++ tests only one d5 fixture: xmask 0x203, zmask 0x200, k 10).
