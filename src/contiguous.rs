//! Dense split-complex kernels: the code the samplers actually spend their time
//! in.
//!
//! # Layout
//!
//! State is a structure of arrays — `re[basis]` and `im[basis]` in two separate
//! `f64` slices, both of length `2^k`. This is what lets a vectorized backend
//! load four or eight consecutive amplitudes without deinterleaving.
//!
//! # Evaluation order is part of the contract
//!
//! Every expression below has a fixed evaluation order, with each product
//! separately rounded. Reassociating or fusing (`mul_add`) changes results in
//! the last few ulps, so SIMD backends are validated against this scalar code.
//! Do not "simplify" the arithmetic.
//!
//! # SIMD dispatch
//!
//! [`fearless_simd::Level`] selects AVX-512 or AVX2 on x86-64 and NEON on
//! AArch64. Unsupported shapes and `TICIT_SIMD=scalar` use the scalar backend.
//! Fearless SIMD owns the target-feature boundary for the dense kernels.
//! The small x86 intrinsic island is limited to whole-vector XOR permutations,
//! integer parity masks, and horizontal reductions that Fearless SIMD 0.6 does
//! not expose directly; all vector memory access uses its checked slice API.
//!
//! # Clamping
//!
//! The probability *kernels* return a raw sum, which rounding can push a hair
//! outside `[0, 1]`. The `*_contiguous` callers clamp; the seam functions do
//! not, so each backend is compared against the unclamped reference.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256d, __m512d, _mm_add_pd, _mm_cvtsd_f64, _mm_hadd_pd, _mm256_add_pd, _mm256_and_pd,
    _mm256_and_si256, _mm256_blendv_pd, _mm256_castpd256_pd128, _mm256_castsi256_pd,
    _mm256_cmpeq_epi64, _mm256_extractf128_pd, _mm256_fmadd_pd, _mm256_fnmadd_pd, _mm256_mul_pd,
    _mm256_permute_pd, _mm256_permute4x64_pd, _mm256_set_epi64x, _mm256_set_pd, _mm256_set1_epi64x,
    _mm256_set1_pd, _mm256_setzero_pd, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_sub_pd,
    _mm256_xor_pd, _mm256_xor_si256, _mm512_add_pd, _mm512_and_si512, _mm512_castpd256_pd512,
    _mm512_castsi512_pd, _mm512_fmadd_pd, _mm512_fnmadd_pd, _mm512_insertf64x4, _mm512_mul_pd,
    _mm512_permutexvar_pd, _mm512_reduce_add_pd, _mm512_set_epi64, _mm512_set1_epi64,
    _mm512_set1_pd, _mm512_setzero_pd, _mm512_slli_epi64, _mm512_srli_epi64, _mm512_sub_pd,
    _mm512_xor_pd, _mm512_xor_si512,
};
use fearless_simd::Level;
#[cfg(target_arch = "aarch64")]
use fearless_simd::aarch64::Neon;
#[cfg(target_arch = "aarch64")]
use fearless_simd::{SimdBase, SimdFloat, f64x2};
#[cfg(target_arch = "x86_64")]
use fearless_simd::{SimdBase, SimdInto, f64x4, f64x8};
use num_complex::Complex64;

use crate::active::{
    INV_SQRT2, PrecomputedActivePauliMeasurementKernel, PrecomputedActivePauliRotationKernel,
    insert_zero_bit,
};

static SIMD_LEVEL: OnceLock<Option<Level>> = OnceLock::new();

fn simd_level() -> Option<Level> {
    *SIMD_LEVEL.get_or_init(|| {
        (!std::env::var("TICIT_SIMD").is_ok_and(|value| value.eq_ignore_ascii_case("scalar")))
            .then(Level::new)
    })
}

/// Name of the dense-kernel backend in use, for diagnostics.
pub fn backend_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    if simd_level().is_some_and(|level| level.as_neon().is_some()) {
        return "neon";
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(level) = simd_level() {
        if level.as_avx512().is_some() {
            return "avx512";
        }
        if level.as_avx2().is_some() {
            return "avx2";
        }
    }
    "scalar"
}

// ==============================================================================
// Rotation
// ==============================================================================

/// `alpha <- exp(-i * phi * P) alpha` over the first `dim` amplitudes.
pub fn rotate_contiguous_active(
    re: &mut [f64],
    im: &mut [f64],
    dim: usize,
    kernel: &PrecomputedActivePauliRotationKernel,
    sign: bool,
) {
    let c = kernel.cos_kernel_angle;
    if kernel.is_diagonal {
        #[cfg(target_arch = "aarch64")]
        if dim >= 2
            && dim.is_multiple_of(2)
            && dim <= re.len()
            && dim <= im.len()
            && let Some(neon) = simd_level().and_then(Level::as_neon)
        {
            rotate_diagonal_neon(
                neon,
                &mut re[..dim],
                &mut im[..dim],
                kernel.action.zmask,
                c,
                kernel.minus_even_coefficient.re,
                kernel.minus_even_coefficient.im,
                sign,
            );
            return;
        }
        // No X part, so every amplitude is just scaled by its own eigenvalue.
        for basis in 0..dim {
            let coefficient = kernel.coefficient(basis, sign);
            let fr = c + coefficient.re;
            let fi = coefficient.im;
            let r = re[basis];
            let i = im[basis];
            re[basis] = fr * r - fi * i;
            im[basis] = fr * i + fi * r;
        }
        return;
    }

    if kernel.uniform_imag_pairs {
        // zmask == 0, so the off-diagonal coefficient is the same purely
        // imaginary number for every pair.
        let coefficient = kernel.coefficient(0, sign);
        rotate_uniform_imag_pairs_soa(
            re,
            im,
            dim,
            kernel.action.xmask,
            kernel.pair_bit,
            c,
            coefficient.im,
        );
        return;
    }

    // General case: each pair gets its own signs, so the two members are
    // enumerated by reinserting the pivot bit.
    let pair_bit = kernel.pair_bit as usize;
    let xmask = kernel.action.xmask as usize;
    for idx in 0..kernel.pair_count {
        let left = insert_zero_bit(idx, pair_bit);
        let right = left ^ xmask;
        let left_odd = kernel.action.phase_odd(left);
        let left_direction = if sign != left_odd { -1.0 } else { 1.0 };
        let right_direction = if kernel.action.xz_overlap_odd {
            -left_direction
        } else {
            left_direction
        };
        let left_re = left_direction * kernel.minus_even_coefficient.re;
        let left_im = left_direction * kernel.minus_even_coefficient.im;
        let right_re = right_direction * kernel.minus_even_coefficient.re;
        let right_im = right_direction * kernel.minus_even_coefficient.im;
        let r0 = re[left];
        let i0 = im[left];
        let r1 = re[right];
        let i1 = im[right];
        re[left] = c * r0 + right_re * r1 - right_im * i1;
        im[left] = c * i0 + right_re * i1 + right_im * r1;
        re[right] = c * r1 + left_re * r0 - left_im * i0;
        im[right] = c * i1 + left_re * i0 + left_im * r0;
    }
}

#[cfg(target_arch = "aarch64")]
fearless_simd::kernel!(
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn rotate_diagonal_neon(
        neon: Neon,
        re: &mut [f64],
        im: &mut [f64],
        zmask: u64,
        cos: f64,
        coefficient_re: f64,
        coefficient_im: f64,
        sign: bool,
    ) {
        let vc = f64x2::splat(neon, cos);
        let cre = f64x2::splat(neon, coefficient_re);
        let cim = f64x2::splat(neon, coefficient_im);
        let mut basis = 0;
        while basis < re.len() {
            let r = f64x2::from_slice(neon, &re[basis..basis + 2]);
            let v = f64x2::from_slice(neon, &im[basis..basis + 2]);
            let direction = branch_direction_neon(neon, basis, zmask, sign);
            let fr = vc + direction * cre;
            let fi = direction * cim;
            (-fi)
                .mul_add(v, fr * r)
                .store_slice(&mut re[basis..basis + 2]);
            fi.mul_add(r, fr * v).store_slice(&mut im[basis..basis + 2]);
            basis += 2;
        }
    }
);

/// The `zmask == 0` pair rotation: a shared `[[c, iq], [iq, c]]` on every pair
/// `(i, i ^ xmask)`.
///
/// Walks blocks of `2 * 2^pair_bit`, which keeps both members of a pair inside
/// one block — the invariant that makes `pair_bit` the *highest* X bit
/// load-bearing.
#[inline]
pub fn rotate_uniform_imag_pairs_soa(
    re: &mut [f64],
    im: &mut [f64],
    dim: usize,
    xmask: u64,
    pair_bit: u32,
    c: f64,
    q: f64,
) {
    #[cfg(target_arch = "x86_64")]
    if dim <= re.len()
        && dim <= im.len()
        && xmask != 0
        && pair_bit < usize::BITS - 1
        && pair_bit == 63 - xmask.leading_zeros()
        && let Some(level) = simd_level()
    {
        if dim >= 8
            && dim.is_multiple_of(8)
            && let Some(avx512) = level.as_avx512()
        {
            rotate_uniform_imag_pairs_avx512(
                avx512,
                &mut re[..dim],
                &mut im[..dim],
                xmask,
                pair_bit,
                c,
                q,
            );
            return;
        }
        if dim >= 4
            && dim.is_multiple_of(4)
            && let Some(avx2) = level.as_avx2()
        {
            rotate_uniform_imag_pairs_avx2(
                avx2,
                &mut re[..dim],
                &mut im[..dim],
                xmask,
                pair_bit,
                c,
                q,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    if dim <= re.len()
        && dim <= im.len()
        && dim >= 2
        && dim.is_multiple_of(2)
        && xmask != 0
        && pair_bit < usize::BITS - 1
        && pair_bit == 63 - xmask.leading_zeros()
        && let Some(neon) = simd_level().and_then(Level::as_neon)
    {
        rotate_uniform_imag_pairs_neon(neon, &mut re[..dim], &mut im[..dim], xmask, pair_bit, c, q);
        return;
    }

    rotate_uniform_imag_pairs_soa_scalar(re, im, dim, xmask, pair_bit, c, q);
}

// NEON has two f64 lanes. The highest X bit splits each block into two halves;
// lower X bits select the partner chunk and, for bit zero, swap its lanes.
// Keeping that mapping explicit avoids gathers while Fearless SIMD owns the
// target-feature and bounds-safety boundary.
#[cfg(target_arch = "aarch64")]
fearless_simd::kernel!(
    #[inline]
    fn rotate_uniform_imag_pairs_neon(
        neon: Neon,
        re: &mut [f64],
        im: &mut [f64],
        xmask: u64,
        pair_bit: u32,
        c: f64,
        q: f64,
    ) {
        let dim = re.len();
        let vc = f64x2::splat(neon, c);
        let vq = f64x2::splat(neon, q);

        if pair_bit == 0 {
            let mut basis = 0;
            while basis < dim {
                let r = f64x2::from_slice(neon, &re[basis..basis + 2]);
                let v = f64x2::from_slice(neon, &im[basis..basis + 2]);
                let swapped_r = r.slide::<1>(r);
                let swapped_v = v.slide::<1>(v);
                (-vq)
                    .mul_add(swapped_v, vc * r)
                    .store_slice(&mut re[basis..basis + 2]);
                vq.mul_add(swapped_r, vc * v)
                    .store_slice(&mut im[basis..basis + 2]);
                basis += 2;
            }
            return;
        }

        let selector = 1usize << pair_bit;
        let step = selector << 1;
        let lower_mask = xmask as usize & (selector - 1);
        let swap_lanes = lower_mask & 1 != 0;
        let chunk_mask = lower_mask & !1;
        let mut block = 0;
        while block < dim {
            let mut offset = 0;
            while offset < selector {
                let i0 = block + offset;
                let i1 = block + selector + (offset ^ chunk_mask);
                let r0 = f64x2::from_slice(neon, &re[i0..i0 + 2]);
                let v0 = f64x2::from_slice(neon, &im[i0..i0 + 2]);
                let mut r1 = f64x2::from_slice(neon, &re[i1..i1 + 2]);
                let mut v1 = f64x2::from_slice(neon, &im[i1..i1 + 2]);
                if swap_lanes {
                    r1 = r1.slide::<1>(r1);
                    v1 = v1.slide::<1>(v1);
                }

                (-vq).mul_add(v1, vc * r0).store_slice(&mut re[i0..i0 + 2]);
                vq.mul_add(r1, vc * v0).store_slice(&mut im[i0..i0 + 2]);
                let mut out_r1 = (-vq).mul_add(v0, vc * r1);
                let mut out_v1 = vq.mul_add(r0, vc * v1);
                if swap_lanes {
                    out_r1 = out_r1.slide::<1>(out_r1);
                    out_v1 = out_v1.slide::<1>(out_v1);
                }
                out_r1.store_slice(&mut re[i1..i1 + 2]);
                out_v1.store_slice(&mut im[i1..i1 + 2]);
                offset += 2;
            }
            block += step;
        }
    }
);

#[cfg(target_arch = "x86_64")]
macro_rules! permute_xor4 {
    ($value:expr, $mask:expr) => {
        match $mask {
            0 => $value,
            1 => _mm256_permute4x64_pd::<0xb1>($value),
            2 => _mm256_permute4x64_pd::<0x4e>($value),
            3 => _mm256_permute4x64_pd::<0x1b>($value),
            _ => unreachable!("four-lane XOR mask"),
        }
    };
}

/// One rotation of a register-resident dim-16 run: the uniform purely
/// imaginary pair rotation `[[c, iq], [iq, c]]` on pairs `(e, e ^ xmask)`.
/// Both sign variants of the imaginary coefficient ride along so one run
/// table serves every shot; bit `i` of the caller's sign mask picks per
/// rotation.
#[derive(Clone, Copy, Debug)]
pub struct UniformImagRunStep {
    pub xmask: u64,
    pub cos: f64,
    pub imag_false: f64,
    pub imag_true: f64,
}

/// One step of the five-rotation diagonal run used by the distillation
/// circuit. Parity signs are resolved once per run instead of once per shot.
#[derive(Clone, Copy, Debug, Default)]
pub struct DiagonalRunStep {
    cos: f64,
    coefficient_re: f64,
    coefficient_im: f64,
    parity_sign: [f64; 32],
}

impl DiagonalRunStep {
    pub fn new(zmask: u64, cos: f64, coefficient: Complex64) -> Self {
        let mut parity_sign = [0.0; 32];
        for (basis, sign) in parity_sign.iter_mut().enumerate() {
            if ((basis as u64 & zmask).count_ones() & 1) != 0 {
                *sign = -0.0;
            }
        }
        Self {
            cos,
            coefficient_re: coefficient.re,
            coefficient_im: coefficient.im,
            parity_sign,
        }
    }
}

/// Whether [`rotate_uniform_imag_run_dim16`] has a vector backend, so run
/// callers can pre-qualify once instead of falling back per shot.
pub fn has_uniform_imag_run_dim16_backend() -> bool {
    #[cfg(target_arch = "x86_64")]
    if let Some(level) = simd_level() {
        return level.as_avx512().is_some() || level.as_avx2().is_some();
    }
    false
}

/// Whether the fixed dim-32 diagonal run has an x86 vector backend.
pub fn has_diagonal_run_dim32_backend() -> bool {
    has_uniform_imag_run_dim16_backend()
}

/// Applies a whole rotation run to one shot's dim-16 state, holding all 16
/// amplitude pairs in registers across the run. Returns `false` (state
/// untouched) when no vector backend applies; callers then use the
/// per-rotation path. Every step must satisfy `1 <= xmask <= 15`.
///
/// Element-wise this performs exactly the arithmetic of
/// [`rotate_uniform_imag_pairs_soa`]'s selected vector kernel per step — same FMA
/// shapes, single rounding per output — so a run is bit-identical to the
/// sequential per-rotation calls it replaces.
pub fn rotate_uniform_imag_run_dim16(
    re: &mut [f64],
    im: &mut [f64],
    steps: &[UniformImagRunStep],
    sign_mask: u32,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if re.len() >= 16
        && im.len() >= 16
        && steps.len() <= 32
        && steps.iter().all(|step| step.xmask >= 1 && step.xmask <= 15)
        && let Some(level) = simd_level()
    {
        if let Some(avx512) = level.as_avx512() {
            rotate_uniform_imag_run_dim16_avx512(
                avx512,
                &mut re[..16],
                &mut im[..16],
                steps,
                sign_mask,
            );
            return true;
        }
        if let Some(avx2) = level.as_avx2() {
            rotate_uniform_imag_run_dim16_avx2(
                avx2,
                &mut re[..16],
                &mut im[..16],
                steps,
                sign_mask,
            );
            return true;
        }
    }
    let _ = (re, im, steps, sign_mask);
    false
}

/// In-register pair rotation where every element's partner lives in the same
/// vector: lane permute, then one FMA pair per component.
#[cfg(target_arch = "x86_64")]
macro_rules! rotate_run_self4 {
    ($r:ident, $v:ident, $lane:expr, $vc:ident, $vq:ident) => {{
        let pr = permute_xor4!($r, $lane);
        let pv = permute_xor4!($v, $lane);
        $r = _mm256_fnmadd_pd($vq, pv, _mm256_mul_pd($vc, $r));
        $v = _mm256_fmadd_pd($vq, pr, _mm256_mul_pd($vc, $v));
    }};
}

/// In-register pair rotation across two state vectors: both directions read
/// the pre-permute partner values, so update order within the pair is free.
#[cfg(target_arch = "x86_64")]
macro_rules! rotate_run_pair4 {
    ($ra:ident, $va:ident, $rb:ident, $vb:ident, $lane:expr, $vc:ident, $vq:ident) => {{
        let pra = permute_xor4!($ra, $lane);
        let pva = permute_xor4!($va, $lane);
        let prb = permute_xor4!($rb, $lane);
        let pvb = permute_xor4!($vb, $lane);
        $ra = _mm256_fnmadd_pd($vq, pvb, _mm256_mul_pd($vc, $ra));
        $va = _mm256_fmadd_pd($vq, prb, _mm256_mul_pd($vc, $va));
        $rb = _mm256_fnmadd_pd($vq, pva, _mm256_mul_pd($vc, $rb));
        $vb = _mm256_fmadd_pd($vq, pra, _mm256_mul_pd($vc, $vb));
    }};
}

// The feature token proves AVX2+FMA; `f64x4::{from,store}_slice` prove every
// memory access is in bounds. The four state vectors cover basis elements
// `4b..4b+4`; element `e` pairs with `e ^ xmask`, which is vector
// `b ^ (xmask >> 2)` at lane `(e ^ xmask) & 3`.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    fn rotate_uniform_imag_run_dim16_avx2(
        avx2: Avx2,
        re: &mut [f64],
        im: &mut [f64],
        steps: &[UniformImagRunStep],
        sign_mask: u32,
    ) {
        let mut r0: __m256d = f64x4::from_slice(avx2, &re[0..4]).into();
        let mut r1: __m256d = f64x4::from_slice(avx2, &re[4..8]).into();
        let mut r2: __m256d = f64x4::from_slice(avx2, &re[8..12]).into();
        let mut r3: __m256d = f64x4::from_slice(avx2, &re[12..16]).into();
        let mut v0: __m256d = f64x4::from_slice(avx2, &im[0..4]).into();
        let mut v1: __m256d = f64x4::from_slice(avx2, &im[4..8]).into();
        let mut v2: __m256d = f64x4::from_slice(avx2, &im[8..12]).into();
        let mut v3: __m256d = f64x4::from_slice(avx2, &im[12..16]).into();

        for (index, step) in steps.iter().enumerate() {
            let vc = _mm256_set1_pd(step.cos);
            let vq = _mm256_set1_pd(if (sign_mask >> index) & 1 == 1 {
                step.imag_true
            } else {
                step.imag_false
            });
            let lane = (step.xmask & 3) as usize;
            match (step.xmask >> 2) & 3 {
                0 => {
                    rotate_run_self4!(r0, v0, lane, vc, vq);
                    rotate_run_self4!(r1, v1, lane, vc, vq);
                    rotate_run_self4!(r2, v2, lane, vc, vq);
                    rotate_run_self4!(r3, v3, lane, vc, vq);
                }
                1 => {
                    rotate_run_pair4!(r0, v0, r1, v1, lane, vc, vq);
                    rotate_run_pair4!(r2, v2, r3, v3, lane, vc, vq);
                }
                2 => {
                    rotate_run_pair4!(r0, v0, r2, v2, lane, vc, vq);
                    rotate_run_pair4!(r1, v1, r3, v3, lane, vc, vq);
                }
                _ => {
                    rotate_run_pair4!(r0, v0, r3, v3, lane, vc, vq);
                    rotate_run_pair4!(r1, v1, r2, v2, lane, vc, vq);
                }
            }
        }

        let r0: f64x4<_> = r0.simd_into(avx2);
        let r1: f64x4<_> = r1.simd_into(avx2);
        let r2: f64x4<_> = r2.simd_into(avx2);
        let r3: f64x4<_> = r3.simd_into(avx2);
        let v0: f64x4<_> = v0.simd_into(avx2);
        let v1: f64x4<_> = v1.simd_into(avx2);
        let v2: f64x4<_> = v2.simd_into(avx2);
        let v3: f64x4<_> = v3.simd_into(avx2);
        r0.store_slice(&mut re[0..4]);
        r1.store_slice(&mut re[4..8]);
        r2.store_slice(&mut re[8..12]);
        r3.store_slice(&mut re[12..16]);
        v0.store_slice(&mut im[0..4]);
        v1.store_slice(&mut im[4..8]);
        v2.store_slice(&mut im[8..12]);
        v3.store_slice(&mut im[12..16]);
    }
);

#[cfg(target_arch = "x86_64")]
macro_rules! nondiagonal_sign_mask4 {
    ($source0:expr, $zmask:expr, $branch:expr) => {{
        let mut parity = _mm256_and_si256(
            _mm256_set_epi64x(
                ($source0 + 3) as i64,
                ($source0 + 2) as i64,
                ($source0 + 1) as i64,
                $source0 as i64,
            ),
            _mm256_set1_epi64x($zmask as i64),
        );
        parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<32>(parity));
        parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<16>(parity));
        parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<8>(parity));
        parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<4>(parity));
        parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<2>(parity));
        parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<1>(parity));
        parity = _mm256_and_si256(parity, _mm256_set1_epi64x(1));
        if $branch {
            parity = _mm256_xor_si256(parity, _mm256_set1_epi64x(1));
        }
        _mm256_castsi256_pd(_mm256_slli_epi64::<63>(parity))
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! nondiagonal_amplitudes4 {
    ($r0:expr, $im0:expr, $r1:expr, $im1:expr, $sign:expr, $c1r:expr, $c1i:expr) => {{
        let c0 = _mm256_set1_pd(INV_SQRT2);
        let c1r = _mm256_xor_pd($c1r, $sign);
        let c1i = _mm256_xor_pd($c1i, $sign);
        let ar = _mm256_fmadd_pd(c1r, $r1, _mm256_mul_pd(c0, $r0));
        let ar = _mm256_fnmadd_pd(c1i, $im1, ar);
        let ai = _mm256_fmadd_pd(c1r, $im1, _mm256_mul_pd(c0, $im0));
        let ai = _mm256_fmadd_pd(c1i, $r1, ai);
        (ar, ai)
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! permute_xor8 {
    ($value:expr, $mask:expr) => {
        match $mask {
            0 => $value,
            1 => _mm512_permutexvar_pd(_mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1), $value),
            2 => _mm512_permutexvar_pd(_mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2), $value),
            3 => _mm512_permutexvar_pd(_mm512_set_epi64(4, 5, 6, 7, 0, 1, 2, 3), $value),
            4 => _mm512_permutexvar_pd(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), $value),
            5 => _mm512_permutexvar_pd(_mm512_set_epi64(2, 3, 0, 1, 6, 7, 4, 5), $value),
            6 => _mm512_permutexvar_pd(_mm512_set_epi64(1, 0, 3, 2, 5, 4, 7, 6), $value),
            7 => _mm512_permutexvar_pd(_mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7), $value),
            _ => unreachable!("eight-lane XOR mask"),
        }
    };
}

/// In-register pair rotation where every element's partner lives in the same
/// AVX-512 vector.
#[cfg(target_arch = "x86_64")]
macro_rules! rotate_run_self8 {
    ($r:ident, $v:ident, $lane:expr, $vc:ident, $vq:ident) => {{
        let pr = permute_xor8!($r, $lane);
        let pv = permute_xor8!($v, $lane);
        $r = _mm512_fnmadd_pd($vq, pv, _mm512_mul_pd($vc, $r));
        $v = _mm512_fmadd_pd($vq, pr, _mm512_mul_pd($vc, $v));
    }};
}

/// In-register pair rotation whose partners cross the two AVX-512 state
/// vectors.
#[cfg(target_arch = "x86_64")]
macro_rules! rotate_run_pair8 {
    ($ra:ident, $va:ident, $rb:ident, $vb:ident, $lane:expr, $vc:ident, $vq:ident) => {{
        let pra = permute_xor8!($ra, $lane);
        let pva = permute_xor8!($va, $lane);
        let prb = permute_xor8!($rb, $lane);
        let pvb = permute_xor8!($vb, $lane);
        $ra = _mm512_fnmadd_pd($vq, pvb, _mm512_mul_pd($vc, $ra));
        $va = _mm512_fmadd_pd($vq, prb, _mm512_mul_pd($vc, $va));
        $rb = _mm512_fnmadd_pd($vq, pva, _mm512_mul_pd($vc, $rb));
        $vb = _mm512_fmadd_pd($vq, pra, _mm512_mul_pd($vc, $vb));
    }};
}

// The AVX-512 token proves the Ice Lake feature set; checked slice loads and
// stores cover all 16 amplitudes. Bit 3 of xmask selects whether partners
// cross the two eight-lane state vectors; the low three bits select lanes.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    fn rotate_uniform_imag_run_dim16_avx512(
        avx512: Avx512,
        re: &mut [f64],
        im: &mut [f64],
        steps: &[UniformImagRunStep],
        sign_mask: u32,
    ) {
        let mut r0: __m512d = f64x8::from_slice(avx512, &re[0..8]).into();
        let mut r1: __m512d = f64x8::from_slice(avx512, &re[8..16]).into();
        let mut v0: __m512d = f64x8::from_slice(avx512, &im[0..8]).into();
        let mut v1: __m512d = f64x8::from_slice(avx512, &im[8..16]).into();

        for (index, step) in steps.iter().enumerate() {
            let vc = _mm512_set1_pd(step.cos);
            let vq = _mm512_set1_pd(if (sign_mask >> index) & 1 == 1 {
                step.imag_true
            } else {
                step.imag_false
            });
            let lane = (step.xmask & 7) as usize;
            if step.xmask < 8 {
                rotate_run_self8!(r0, v0, lane, vc, vq);
                rotate_run_self8!(r1, v1, lane, vc, vq);
            } else {
                rotate_run_pair8!(r0, v0, r1, v1, lane, vc, vq);
            }
        }

        let r0: f64x8<_> = r0.simd_into(avx512);
        let r1: f64x8<_> = r1.simd_into(avx512);
        let v0: f64x8<_> = v0.simd_into(avx512);
        let v1: f64x8<_> = v1.simd_into(avx512);
        r0.store_slice(&mut re[0..8]);
        r1.store_slice(&mut re[8..16]);
        v0.store_slice(&mut im[0..8]);
        v1.store_slice(&mut im[8..16]);
    }
);

#[cfg(target_arch = "x86_64")]
macro_rules! nondiagonal_sign_mask8 {
    ($source0:expr, $zmask:expr, $branch:expr) => {{
        let mut parity = _mm512_and_si512(
            _mm512_set_epi64(
                ($source0 + 7) as i64,
                ($source0 + 6) as i64,
                ($source0 + 5) as i64,
                ($source0 + 4) as i64,
                ($source0 + 3) as i64,
                ($source0 + 2) as i64,
                ($source0 + 1) as i64,
                $source0 as i64,
            ),
            _mm512_set1_epi64($zmask as i64),
        );
        parity = _mm512_xor_si512(parity, _mm512_srli_epi64::<32>(parity));
        parity = _mm512_xor_si512(parity, _mm512_srli_epi64::<16>(parity));
        parity = _mm512_xor_si512(parity, _mm512_srli_epi64::<8>(parity));
        parity = _mm512_xor_si512(parity, _mm512_srli_epi64::<4>(parity));
        parity = _mm512_xor_si512(parity, _mm512_srli_epi64::<2>(parity));
        parity = _mm512_xor_si512(parity, _mm512_srli_epi64::<1>(parity));
        parity = _mm512_and_si512(parity, _mm512_set1_epi64(1));
        if $branch {
            parity = _mm512_xor_si512(parity, _mm512_set1_epi64(1));
        }
        _mm512_castsi512_pd(_mm512_slli_epi64::<63>(parity))
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! rotate_diagonal_run_step4 {
    ($avx2:expr, $r:ident, $v:ident, $basis:expr, $step:expr, $sign:expr) => {{
        let step = &$step;
        let parity: __m256d =
            f64x4::from_slice($avx2, &step.parity_sign[$basis..$basis + 4]).into();
        let direction = _mm256_xor_pd(parity, $sign);
        let fr = _mm256_add_pd(
            _mm256_set1_pd(step.cos),
            _mm256_xor_pd(_mm256_set1_pd(step.coefficient_re), direction),
        );
        let fi = _mm256_xor_pd(_mm256_set1_pd(step.coefficient_im), direction);
        let old_r = $r;
        let old_v = $v;
        $r = _mm256_sub_pd(_mm256_mul_pd(fr, old_r), _mm256_mul_pd(fi, old_v));
        $v = _mm256_add_pd(_mm256_mul_pd(fr, old_v), _mm256_mul_pd(fi, old_r));
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! rotate_diagonal_run_step8 {
    ($avx512:expr, $r:ident, $v:ident, $basis:expr, $step:expr, $sign:expr) => {{
        let step = &$step;
        let parity: __m512d =
            f64x8::from_slice($avx512, &step.parity_sign[$basis..$basis + 8]).into();
        let direction = _mm512_xor_pd(parity, $sign);
        let fr = _mm512_add_pd(
            _mm512_set1_pd(step.cos),
            _mm512_xor_pd(_mm512_set1_pd(step.coefficient_re), direction),
        );
        let fi = _mm512_xor_pd(_mm512_set1_pd(step.coefficient_im), direction);
        let old_r = $r;
        let old_v = $v;
        $r = _mm512_sub_pd(_mm512_mul_pd(fr, old_r), _mm512_mul_pd(fi, old_v));
        $v = _mm512_add_pd(_mm512_mul_pd(fr, old_v), _mm512_mul_pd(fi, old_r));
    }};
}

/// Applies the distillation circuit's five diagonal rotations while each
/// dim-32 amplitude chunk remains in registers.
pub fn rotate_diagonal_run_dim32(
    re: &mut [f64],
    im: &mut [f64],
    steps: &[DiagonalRunStep; 5],
    sign_mask: u32,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if re.len() >= 32
        && im.len() >= 32
        && let Some(level) = simd_level()
    {
        if let Some(avx512) = level.as_avx512() {
            rotate_diagonal_run_dim32_avx512(
                avx512,
                &mut re[..32],
                &mut im[..32],
                steps,
                sign_mask,
            );
            return true;
        }
        if let Some(avx2) = level.as_avx2() {
            rotate_diagonal_run_dim32_avx2(avx2, &mut re[..32], &mut im[..32], steps, sign_mask);
            return true;
        }
    }
    let _ = (re, im, steps, sign_mask);
    false
}

#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    fn rotate_diagonal_run_dim32_avx2(
        avx2: Avx2,
        re: &mut [f64],
        im: &mut [f64],
        steps: &[DiagonalRunStep; 5],
        sign_mask: u32,
    ) {
        let mut basis = 0;
        while basis < 32 {
            let mut r: __m256d = f64x4::from_slice(avx2, &re[basis..basis + 4]).into();
            let mut v: __m256d = f64x4::from_slice(avx2, &im[basis..basis + 4]).into();
            for (index, step) in steps.iter().enumerate() {
                let sign = _mm256_set1_pd(if (sign_mask >> index) & 1 == 0 {
                    0.0
                } else {
                    -0.0
                });
                rotate_diagonal_run_step4!(avx2, r, v, basis, step, sign);
            }
            let r: f64x4<_> = r.simd_into(avx2);
            let v: f64x4<_> = v.simd_into(avx2);
            r.store_slice(&mut re[basis..basis + 4]);
            v.store_slice(&mut im[basis..basis + 4]);
            basis += 4;
        }
    }
);

#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    fn rotate_diagonal_run_dim32_avx512(
        avx512: Avx512,
        re: &mut [f64],
        im: &mut [f64],
        steps: &[DiagonalRunStep; 5],
        sign_mask: u32,
    ) {
        let mut basis = 0;
        while basis < 32 {
            let mut r: __m512d = f64x8::from_slice(avx512, &re[basis..basis + 8]).into();
            let mut v: __m512d = f64x8::from_slice(avx512, &im[basis..basis + 8]).into();
            for (index, step) in steps.iter().enumerate() {
                let sign = _mm512_set1_pd(if (sign_mask >> index) & 1 == 0 {
                    0.0
                } else {
                    -0.0
                });
                rotate_diagonal_run_step8!(avx512, r, v, basis, step, sign);
            }
            let r: f64x8<_> = r.simd_into(avx512);
            let v: f64x8<_> = v.simd_into(avx512);
            r.store_slice(&mut re[basis..basis + 8]);
            v.store_slice(&mut im[basis..basis + 8]);
            basis += 8;
        }
    }
);

#[cfg(target_arch = "x86_64")]
macro_rules! nondiagonal_amplitudes8 {
    ($r0:expr, $im0:expr, $r1:expr, $im1:expr, $sign:expr, $c1r:expr, $c1i:expr) => {{
        let c0 = _mm512_set1_pd(INV_SQRT2);
        let c1r = _mm512_xor_pd($c1r, $sign);
        let c1i = _mm512_xor_pd($c1i, $sign);
        let ar = _mm512_fmadd_pd(c1r, $r1, _mm512_mul_pd(c0, $r0));
        let ar = _mm512_fnmadd_pd(c1i, $im1, ar);
        let ai = _mm512_fmadd_pd(c1r, $im1, _mm512_mul_pd(c0, $im0));
        let ai = _mm512_fmadd_pd(c1i, $r1, ai);
        (ar, ai)
    }};
}

// The feature token proves the Ice Lake AVX-512 feature set, while
// `f64x8::{from,store}_slice` prove every memory access is in bounds.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    fn rotate_uniform_imag_pairs_avx512(
        avx512: Avx512,
        re: &mut [f64],
        im: &mut [f64],
        xmask: u64,
        pair_bit: u32,
        c: f64,
        q: f64,
    ) {
        let dim = re.len();
        let vc = _mm512_set1_pd(c);
        let vq = _mm512_set1_pd(q);

        if pair_bit < 3 {
            let lane_mask = xmask as usize;
            let mut basis = 0;
            while basis < dim {
                let r: __m512d = f64x8::from_slice(avx512, &re[basis..basis + 8]).into();
                let v: __m512d = f64x8::from_slice(avx512, &im[basis..basis + 8]).into();
                let swapped_r = permute_xor8!(r, lane_mask);
                let swapped_v = permute_xor8!(v, lane_mask);
                let out_r = _mm512_fnmadd_pd(vq, swapped_v, _mm512_mul_pd(vc, r));
                let out_v = _mm512_fmadd_pd(vq, swapped_r, _mm512_mul_pd(vc, v));
                let out_r: f64x8<_> = out_r.simd_into(avx512);
                let out_v: f64x8<_> = out_v.simd_into(avx512);
                out_r.store_slice(&mut re[basis..basis + 8]);
                out_v.store_slice(&mut im[basis..basis + 8]);
                basis += 8;
            }
            return;
        }

        let selector = 1usize << pair_bit;
        let step = selector << 1;
        let lower_mask = xmask as usize & (selector - 1);
        let lane_mask = lower_mask & 7;
        let chunk_mask = lower_mask & !7;
        let mut block = 0;
        while block < dim {
            let mut offset = 0;
            while offset < selector {
                let i0 = block + offset;
                let i1 = block + selector + (offset ^ chunk_mask);
                let r0: __m512d = f64x8::from_slice(avx512, &re[i0..i0 + 8]).into();
                let im0: __m512d = f64x8::from_slice(avx512, &im[i0..i0 + 8]).into();
                let r1: __m512d = f64x8::from_slice(avx512, &re[i1..i1 + 8]).into();
                let im1: __m512d = f64x8::from_slice(avx512, &im[i1..i1 + 8]).into();
                let r1 = permute_xor8!(r1, lane_mask);
                let im1 = permute_xor8!(im1, lane_mask);

                let out_r0 = _mm512_fnmadd_pd(vq, im1, _mm512_mul_pd(vc, r0));
                let out_im0 = _mm512_fmadd_pd(vq, r1, _mm512_mul_pd(vc, im0));
                let out_r1 = _mm512_fnmadd_pd(vq, im0, _mm512_mul_pd(vc, r1));
                let out_im1 = _mm512_fmadd_pd(vq, r0, _mm512_mul_pd(vc, im1));

                let out_r0: f64x8<_> = out_r0.simd_into(avx512);
                let out_im0: f64x8<_> = out_im0.simd_into(avx512);
                let out_r1: f64x8<_> = permute_xor8!(out_r1, lane_mask).simd_into(avx512);
                let out_im1: f64x8<_> = permute_xor8!(out_im1, lane_mask).simd_into(avx512);
                out_r0.store_slice(&mut re[i0..i0 + 8]);
                out_im0.store_slice(&mut im[i0..i0 + 8]);
                out_r1.store_slice(&mut re[i1..i1 + 8]);
                out_im1.store_slice(&mut im[i1..i1 + 8]);
                offset += 8;
            }
            block += step;
        }
    }
);

// The feature token proves AVX2/FMA support, while `f64x4::{from,store}_slice`
// prove every memory access is in bounds. The kernel therefore needs no local
// unsafe block; Fearless SIMD contains the target-feature boundary.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    fn rotate_uniform_imag_pairs_avx2(
        avx2: Avx2,
        re: &mut [f64],
        im: &mut [f64],
        xmask: u64,
        pair_bit: u32,
        c: f64,
        q: f64,
    ) {
        let dim = re.len();
        let vc = _mm256_set1_pd(c);
        let vq = _mm256_set1_pd(q);

        if pair_bit == 0 {
            let mut basis = 0;
            while basis < dim {
                let r: __m256d = f64x4::from_slice(avx2, &re[basis..basis + 4]).into();
                let v: __m256d = f64x4::from_slice(avx2, &im[basis..basis + 4]).into();
                let swapped_r = _mm256_permute_pd::<0b0101>(r);
                let swapped_v = _mm256_permute_pd::<0b0101>(v);
                let out_r = _mm256_fnmadd_pd(vq, swapped_v, _mm256_mul_pd(vc, r));
                let out_v = _mm256_fmadd_pd(vq, swapped_r, _mm256_mul_pd(vc, v));
                let out_r: f64x4<_> = out_r.simd_into(avx2);
                let out_v: f64x4<_> = out_v.simd_into(avx2);
                out_r.store_slice(&mut re[basis..basis + 4]);
                out_v.store_slice(&mut im[basis..basis + 4]);
                basis += 4;
            }
            return;
        }

        if pair_bit == 1 {
            let lane_mask = xmask as usize & 3;
            let mut basis = 0;
            while basis < dim {
                let r: __m256d = f64x4::from_slice(avx2, &re[basis..basis + 4]).into();
                let v: __m256d = f64x4::from_slice(avx2, &im[basis..basis + 4]).into();
                let swapped_r = permute_xor4!(r, lane_mask);
                let swapped_v = permute_xor4!(v, lane_mask);
                let out_r = _mm256_fnmadd_pd(vq, swapped_v, _mm256_mul_pd(vc, r));
                let out_v = _mm256_fmadd_pd(vq, swapped_r, _mm256_mul_pd(vc, v));
                let out_r: f64x4<_> = out_r.simd_into(avx2);
                let out_v: f64x4<_> = out_v.simd_into(avx2);
                out_r.store_slice(&mut re[basis..basis + 4]);
                out_v.store_slice(&mut im[basis..basis + 4]);
                basis += 4;
            }
            return;
        }

        let selector = 1usize << pair_bit;
        let step = selector << 1;
        let lower_mask = xmask as usize & (selector - 1);
        let lane_mask = lower_mask & 3;
        let chunk_mask = lower_mask & !3;
        let mut block = 0;
        while block < dim {
            let mut offset = 0;
            while offset < selector {
                let i0 = block + offset;
                let i1 = block + selector + (offset ^ chunk_mask);
                let r0: __m256d = f64x4::from_slice(avx2, &re[i0..i0 + 4]).into();
                let im0: __m256d = f64x4::from_slice(avx2, &im[i0..i0 + 4]).into();
                let r1: __m256d = f64x4::from_slice(avx2, &re[i1..i1 + 4]).into();
                let im1: __m256d = f64x4::from_slice(avx2, &im[i1..i1 + 4]).into();
                let r1 = permute_xor4!(r1, lane_mask);
                let im1 = permute_xor4!(im1, lane_mask);

                let out_r0 = _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0));
                let out_im0 = _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0));
                let out_r1 = _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1));
                let out_im1 = _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1));

                let out_r0: f64x4<_> = out_r0.simd_into(avx2);
                let out_im0: f64x4<_> = out_im0.simd_into(avx2);
                let out_r1: f64x4<_> = permute_xor4!(out_r1, lane_mask).simd_into(avx2);
                let out_im1: f64x4<_> = permute_xor4!(out_im1, lane_mask).simd_into(avx2);
                out_r0.store_slice(&mut re[i0..i0 + 4]);
                out_im0.store_slice(&mut im[i0..i0 + 4]);
                out_r1.store_slice(&mut re[i1..i1 + 4]);
                out_im1.store_slice(&mut im[i1..i1 + 4]);
                offset += 4;
            }
            block += step;
        }
    }
);

fn rotate_uniform_imag_pairs_soa_scalar(
    re: &mut [f64],
    im: &mut [f64],
    dim: usize,
    xmask: u64,
    pair_bit: u32,
    c: f64,
    q: f64,
) {
    let selector = 1usize << pair_bit;
    let step = selector << 1;
    let xmask = xmask as usize;
    let mut block = 0;
    while block < dim {
        for offset in 0..selector {
            let i0 = block + offset;
            let i1 = i0 ^ xmask;
            let r0 = re[i0];
            let im0 = im[i0];
            let r1 = re[i1];
            let im1 = im[i1];
            re[i0] = c * r0 - q * im1;
            im[i0] = c * im0 + q * r1;
            re[i1] = c * r1 - q * im0;
            im[i1] = c * im1 + q * r0;
        }
        block += step;
    }
}

// ==============================================================================
// Promotion
// ==============================================================================

/// Doubles the state as a new highest qubit is promoted out of the dormant set:
/// `alpha' = [c * alpha, -i q * alpha]`.
///
/// `re` and `im` must have room for `2 * dim` amplitudes.
pub fn promote_contiguous_active(re: &mut [f64], im: &mut [f64], dim: usize, c: f64, q: f64) {
    #[cfg(target_arch = "aarch64")]
    if dim >= 2
        && dim.is_multiple_of(2)
        && re.len() >= dim << 1
        && im.len() >= dim << 1
        && let Some(neon) = simd_level().and_then(Level::as_neon)
    {
        promote_contiguous_active_neon(neon, re, im, dim, c, q);
        return;
    }
    let (old_re, new_re) = re.split_at_mut(dim);
    let (old_im, new_im) = im.split_at_mut(dim);
    for (((r, i), new_r), new_i) in old_re
        .iter_mut()
        .zip(old_im)
        .zip(&mut new_re[..dim])
        .zip(&mut new_im[..dim])
    {
        let (old_r, old_i) = (*r, *i);
        *r = c * old_r;
        *i = c * old_i;
        *new_r = -q * old_i;
        *new_i = q * old_r;
    }
}

#[cfg(target_arch = "aarch64")]
fearless_simd::kernel!(
    #[inline]
    fn promote_contiguous_active_neon(
        neon: Neon,
        re: &mut [f64],
        im: &mut [f64],
        dim: usize,
        c: f64,
        q: f64,
    ) {
        let (old_re, new_re) = re.split_at_mut(dim);
        let (old_im, new_im) = im.split_at_mut(dim);
        let vc = f64x2::splat(neon, c);
        let vq = f64x2::splat(neon, q);
        let mut index = 0;
        while index < dim {
            let r = f64x2::from_slice(neon, &old_re[index..index + 2]);
            let v = f64x2::from_slice(neon, &old_im[index..index + 2]);
            (vc * r).store_slice(&mut old_re[index..index + 2]);
            (vc * v).store_slice(&mut old_im[index..index + 2]);
            ((-vq) * v).store_slice(&mut new_re[index..index + 2]);
            (vq * r).store_slice(&mut new_im[index..index + 2]);
            index += 2;
        }
    }
);

// ==============================================================================
// Measurement probability
// ==============================================================================

/// Probability of `branch` for a diagonal Pauli: the mass already sitting in the
/// selected eigenspace.
pub fn diagonal_probability_contiguous(
    re: &[f64],
    im: &[f64],
    kernel: &PrecomputedActivePauliMeasurementKernel,
    branch: bool,
) -> f64 {
    // `diagonal_source` enumerates exactly the bases with
    // `parity(b & zmask) == diagonal_phase_bit ^ branch` (the pivot bit is
    // chosen to make the overall parity constant), so the probability is a
    // parity-masked norm over the full state — no gather needed. The SIMD
    // sum reassociates the additions (tolerance-class, like the other
    // probability kernels); `TICIT_SIMD=scalar` keeps the exact gather order.
    #[cfg(target_arch = "x86_64")]
    {
        let dim = kernel.out_dim << 1;
        let zmask = kernel.action.zmask;
        if dim >= 4
            && dim.is_multiple_of(4)
            && dim <= re.len()
            && dim <= im.len()
            && kernel.pivot < 63
            && (zmask >> kernel.pivot) & 1 == 1
            && let Some(level) = simd_level()
            && let Some(avx2) = level.as_avx2()
        {
            let target = (kernel.diagonal_source(0, branch) >> kernel.pivot) & 1 == 1;
            let probability =
                diagonal_probability_masked_avx2(avx2, &re[..dim], &im[..dim], zmask, target);
            return probability.clamp(0.0, 1.0);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let dim = kernel.out_dim << 1;
        let zmask = kernel.action.zmask;
        if dim >= 2
            && dim.is_multiple_of(2)
            && dim <= re.len()
            && dim <= im.len()
            && kernel.pivot < 63
            && (zmask >> kernel.pivot) & 1 == 1
            && let Some(neon) = simd_level().and_then(Level::as_neon)
        {
            let target = (kernel.diagonal_source(0, branch) >> kernel.pivot) & 1 == 1;
            return diagonal_probability_masked_neon(neon, &re[..dim], &im[..dim], zmask, target)
                .clamp(0.0, 1.0);
        }
    }
    let mut probability = 0.0;
    for idx in 0..kernel.out_dim {
        let source = kernel.diagonal_source(idx, branch);
        probability += re[source] * re[source] + im[source] * im[source];
    }
    probability.clamp(0.0, 1.0)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn parity_selection_neon(neon: Neon, basis: usize, zmask: u64, target: bool) -> f64x2<Neon> {
    f64x2::from_fn(neon, |lane| {
        f64::from((((basis + lane) as u64 & zmask).count_ones() & 1 != 0) == target)
    })
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn branch_direction_neon(neon: Neon, source0: usize, zmask: u64, branch: bool) -> f64x2<Neon> {
    f64x2::from_fn(neon, |lane| {
        let odd = (((source0 + lane) as u64 & zmask).count_ones() & 1) != 0;
        if branch != odd { -1.0 } else { 1.0 }
    })
}

// Four accumulators hide the floating-point dependency latency. SIMD
// reassociates the norm sum, so callers compare this tolerance-class result
// against the scalar oracle rather than requiring bit identity.
#[cfg(target_arch = "aarch64")]
fearless_simd::kernel!(
    #[inline]
    fn diagonal_probability_masked_neon(
        neon: Neon,
        re: &[f64],
        im: &[f64],
        zmask: u64,
        target: bool,
    ) -> f64 {
        let zero = f64x2::splat(neon, 0.0);
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        let mut basis = 0;
        let mut chunk = 0;
        while basis < re.len() {
            let r = f64x2::from_slice(neon, &re[basis..basis + 2]);
            let v = f64x2::from_slice(neon, &im[basis..basis + 2]);
            let mass = r.mul_add(r, v * v) * parity_selection_neon(neon, basis, zmask, target);
            match chunk & 3 {
                0 => a0 += mass,
                1 => a1 += mass,
                2 => a2 += mass,
                _ => a3 += mass,
            }
            basis += 2;
            chunk += 1;
        }
        let total = a0 + a1 + a2 + a3;
        total[0] + total[1]
    }
);

// The feature token proves AVX2+FMA; all loads use checked slices. Lane
// parity folds `b & zmask` to bit 0, and `cmpeq` widens the match into a
// full-lane mask over the squared amplitudes.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    fn diagonal_probability_masked_avx2(
        avx2: Avx2,
        re: &[f64],
        im: &[f64],
        zmask: u64,
        target: bool,
    ) -> f64 {
        let dim = re.len();
        let target_lanes = _mm256_set1_epi64x(i64::from(target));
        let mut acc = _mm256_setzero_pd();
        let mut basis = 0;
        while basis < dim {
            let r: __m256d = f64x4::from_slice(avx2, &re[basis..basis + 4]).into();
            let v: __m256d = f64x4::from_slice(avx2, &im[basis..basis + 4]).into();
            let mut parity = _mm256_and_si256(
                _mm256_set_epi64x(
                    (basis + 3) as i64,
                    (basis + 2) as i64,
                    (basis + 1) as i64,
                    basis as i64,
                ),
                _mm256_set1_epi64x(zmask as i64),
            );
            parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<32>(parity));
            parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<16>(parity));
            parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<8>(parity));
            parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<4>(parity));
            parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<2>(parity));
            parity = _mm256_xor_si256(parity, _mm256_srli_epi64::<1>(parity));
            parity = _mm256_and_si256(parity, _mm256_set1_epi64x(1));
            let keep = _mm256_castsi256_pd(_mm256_cmpeq_epi64(parity, target_lanes));
            let squared = _mm256_fmadd_pd(v, v, _mm256_mul_pd(r, r));
            acc = _mm256_add_pd(acc, _mm256_and_pd(keep, squared));
            basis += 4;
        }
        let pair = _mm_add_pd(_mm256_castpd256_pd128(acc), _mm256_extractf128_pd::<1>(acc));
        _mm_cvtsd_f64(_mm_hadd_pd(pair, pair))
    }
);

/// Probability of `branch` for a Pauli with an X component: the norm of the
/// projected-and-compacted state.
pub fn nondiagonal_probability_contiguous(
    re: &[f64],
    im: &[f64],
    kernel: &PrecomputedActivePauliMeasurementKernel,
    branch: bool,
) -> f64 {
    let probability = measure_nondiagonal_probability_soa(
        re,
        im,
        kernel.out_dim << 1,
        kernel.action.xmask,
        kernel.action.zmask,
        kernel.pivot,
        kernel.nondiagonal_coefficient1_even,
        branch,
    );
    probability.clamp(0.0, 1.0)
}

// The Ice Lake-level token proves every AVX-512 feature used here; all
// eight-lane loads use checked slices.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn measure_nondiagonal_probability_avx512(
        avx512: Avx512,
        re: &[f64],
        im: &[f64],
        xmask: u64,
        zmask: u64,
        pivot: usize,
        coefficient_re: f64,
        coefficient_im: f64,
        branch: bool,
    ) -> f64 {
        let dim = re.len();
        let selector = 1usize << pivot;
        let step = selector << 1;
        let lower_mask = xmask as usize & (selector - 1);
        let lane_mask = lower_mask & 7;
        let chunk_mask = lower_mask & !7;
        let coefficient_re = _mm512_set1_pd(coefficient_re);
        let coefficient_im = _mm512_set1_pd(coefficient_im);
        let mut probability = _mm512_setzero_pd();
        let mut block = 0;
        while block < dim {
            let mut offset = 0;
            while offset < selector {
                let source0 = block + offset;
                let source1 = block + selector + (offset ^ chunk_mask);
                let r0: __m512d = f64x8::from_slice(avx512, &re[source0..source0 + 8]).into();
                let im0: __m512d = f64x8::from_slice(avx512, &im[source0..source0 + 8]).into();
                let r1: __m512d = f64x8::from_slice(avx512, &re[source1..source1 + 8]).into();
                let im1: __m512d = f64x8::from_slice(avx512, &im[source1..source1 + 8]).into();
                let r1 = permute_xor8!(r1, lane_mask);
                let im1 = permute_xor8!(im1, lane_mask);
                let sign = nondiagonal_sign_mask8!(source0, zmask, branch);
                let (ar, ai) = nondiagonal_amplitudes8!(
                    r0,
                    im0,
                    r1,
                    im1,
                    sign,
                    coefficient_re,
                    coefficient_im
                );
                probability = _mm512_fmadd_pd(ar, ar, probability);
                probability = _mm512_fmadd_pd(ai, ai, probability);
                offset += 8;
            }
            block += step;
        }
        _mm512_reduce_add_pd(probability)
    }
);

// AVX2/FMA is proved by the token; all four-lane loads use checked slices.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn measure_nondiagonal_probability_avx2(
        avx2: Avx2,
        re: &[f64],
        im: &[f64],
        xmask: u64,
        zmask: u64,
        pivot: usize,
        coefficient_re: f64,
        coefficient_im: f64,
        branch: bool,
    ) -> f64 {
        let dim = re.len();
        let selector = 1usize << pivot;
        let step = selector << 1;
        let lower_mask = xmask as usize & (selector - 1);
        let lane_mask = lower_mask & 3;
        let chunk_mask = lower_mask & !3;
        let coefficient_re = _mm256_set1_pd(coefficient_re);
        let coefficient_im = _mm256_set1_pd(coefficient_im);
        let mut probability = _mm256_setzero_pd();
        let mut block = 0;
        while block < dim {
            let mut offset = 0;
            while offset < selector {
                let source0 = block + offset;
                let source1 = block + selector + (offset ^ chunk_mask);
                let r0: __m256d = f64x4::from_slice(avx2, &re[source0..source0 + 4]).into();
                let im0: __m256d = f64x4::from_slice(avx2, &im[source0..source0 + 4]).into();
                let r1: __m256d = f64x4::from_slice(avx2, &re[source1..source1 + 4]).into();
                let im1: __m256d = f64x4::from_slice(avx2, &im[source1..source1 + 4]).into();
                let r1 = permute_xor4!(r1, lane_mask);
                let im1 = permute_xor4!(im1, lane_mask);
                let sign = nondiagonal_sign_mask4!(source0, zmask, branch);
                let (ar, ai) = nondiagonal_amplitudes4!(
                    r0,
                    im0,
                    r1,
                    im1,
                    sign,
                    coefficient_re,
                    coefficient_im
                );
                probability = _mm256_fmadd_pd(ar, ar, probability);
                probability = _mm256_fmadd_pd(ai, ai, probability);
                offset += 4;
            }
            block += step;
        }
        let pair = _mm_add_pd(
            _mm256_castpd256_pd128(probability),
            _mm256_extractf128_pd::<1>(probability),
        );
        _mm_cvtsd_f64(_mm_hadd_pd(pair, pair))
    }
);

/// SIMD seam: unclamped `sum |(<0| + <1|) alpha|^2` over the `dim / 2` outputs.
#[allow(clippy::too_many_arguments)]
pub fn measure_nondiagonal_probability_soa(
    re: &[f64],
    im: &[f64],
    dim: usize,
    xmask: u64,
    zmask: u64,
    pivot: usize,
    coefficient1_even: Complex64,
    branch: bool,
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if let Some(level) = simd_level() {
        if pivot >= 3
            && let Some(avx512) = level.as_avx512()
        {
            return measure_nondiagonal_probability_avx512(
                avx512,
                &re[..dim],
                &im[..dim],
                xmask,
                zmask,
                pivot,
                coefficient1_even.re,
                coefficient1_even.im,
                branch,
            );
        }
        if pivot >= 2
            && let Some(avx2) = level.as_avx2()
        {
            return measure_nondiagonal_probability_avx2(
                avx2,
                &re[..dim],
                &im[..dim],
                xmask,
                zmask,
                pivot,
                coefficient1_even.re,
                coefficient1_even.im,
                branch,
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    if pivot >= 1
        && dim.is_multiple_of(2)
        && let Some(neon) = simd_level().and_then(Level::as_neon)
    {
        return measure_nondiagonal_probability_neon(
            neon,
            &re[..dim],
            &im[..dim],
            xmask,
            zmask,
            pivot,
            coefficient1_even.re,
            coefficient1_even.im,
            branch,
        );
    }

    measure_nondiagonal_probability_soa_scalar(
        re,
        im,
        dim,
        xmask,
        zmask,
        pivot,
        coefficient1_even,
        branch,
    )
}

#[cfg(target_arch = "aarch64")]
fearless_simd::kernel!(
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn measure_nondiagonal_probability_neon(
        neon: Neon,
        re: &[f64],
        im: &[f64],
        xmask: u64,
        zmask: u64,
        pivot: usize,
        coefficient_re: f64,
        coefficient_im: f64,
        branch: bool,
    ) -> f64 {
        let selector = 1usize << pivot;
        let step = selector << 1;
        let lower_mask = xmask as usize & (selector - 1);
        let swap_lanes = lower_mask & 1 != 0;
        let chunk_mask = lower_mask & !1;
        let c0 = f64x2::splat(neon, INV_SQRT2);
        let cre = f64x2::splat(neon, coefficient_re);
        let cim = f64x2::splat(neon, coefficient_im);
        let zero = f64x2::splat(neon, 0.0);
        let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
        let mut block = 0;
        let mut chunk = 0;
        while block < re.len() {
            let mut offset = 0;
            while offset < selector {
                let source0 = block + offset;
                let source1 = block + selector + (offset ^ chunk_mask);
                let r0 = f64x2::from_slice(neon, &re[source0..source0 + 2]);
                let v0 = f64x2::from_slice(neon, &im[source0..source0 + 2]);
                let mut r1 = f64x2::from_slice(neon, &re[source1..source1 + 2]);
                let mut v1 = f64x2::from_slice(neon, &im[source1..source1 + 2]);
                if swap_lanes {
                    r1 = r1.slide::<1>(r1);
                    v1 = v1.slide::<1>(v1);
                }
                let direction = branch_direction_neon(neon, source0, zmask, branch);
                let c1r = direction * cre;
                let c1i = direction * cim;
                let ar = (-c1i).mul_add(v1, c1r.mul_add(r1, c0 * r0));
                let ai = c1i.mul_add(r1, c1r.mul_add(v1, c0 * v0));
                let mass = ar.mul_add(ar, ai * ai);
                match chunk & 3 {
                    0 => a0 += mass,
                    1 => a1 += mass,
                    2 => a2 += mass,
                    _ => a3 += mass,
                }
                offset += 2;
                chunk += 1;
            }
            block += step;
        }
        let total = a0 + a1 + a2 + a3;
        total[0] + total[1]
    }
);

#[allow(clippy::too_many_arguments)]
fn measure_nondiagonal_probability_soa_scalar(
    re: &[f64],
    im: &[f64],
    dim: usize,
    xmask: u64,
    zmask: u64,
    pivot: usize,
    coefficient1_even: Complex64,
    branch: bool,
) -> f64 {
    let out_dim = dim >> 1;
    let xmask = xmask as usize;
    let mut probability = 0.0;
    for idx in 0..out_dim {
        let source0 = insert_zero_bit(idx, pivot);
        let source1 = source0 ^ xmask;
        let odd = ((source0 as u64 & zmask).count_ones() & 1) != 0;
        let direction = if branch != odd { -1.0 } else { 1.0 };
        let c1r = direction * coefficient1_even.re;
        let c1i = direction * coefficient1_even.im;
        let ar = INV_SQRT2 * re[source0] + c1r * re[source1] - c1i * im[source1];
        let ai = INV_SQRT2 * im[source0] + c1r * im[source1] + c1i * re[source1];
        probability += ar * ar + ai * ai;
    }
    probability
}

// ==============================================================================
// Projection
// ==============================================================================

/// Collapses onto `branch` for a diagonal Pauli, compacting `2^k` amplitudes
/// into the low `2^(k-1)`.
///
/// Safe in place: every gathered source index is at least its output index.
pub fn project_diagonal_contiguous(
    re: &mut [f64],
    im: &mut [f64],
    kernel: &PrecomputedActivePauliMeasurementKernel,
    branch: bool,
    invnorm: f64,
) {
    // Each output block's two candidate source blocks (pivot bit clear/set)
    // are contiguous for `pivot >= 2`, so the gather becomes two loads and a
    // parity blend. Per element this is the same single `* invnorm`
    // rounding, so the vector path is bit-identical to the scalar gather.
    #[cfg(target_arch = "x86_64")]
    {
        let out_dim = kernel.out_dim;
        let pivot = kernel.pivot;
        let zmask = kernel.action.zmask;
        if out_dim >= 4
            && out_dim.is_multiple_of(4)
            && (2..63).contains(&pivot)
            && (zmask >> pivot) & 1 == 1
            && (out_dim << 1) <= re.len()
            && (out_dim << 1) <= im.len()
            && let Some(level) = simd_level()
            && let Some(avx2) = level.as_avx2()
        {
            let target = (kernel.diagonal_source(0, branch) >> pivot) & 1 == 1;
            project_diagonal_blend_avx2(
                avx2,
                re,
                im,
                out_dim,
                pivot as u32,
                zmask & !(1u64 << pivot),
                target,
                invnorm,
            );
            return;
        }
    }
    for idx in 0..kernel.out_dim {
        let source = kernel.diagonal_source(idx, branch);
        re[idx] = re[source] * invnorm;
        im[idx] = im[source] * invnorm;
    }
}

// The feature token proves AVX2+FMA; all accesses use checked slices. The
// sign mask picks the pivot-set candidate exactly where the scalar gather's
// `pivot_value` is one; sources always sit at or above the write index, so
// the in-place compaction never reads a clobbered block.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn project_diagonal_blend_avx2(
        avx2: Avx2,
        re: &mut [f64],
        im: &mut [f64],
        out_dim: usize,
        pivot: u32,
        z_without_pivot: u64,
        target: bool,
        invnorm: f64,
    ) {
        let selector = 1usize << pivot;
        let scale = _mm256_set1_pd(invnorm);
        let mut idx = 0;
        while idx < out_dim {
            let base = insert_zero_bit(idx, pivot as usize);
            let low_r: __m256d = f64x4::from_slice(avx2, &re[base..base + 4]).into();
            let low_v: __m256d = f64x4::from_slice(avx2, &im[base..base + 4]).into();
            let high_r: __m256d =
                f64x4::from_slice(avx2, &re[base + selector..base + selector + 4]).into();
            let high_v: __m256d =
                f64x4::from_slice(avx2, &im[base + selector..base + selector + 4]).into();
            let mask = nondiagonal_sign_mask4!(base, z_without_pivot, target);
            let out_r = _mm256_mul_pd(_mm256_blendv_pd(low_r, high_r, mask), scale);
            let out_v = _mm256_mul_pd(_mm256_blendv_pd(low_v, high_v, mask), scale);
            let out_r: f64x4<_> = out_r.simd_into(avx2);
            let out_v: f64x4<_> = out_v.simd_into(avx2);
            out_r.store_slice(&mut re[idx..idx + 4]);
            out_v.store_slice(&mut im[idx..idx + 4]);
            idx += 4;
        }
    }
);

/// Collapses onto `branch` for a Pauli with an X component.
///
/// Each output mixes two inputs, so this cannot run in place; it writes the
/// compacted state into the caller's scratch and copies it back.
pub fn project_nondiagonal_contiguous(
    re: &mut [f64],
    im: &mut [f64],
    scratch_re: &mut [f64],
    scratch_im: &mut [f64],
    kernel: &PrecomputedActivePauliMeasurementKernel,
    branch: bool,
    invnorm: f64,
) {
    let out_dim = kernel.out_dim;
    project_nondiagonal_soa(
        re,
        im,
        scratch_re,
        scratch_im,
        out_dim << 1,
        kernel.action.xmask,
        kernel.action.zmask,
        kernel.pivot,
        kernel.nondiagonal_coefficient1_even,
        branch,
        invnorm,
    );
    re[..out_dim].copy_from_slice(&scratch_re[..out_dim]);
    im[..out_dim].copy_from_slice(&scratch_im[..out_dim]);
}

// The Ice Lake-level token proves every AVX-512 feature used here; all loads
// and stores use checked slices.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn project_nondiagonal_avx512(
        avx512: Avx512,
        re: &[f64],
        im: &[f64],
        out_re: &mut [f64],
        out_im: &mut [f64],
        xmask: u64,
        zmask: u64,
        pivot: usize,
        coefficient_re: f64,
        coefficient_im: f64,
        branch: bool,
        invnorm: f64,
    ) {
        let dim = re.len();
        let selector = 1usize << pivot;
        let step = selector << 1;
        let lower_mask = xmask as usize & (selector - 1);
        let lane_mask = lower_mask & 7;
        let chunk_mask = lower_mask & !7;
        let coefficient_re = _mm512_set1_pd(coefficient_re);
        let coefficient_im = _mm512_set1_pd(coefficient_im);
        let invnorm = _mm512_set1_pd(invnorm);
        let mut block = 0;
        while block < dim {
            let output_base = block >> 1;
            let mut offset = 0;
            while offset < selector {
                let source0 = block + offset;
                let source1 = block + selector + (offset ^ chunk_mask);
                let r0: __m512d = f64x8::from_slice(avx512, &re[source0..source0 + 8]).into();
                let im0: __m512d = f64x8::from_slice(avx512, &im[source0..source0 + 8]).into();
                let r1: __m512d = f64x8::from_slice(avx512, &re[source1..source1 + 8]).into();
                let im1: __m512d = f64x8::from_slice(avx512, &im[source1..source1 + 8]).into();
                let r1 = permute_xor8!(r1, lane_mask);
                let im1 = permute_xor8!(im1, lane_mask);
                let sign = nondiagonal_sign_mask8!(source0, zmask, branch);
                let (ar, ai) = nondiagonal_amplitudes8!(
                    r0,
                    im0,
                    r1,
                    im1,
                    sign,
                    coefficient_re,
                    coefficient_im
                );
                let ar: f64x8<_> = _mm512_mul_pd(ar, invnorm).simd_into(avx512);
                let ai: f64x8<_> = _mm512_mul_pd(ai, invnorm).simd_into(avx512);
                ar.store_slice(&mut out_re[output_base + offset..output_base + offset + 8]);
                ai.store_slice(&mut out_im[output_base + offset..output_base + offset + 8]);
                offset += 8;
            }
            block += step;
        }
    }
);

// AVX2/FMA is proved by the token; all loads and stores use checked slices.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn project_nondiagonal_avx2(
        avx2: Avx2,
        re: &[f64],
        im: &[f64],
        out_re: &mut [f64],
        out_im: &mut [f64],
        xmask: u64,
        zmask: u64,
        pivot: usize,
        coefficient_re: f64,
        coefficient_im: f64,
        branch: bool,
        invnorm: f64,
    ) {
        let dim = re.len();
        let selector = 1usize << pivot;
        let step = selector << 1;
        let lower_mask = xmask as usize & (selector - 1);
        let lane_mask = lower_mask & 3;
        let chunk_mask = lower_mask & !3;
        let coefficient_re = _mm256_set1_pd(coefficient_re);
        let coefficient_im = _mm256_set1_pd(coefficient_im);
        let invnorm = _mm256_set1_pd(invnorm);
        let mut block = 0;
        while block < dim {
            let output_base = block >> 1;
            let mut offset = 0;
            while offset < selector {
                let source0 = block + offset;
                let source1 = block + selector + (offset ^ chunk_mask);
                let r0: __m256d = f64x4::from_slice(avx2, &re[source0..source0 + 4]).into();
                let im0: __m256d = f64x4::from_slice(avx2, &im[source0..source0 + 4]).into();
                let r1: __m256d = f64x4::from_slice(avx2, &re[source1..source1 + 4]).into();
                let im1: __m256d = f64x4::from_slice(avx2, &im[source1..source1 + 4]).into();
                let r1 = permute_xor4!(r1, lane_mask);
                let im1 = permute_xor4!(im1, lane_mask);
                let sign = nondiagonal_sign_mask4!(source0, zmask, branch);
                let (ar, ai) = nondiagonal_amplitudes4!(
                    r0,
                    im0,
                    r1,
                    im1,
                    sign,
                    coefficient_re,
                    coefficient_im
                );
                let ar: f64x4<_> = _mm256_mul_pd(ar, invnorm).simd_into(avx2);
                let ai: f64x4<_> = _mm256_mul_pd(ai, invnorm).simd_into(avx2);
                ar.store_slice(&mut out_re[output_base + offset..output_base + offset + 4]);
                ai.store_slice(&mut out_im[output_base + offset..output_base + offset + 4]);
                offset += 4;
            }
            block += step;
        }
    }
);

/// SIMD seam: the projection matching
/// [`measure_nondiagonal_probability_soa`], scaled by `invnorm`.
#[allow(clippy::too_many_arguments)]
pub fn project_nondiagonal_soa(
    re: &[f64],
    im: &[f64],
    out_re: &mut [f64],
    out_im: &mut [f64],
    dim: usize,
    xmask: u64,
    zmask: u64,
    pivot: usize,
    coefficient1_even: Complex64,
    branch: bool,
    invnorm: f64,
) {
    let out_dim = dim >> 1;
    #[cfg(target_arch = "x86_64")]
    if let Some(level) = simd_level() {
        if pivot >= 3
            && let Some(avx512) = level.as_avx512()
        {
            project_nondiagonal_avx512(
                avx512,
                &re[..dim],
                &im[..dim],
                &mut out_re[..out_dim],
                &mut out_im[..out_dim],
                xmask,
                zmask,
                pivot,
                coefficient1_even.re,
                coefficient1_even.im,
                branch,
                invnorm,
            );
            return;
        }
        if pivot >= 2
            && let Some(avx2) = level.as_avx2()
        {
            project_nondiagonal_avx2(
                avx2,
                &re[..dim],
                &im[..dim],
                &mut out_re[..out_dim],
                &mut out_im[..out_dim],
                xmask,
                zmask,
                pivot,
                coefficient1_even.re,
                coefficient1_even.im,
                branch,
                invnorm,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    if pivot >= 1
        && dim.is_multiple_of(2)
        && let Some(neon) = simd_level().and_then(Level::as_neon)
    {
        project_nondiagonal_neon(
            neon,
            &re[..dim],
            &im[..dim],
            &mut out_re[..out_dim],
            &mut out_im[..out_dim],
            xmask,
            zmask,
            pivot,
            coefficient1_even.re,
            coefficient1_even.im,
            branch,
            invnorm,
        );
        return;
    }

    project_nondiagonal_soa_scalar(
        re,
        im,
        out_re,
        out_im,
        dim,
        xmask,
        zmask,
        pivot,
        coefficient1_even,
        branch,
        invnorm,
    );
}

#[cfg(target_arch = "aarch64")]
fearless_simd::kernel!(
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn project_nondiagonal_neon(
        neon: Neon,
        re: &[f64],
        im: &[f64],
        out_re: &mut [f64],
        out_im: &mut [f64],
        xmask: u64,
        zmask: u64,
        pivot: usize,
        coefficient_re: f64,
        coefficient_im: f64,
        branch: bool,
        invnorm: f64,
    ) {
        let selector = 1usize << pivot;
        let step = selector << 1;
        let lower_mask = xmask as usize & (selector - 1);
        let swap_lanes = lower_mask & 1 != 0;
        let chunk_mask = lower_mask & !1;
        let c0 = f64x2::splat(neon, INV_SQRT2);
        let cre = f64x2::splat(neon, coefficient_re);
        let cim = f64x2::splat(neon, coefficient_im);
        let scale = f64x2::splat(neon, invnorm);
        let mut block = 0;
        while block < re.len() {
            let output_base = block >> 1;
            let mut offset = 0;
            while offset < selector {
                let source0 = block + offset;
                let source1 = block + selector + (offset ^ chunk_mask);
                let r0 = f64x2::from_slice(neon, &re[source0..source0 + 2]);
                let v0 = f64x2::from_slice(neon, &im[source0..source0 + 2]);
                let mut r1 = f64x2::from_slice(neon, &re[source1..source1 + 2]);
                let mut v1 = f64x2::from_slice(neon, &im[source1..source1 + 2]);
                if swap_lanes {
                    r1 = r1.slide::<1>(r1);
                    v1 = v1.slide::<1>(v1);
                }
                let direction = branch_direction_neon(neon, source0, zmask, branch);
                let c1r = direction * cre;
                let c1i = direction * cim;
                let ar = (-c1i).mul_add(v1, c1r.mul_add(r1, c0 * r0)) * scale;
                let ai = c1i.mul_add(r1, c1r.mul_add(v1, c0 * v0)) * scale;
                ar.store_slice(&mut out_re[output_base + offset..output_base + offset + 2]);
                ai.store_slice(&mut out_im[output_base + offset..output_base + offset + 2]);
                offset += 2;
            }
            block += step;
        }
    }
);

#[allow(clippy::too_many_arguments)]
fn project_nondiagonal_soa_scalar(
    re: &[f64],
    im: &[f64],
    out_re: &mut [f64],
    out_im: &mut [f64],
    dim: usize,
    xmask: u64,
    zmask: u64,
    pivot: usize,
    coefficient1_even: Complex64,
    branch: bool,
    invnorm: f64,
) {
    let out_dim = dim >> 1;
    let xmask = xmask as usize;
    for idx in 0..out_dim {
        let source0 = insert_zero_bit(idx, pivot);
        let source1 = source0 ^ xmask;
        let odd = ((source0 as u64 & zmask).count_ones() & 1) != 0;
        let direction = if branch != odd { -1.0 } else { 1.0 };
        let c1r = direction * coefficient1_even.re;
        let c1i = direction * coefficient1_even.im;
        let ar = INV_SQRT2 * re[source0] + c1r * re[source1] - c1i * im[source1];
        let ai = INV_SQRT2 * im[source0] + c1r * im[source1] + c1i * re[source1];
        out_re[idx] = ar * invnorm;
        out_im[idx] = ai * invnorm;
    }
}

/// One dormant-qubit promotion inside a register-resident run:
/// `alpha' = [c * alpha, -i q * alpha]`, doubling the state. Both sign
/// variants of `q` ride along like [`UniformImagRunStep`]'s.
#[derive(Clone, Copy, Debug)]
pub struct PromotionRunStep {
    pub cos: f64,
    pub imag_false: f64,
    pub imag_true: f64,
}

/// Applies a promotion prefix followed by a rotation run to one shot,
/// register-resident throughout: the pre-promotion state (`start_dim`
/// amplitudes, 2/4/8) is loaded, each promotion doubles it in registers
/// (element-wise `new_re = -q * old_im`, `new_im = q * old_re`,
/// `old *= c` — exactly [`promote_contiguous_active`]'s arithmetic), the
/// rotations run as in [`rotate_uniform_imag_run_dim16`], and the dim-16
/// state is stored once. Sign-mask bits cover promotions first, then
/// rotations. Returns `false` (state untouched) without a vector backend.
pub fn promote_rotate_uniform_imag_run_dim16(
    re: &mut [f64],
    im: &mut [f64],
    start_dim: usize,
    promotions: &[PromotionRunStep],
    steps: &[UniformImagRunStep],
    sign_mask: u32,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if re.len() >= 16
        && im.len() >= 16
        && matches!(start_dim, 2 | 4 | 8)
        && start_dim << promotions.len() == 16
        && promotions.len() + steps.len() <= 32
        && steps.iter().all(|step| step.xmask >= 1 && step.xmask <= 15)
        && let Some(level) = simd_level()
    {
        if let Some(avx512) = level.as_avx512() {
            promote_rotate_uniform_imag_run_dim16_avx512(
                avx512,
                &mut re[..16],
                &mut im[..16],
                start_dim,
                promotions,
                steps,
                sign_mask,
            );
            return true;
        }
        if let Some(avx2) = level.as_avx2() {
            promote_rotate_uniform_imag_run_dim16_avx2(
                avx2,
                &mut re[..16],
                &mut im[..16],
                start_dim,
                promotions,
                steps,
                sign_mask,
            );
            return true;
        }
    }
    let _ = (re, im, start_dim, promotions, steps, sign_mask);
    false
}

// The promotion ladder reaches eight amplitudes before the final widening:
// start-dim 2 uses one scalar-exact rung, start-dim 4 loads one YMM, and
// start-dim 8 loads one ZMM. The final promotion creates the second ZMM;
// rotations then keep the whole dim-16 state in four registers.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    fn promote_rotate_uniform_imag_run_dim16_avx512(
        avx512: Avx512,
        re: &mut [f64],
        im: &mut [f64],
        start_dim: usize,
        promotions: &[PromotionRunStep],
        steps: &[UniformImagRunStep],
        sign_mask: u32,
    ) {
        let mut promo = 0usize;
        let (mut r0, mut v0): (__m512d, __m512d);

        if start_dim == 8 {
            r0 = f64x8::from_slice(avx512, &re[0..8]).into();
            v0 = f64x8::from_slice(avx512, &im[0..8]).into();
        } else {
            let (lower_r, lower_v) = if start_dim == 2 {
                let step = &promotions[0];
                let q = if sign_mask & 1 == 1 {
                    step.imag_true
                } else {
                    step.imag_false
                };
                promo = 1;
                let (c, nq) = (step.cos, -q);
                let (re0, re1, im0, im1) = (re[0], re[1], im[0], im[1]);
                (
                    _mm256_set_pd(nq * im1, nq * im0, c * re1, c * re0),
                    _mm256_set_pd(q * re1, q * re0, c * im1, c * im0),
                )
            } else {
                (
                    _mm256_set_pd(re[3], re[2], re[1], re[0]),
                    _mm256_set_pd(im[3], im[2], im[1], im[0]),
                )
            };

            let step = &promotions[promo];
            let q = if (sign_mask >> promo) & 1 == 1 {
                step.imag_true
            } else {
                step.imag_false
            };
            promo += 1;
            let vc = _mm256_set1_pd(step.cos);
            let vq = _mm256_set1_pd(q);
            let vnq = _mm256_set1_pd(-q);
            let upper_r = _mm256_mul_pd(vnq, lower_v);
            let upper_v = _mm256_mul_pd(vq, lower_r);
            r0 = _mm512_insertf64x4::<1>(
                _mm512_castpd256_pd512(_mm256_mul_pd(vc, lower_r)),
                upper_r,
            );
            v0 = _mm512_insertf64x4::<1>(
                _mm512_castpd256_pd512(_mm256_mul_pd(vc, lower_v)),
                upper_v,
            );
        }

        let step = &promotions[promo];
        let q = if (sign_mask >> promo) & 1 == 1 {
            step.imag_true
        } else {
            step.imag_false
        };
        let vc = _mm512_set1_pd(step.cos);
        let vq = _mm512_set1_pd(q);
        let vnq = _mm512_set1_pd(-q);
        let mut r1 = _mm512_mul_pd(vnq, v0);
        let mut v1 = _mm512_mul_pd(vq, r0);
        r0 = _mm512_mul_pd(vc, r0);
        v0 = _mm512_mul_pd(vc, v0);

        for (index, step) in steps.iter().enumerate() {
            let vc = _mm512_set1_pd(step.cos);
            let vq = _mm512_set1_pd(if (sign_mask >> (promotions.len() + index)) & 1 == 1 {
                step.imag_true
            } else {
                step.imag_false
            });
            let lane = (step.xmask & 7) as usize;
            if step.xmask < 8 {
                rotate_run_self8!(r0, v0, lane, vc, vq);
                rotate_run_self8!(r1, v1, lane, vc, vq);
            } else {
                rotate_run_pair8!(r0, v0, r1, v1, lane, vc, vq);
            }
        }

        let r0: f64x8<_> = r0.simd_into(avx512);
        let r1: f64x8<_> = r1.simd_into(avx512);
        let v0: f64x8<_> = v0.simd_into(avx512);
        let v1: f64x8<_> = v1.simd_into(avx512);
        r0.store_slice(&mut re[0..8]);
        r1.store_slice(&mut re[8..16]);
        v0.store_slice(&mut im[0..8]);
        v1.store_slice(&mut im[8..16]);
    }
);

// Same proof obligations as the pure-rotation run kernel. The promotion
// ladder keeps the register naming invariant — after promoting past dim `d`
// the live vectors are exactly `r0..r{d/8}`/`v0..v{d/8}` — so the rotation
// loop below is byte-for-byte the one in `rotate_uniform_imag_run_dim16_avx2`
// with the sign bits offset past the promotions.
#[cfg(target_arch = "x86_64")]
fearless_simd::kernel!(
    #[inline]
    fn promote_rotate_uniform_imag_run_dim16_avx2(
        avx2: Avx2,
        re: &mut [f64],
        im: &mut [f64],
        start_dim: usize,
        promotions: &[PromotionRunStep],
        steps: &[UniformImagRunStep],
        sign_mask: u32,
    ) {
        let mut promo = 0usize;

        let mut r1 = _mm256_setzero_pd();
        let mut r2 = _mm256_setzero_pd();
        let mut r3 = _mm256_setzero_pd();
        let mut v1 = _mm256_setzero_pd();
        let mut v2 = _mm256_setzero_pd();
        let mut v3 = _mm256_setzero_pd();
        let (mut r0, mut v0);

        let mut dim = start_dim;
        if dim == 2 {
            // Scalar products here match `promote_contiguous_active`'s
            // element order exactly; the lanes are only packed differently.
            let step = &promotions[0];
            let q = if (sign_mask >> promo) & 1 == 1 {
                step.imag_true
            } else {
                step.imag_false
            };
            promo += 1;
            let (c, nq) = (step.cos, -q);
            let (re0, re1, im0, im1) = (re[0], re[1], im[0], im[1]);
            r0 = _mm256_set_pd(nq * im1, nq * im0, c * re1, c * re0);
            v0 = _mm256_set_pd(q * re1, q * re0, c * im1, c * im0);
            dim = 4;
        } else {
            r0 = f64x4::from_slice(avx2, &re[0..4]).into();
            v0 = f64x4::from_slice(avx2, &im[0..4]).into();
            if dim == 8 {
                r1 = f64x4::from_slice(avx2, &re[4..8]).into();
                v1 = f64x4::from_slice(avx2, &im[4..8]).into();
            }
        }
        if dim == 4 {
            let step = &promotions[promo];
            let q = if (sign_mask >> promo) & 1 == 1 {
                step.imag_true
            } else {
                step.imag_false
            };
            promo += 1;
            let vc = _mm256_set1_pd(step.cos);
            let vq = _mm256_set1_pd(q);
            let vnq = _mm256_set1_pd(-q);
            r1 = _mm256_mul_pd(vnq, v0);
            v1 = _mm256_mul_pd(vq, r0);
            r0 = _mm256_mul_pd(vc, r0);
            v0 = _mm256_mul_pd(vc, v0);
            dim = 8;
        }
        if dim == 8 {
            let step = &promotions[promo];
            let q = if (sign_mask >> promo) & 1 == 1 {
                step.imag_true
            } else {
                step.imag_false
            };
            let vc = _mm256_set1_pd(step.cos);
            let vq = _mm256_set1_pd(q);
            let vnq = _mm256_set1_pd(-q);
            r2 = _mm256_mul_pd(vnq, v0);
            r3 = _mm256_mul_pd(vnq, v1);
            v2 = _mm256_mul_pd(vq, r0);
            v3 = _mm256_mul_pd(vq, r1);
            r0 = _mm256_mul_pd(vc, r0);
            r1 = _mm256_mul_pd(vc, r1);
            v0 = _mm256_mul_pd(vc, v0);
            v1 = _mm256_mul_pd(vc, v1);
        }

        for (index, step) in steps.iter().enumerate() {
            let vc = _mm256_set1_pd(step.cos);
            let vq = _mm256_set1_pd(if (sign_mask >> (promotions.len() + index)) & 1 == 1 {
                step.imag_true
            } else {
                step.imag_false
            });
            let lane = (step.xmask & 3) as usize;
            match (step.xmask >> 2) & 3 {
                0 => {
                    rotate_run_self4!(r0, v0, lane, vc, vq);
                    rotate_run_self4!(r1, v1, lane, vc, vq);
                    rotate_run_self4!(r2, v2, lane, vc, vq);
                    rotate_run_self4!(r3, v3, lane, vc, vq);
                }
                1 => {
                    rotate_run_pair4!(r0, v0, r1, v1, lane, vc, vq);
                    rotate_run_pair4!(r2, v2, r3, v3, lane, vc, vq);
                }
                2 => {
                    rotate_run_pair4!(r0, v0, r2, v2, lane, vc, vq);
                    rotate_run_pair4!(r1, v1, r3, v3, lane, vc, vq);
                }
                _ => {
                    rotate_run_pair4!(r0, v0, r3, v3, lane, vc, vq);
                    rotate_run_pair4!(r1, v1, r2, v2, lane, vc, vq);
                }
            }
        }

        let r0: f64x4<_> = r0.simd_into(avx2);
        let r1: f64x4<_> = r1.simd_into(avx2);
        let r2: f64x4<_> = r2.simd_into(avx2);
        let r3: f64x4<_> = r3.simd_into(avx2);
        let v0: f64x4<_> = v0.simd_into(avx2);
        let v1: f64x4<_> = v1.simd_into(avx2);
        let v2: f64x4<_> = v2.simd_into(avx2);
        let v3: f64x4<_> = v3.simd_into(avx2);
        r0.store_slice(&mut re[0..4]);
        r1.store_slice(&mut re[4..8]);
        r2.store_slice(&mut re[8..12]);
        r3.store_slice(&mut re[12..16]);
        v0.store_slice(&mut im[0..4]);
        v1.store_slice(&mut im[4..8]);
        v2.store_slice(&mut im[8..12]);
        v3.store_slice(&mut im[12..16]);
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active::ActivePauliAction;
    use crate::pauli::{pauli_x, pauli_y, pauli_z};
    use num_complex::Complex64;

    fn split(alpha: &[Complex64]) -> (Vec<f64>, Vec<f64>) {
        (
            alpha.iter().map(|value| value.re).collect(),
            alpha.iter().map(|value| value.im).collect(),
        )
    }

    fn sample_alpha(k: usize) -> Vec<Complex64> {
        (0..1usize << k)
            .map(|basis| {
                Complex64::new(
                    0.01 * (basis as f64 + 1.0),
                    -0.02 * ((basis % 7) as f64 + 1.0),
                )
            })
            .collect()
    }

    /// Every four-qubit diagonal (Z-only) measurement shape: the blend
    /// projection must be bit-identical to the scalar gather, and the masked
    /// probability must agree within SIMD tolerance, on both branches.
    #[test]
    fn diagonal_kernels_match_the_scalar_gather_for_every_z_mask() {
        let (initial_re, initial_im) = split(&sample_alpha(4));
        for zmask in 1u64..16 {
            let mut pauli = crate::pauli::pauli_identity(4);
            for qubit in 0..4 {
                if (zmask >> qubit) & 1 == 1 {
                    pauli = &pauli * &pauli_z(4, qubit);
                }
            }
            let kernel =
                PrecomputedActivePauliMeasurementKernel::from_pauli(&pauli).expect("Z-only Pauli");
            assert!(kernel.is_diagonal);
            for branch in [false, true] {
                let mut expected_probability = 0.0;
                for idx in 0..kernel.out_dim {
                    let source = kernel.diagonal_source(idx, branch);
                    expected_probability += initial_re[source] * initial_re[source]
                        + initial_im[source] * initial_im[source];
                }
                let probability =
                    diagonal_probability_contiguous(&initial_re, &initial_im, &kernel, branch);
                assert!(
                    (probability - expected_probability.clamp(0.0, 1.0)).abs() < 1e-14,
                    "zmask {zmask} branch {branch}"
                );

                let (mut expected_re, mut expected_im) = (initial_re.clone(), initial_im.clone());
                for idx in 0..kernel.out_dim {
                    let source = kernel.diagonal_source(idx, branch);
                    expected_re[idx] = initial_re[source] * 1.25;
                    expected_im[idx] = initial_im[source] * 1.25;
                }
                let (mut actual_re, mut actual_im) = (initial_re.clone(), initial_im.clone());
                project_diagonal_contiguous(&mut actual_re, &mut actual_im, &kernel, branch, 1.25);
                assert_eq!(
                    actual_re[..kernel.out_dim],
                    expected_re[..kernel.out_dim],
                    "zmask {zmask} branch {branch}"
                );
                assert_eq!(
                    actual_im[..kernel.out_dim],
                    expected_im[..kernel.out_dim],
                    "zmask {zmask} branch {branch}"
                );
            }
        }
    }

    /// The register-resident run kernel must be bit-identical to sequential
    /// per-rotation vector calls — same FMA shapes, one rounding per output —
    /// across every four-qubit X mask and mixed-mask runs.
    #[test]
    fn register_run_matches_sequential_rotations_exactly() {
        if !has_uniform_imag_run_dim16_backend() {
            return;
        }
        let (initial_re, initial_im) = split(&sample_alpha(4));
        // One run visiting all 15 masks with distinct angles and alternating
        // sign bits, plus each mask alone under both sign selections.
        let mut runs: Vec<(Vec<UniformImagRunStep>, u32)> = (1u64..16)
            .flat_map(|xmask| {
                let step = UniformImagRunStep {
                    xmask,
                    cos: 0.91,
                    imag_false: -0.17,
                    imag_true: 0.29,
                };
                [(vec![step], 0u32), (vec![step], 1u32)]
            })
            .collect();
        runs.push((
            (1u64..16)
                .map(|xmask| UniformImagRunStep {
                    xmask,
                    cos: (0.05 * xmask as f64).cos(),
                    imag_false: -0.23,
                    imag_true: 0.31,
                })
                .collect(),
            0b101_0101_0101_0101,
        ));
        for (steps, sign_mask) in runs {
            let (mut expected_re, mut expected_im) = (initial_re.clone(), initial_im.clone());
            let (mut actual_re, mut actual_im) = (initial_re.clone(), initial_im.clone());
            for (index, step) in steps.iter().enumerate() {
                rotate_uniform_imag_pairs_soa(
                    &mut expected_re,
                    &mut expected_im,
                    16,
                    step.xmask,
                    63 - step.xmask.leading_zeros(),
                    step.cos,
                    if (sign_mask >> index) & 1 == 1 {
                        step.imag_true
                    } else {
                        step.imag_false
                    },
                );
            }
            assert!(rotate_uniform_imag_run_dim16(
                &mut actual_re,
                &mut actual_im,
                &steps,
                sign_mask,
            ));
            assert_eq!(actual_re, expected_re, "run of {} steps", steps.len());
            assert_eq!(actual_im, expected_im, "run of {} steps", steps.len());
        }
    }

    #[test]
    fn diagonal_run_matches_sequential_rotations_exactly() {
        if !has_diagonal_run_dim32_backend() {
            return;
        }
        let zmasks = [1u64, 3, 7, 18, 31];
        let angles = [0.11, -0.23, 0.37, -0.41, 0.53];
        let mut kernels = [PrecomputedActivePauliRotationKernel::default(); 5];
        let mut steps = [DiagonalRunStep::default(); 5];
        for index in 0..5 {
            let mut pauli = crate::pauli::pauli_identity(5);
            for qubit in 0..5 {
                if (zmasks[index] >> qubit) & 1 == 1 {
                    pauli = &pauli * &pauli_z(5, qubit);
                }
            }
            let action = ActivePauliAction::new(&pauli).expect("Hermitian diagonal Pauli");
            let kernel = PrecomputedActivePauliRotationKernel::new(&action, angles[index])
                .expect("five-qubit kernel");
            steps[index] = DiagonalRunStep::new(
                action.zmask,
                kernel.cos_kernel_angle,
                kernel.minus_even_coefficient,
            );
            kernels[index] = kernel;
        }

        let (initial_re, initial_im) = split(&sample_alpha(5));
        for sign_mask in 0u32..32 {
            let (mut expected_re, mut expected_im) = (initial_re.clone(), initial_im.clone());
            let (mut actual_re, mut actual_im) = (initial_re.clone(), initial_im.clone());
            for (index, kernel) in kernels.iter().enumerate() {
                rotate_contiguous_active(
                    &mut expected_re,
                    &mut expected_im,
                    32,
                    kernel,
                    (sign_mask >> index) & 1 != 0,
                );
            }
            assert!(rotate_diagonal_run_dim32(
                &mut actual_re,
                &mut actual_im,
                &steps,
                sign_mask,
            ));
            assert_eq!(actual_re, expected_re, "sign mask {sign_mask:#07b}");
            assert_eq!(actual_im, expected_im, "sign mask {sign_mask:#07b}");
        }
    }

    #[test]
    fn promotion_run_matches_sequential_promotions_and_rotations_exactly() {
        if !has_uniform_imag_run_dim16_backend() {
            return;
        }
        // Distinct angle per promotion rung so a mixed-up rung order cannot
        // cancel out; 99.0 sentinels catch any slot the kernels fail to write.
        let promo_ladder = [
            PromotionRunStep {
                cos: 0.93,
                imag_false: -0.36,
                imag_true: 0.36,
            },
            PromotionRunStep {
                cos: 0.87,
                imag_false: -0.49,
                imag_true: 0.49,
            },
            PromotionRunStep {
                cos: 0.79,
                imag_false: -0.61,
                imag_true: 0.61,
            },
        ];
        let rotation_runs: [&[UniformImagRunStep]; 2] = [
            &[UniformImagRunStep {
                xmask: 9,
                cos: 0.95,
                imag_false: -0.31,
                imag_true: 0.31,
            }],
            &[
                UniformImagRunStep {
                    xmask: 1,
                    cos: 0.97,
                    imag_false: -0.11,
                    imag_true: 0.13,
                },
                UniformImagRunStep {
                    xmask: 5,
                    cos: 0.89,
                    imag_false: -0.19,
                    imag_true: 0.23,
                },
                UniformImagRunStep {
                    xmask: 10,
                    cos: 0.83,
                    imag_false: -0.27,
                    imag_true: 0.33,
                },
                UniformImagRunStep {
                    xmask: 15,
                    cos: 0.71,
                    imag_false: -0.41,
                    imag_true: 0.43,
                },
            ],
        ];
        for start_k in 1usize..=3 {
            let start_dim = 1 << start_k;
            let promotions = &promo_ladder[..4 - start_k];
            let (state_re, state_im) = split(&sample_alpha(start_k));
            for steps in rotation_runs {
                for promo_bits in 0u32..1 << promotions.len() {
                    for rot_bits in [0u32, 0b0101, 0b1111] {
                        let sign_mask = promo_bits | (rot_bits << promotions.len());
                        let mut expected_re = vec![99.0f64; 16];
                        let mut expected_im = vec![99.0f64; 16];
                        expected_re[..start_dim].copy_from_slice(&state_re);
                        expected_im[..start_dim].copy_from_slice(&state_im);
                        let mut actual_re = expected_re.clone();
                        let mut actual_im = expected_im.clone();

                        let mut dim = start_dim;
                        for (index, step) in promotions.iter().enumerate() {
                            let q = if (sign_mask >> index) & 1 == 1 {
                                step.imag_true
                            } else {
                                step.imag_false
                            };
                            promote_contiguous_active(
                                &mut expected_re,
                                &mut expected_im,
                                dim,
                                step.cos,
                                q,
                            );
                            dim *= 2;
                        }
                        for (index, step) in steps.iter().enumerate() {
                            let q = if (sign_mask >> (promotions.len() + index)) & 1 == 1 {
                                step.imag_true
                            } else {
                                step.imag_false
                            };
                            rotate_uniform_imag_pairs_soa(
                                &mut expected_re,
                                &mut expected_im,
                                16,
                                step.xmask,
                                63 - step.xmask.leading_zeros(),
                                step.cos,
                                q,
                            );
                        }

                        assert!(promote_rotate_uniform_imag_run_dim16(
                            &mut actual_re,
                            &mut actual_im,
                            start_dim,
                            promotions,
                            steps,
                            sign_mask,
                        ));
                        assert_eq!(
                            actual_re,
                            expected_re,
                            "start dim {start_dim}, {} rotations, mask {sign_mask:b}",
                            steps.len()
                        );
                        assert_eq!(
                            actual_im,
                            expected_im,
                            "start dim {start_dim}, {} rotations, mask {sign_mask:b}",
                            steps.len()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn uniform_imag_dispatch_matches_scalar_for_every_four_qubit_x_mask() {
        let (initial_re, initial_im) = split(&sample_alpha(4));
        for xmask in 1u64..16 {
            let pair_bit = 63 - xmask.leading_zeros();
            let (mut expected_re, mut expected_im) = (initial_re.clone(), initial_im.clone());
            let (mut actual_re, mut actual_im) = (initial_re.clone(), initial_im.clone());
            rotate_uniform_imag_pairs_soa_scalar(
                &mut expected_re,
                &mut expected_im,
                16,
                xmask,
                pair_bit,
                0.91,
                -0.17,
            );
            rotate_uniform_imag_pairs_soa(
                &mut actual_re,
                &mut actual_im,
                16,
                xmask,
                pair_bit,
                0.91,
                -0.17,
            );
            for basis in 0..16 {
                assert!(
                    (actual_re[basis] - expected_re[basis]).abs() < 1e-14
                        && (actual_im[basis] - expected_im[basis]).abs() < 1e-14,
                    "xmask {xmask:#x}, basis {basis} diverged"
                );
            }
        }
    }

    #[test]
    fn promotion_writes_the_new_half() {
        let mut re = vec![1.0, 2.0, 0.0, 0.0];
        let mut im = vec![3.0, 4.0, 0.0, 0.0];
        promote_contiguous_active(&mut re, &mut im, 2, 0.5, 0.25);
        assert_eq!(re, vec![0.5, 1.0, -0.75, -1.0]);
        assert_eq!(im, vec![1.5, 2.0, 0.25, 0.5]);
    }

    #[test]
    fn diagonal_branch_probabilities_sum_to_one() {
        let k = 3;
        let alpha = sample_alpha(k);
        let norm: f64 = alpha.iter().map(|value| value.norm_sqr()).sum();
        let alpha: Vec<Complex64> = alpha.iter().map(|value| value / norm.sqrt()).collect();
        let (re, im) = split(&alpha);
        let kernel =
            PrecomputedActivePauliMeasurementKernel::from_pauli(&(&pauli_z(k, 0) * &pauli_z(k, 2)))
                .expect("diagonal measurement kernel");
        let p_true = diagonal_probability_contiguous(&re, &im, &kernel, true);
        let p_false = diagonal_probability_contiguous(&re, &im, &kernel, false);
        assert!((p_true + p_false - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nondiagonal_branch_probabilities_sum_to_one() {
        let k = 3;
        let alpha = sample_alpha(k);
        let norm: f64 = alpha.iter().map(|value| value.norm_sqr()).sum();
        let alpha: Vec<Complex64> = alpha.iter().map(|value| value / norm.sqrt()).collect();
        let (re, im) = split(&alpha);
        let kernel =
            PrecomputedActivePauliMeasurementKernel::from_pauli(&(&pauli_x(k, 0) * &pauli_y(k, 2)))
                .expect("nondiagonal measurement kernel");
        let p_true = nondiagonal_probability_contiguous(&re, &im, &kernel, true);
        let p_false = nondiagonal_probability_contiguous(&re, &im, &kernel, false);
        assert!((p_true + p_false - 1.0).abs() < 1e-12);
    }
}
