//! Active-state kernels over
//! all live shots, in both layouts.
//!
//! Shot-major delegates to the contiguous kernels per shot;
//! basis-major vectorizes across the shot lanes with per-shot sign handled
//! branch-free — signs become a ±coefficient lane vector, and per-shot branch
//! divergence becomes a per-lane source-index or coefficient select.

use super::symbols::assign_batch_symbol;
use super::{
    BatchFactoredExecutorState, XMASK_ROTATION_PAIR_THRESHOLD, batch_live_word_mask,
    batch_shot_mask, batch_shot_word, runtime_batch_word_count,
};
use crate::active::{
    PrecomputedActivePauliMeasurementKernel, PrecomputedActivePauliRotationKernel, active_length,
    insert_zero_bit,
};
use crate::contiguous::{
    diagonal_probability_contiguous, nondiagonal_probability_contiguous,
    project_diagonal_contiguous, project_nondiagonal_contiguous, promote_contiguous_active,
    rotate_contiguous_active,
};
use crate::errors::{Result, TicitError};
use crate::random::sample_bernoulli;

const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BatchSignMode {
    AllMinus,
    AllPlus,
    Mixed,
}

pub fn batch_bit_at(bits: &[u64], shot: usize) -> bool {
    let word = batch_shot_word(shot);
    word < bits.len() && (bits[word] & batch_shot_mask(shot)) != 0
}

/// Materializes a per-shot scalar from a sign bit-plane. Only
/// `[0, active_shots)` lanes are written; the tail stays stale by design.
pub(crate) fn fill_shot_coefficient_scalars(
    scalars: &mut Vec<f64>,
    runtime: &BatchFactoredExecutorState,
    sign_bits: &[u64],
    minus_coeff: f64,
    plus_coeff: f64,
) {
    let lanes = runtime.active_pitch;
    if scalars.len() < lanes {
        scalars.resize(lanes, 0.0);
    }
    let nwords = runtime_batch_word_count(runtime);
    for word in 0..nwords {
        let bits = if word < sign_bits.len() {
            sign_bits[word]
        } else {
            0
        };
        let base_shot = word << 6;
        let live = 64.min(runtime.active_shots - base_shot);
        for bit in 0..live {
            scalars[base_shot + bit] = if (bits >> bit) & 1 != 0 {
                plus_coeff
            } else {
                minus_coeff
            };
        }
    }
}

pub(crate) fn batch_sign_mode(
    runtime: &BatchFactoredExecutorState,
    sign_bits: &[u64],
) -> BatchSignMode {
    let mut saw_zero = false;
    let mut saw_one = false;
    let nwords = runtime_batch_word_count(runtime);
    for word in 0..nwords {
        let live = batch_live_word_mask(runtime, word);
        let bits = if word < sign_bits.len() {
            sign_bits[word] & live
        } else {
            0
        };
        saw_one = saw_one || bits != 0;
        saw_zero = saw_zero || bits != live;
        if saw_one && saw_zero {
            return BatchSignMode::Mixed;
        }
    }
    if saw_one {
        BatchSignMode::AllPlus
    } else {
        BatchSignMode::AllMinus
    }
}

fn finish_active_measurement_branch(
    runtime: &mut BatchFactoredExecutorState,
    branch_condition: i32,
    branch_bits: &[u64],
) -> Result<()> {
    runtime.k -= 1;
    runtime.ndormant += 1;
    assign_batch_symbol(runtime, branch_condition, branch_bits)
}

/// Copies only the live lanes of the first `out_dim` basis rows back from
/// scratch; padded lanes keep stale values.
fn copy_projected_active_prefix_from_scratch(
    runtime: &mut BatchFactoredExecutorState,
    out_dim: usize,
) {
    let pitch = runtime.active_pitch;
    let live = runtime.active_shots;
    for basis in 0..out_dim {
        let base = basis * pitch;
        runtime.active_re[base..base + live]
            .copy_from_slice(&runtime.scratch_re[base..base + live]);
        runtime.active_im[base..base + live]
            .copy_from_slice(&runtime.scratch_im[base..base + live]);
    }
}

// ==============================================================================
// Shot-major access
// ==============================================================================

fn shot_major_range(runtime: &BatchFactoredExecutorState, shot: usize) -> std::ops::Range<usize> {
    let base = shot * runtime.active_stride;
    base..base + runtime.active_stride
}

// ==============================================================================
// Rotation
// ==============================================================================

fn rotate_uniform_imag_pairs_basis_major(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &PrecomputedActivePauliRotationKernel,
    q_by_shot: &[f64],
    walk_pairs: bool,
) {
    // Same arithmetic either way; `walk_pairs` selects the insert-zero pair
    // walk (small pair counts) vs the block/offset walk over the full dim.
    let pitch = runtime.active_pitch;
    let shots = runtime.active_shots;
    let c = kernel.cos_kernel_angle;
    let xmask = kernel.action.xmask as usize;
    let apply = |re: &mut [f64], im: &mut [f64], i0: usize, i1: usize| {
        let base0 = i0 * pitch;
        let base1 = i1 * pitch;
        for shot in 0..shots {
            let q = q_by_shot[shot];
            let r0 = re[base0 + shot];
            let i0v = im[base0 + shot];
            let r1 = re[base1 + shot];
            let i1v = im[base1 + shot];
            re[base0 + shot] = c * r0 - q * i1v;
            im[base0 + shot] = c * i0v + q * r1;
            re[base1 + shot] = c * r1 - q * i0v;
            im[base1 + shot] = c * i1v + q * r0;
        }
    };
    let mut re = std::mem::take(&mut runtime.active_re);
    let mut im = std::mem::take(&mut runtime.active_im);
    if walk_pairs {
        for idx in 0..kernel.pair_count {
            let i0 = insert_zero_bit(idx, kernel.pair_bit as usize);
            apply(&mut re, &mut im, i0, i0 ^ xmask);
        }
    } else {
        let selector = 1usize << kernel.pair_bit;
        let dim = kernel.pair_count << 1;
        let step = selector << 1;
        let mut block = 0;
        while block < dim {
            for offset in 0..selector {
                let i0 = block + offset;
                apply(&mut re, &mut im, i0, i0 ^ xmask);
            }
            block += step;
        }
    }
    runtime.active_re = re;
    runtime.active_im = im;
}

fn rotate_uniform_imag_pairs_batch(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &PrecomputedActivePauliRotationKernel,
    sign_bits: &[u64],
) {
    let mode = batch_sign_mode(runtime, sign_bits);
    let minus_q = kernel.coefficient(0, false).im;
    let plus_q = kernel.coefficient(0, true).im;
    let mut coeffs = std::mem::take(&mut runtime.shot_coefficient_scalars);
    if mode != BatchSignMode::Mixed {
        let q = if mode == BatchSignMode::AllPlus {
            plus_q
        } else {
            minus_q
        };
        if coeffs.len() < runtime.active_pitch {
            coeffs.resize(runtime.active_pitch, 0.0);
        }
        coeffs[..runtime.active_pitch].fill(q);
        rotate_uniform_imag_pairs_basis_major(runtime, kernel, &coeffs, false);
        runtime.shot_coefficient_scalars = coeffs;
        return;
    }
    fill_shot_coefficient_scalars(&mut coeffs, runtime, sign_bits, minus_q, plus_q);
    let walk_pairs = kernel.pair_count < XMASK_ROTATION_PAIR_THRESHOLD && runtime.active_pitch != 2;
    rotate_uniform_imag_pairs_basis_major(runtime, kernel, &coeffs, walk_pairs);
    runtime.shot_coefficient_scalars = coeffs;
}

fn rotate_compact_pairs_batch(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &PrecomputedActivePauliRotationKernel,
    sign_bits: &[u64],
) {
    let pitch = runtime.active_pitch;
    let shots = runtime.active_shots;
    let c = kernel.cos_kernel_angle;
    let mut directions = std::mem::take(&mut runtime.shot_coefficient_scalars);
    fill_shot_coefficient_scalars(&mut directions, runtime, sign_bits, 1.0, -1.0);
    for idx in 0..kernel.pair_count {
        let left = insert_zero_bit(idx, kernel.pair_bit as usize);
        let right = left ^ kernel.action.xmask as usize;
        let left_parity = if kernel.action.phase_odd(left) {
            -1.0
        } else {
            1.0
        };
        let right_parity = if kernel.action.xz_overlap_odd {
            -left_parity
        } else {
            left_parity
        };
        let left_minus_re = left_parity * kernel.minus_even_coefficient.re;
        let left_minus_im = left_parity * kernel.minus_even_coefficient.im;
        let right_minus_re = right_parity * kernel.minus_even_coefficient.re;
        let right_minus_im = right_parity * kernel.minus_even_coefficient.im;
        let left_base = left * pitch;
        let right_base = right * pitch;
        for shot in 0..shots {
            let direction = directions[shot];
            let left_re = direction * left_minus_re;
            let left_im = direction * left_minus_im;
            let right_re = direction * right_minus_re;
            let right_im = direction * right_minus_im;
            let r0 = runtime.active_re[left_base + shot];
            let i0 = runtime.active_im[left_base + shot];
            let r1 = runtime.active_re[right_base + shot];
            let i1 = runtime.active_im[right_base + shot];
            runtime.active_re[left_base + shot] = c * r0 + right_re * r1 - right_im * i1;
            runtime.active_im[left_base + shot] = c * i0 + right_re * i1 + right_im * r1;
            runtime.active_re[right_base + shot] = c * r1 + left_re * r0 - left_im * i0;
            runtime.active_im[right_base + shot] = c * i1 + left_re * i0 + left_im * r0;
        }
    }
    runtime.shot_coefficient_scalars = directions;
}

pub(crate) fn rotate_pauli_batch(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &PrecomputedActivePauliRotationKernel,
    sign_bits: &[u64],
) -> Result<()> {
    if kernel.action.nqubits != runtime.k {
        return Err(TicitError::new(
            "rotation kernel dimension does not match batch active state",
        ));
    }
    let dim = active_length(runtime.k)?;
    if runtime.dense_shot_major_active {
        for shot in 0..runtime.active_shots {
            let range = shot_major_range(runtime, shot);
            let sign = batch_bit_at(sign_bits, shot);
            let (re, im) = (
                &mut runtime.active_re[range.clone()],
                &mut runtime.active_im[range],
            );
            rotate_contiguous_active(re, im, dim, kernel, sign);
        }
        return Ok(());
    }
    if runtime.active_pitch == 1 {
        let sign = batch_bit_at(sign_bits, 0);
        rotate_contiguous_active(
            &mut runtime.active_re,
            &mut runtime.active_im,
            dim,
            kernel,
            sign,
        );
        return Ok(());
    }
    if kernel.is_diagonal {
        let pitch = runtime.active_pitch;
        let shots = runtime.active_shots;
        let c = kernel.cos_kernel_angle;
        let mut directions = std::mem::take(&mut runtime.shot_coefficient_scalars);
        fill_shot_coefficient_scalars(&mut directions, runtime, sign_bits, 1.0, -1.0);
        for basis in 0..dim {
            let minus_coefficient = kernel.coefficient(basis, false);
            let base = basis * pitch;
            for shot in 0..shots {
                let fr = c + directions[shot] * minus_coefficient.re;
                let fi = directions[shot] * minus_coefficient.im;
                let r = runtime.active_re[base + shot];
                let i = runtime.active_im[base + shot];
                runtime.active_re[base + shot] = fr * r - fi * i;
                runtime.active_im[base + shot] = fr * i + fi * r;
            }
        }
        runtime.shot_coefficient_scalars = directions;
        return Ok(());
    }
    if kernel.uniform_imag_pairs {
        rotate_uniform_imag_pairs_batch(runtime, kernel, sign_bits);
        return Ok(());
    }
    rotate_compact_pairs_batch(runtime, kernel, sign_bits);
    Ok(())
}

// ==============================================================================
// Promotion
// ==============================================================================

pub(crate) fn promote_first_dormant_rotation_batch(
    runtime: &mut BatchFactoredExecutorState,
    kernel_angle: f64,
    sign_bits: &[u64],
) -> Result<()> {
    if runtime.ndormant == 0 {
        return Err(TicitError::new(
            "cannot promote a dormant qubit when none remain",
        ));
    }
    let dim = active_length(runtime.k)?;
    let promoted_dim = 2 * dim;
    if runtime.dense_shot_major_active && runtime.active_stride < promoted_dim {
        return Err(TicitError::new(
            "batch active shot-major stride is too short for dormant promotion",
        ));
    }
    if !runtime.dense_shot_major_active
        && runtime.active_re.len() < promoted_dim * runtime.active_pitch
    {
        return Err(TicitError::new(
            "batch active storage has too few columns for dormant promotion",
        ));
    }
    let c = kernel_angle.cos();
    let s = kernel_angle.sin();
    if runtime.dense_shot_major_active {
        for shot in 0..runtime.active_shots {
            let q = if batch_bit_at(sign_bits, shot) { s } else { -s };
            let range = shot_major_range(runtime, shot);
            let (re, im) = (
                &mut runtime.active_re[range.clone()],
                &mut runtime.active_im[range],
            );
            promote_contiguous_active(re, im, dim, c, q);
        }
        runtime.k += 1;
        runtime.ndormant -= 1;
        return Ok(());
    }
    if runtime.active_pitch == 1 {
        let q = if batch_bit_at(sign_bits, 0) { s } else { -s };
        promote_contiguous_active(&mut runtime.active_re, &mut runtime.active_im, dim, c, q);
        runtime.k += 1;
        runtime.ndormant -= 1;
        return Ok(());
    }
    let mut coeffs = std::mem::take(&mut runtime.shot_coefficient_scalars);
    fill_shot_coefficient_scalars(&mut coeffs, runtime, sign_bits, -s, s);
    let pitch = runtime.active_pitch;
    let shots = runtime.active_shots;
    for basis in 0..dim {
        let base0 = basis * pitch;
        let base1 = (dim + basis) * pitch;
        for shot in 0..shots {
            let q = coeffs[shot];
            let r = runtime.active_re[base0 + shot];
            let i = runtime.active_im[base0 + shot];
            runtime.active_re[base0 + shot] = c * r;
            runtime.active_im[base0 + shot] = c * i;
            runtime.active_re[base1 + shot] = -q * i;
            runtime.active_im[base1 + shot] = q * r;
        }
    }
    runtime.shot_coefficient_scalars = coeffs;
    runtime.k += 1;
    runtime.ndormant -= 1;
    Ok(())
}

// ==============================================================================
// Measurement
// ==============================================================================

fn measure_nondiagonal_true_prob_batch(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &PrecomputedActivePauliMeasurementKernel,
) {
    let pitch = runtime.active_pitch;
    let shots = runtime.active_shots;
    runtime.branch_prob_true[..shots].fill(0.0);
    for idx in 0..kernel.out_dim {
        let source0 = kernel.nondiagonal_source0(idx);
        let source1 = kernel.nondiagonal_source1(idx);
        let coefficient1 = kernel.nondiagonal_coefficient1(idx, true);
        let base0 = source0 * pitch;
        let base1 = source1 * pitch;
        for shot in 0..shots {
            let ar = INV_SQRT2 * runtime.active_re[base0 + shot]
                + coefficient1.re * runtime.active_re[base1 + shot]
                - coefficient1.im * runtime.active_im[base1 + shot];
            let ai = INV_SQRT2 * runtime.active_im[base0 + shot]
                + coefficient1.re * runtime.active_im[base1 + shot]
                + coefficient1.im * runtime.active_re[base1 + shot];
            runtime.branch_prob_true[shot] += ar * ar + ai * ai;
        }
    }
}

fn compute_active_measurement_true_prob_batch(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &PrecomputedActivePauliMeasurementKernel,
) {
    if runtime.dense_shot_major_active {
        for shot in 0..runtime.active_shots {
            let range = shot_major_range(runtime, shot);
            let (re, im) = (&runtime.active_re[range.clone()], &runtime.active_im[range]);
            runtime.branch_prob_true[shot] = if kernel.is_diagonal {
                diagonal_probability_contiguous(re, im, kernel, true)
            } else {
                nondiagonal_probability_contiguous(re, im, kernel, true)
            };
        }
        return;
    }
    if runtime.active_pitch == 1 {
        runtime.branch_prob_true[0] = if kernel.is_diagonal {
            diagonal_probability_contiguous(&runtime.active_re, &runtime.active_im, kernel, true)
        } else {
            nondiagonal_probability_contiguous(&runtime.active_re, &runtime.active_im, kernel, true)
        };
        return;
    }
    if !kernel.is_diagonal {
        measure_nondiagonal_true_prob_batch(runtime, kernel);
        return;
    }
    let pitch = runtime.active_pitch;
    let shots = runtime.active_shots;
    runtime.branch_prob_true[..shots].fill(0.0);
    for idx in 0..kernel.out_dim {
        let base = kernel.diagonal_source(idx, true) * pitch;
        for shot in 0..shots {
            let r = runtime.active_re[base + shot];
            let i = runtime.active_im[base + shot];
            runtime.branch_prob_true[shot] += r * r + i * i;
        }
    }
}

fn project_nondiagonal_batch(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &PrecomputedActivePauliMeasurementKernel,
    branch_bits: &[u64],
) {
    let pitch = runtime.active_pitch;
    let shots = runtime.active_shots;
    let mut directions = std::mem::take(&mut runtime.shot_coefficient_scalars);
    fill_shot_coefficient_scalars(&mut directions, runtime, branch_bits, 1.0, -1.0);
    for idx in 0..kernel.out_dim {
        let source0 = kernel.nondiagonal_source0(idx);
        let source1 = kernel.nondiagonal_source1(idx);
        let false_coefficient1 = kernel.nondiagonal_coefficient1(idx, false);
        let true_coefficient1 = -false_coefficient1;
        let base0 = source0 * pitch;
        let base1 = source1 * pitch;
        let out_base = idx * pitch;
        for shot in 0..shots {
            let coefficient1 = if directions[shot] < 0.0 {
                true_coefficient1
            } else {
                false_coefficient1
            };
            let ar = INV_SQRT2 * runtime.active_re[base0 + shot]
                + coefficient1.re * runtime.active_re[base1 + shot]
                - coefficient1.im * runtime.active_im[base1 + shot];
            let ai = INV_SQRT2 * runtime.active_im[base0 + shot]
                + coefficient1.re * runtime.active_im[base1 + shot]
                + coefficient1.im * runtime.active_re[base1 + shot];
            runtime.scratch_re[out_base + shot] = ar * runtime.branch_invnorms[shot];
            runtime.scratch_im[out_base + shot] = ai * runtime.branch_invnorms[shot];
        }
    }
    runtime.shot_coefficient_scalars = directions;
    copy_projected_active_prefix_from_scratch(runtime, kernel.out_dim);
}

/// Draws one branch per live shot in ascending shot order, packing branch
/// bits and computing per-shot inverse norms.
fn sample_batch_measurement_branches_from_true(
    runtime: &mut BatchFactoredExecutorState,
    branch_bits: &mut Vec<u64>,
) -> Result<()> {
    if branch_bits.len() < runtime.batch_words {
        branch_bits.resize(runtime.batch_words, 0);
    }
    let nwords = runtime_batch_word_count(runtime);
    for word in 0..nwords {
        let base_shot = word << 6;
        let live = 64.min(runtime.active_shots - base_shot);
        let mut packed = 0u64;
        for bit in 0..live {
            let shot = base_shot + bit;
            let pt = runtime.branch_prob_true[shot].clamp(0.0, 1.0);
            let branch = sample_bernoulli(&mut runtime.rng_state, pt)?;
            if branch {
                packed |= 1u64 << bit;
            }
            let probability = if branch { pt } else { 1.0 - pt };
            if probability <= 0.0 {
                return Err(TicitError::new(
                    "sampled an impossible active measurement branch",
                ));
            }
            runtime.branch_invnorms[shot] = 1.0 / probability.sqrt();
        }
        branch_bits[word] = packed;
    }
    branch_bits[nwords..].fill(0);
    // The invnorm tail is deliberately 1.0 so padded lanes stay finite.
    runtime.branch_invnorms[runtime.active_shots..runtime.active_pitch].fill(1.0);
    Ok(())
}

/// Shot-major branch measurement: probability, draw and projection are
/// interleaved per shot, but the draw order is identical to the basis-major
/// path. Branch bits are left in `branch_scratch` for the record fast path.
fn measure_shot_major_active_branch_batch(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &PrecomputedActivePauliMeasurementKernel,
    branch_condition: i32,
    branch_scratch: &mut Vec<u64>,
) -> Result<()> {
    if branch_scratch.len() < runtime.batch_words {
        branch_scratch.resize(runtime.batch_words, 0);
    }
    let nwords = runtime_batch_word_count(runtime);
    for word in 0..nwords {
        let base_shot = word << 6;
        let live = 64.min(runtime.active_shots - base_shot);
        let mut packed = 0u64;
        for bit in 0..live {
            let shot = base_shot + bit;
            let range = shot_major_range(runtime, shot);
            let pt = {
                let (re, im) = (
                    &runtime.active_re[range.clone()],
                    &runtime.active_im[range.clone()],
                );
                if kernel.is_diagonal {
                    diagonal_probability_contiguous(re, im, kernel, true)
                } else {
                    nondiagonal_probability_contiguous(re, im, kernel, true)
                }
            };
            let branch = sample_bernoulli(&mut runtime.rng_state, pt)?;
            if branch {
                packed |= 1u64 << bit;
            }
            let probability_sampled = if branch { pt } else { 1.0 - pt };
            if probability_sampled <= 0.0 {
                return Err(TicitError::new(
                    "sampled an impossible active measurement branch",
                ));
            }
            let invnorm = 1.0 / probability_sampled.sqrt();
            if kernel.is_diagonal {
                let (re, im) = (
                    &mut runtime.active_re[range.clone()],
                    &mut runtime.active_im[range],
                );
                project_diagonal_contiguous(re, im, kernel, branch, invnorm);
            } else {
                // The scratch row belongs to the same shot, so both slices are
                // disjoint by construction.
                let scratch_range = range.clone();
                let mut scratch_re = std::mem::take(&mut runtime.scratch_re);
                let mut scratch_im = std::mem::take(&mut runtime.scratch_im);
                project_nondiagonal_contiguous(
                    &mut runtime.active_re[range.clone()],
                    &mut runtime.active_im[range],
                    &mut scratch_re[scratch_range.clone()],
                    &mut scratch_im[scratch_range],
                    kernel,
                    branch,
                    invnorm,
                );
                runtime.scratch_re = scratch_re;
                runtime.scratch_im = scratch_im;
            }
        }
        branch_scratch[word] = packed;
    }
    branch_scratch[nwords..].fill(0);
    finish_active_measurement_branch(runtime, branch_condition, branch_scratch)
}

/// Samples the branch for every live shot, leaving branch bits in
/// `branch_scratch` (the caller's working buffer).
pub(crate) fn measure_precomputed_active_pauli_branch_batch(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &PrecomputedActivePauliMeasurementKernel,
    branch_condition: i32,
    branch_scratch: &mut Vec<u64>,
) -> Result<()> {
    if runtime.k == 0 {
        return Err(TicitError::new(
            "cannot measure an active Pauli when k == 0",
        ));
    }
    if kernel.action.nqubits != runtime.k {
        return Err(TicitError::new(
            "measurement kernel dimension does not match batch active state",
        ));
    }
    if runtime.dense_shot_major_active {
        return measure_shot_major_active_branch_batch(
            runtime,
            kernel,
            branch_condition,
            branch_scratch,
        );
    }
    if runtime.active_pitch == 1 {
        runtime.branch_prob_true[0] = if kernel.is_diagonal {
            diagonal_probability_contiguous(&runtime.active_re, &runtime.active_im, kernel, true)
        } else {
            nondiagonal_probability_contiguous(&runtime.active_re, &runtime.active_im, kernel, true)
        };
        sample_batch_measurement_branches_from_true(runtime, branch_scratch)?;
        let branch = batch_bit_at(branch_scratch, 0);
        let invnorm = runtime.branch_invnorms[0];
        if kernel.is_diagonal {
            project_diagonal_contiguous(
                &mut runtime.active_re,
                &mut runtime.active_im,
                kernel,
                branch,
                invnorm,
            );
        } else {
            let mut scratch_re = std::mem::take(&mut runtime.scratch_re);
            let mut scratch_im = std::mem::take(&mut runtime.scratch_im);
            project_nondiagonal_contiguous(
                &mut runtime.active_re,
                &mut runtime.active_im,
                &mut scratch_re,
                &mut scratch_im,
                kernel,
                branch,
                invnorm,
            );
            runtime.scratch_re = scratch_re;
            runtime.scratch_im = scratch_im;
        }
        return finish_active_measurement_branch(runtime, branch_condition, branch_scratch);
    }
    compute_active_measurement_true_prob_batch(runtime, kernel);
    sample_batch_measurement_branches_from_true(runtime, branch_scratch)?;
    if kernel.is_diagonal {
        // Divergent branches become a per-lane source-index gather.
        let pitch = runtime.active_pitch;
        let shots = runtime.active_shots;
        let mut directions = std::mem::take(&mut runtime.shot_coefficient_scalars);
        fill_shot_coefficient_scalars(&mut directions, runtime, branch_scratch, 1.0, -1.0);
        for idx in 0..kernel.out_dim {
            let out_base = idx * pitch;
            for shot in 0..shots {
                let source = kernel.diagonal_source(idx, directions[shot] < 0.0);
                let source_lane = source * pitch + shot;
                runtime.active_re[out_base + shot] =
                    runtime.active_re[source_lane] * runtime.branch_invnorms[shot];
                runtime.active_im[out_base + shot] =
                    runtime.active_im[source_lane] * runtime.branch_invnorms[shot];
            }
        }
        runtime.shot_coefficient_scalars = directions;
    } else {
        project_nondiagonal_batch(runtime, kernel, branch_scratch);
    }
    finish_active_measurement_branch(runtime, branch_condition, branch_scratch)
}

/// Non-destructive expectation: `±(1 − 2·P(true))` per shot, no collapse.
pub(crate) fn measure_precomputed_active_pauli_expectation_batch(
    runtime: &mut BatchFactoredExecutorState,
    kernel: &PrecomputedActivePauliMeasurementKernel,
    outcome_bits: &[u64],
    exp_val: i32,
) -> Result<()> {
    if runtime.k == 0 {
        return Err(TicitError::new(
            "cannot measure an active Pauli when k == 0",
        ));
    }
    if kernel.action.nqubits != runtime.k {
        return Err(TicitError::new(
            "measurement kernel dimension does not match batch active state",
        ));
    }
    if exp_val < 0 || exp_val as usize >= runtime.nexpvals {
        return Err(TicitError::new("expectation value index is out of range"));
    }
    compute_active_measurement_true_prob_batch(runtime, kernel);
    let base = exp_val as usize * runtime.batches;
    for shot in 0..runtime.active_shots {
        let sign = if batch_bit_at(outcome_bits, shot) {
            -1.0
        } else {
            1.0
        };
        let probability = runtime.branch_prob_true[shot].clamp(0.0, 1.0);
        runtime.exp_values[base + shot] = sign * (1.0 - 2.0 * probability);
    }
    Ok(())
}
