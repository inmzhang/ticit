//! The sampling RNG: SplitMix64 plus the three draw primitives every sampler
//! shares.
//!
//! The sequence of *draws* is part of the reproducibility contract.
//! `sample_bernoulli` consumes nothing for deterministic probabilities,
//! geometric gap draws consume exactly one draw including on the iteration that
//! terminates a skip loop, and `sample_categorical_row` always consumes exactly one.

use crate::bits::check_probability;
use crate::errors::{Result, TicitError};

/// SplitMix64. The one generator used everywhere; state is the bare `u64`.
pub fn next_random_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Uniform in `[0, 1)` with 53 random bits, exactly `(next >> 11) * 2^-53`.
pub fn rand_float(state: &mut u64) -> f64 {
    ((next_random_u64(state) >> 11) as f64) * f64::from_bits(0x3CA0_0000_0000_0000)
}

/// One biased coin. Draws nothing at all when the outcome is certain.
pub fn sample_bernoulli(rng_state: &mut u64, probability: f64) -> Result<bool> {
    let p = check_probability(probability)?;
    if p <= 0.0 {
        return Ok(false);
    }
    if p >= 1.0 {
        return Ok(true);
    }
    Ok(rand_float(rng_state) < p)
}

/// One draw; walks the cumulative sum with an inclusive comparison and falls
/// through to the last row, so rounding shortfall lands on the final entry.
pub fn sample_categorical_row(rng_state: &mut u64, probabilities: &[f64]) -> usize {
    let r = rand_float(rng_state);
    let mut cumulative = 0.0;
    for (i, &probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if r <= cumulative {
            return i;
        }
    }
    probabilities.len() - 1
}

/// `ln(1 - p)`, the divisor shared by every gap drawn at probability `p`.
/// Skip loops hoist this: `log1p` costs as much as the draw itself.
pub fn geometric_gap_denominator(probability: f64) -> Result<f64> {
    if !(probability > 0.0 && probability < 1.0) {
        return Err(TicitError::new(
            "geometric gap probability must be in (0, 1)",
        ));
    }
    Ok((-probability).ln_1p())
}

/// One gap draw against a precomputed [`geometric_gap_denominator`]. The
/// division is kept (not a reciprocal multiply) so results stay bit-identical
/// to the unhoisted form.
pub fn sample_geometric_gap_with_denominator(rng_state: &mut u64, denominator: f64) -> f64 {
    let u = rand_float(rng_state).max(f64::MIN_POSITIVE);
    let gap = (u.ln() / denominator).floor();
    if !gap.is_finite() || gap >= i32::MAX as f64 {
        return i32::MAX as f64;
    }
    gap
}

/// Derives a seed for one deterministic work unit. `base` picks the purpose
/// (exogenous vs branch), and `block_index` identifies work within `seed`.
pub fn block_seed(base: u64, seed: u64, block_index: u64) -> u64 {
    base ^ 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(block_index.wrapping_add(1))
        ^ 0xbf58_476d_1ce4_e5b9u64.wrapping_mul(seed.wrapping_add(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First outputs of SplitMix64 from small seeds, evaluated by hand from
    /// the reference arithmetic (state += gamma; three xor-multiply mixes).
    #[test]
    fn splitmix64_matches_the_reference_sequence() {
        // Seed 0: state becomes the golden gamma itself on the first step.
        // These values are the published SplitMix64 test vector.
        let mut state = 0u64;
        assert_eq!(next_random_u64(&mut state), 0xe220a8397b1dcdaf);
        assert_eq!(next_random_u64(&mut state), 0x6e789e6aa1b965f4);
        assert_eq!(next_random_u64(&mut state), 0x06c45d188009454f);
        assert_eq!(state, 0x9e3779b97f4a7c15u64.wrapping_mul(3));

        let mut state = 1u64;
        assert_eq!(next_random_u64(&mut state), 0x910a2dec89025cc1);
        // Verified against the C++ header linked from libsymft_cpp.a.
        let mut state = 42u64;
        assert_eq!(next_random_u64(&mut state), 0xbdd732262feb6e95);
    }

    #[test]
    fn rand_float_uses_the_high_53_bits() {
        // Same draw as above, so the expected value is (draw >> 11) * 2^-53.
        let mut state = 0u64;
        let expected = ((0xe220a8397b1dcdafu64 >> 11) as f64) * (0.5f64).powi(53);
        let drawn = rand_float(&mut state);
        assert_eq!(drawn, expected);
        // Pinned against the C++ implementation (%.17g of the same draw).
        #[allow(clippy::excessive_precision)]
        let pinned = 0.88331080821364261_f64;
        assert_eq!(drawn, pinned);
    }

    #[test]
    fn certain_bernoulli_consumes_no_randomness() {
        let mut state = 7u64;
        assert!(!sample_bernoulli(&mut state, 0.0).expect("valid probability"));
        assert!(sample_bernoulli(&mut state, 1.0).expect("valid probability"));
        assert_eq!(
            state, 7,
            "deterministic outcomes must not advance the state"
        );
        sample_bernoulli(&mut state, 0.5).expect("valid probability");
        assert_ne!(state, 7);
    }

    #[test]
    fn bernoulli_rejects_invalid_probability() {
        let mut state = 1u64;
        assert!(sample_bernoulli(&mut state, -0.1).is_err());
        assert!(sample_bernoulli(&mut state, f64::NAN).is_err());
    }

    #[test]
    fn categorical_walk_is_inclusive_and_falls_through() {
        // A distribution summing short of 1: any draw beyond the sum takes the
        // last row via fall-through rather than an out-of-range index.
        let mut state = 0u64;
        let row = sample_categorical_row(&mut state, &[0.0, 0.0]);
        assert_eq!(row, 1);
        // First entry 1.0 always wins on the inclusive comparison.
        let mut state = 0u64;
        assert_eq!(sample_categorical_row(&mut state, &[1.0, 0.0]), 0);
    }

    #[test]
    fn block_seed_separates_seeds_and_blocks() {
        let a = block_seed(0x7eed0000, 0, 0);
        assert_ne!(a, block_seed(0x7eed0000, 1, 0));
        assert_ne!(a, block_seed(0x7eed0000, 0, 1));
        assert_ne!(a, block_seed(0x5eed1234, 0, 0));
        // Spot value from the definition, evaluated directly.
        assert_eq!(
            block_seed(2, 3, 4),
            2u64 ^ 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(5)
                ^ 0xbf58_476d_1ce4_e5b9u64.wrapping_mul(4)
        );
    }
}
