//! Packed-bit addressing and normalization helpers.

use crate::errors::{Result, TicitError};

pub(crate) const WORD_BITS: usize = 64;

pub(crate) fn nwords_for(nbits: usize) -> usize {
    nbits.div_ceil(WORD_BITS)
}

pub(crate) fn bit_word_count(nbits: usize) -> usize {
    nwords_for(nbits)
}

pub(crate) fn packed_bit(words: &[u64], bit_index: usize) -> bool {
    let word = bit_index / WORD_BITS;
    word < words.len() && (words[word] & (1u64 << (bit_index % WORD_BITS))) != 0
}

pub(crate) fn set_packed_bit(words: &mut Vec<u64>, bit_index: usize, value: bool) {
    let word = bit_index / WORD_BITS;
    if word >= words.len() {
        words.resize(word + 1, 0);
    }
    let mask = 1u64 << (bit_index % WORD_BITS);
    if value {
        words[word] |= mask;
    } else {
        words[word] &= !mask;
    }
}

pub(crate) fn packed_bits(bits: &[bool]) -> Vec<u64> {
    let mut out = vec![0; bit_word_count(bits.len())];
    for (index, &bit) in bits.iter().enumerate() {
        if bit {
            set_packed_bit(&mut out, index, true);
        }
    }
    out
}

/// Bit position of qubit `q` inside its word.
#[inline]
pub(crate) fn bit_mask(q: usize) -> u64 {
    1u64 << (q & 63)
}

/// Word holding qubit `q`.
#[inline]
pub(crate) fn word_index(q: usize) -> usize {
    q >> 6
}

/// Panics unless `q` names a qubit of an `nqubits`-qubit register.
#[inline]
pub(crate) fn check_qubit(nqubits: usize, q: usize) -> usize {
    assert!(q < nqubits, "qubit index out of range");
    q
}

#[inline]
pub(crate) fn is_odd_popcount(value: u64) -> bool {
    value.count_ones() % 2 == 1
}

/// Condition ids are 1-based and dense, so symbol `c` lives at bit `c - 1` of
/// the shot's symbol bitset. Every sampler indexes presampled storage this way.
#[inline]
pub(crate) fn symbol_bit_mask(condition: i32) -> u64 {
    assert!(condition > 0, "condition id must be positive");
    1u64 << ((condition - 1) & 63)
}

#[inline]
pub(crate) fn symbol_word_index(condition: i32) -> usize {
    assert!(condition > 0, "condition id must be positive");
    ((condition - 1) >> 6) as usize
}

/// Words needed for a dense table of `count` 1-based symbol/record/detector
/// ids.
#[inline]
pub(crate) fn symbol_word_count(count: usize) -> usize {
    count.div_ceil(64)
}

pub(crate) fn check_probability(probability: f64) -> Result<f64> {
    // NaN fails the range check, as it must.
    if !(0.0..=1.0).contains(&probability) {
        return Err(TicitError::new("probability must be between 0 and 1"));
    }
    Ok(probability)
}

/// XORs one condition into a sorted, duplicate-free condition list: present
/// means remove, absent means insert.
pub(crate) fn toggle_condition(conditions: &mut Vec<i32>, condition: i32) {
    assert!(condition > 0, "condition id must be positive");
    match conditions.binary_search(&condition) {
        Ok(index) => {
            conditions.remove(index);
        }
        Err(index) => conditions.insert(index, condition),
    }
}

/// [`normalize_conditions`] in place, by sorting rather than by repeated
/// insertion.
///
/// The planner's symbol substitution concatenates whole condition lists and then
/// re-canonicalizes, so it needs the sort-based form: the insertion-based one is
/// quadratic on long unsorted inputs.
pub(crate) fn normalize_xor_conditions(conditions: &mut Vec<i32>) {
    conditions.sort_unstable();
    let mut write = 0;
    let mut read = 0;
    while read < conditions.len() {
        let mut end = read + 1;
        while end < conditions.len() && conditions[end] == conditions[read] {
            end += 1;
        }
        if (end - read) % 2 == 1 {
            conditions[write] = conditions[read];
            write += 1;
        }
        read = end;
    }
    conditions.truncate(write);
}

/// Canonicalizes a condition list to strictly ascending order with
/// even-multiplicity entries cancelled (`s ^ s == 0`).
pub(crate) fn normalize_conditions(conditions: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(conditions.len());
    for &condition in conditions {
        toggle_condition(&mut out, condition);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_addressing_is_one_based() {
        assert_eq!(symbol_word_index(1), 0);
        assert_eq!(symbol_bit_mask(1), 1);
        assert_eq!(symbol_word_index(64), 0);
        assert_eq!(symbol_bit_mask(64), 1u64 << 63);
        assert_eq!(symbol_word_index(65), 1);
        assert_eq!(symbol_bit_mask(65), 1);
    }

    #[test]
    #[should_panic(expected = "condition id must be positive")]
    fn symbol_bit_mask_rejects_zero() {
        symbol_bit_mask(0);
    }

    #[test]
    fn normalize_sorts_and_cancels_pairs() {
        assert_eq!(normalize_conditions(&[3, 1, 3, 2, 1, 1]), vec![1, 2]);
        assert!(normalize_conditions(&[7, 7]).is_empty());
    }

    #[test]
    fn in_place_normalization_agrees_with_the_insertion_form() {
        let mut conditions = vec![3, 1, 3, 2, 1, 1];
        normalize_xor_conditions(&mut conditions);
        assert_eq!(conditions, vec![1, 2]);
        let mut pair = vec![7, 7];
        normalize_xor_conditions(&mut pair);
        assert!(pair.is_empty());
    }

    #[test]
    fn probabilities_must_be_in_range() {
        assert!(check_probability(0.0).is_ok());
        assert!(check_probability(1.0).is_ok());
        assert!(check_probability(-0.0001).is_err());
        assert!(check_probability(1.0001).is_err());
        assert!(check_probability(f64::NAN).is_err());
    }

    #[test]
    fn packed_bits_are_lsb_first() {
        assert_eq!(nwords_for(65), 2);
        let mut words = packed_bits(&[false, true, true]);
        assert_eq!(words, vec![0b110]);
        assert!(!packed_bit(&words, 99));
        set_packed_bit(&mut words, 65, true);
        assert!(packed_bit(&words, 65));
    }
}
