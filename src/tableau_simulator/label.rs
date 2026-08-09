//! Destabilizer-coset labels and the word masks the frame math runs against.
//!
//! A label $c \in \{0,1\}^n$ indexes a term $D^c\lvert\psi\_0\rangle$ of the stabilizer frame (see
//! [`crate::tableau_simulator::TableauSimulator`]). Equivalently it is a computational-basis
//! index of the rotated state $\lvert\chi\rangle = R^\dagger\lvert\psi\rangle$. Physical circuits reach hundreds
//! of qubits, so labels are packed `u64` word vectors.
//!
//! The hot inner loops (`T` and measurement) touch every label with the coset
//! shift $c \oplus a$ and the sign parity $\langle b,c\rangle$. Both $a$ and $b$ are stored as
//! labels of the same width so those become whole-word operations
//! ($O(\lceil n/64\rceil)$) instead of per-set-bit index work — the difference
//! between ~8 and ~250 operations per term at $n = 500$.
//!
//! # Why the width is a type
//!
//! A runtime-width label costs three things at once in the innermost loop: a
//! dynamic trip count the optimizer could not unroll, an inline-vs-spilled
//! branch on every deref (including inside the hash map's equality probe), and
//! a 72-byte key that made a rank-4096 map four times larger than the data it
//! carried.
//!
//! So the width becomes a type parameter. [`Key<W>`] is $W$ words of inline,
//! `Copy` storage; a register of $\lceil n/64\rceil$ words is rounded up to the next
//! supported $W \in \{1, 2, 4, 8\}$ and the padding words are held at zero.
//! Padding is transparent because every operation here is bitwise or a parity,
//! for which a zero word is the identity — and in exchange $n = 128$ runs on a
//! 16-byte key with a two-iteration loop the compiler unrolls flat. Registers
//! past 512 qubits fall back to [`Label`], the heap-allocated runtime-width
//! label, which keeps the specialization table finite.
//!
//! [`LabelKey`] is the interface the amplitude engine is generic over; the
//! [`Width`] class is how the simulator picks an implementation.

use std::hash::{Hash, Hasher};

/// Parity of a word's set bits.
///
/// `count_ones` is one `popcnt` instruction where the target has it and a
/// ~12-operation SWAR sequence where it does not; either beats a shift-fold
/// chain, and the whole call folds into the caller.
#[inline]
fn parity(word: u64) -> bool {
    word.count_ones() & 1 == 1
}

/// Which [`LabelKey`] implementation a register width selects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Width {
    W1,
    W2,
    W4,
    W8,
    /// Runtime width — registers past 512 qubits.
    Wide,
}

impl Width {
    /// The class holding a register of `words` words.
    pub(crate) fn for_words(words: usize) -> Self {
        match words {
            0 | 1 => Width::W1,
            2 => Width::W2,
            3 | 4 => Width::W4,
            5..=8 => Width::W8,
            _ => Width::Wide,
        }
    }
}

/// The label operations the amplitude engine runs on, over any width.
///
/// Implementors carry *at least* the register's `⌈n/64⌉` words; the surplus is
/// always zero, so index-free operations (`xor`, `dot_parity`, `is_zero`) may
/// run over the whole storage without checking where the register ends.
pub(crate) trait LabelKey: Clone + Eq + Hash + std::fmt::Debug {
    /// The all-zero label on `words` words.
    fn zeros(words: usize) -> Self;

    /// A label of width `words` carrying `src`'s bits. `src` may be narrower
    /// (register growth re-keys through here); the rest reads as zero.
    fn from_words(src: &[u64], words: usize) -> Self;

    /// The backing words, for handing a mask to the frame's row math. Exactly
    /// `words` long, which is what the frame asserts against its own width.
    fn as_slice(&self) -> &[u64];

    /// The backing words, for letting the frame write a mask in place.
    fn as_mut_slice(&mut self) -> &mut [u64];

    /// `self ⊕ mask` word-wise (the coset shift `c ⊕ a`).
    fn xor(&self, mask: &Self) -> Self;

    /// Parity `⟨self, mask⟩ = ⊕_i (self_i ∧ mask_i)`, used for the
    /// `(−1)^{⟨b,c⟩}` phases. Uses `parity(a) ⊕ parity(b) = parity(a ⊕ b)` to
    /// fold the per-word AND results before a single pop-count.
    fn dot_parity(&self, mask: &Self) -> bool;

    /// Test bit `index`.
    fn get(&self, index: usize) -> bool;

    /// Toggle bit `index` (the pivot bit of the measurement's `Z_p` factor).
    fn flip(&mut self, index: usize);

    /// Index of the lowest set bit, or `None` if all zero (measurement pivot).
    fn first_set_bit(&self) -> Option<usize>;

    /// Whether every bit is zero (a frame-diagonal decomposition, `a = 0`).
    fn is_zero(&self) -> bool;

    /// Build a mask with `support`'s bits set — only how the tests spell one
    /// out; the frame writes its masks word-wise through [`Self::as_mut_slice`].
    #[cfg(test)]
    fn mask_from_support(words: usize, support: impl Iterator<Item = usize>) -> Self
    where
        Self: Sized,
    {
        let mut mask = Self::zeros(words);
        for index in support {
            mask.flip(index);
        }
        mask
    }
}

// ==============================================================================
// Fixed-width keys
// ==============================================================================

/// A label of exactly `W` machine words, held inline and copied by value.
///
/// The const parameter is the whole point: `W` is a compile-time trip count, so
/// the loops below unroll into straight-line code and the key is a `Copy`
/// `8·W`-byte value the hash map can compare with a single fixed-size memcmp.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Key<const W: usize>([u64; W]);

/// Fold the words through `write_u64` rather than hashing the array as a byte
/// slice.
///
/// `[u64; W]`'s derived `Hash` goes through the slice impl, which prefixes a
/// length that is constant across every key in a given map — pure overhead —
/// and then hands the hasher a byte buffer it has to re-chunk. Folding words
/// directly avoids hashing a redundant length.
impl<const W: usize> Hash for Key<W> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &word in &self.0 {
            state.write_u64(word);
        }
    }
}

impl<const W: usize> LabelKey for Key<W> {
    #[inline]
    fn zeros(words: usize) -> Self {
        debug_assert!(words <= W, "register of {words} words needs a wider key");
        Key([0; W])
    }

    #[inline]
    fn from_words(src: &[u64], words: usize) -> Self {
        debug_assert!(words <= W, "register of {words} words needs a wider key");
        debug_assert!(src.len() <= W, "source label is wider than the key");
        let mut out = [0u64; W];
        out[..src.len()].copy_from_slice(src);
        Key(out)
    }

    #[inline]
    fn as_slice(&self) -> &[u64] {
        &self.0
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u64] {
        &mut self.0
    }

    #[inline]
    fn xor(&self, mask: &Self) -> Self {
        Key(std::array::from_fn(|i| self.0[i] ^ mask.0[i]))
    }

    #[inline]
    fn dot_parity(&self, mask: &Self) -> bool {
        let mut acc = 0u64;
        for i in 0..W {
            acc ^= self.0[i] & mask.0[i];
        }
        parity(acc)
    }

    #[inline]
    fn get(&self, index: usize) -> bool {
        (self.0[index >> 6] >> (index & 63)) & 1 == 1
    }

    #[inline]
    fn flip(&mut self, index: usize) {
        self.0[index >> 6] ^= 1u64 << (index & 63);
    }

    #[inline]
    fn first_set_bit(&self) -> Option<usize> {
        for (i, &word) in self.0.iter().enumerate() {
            if word != 0 {
                return Some(i * 64 + word.trailing_zeros() as usize);
            }
        }
        None
    }

    #[inline]
    fn is_zero(&self) -> bool {
        // OR-fold rather than `all`: no branch per word, and at `W = 1` it is a
        // single compare.
        self.0.iter().fold(0u64, |acc, &word| acc | word) == 0
    }
}

// ==============================================================================
// Runtime-width fallback
// ==============================================================================

/// A label of runtime width, for registers past [`MAX_INLINE_QUBITS`].
///
/// `Box<[u64]>` rather than an inline-capacity vector: at these widths the
/// words never fit inline anyway, so the 16-byte fat pointer is the whole key
/// and the map stays dense.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Label(Box<[u64]>);

/// Word-fold, matching [`Key`]'s: the width is a property of the simulator, not
/// of an individual label, so hashing it would be constant overhead.
impl Hash for Label {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &word in self.0.iter() {
            state.write_u64(word);
        }
    }
}

impl LabelKey for Label {
    fn zeros(words: usize) -> Self {
        Label(vec![0u64; words].into_boxed_slice())
    }

    fn from_words(src: &[u64], words: usize) -> Self {
        debug_assert!(src.len() <= words, "source label is wider than the target");
        let mut out = vec![0u64; words];
        out[..src.len()].copy_from_slice(src);
        Label(out.into_boxed_slice())
    }

    #[inline]
    fn as_slice(&self) -> &[u64] {
        &self.0
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u64] {
        &mut self.0
    }

    fn xor(&self, mask: &Self) -> Self {
        debug_assert_eq!(self.0.len(), mask.0.len());
        let mut out = self.0.clone();
        for (o, &m) in out.iter_mut().zip(mask.0.iter()) {
            *o ^= m;
        }
        Label(out)
    }

    #[inline]
    fn dot_parity(&self, mask: &Self) -> bool {
        debug_assert_eq!(self.0.len(), mask.0.len());
        let mut acc = 0u64;
        for (&a, &b) in self.0.iter().zip(mask.0.iter()) {
            acc ^= a & b;
        }
        parity(acc)
    }

    #[inline]
    fn get(&self, index: usize) -> bool {
        (self.0[index >> 6] >> (index & 63)) & 1 == 1
    }

    #[inline]
    fn flip(&mut self, index: usize) {
        self.0[index >> 6] ^= 1u64 << (index & 63);
    }

    #[inline]
    fn first_set_bit(&self) -> Option<usize> {
        self.0
            .iter()
            .position(|&word| word != 0)
            .map(|i| i * 64 + self.0[i].trailing_zeros() as usize)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|&word| word == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Padding words must never change an answer, so a narrow register held in
    /// a wider key has to agree with the exact-width one bit for bit.
    #[test]
    fn padding_words_are_transparent() {
        let bits = [3usize, 64, 65, 100];
        let narrow = Key::<2>::mask_from_support(2, bits.into_iter());
        let padded = Key::<8>::mask_from_support(2, bits.into_iter());
        assert_eq!(narrow.first_set_bit(), padded.first_set_bit());
        assert_eq!(narrow.is_zero(), padded.is_zero());

        let other = [3usize, 65];
        let narrow_mask = Key::<2>::mask_from_support(2, other.into_iter());
        let padded_mask = Key::<8>::mask_from_support(2, other.into_iter());
        assert_eq!(
            narrow.dot_parity(&narrow_mask),
            padded.dot_parity(&padded_mask)
        );
        assert_eq!(
            narrow.xor(&narrow_mask).as_slice(),
            &padded.xor(&padded_mask).as_slice()[..2]
        );
    }

    /// The wide fallback is the same label type as the fixed-width keys, and a
    /// divergence there would only show up on registers past 512 qubits.
    #[test]
    fn wide_labels_agree_with_fixed_width_keys() {
        let bits = [0usize, 63, 64, 127];
        let fixed = Key::<2>::mask_from_support(2, bits.into_iter());
        let wide = Label::mask_from_support(2, bits.into_iter());
        assert_eq!(fixed.as_slice(), wide.as_slice());
        assert_eq!(fixed.first_set_bit(), wide.first_set_bit());

        let mask_bits = [63usize, 64];
        let fixed_mask = Key::<2>::mask_from_support(2, mask_bits.into_iter());
        let wide_mask = Label::mask_from_support(2, mask_bits.into_iter());
        assert_eq!(fixed.dot_parity(&fixed_mask), wide.dot_parity(&wide_mask));
        assert_eq!(
            fixed.xor(&fixed_mask).as_slice(),
            wide.xor(&wide_mask).as_slice()
        );
    }

    #[test]
    fn width_classes_round_up() {
        for (words, want) in [
            (1usize, Width::W1),
            (2, Width::W2),
            (3, Width::W4),
            (4, Width::W4),
            (5, Width::W8),
            (8, Width::W8),
            (9, Width::Wide),
        ] {
            assert_eq!(Width::for_words(words), want, "{words} words");
        }
    }
}
