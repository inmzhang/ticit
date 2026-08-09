//! Packed Pauli strings.
//!
//! # Layout contract
//!
//! `x` and `z` are LSB-first bitsets of exactly `nwords_for(nqubits)` words:
//! qubit `q` lives in word `q >> 6` at bit `q & 63`. Downstream planners and
//! SIMD kernels reinterpret `x[0]` / `z[0]` wholesale as active-basis masks, so
//! this layout is load-bearing and must not change.
//!
//! # Phase convention
//!
//! The represented operator is $i^{\mathrm{phase}} \prod\_q X\_q^{x\_q} Z\_q^{z\_q}$. A `Y`
//! therefore carries an implicit `i` in its body, which the stored phase
//! cancels: `pauli_y` stores `x=1, z=1, phase=1`. Everything that reports a
//! "coefficient" phase (`Display`, [`measurement_phase_sign`],
//! [`pauli_squares_to_identity`]) subtracts the body's Y count first.

use std::fmt;
use std::ops::Mul;

use crate::bits::nwords_for;
use crate::bits::{bit_mask, check_qubit, is_odd_popcount, word_index};
use crate::errors::{Result, TicitError};

/// A Pauli operator on `nqubits` qubits, with a two-bit phase exponent of `i`.
///
/// Equality is structural — it compares the stored phase byte and both bitsets,
/// not the operator up to phase.
///
/// # Examples
///
/// ```
/// use ticit::{neg, pauli_string};
///
/// let xyz = pauli_string("XYZ")?;
/// assert_eq!(xyz.to_string(), "XYZ");
/// assert_eq!(neg(xyz).to_string(), "-XYZ");
/// # Ok::<(), ticit::TicitError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PauliString {
    /// Number of qubits in the operator.
    pub nqubits: usize,
    /// Packed LSB-first `X` bits; qubit `q` is bit `q & 63` of word `q >> 6`.
    pub x: Vec<u64>,
    /// Packed LSB-first `Z` bits, with the same layout as [`x`](Self::x).
    pub z: Vec<u64>,
    // Kept private so it always stays canonical (0..=3); the derived `PartialEq`
    // would otherwise distinguish equal operators with differently-encoded phases.
    phase: u8,
}

impl PauliString {
    /// Identity on `nqubits` qubits.
    pub fn new(nqubits: usize) -> Self {
        let nwords = nwords_for(nqubits);
        Self {
            nqubits,
            x: vec![0; nwords],
            z: vec![0; nwords],
            phase: 0,
        }
    }

    /// Returns whether qubit `q` has an `X` component.
    ///
    /// # Panics
    ///
    /// Panics if `q >= self.nqubits`.
    #[must_use]
    pub fn xbit(&self, q: usize) -> bool {
        check_qubit(self.nqubits, q);
        (self.x[word_index(q)] & bit_mask(q)) != 0
    }

    /// Returns whether qubit `q` has a `Z` component.
    ///
    /// # Panics
    ///
    /// Panics if `q >= self.nqubits`.
    #[must_use]
    pub fn zbit(&self, q: usize) -> bool {
        check_qubit(self.nqubits, q);
        (self.z[word_index(q)] & bit_mask(q)) != 0
    }

    /// Sets or clears the `X` component on qubit `q`.
    ///
    /// # Panics
    ///
    /// Panics if `q >= self.nqubits`.
    pub fn set_xbit(&mut self, q: usize, value: bool) {
        check_qubit(self.nqubits, q);
        let word = &mut self.x[word_index(q)];
        if value {
            *word |= bit_mask(q);
        } else {
            *word &= !bit_mask(q);
        }
    }

    /// Sets or clears the `Z` component on qubit `q`.
    ///
    /// # Panics
    ///
    /// Panics if `q >= self.nqubits`.
    pub fn set_zbit(&mut self, q: usize, value: bool) {
        check_qubit(self.nqubits, q);
        let word = &mut self.z[word_index(q)];
        if value {
            *word |= bit_mask(q);
        } else {
            *word &= !bit_mask(q);
        }
    }

    /// Exponent of `i` in the stored phase, in `0..=3`.
    pub fn phase_exponent(&self) -> i32 {
        i32::from(self.phase & 3)
    }

    /// Sets the phase exponent modulo 4; negative inputs wrap the same way the
    /// C++ `& 3` on a signed int does.
    pub fn set_phase(&mut self, phase_exponent: i32) {
        self.phase = (phase_exponent & 3) as u8;
    }

    /// Adds `delta` to the stored phase exponent modulo four.
    pub fn phase_shift(&mut self, delta: i32) {
        self.set_phase(self.phase_exponent() + delta);
    }

    /// Returns whether any qubit carries `X`, `Y`, or `Z` instead of identity.
    #[must_use]
    pub fn has_nonidentity_body(&self) -> bool {
        self.x.iter().chain(&self.z).any(|&word| word != 0)
    }

    /// Body equality, ignoring phase. Rotation fusion uses this as the identity
    /// test for "same Pauli, possibly different sign".
    pub fn same_body(&self, other: &Self) -> bool {
        self.nqubits == other.nqubits && self.x == other.x && self.z == other.z
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let coefficient_phase = (self.phase_exponent() - pauli_body_y_count(self)) & 3;
        match coefficient_phase {
            1 => f.write_str("i*")?,
            2 => f.write_str("-")?,
            3 => f.write_str("-i*")?,
            _ => {}
        }
        if self.nqubits == 0 {
            return f.write_str("I");
        }
        for q in 0..self.nqubits {
            let symbol = match (self.xbit(q), self.zbit(q)) {
                (true, true) => 'Y',
                (true, false) => 'X',
                (false, true) => 'Z',
                (false, false) => 'I',
            };
            write!(f, "{symbol}")?;
        }
        Ok(())
    }
}

impl Mul<&PauliString> for &PauliString {
    type Output = PauliString;

    /// Panics if the operands act on different numbers of qubits: every caller
    /// multiplies rows of one tableau, so a mismatch is a wiring bug.
    fn mul(self, rhs: &PauliString) -> PauliString {
        assert_eq!(
            self.nqubits, rhs.nqubits,
            "Pauli strings act on different numbers of qubits"
        );
        let mut out = PauliString::new(self.nqubits);
        // Commuting the right operand's X past the left operand's Z costs a
        // factor of -1 per overlapping qubit, i.e. i^2 per odd popcount.
        let mut carry = 0u32;
        for (&lhs_z, &rhs_x) in self.z.iter().zip(&rhs.x) {
            carry += (lhs_z & rhs_x).count_ones();
        }
        out.set_phase(self.phase_exponent() + rhs.phase_exponent() + 2 * (carry as i32 & 1));
        out.x
            .iter_mut()
            .zip(self.x.iter().zip(&rhs.x))
            .for_each(|(out_word, (&a, &b))| *out_word = a ^ b);
        out.z
            .iter_mut()
            .zip(self.z.iter().zip(&rhs.z))
            .for_each(|(out_word, (&a, &b))| *out_word = a ^ b);
        out
    }
}

impl Mul for PauliString {
    type Output = PauliString;

    fn mul(self, rhs: PauliString) -> PauliString {
        &self * &rhs
    }
}

/// Constructs identity on `nqubits` qubits.
#[must_use]
pub fn pauli_identity(nqubits: usize) -> PauliString {
    PauliString::new(nqubits)
}

/// Constructs `X` on qubit `q` and identity elsewhere.
///
/// # Panics
///
/// Panics if `q >= nqubits`.
#[must_use]
pub fn pauli_x(nqubits: usize, q: usize) -> PauliString {
    let mut out = PauliString::new(nqubits);
    out.set_xbit(q, true);
    out
}

/// Constructs `Y` on qubit `q` and identity elsewhere.
///
/// # Panics
///
/// Panics if `q >= nqubits`.
#[must_use]
pub fn pauli_y(nqubits: usize, q: usize) -> PauliString {
    let mut out = PauliString::new(nqubits);
    out.set_xbit(q, true);
    out.set_zbit(q, true);
    out.set_phase(1);
    out
}

/// Constructs `Z` on qubit `q` and identity elsewhere.
///
/// # Panics
///
/// Panics if `q >= nqubits`.
#[must_use]
pub fn pauli_z(nqubits: usize, q: usize) -> PauliString {
    let mut out = PauliString::new(nqubits);
    out.set_zbit(q, true);
    out
}

/// Parses a dense Pauli string such as `"IXYZ"`; `'_'` is an alias for `'I'`
/// and lowercase is accepted. String position is the qubit index.
pub fn pauli_string(ops: &str) -> Result<PauliString> {
    let bytes = ops.as_bytes();
    let mut out = PauliString::new(bytes.len());
    for (q, byte) in bytes.iter().enumerate() {
        match byte.to_ascii_uppercase() {
            b'I' | b'_' => {}
            b'X' => out.set_xbit(q, true),
            b'Z' => out.set_zbit(q, true),
            b'Y' => {
                out.set_xbit(q, true);
                out.set_zbit(q, true);
                out.phase_shift(1);
            }
            _ => return Err(TicitError::unsupported("unsupported Pauli character")),
        }
    }
    Ok(out)
}

/// Multiplies by -1.
pub fn neg(mut pauli: PauliString) -> PauliString {
    pauli.phase_shift(2);
    pauli
}

/// Symplectic inner product: true when `a` and `b` anticommute.
pub fn pauli_anticommutes(a: &PauliString, b: &PauliString) -> bool {
    assert_eq!(
        a.nqubits, b.nqubits,
        "Pauli strings act on different numbers of qubits"
    );
    // Every per-qubit anticommutation is folded into one accumulator before the
    // parity is taken; per-word parities would lose cross-word cancellation.
    let mut parity_bits = 0u64;
    for ((&ax, &az), (&bx, &bz)) in a.x.iter().zip(&a.z).zip(b.x.iter().zip(&b.z)) {
        parity_bits ^= (ax & bz) ^ (az & bx);
    }
    is_odd_popcount(parity_bits)
}

/// Number of qubits carrying a `Y`, i.e. the implicit `i` count in the body.
pub fn pauli_body_y_count(pauli: &PauliString) -> i32 {
    pauli
        .x
        .iter()
        .zip(&pauli.z)
        .map(|(&x, &z)| (x & z).count_ones() as i32)
        .sum()
}

/// True when the operator is Hermitian, i.e. its coefficient is real.
pub fn pauli_squares_to_identity(pauli: &PauliString) -> bool {
    ((pauli.phase_exponent() - pauli_body_y_count(pauli)) & 1) == 0
}

/// Extracts the sign of a Hermitian Pauli: `false` for `+P`, `true` for `-P`.
///
/// Fails for a non-Hermitian Pauli, which is reachable from circuit input (a
/// measured Pauli product whose factors leave an `i` behind).
pub fn measurement_phase_sign(pauli: &PauliString) -> Result<bool> {
    match (pauli.phase_exponent() - pauli_body_y_count(pauli)) & 3 {
        0 => Ok(false),
        2 => Ok(true),
        _ => Err(TicitError::new(
            "Pauli measurement requires a Hermitian Pauli with real coefficient",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(ops: &str) -> PauliString {
        pauli_string(ops).expect("test Pauli literals are valid")
    }

    #[test]
    fn single_qubit_products_carry_the_expected_phase() {
        assert_eq!(pauli_x(1, 0) * pauli_x(1, 0), pauli_identity(1));
        assert_eq!(pauli_y(1, 0) * pauli_y(1, 0), pauli_identity(1));

        let mut xz = pauli_y(1, 0);
        xz.phase_shift(3);
        assert_eq!(pauli_x(1, 0) * pauli_z(1, 0), xz);
        assert_eq!(xz.phase_exponent(), 0);
    }

    #[test]
    fn x_anticommutes_with_z() {
        assert!(pauli_anticommutes(&pauli_x(1, 0), &pauli_z(1, 0)));
        assert!(!pauli_anticommutes(&pauli_x(1, 0), &pauli_x(1, 0)));
        assert!(!pauli_anticommutes(&pauli_z(2, 0), &pauli_x(2, 1)));
    }

    #[test]
    fn cross_word_anticommutations_cancel_in_one_accumulator() {
        let mut xx = pauli_identity(65);
        xx.set_xbit(0, true);
        xx.set_xbit(64, true);
        let mut zz = pauli_identity(65);
        zz.set_zbit(0, true);
        zz.set_zbit(64, true);
        assert!(!pauli_anticommutes(&xx, &zz));

        zz.set_zbit(0, false);
        assert!(pauli_anticommutes(&xx, &zz));
    }

    #[test]
    fn pauli_string_bit_layout() {
        let ps = parse("IXYZ");
        assert_eq!(ps.nqubits, 4);
        assert!(!ps.xbit(0) && !ps.zbit(0));
        assert!(ps.xbit(1) && !ps.zbit(1));
        assert!(ps.xbit(2) && ps.zbit(2));
        assert!(!ps.xbit(3) && ps.zbit(3));
        assert_eq!(ps.phase_exponent(), 1);
    }

    #[test]
    fn word_vectors_are_sized_by_qubit_count() {
        for nqubits in [0usize, 1, 63, 64, 65, 129] {
            let pauli = pauli_identity(nqubits);
            assert_eq!(pauli.x.len(), nwords_for(nqubits));
            assert_eq!(pauli.z.len(), pauli.x.len());
        }
        let mut pauli = pauli_identity(130);
        pauli.set_xbit(65, true);
        pauli.set_zbit(129, true);
        assert_eq!(pauli.x, vec![0, 1 << 1, 0]);
        assert_eq!(pauli.z, vec![0, 0, 1 << 1]);
    }

    #[test]
    fn parser_accepts_underscore_and_lowercase() {
        assert_eq!(parse("_X"), parse("IX"));
        assert_eq!(parse("xyz"), parse("XYZ"));
        assert!(parse("").x.is_empty());
    }

    #[test]
    fn parser_rejects_unknown_characters() {
        let error = pauli_string("XQ").expect_err("Q is not a Pauli");
        assert_eq!(error.to_string(), "unsupported Pauli character");
        assert!(matches!(error, TicitError::Unsupported { .. }));
    }

    #[test]
    fn display_renders_the_coefficient_phase() {
        assert_eq!(parse("IXYZ").to_string(), "IXYZ");
        assert_eq!(pauli_identity(0).to_string(), "I");
        assert_eq!(neg(parse("XY")).to_string(), "-XY");

        let mut phased = parse("X");
        phased.set_phase(1);
        assert_eq!(phased.to_string(), "i*X");
        phased.set_phase(3);
        assert_eq!(phased.to_string(), "-i*X");
        assert_eq!(pauli_y(1, 0).to_string(), "Y");
    }

    #[test]
    fn body_predicates() {
        assert!(!pauli_identity(4).has_nonidentity_body());
        assert!(pauli_x(4, 3).has_nonidentity_body());
        assert!(!neg(pauli_identity(4)).has_nonidentity_body());

        assert!(pauli_x(2, 0).same_body(&neg(pauli_x(2, 0))));
        assert!(!pauli_x(2, 0).same_body(&pauli_z(2, 0)));
        assert!(!pauli_x(2, 0).same_body(&pauli_x(3, 0)));

        assert_eq!(pauli_body_y_count(&parse("YXY")), 2);
        assert_eq!(pauli_body_y_count(&pauli_identity(70)), 0);
        let mut wide = pauli_identity(70);
        wide.set_xbit(69, true);
        wide.set_zbit(69, true);
        assert_eq!(pauli_body_y_count(&wide), 1);
    }

    #[test]
    fn hermiticity_and_measurement_sign() {
        assert!(pauli_squares_to_identity(&parse("XYZ")));
        assert!(pauli_squares_to_identity(&neg(parse("XYZ"))));

        assert_eq!(measurement_phase_sign(&parse("XZ")), Ok(false));
        assert_eq!(measurement_phase_sign(&neg(parse("XZ"))), Ok(true));
        assert_eq!(measurement_phase_sign(&pauli_y(1, 0)), Ok(false));

        let non_hermitian = pauli_x(1, 0) * pauli_z(1, 0);
        assert!(!pauli_squares_to_identity(&non_hermitian));
        let error = measurement_phase_sign(&non_hermitian).expect_err("i*P is not Hermitian");
        assert_eq!(
            error.to_string(),
            "Pauli measurement requires a Hermitian Pauli with real coefficient"
        );
    }

    #[test]
    fn products_are_structural_not_semantic() {
        assert_ne!(pauli_x(1, 0), neg(pauli_x(1, 0)));
        let mut wrapped = pauli_x(1, 0);
        wrapped.set_phase(6);
        assert_eq!(wrapped.phase_exponent(), 2);
        assert_eq!(wrapped, neg(pauli_x(1, 0)));
        wrapped.set_phase(-1);
        assert_eq!(wrapped.phase_exponent(), 3);
    }

    #[test]
    fn multi_word_products_track_carries() {
        let mut left = pauli_identity(130);
        left.set_zbit(0, true);
        left.set_zbit(70, true);
        let mut right = pauli_identity(130);
        right.set_xbit(0, true);
        right.set_xbit(70, true);
        let product = &left * &right;
        assert_eq!(product.phase_exponent(), 0);
        assert!(product.xbit(0) && product.zbit(0));
        assert!(product.xbit(70) && product.zbit(70));

        right.set_xbit(0, false);
        assert_eq!((&left * &right).phase_exponent(), 2);
    }

    #[test]
    #[should_panic(expected = "qubit index out of range")]
    fn qubit_index_is_bounds_checked() {
        let _ = pauli_identity(3).xbit(3);
    }

    #[test]
    #[should_panic(expected = "Pauli strings act on different numbers of qubits")]
    fn products_require_matching_registers() {
        let _ = pauli_x(2, 0) * pauli_x(3, 0);
    }

    #[test]
    #[should_panic(expected = "Pauli strings act on different numbers of qubits")]
    fn anticommutation_requires_matching_registers() {
        pauli_anticommutes(&pauli_x(2, 0), &pauli_z(3, 0));
    }
}
