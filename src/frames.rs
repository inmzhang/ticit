//! Clifford tableaus and symbolic Pauli frames.
//!
//! Two independent "frames" live here:
//!
//! * [`CliffordFrame`] — the deterministic Clifford `U` absorbed so far, stored
//!   as the images `U† X_q U` (rows `0..n`) and `U† Z_q U` (rows `n..2n`).
//!   [`preimage`] therefore returns `U† P U`, the forward tableau of the
//!   inverse gate.
//! * [`ActivePauliFrame`] — a list of Pauli corrections, each gated on a
//!   condition symbol, that have been pushed past the Clifford. Conjugating a
//!   Pauli through it never changes the body, only the symbolic sign.

use std::cell::RefCell;

use crate::bits::{bit_mask, check_qubit};
use crate::errors::{Result, TicitError};
use crate::pauli::{PauliString, pauli_anticommutes};
#[cfg(test)]
use crate::symbolic::symbolic_bool;
use crate::symbolic::{SymbolicBool, SymbolicContext};

// ==============================================================================
// Conditional and symbolically-signed Pauli strings
// ==============================================================================

/// A Pauli that is applied only when condition symbol `condition` is true.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditionalPauliString {
    pub pauli: PauliString,
    pub condition: i32,
}

impl ConditionalPauliString {
    /// Panics on a non-positive condition id — ids come from a
    /// [`SymbolicContext`], so anything else is a programming bug.
    pub fn new(pauli: PauliString, condition: i32) -> Self {
        assert!(condition > 0, "condition id must be positive");
        Self { pauli, condition }
    }
}

/// A Pauli whose sign is a symbolic expression rather than a known bit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SymbolicPauliString {
    pub pauli: PauliString,
    pub sign: SymbolicBool,
}

impl SymbolicPauliString {
    #[cfg(test)]
    pub fn new(pauli: PauliString) -> Self {
        Self {
            pauli,
            sign: SymbolicBool::default(),
        }
    }

    pub fn with_sign(pauli: PauliString, sign: SymbolicBool) -> Self {
        Self { pauli, sign }
    }
}

// ==============================================================================
// Active Pauli frame
// ==============================================================================

/// A queue of conditional Pauli corrections on `k` active qubits.
///
/// Alongside the term list it keeps a bitset transpose: terms are packed 64 to a
/// block, and each block occupies `k` consecutive words indexed `[block*k + q]`
/// with term `t` at bit `t & 63`. Conjugation then costs one word-XOR per
/// support qubit per block instead of a scan over every term.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ActivePauliFrame {
    pub k: usize,
    pub terms: Vec<ConditionalPauliString>,
    x_term_blocks: Vec<u64>,
    z_term_blocks: Vec<u64>,
}

impl ActivePauliFrame {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            terms: Vec::new(),
            x_term_blocks: Vec::new(),
            z_term_blocks: Vec::new(),
        }
    }

    /// Appends a correction gated on `condition`.
    pub fn add_pauli(
        &mut self,
        pauli: &PauliString,
        condition: i32,
        context: &mut SymbolicContext,
    ) -> ConditionalPauliString {
        assert_eq!(
            pauli.nqubits, self.k,
            "Pauli string dimension does not match active Pauli frame"
        );
        let term = self.terms.len();
        if term & 63 == 0 {
            self.x_term_blocks
                .resize(self.x_term_blocks.len() + self.k, 0);
            self.z_term_blocks
                .resize(self.z_term_blocks.len() + self.k, 0);
        }
        let base = (term >> 6) * self.k;
        let mask = bit_mask(term);
        for (word, (&x_word, &z_word)) in pauli.x.iter().zip(&pauli.z).enumerate() {
            let mut x_bits = x_word;
            while x_bits != 0 {
                let q = word * 64 + x_bits.trailing_zeros() as usize;
                if q < self.k {
                    self.x_term_blocks[base + q] |= mask;
                }
                x_bits &= x_bits - 1;
            }
            let mut z_bits = z_word;
            while z_bits != 0 {
                let q = word * 64 + z_bits.trailing_zeros() as usize;
                if q < self.k {
                    self.z_term_blocks[base + q] |= mask;
                }
                z_bits &= z_bits - 1;
            }
        }
        let entry = ConditionalPauliString::new(pauli.clone(), condition);
        self.terms.push(entry.clone());
        context.bump_next_condition(condition);
        entry
    }

    /// Appends a correction gated on a newly minted condition.
    #[cfg(test)]
    pub fn add_pauli_fresh(
        &mut self,
        pauli: &PauliString,
        context: &mut SymbolicContext,
    ) -> ConditionalPauliString {
        let condition = context.fresh_condition();
        self.add_pauli(pauli, condition, context)
    }
}

/// Conjugates `pauli` by the whole frame, collecting the sign contributed by
/// every anticommuting correction.
pub fn conjugate_by(frame: &ActivePauliFrame, pauli: &PauliString) -> SymbolicPauliString {
    assert_eq!(
        pauli.nqubits, frame.k,
        "Pauli string dimension does not match active Pauli frame"
    );
    let mut x_qubits = Vec::new();
    let mut z_qubits = Vec::new();
    for (word, (&x_word, &z_word)) in pauli.x.iter().zip(&pauli.z).enumerate() {
        let mut x_bits = x_word;
        while x_bits != 0 {
            x_qubits.push(word * 64 + x_bits.trailing_zeros() as usize);
            x_bits &= x_bits - 1;
        }
        let mut z_bits = z_word;
        while z_bits != 0 {
            z_qubits.push(word * 64 + z_bits.trailing_zeros() as usize);
            z_bits &= z_bits - 1;
        }
    }

    let mut conditions = Vec::new();
    let blocks = frame.terms.len().div_ceil(64);
    for block in 0..blocks {
        let base = block * frame.k;
        let mut anticommuting = 0u64;
        for &q in &x_qubits {
            anticommuting ^= frame.z_term_blocks[base + q];
        }
        for &q in &z_qubits {
            anticommuting ^= frame.x_term_blocks[base + q];
        }
        while anticommuting != 0 {
            let term = block * 64 + anticommuting.trailing_zeros() as usize;
            if term < frame.terms.len() {
                conditions.push(frame.terms[term].condition);
            }
            anticommuting &= anticommuting - 1;
        }
    }

    // The same condition can gate several terms, so cancel even multiplicities
    // rather than merely deduplicating.
    conditions.sort_unstable();
    let mut normalized = Vec::with_capacity(conditions.len());
    let mut start = 0;
    while start < conditions.len() {
        let mut end = start + 1;
        while end < conditions.len() && conditions[end] == conditions[start] {
            end += 1;
        }
        if (end - start) % 2 == 1 {
            normalized.push(conditions[start]);
        }
        start = end;
    }

    SymbolicPauliString::with_sign(
        pauli.clone(),
        SymbolicBool {
            constant: false,
            conditions: normalized,
        },
    )
}

// ==============================================================================
// Dormant state
// ==============================================================================

/// Classical bits for the `d` qubits that have been measured out of the active
/// simulation but may still be promoted back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DormantState {
    pub d: usize,
    pub bits: Vec<SymbolicBool>,
}

impl DormantState {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            bits: vec![SymbolicBool::default(); d],
        }
    }

    /// Adopts an existing bit vector, bumping the context past every symbol it
    /// mentions.
    pub fn from_bits(bits: Vec<SymbolicBool>, context: &mut SymbolicContext) -> Self {
        for bit in &bits {
            context.bump_next_condition_for(bit);
        }
        Self {
            d: bits.len(),
            bits,
        }
    }
}

// ==============================================================================
// Clifford frame
// ==============================================================================

/// One word of a row's support: which qubits in word `index` carry an X and a Z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SupportWord {
    index: usize,
    x_mask: u64,
    z_mask: u64,
}

#[derive(Clone, Debug, Default)]
struct SupportCache {
    rows: Vec<Vec<SupportWord>>,
    valid: bool,
}

#[derive(Clone, Debug, Default)]
struct CoordinateCache {
    x_columns: Vec<u64>,
    z_columns: Vec<u64>,
    valid: bool,
}

/// A Clifford `U` stored as a tableau of `2n` Pauli rows.
///
/// Row `xrow(q)` holds `U† X_q U` and row `zrow(q)` holds `U† Z_q U`.
///
/// Two memoized views of the rows are rebuilt lazily after any change to the
/// bitsets: a per-row list of non-empty words (used by [`preimage`]) and a
/// transposed row-occupancy bitset (used by [`coordinates_in_frame`]). They live
/// behind [`RefCell`]s so that both queries take `&self`;
/// the frame is consequently `Send` but not `Sync`, which is fine because
/// planning is single-threaded and the sampler consumes the plan, not the frame.
///
/// Writing [`rows`](Self::rows) directly leaves those caches stale — either go
/// through [`copy_pauli_to_row`](Self::copy_pauli_to_row) or call
/// [`invalidate_support_cache`](Self::invalidate_support_cache) afterwards.
/// Phase-only edits are exempt: a phase never changes a row's support.
#[derive(Clone, Debug, Default)]
pub struct CliffordFrame {
    pub nqubits: usize,
    pub rows: Vec<PauliString>,
    support: RefCell<SupportCache>,
    coordinates: RefCell<CoordinateCache>,
}

impl PartialEq for CliffordFrame {
    /// Compares the tableau itself, phases included; the caches are derived.
    fn eq(&self, other: &Self) -> bool {
        self.nqubits == other.nqubits && self.rows == other.rows
    }
}

impl Eq for CliffordFrame {}

impl CliffordFrame {
    /// The identity Clifford on `nqubits` qubits.
    pub fn new(nqubits: usize) -> Self {
        let mut frame = Self {
            nqubits,
            rows: vec![PauliString::new(nqubits); 2 * nqubits],
            support: RefCell::new(SupportCache::default()),
            coordinates: RefCell::new(CoordinateCache::default()),
        };
        for q in 0..nqubits {
            let (x, z) = (frame.xrow(q), frame.zrow(q));
            frame.rows[x].set_xbit(q, true);
            frame.rows[z].set_zbit(q, true);
        }
        frame
    }

    pub fn xrow(&self, q: usize) -> usize {
        check_qubit(self.nqubits, q)
    }

    pub fn zrow(&self, q: usize) -> usize {
        self.nqubits + check_qubit(self.nqubits, q)
    }

    /// The sanctioned way to overwrite a row: it invalidates the caches.
    pub fn copy_pauli_to_row(&mut self, row: usize, pauli: &PauliString) {
        assert!(
            pauli.nqubits == self.nqubits && row < self.rows.len(),
            "invalid Clifford frame row assignment"
        );
        self.rows[row] = pauli.clone();
        self.invalidate_support_cache();
    }

    pub fn invalidate_support_cache(&mut self) {
        self.support.get_mut().valid = false;
        self.coordinates.get_mut().valid = false;
    }

    /// Grow the frame by tensoring untouched `|0⟩` qubits onto the register.
    pub(crate) fn grow_to(&mut self, nqubits: usize) {
        if nqubits <= self.nqubits {
            return;
        }
        let old = std::mem::replace(self, Self::new(nqubits));
        for q in 0..old.nqubits {
            for (src, dst) in [(old.xrow(q), self.xrow(q)), (old.zrow(q), self.zrow(q))] {
                let mut row = old.rows[src].clone();
                row.nqubits = nqubits;
                row.x.resize(self.rows[dst].x.len(), 0);
                row.z.resize(self.rows[dst].z.len(), 0);
                self.rows[dst] = row;
            }
        }
        self.invalidate_support_cache();
    }

    fn ensure_support_words(&self) {
        if self.support.borrow().valid {
            return;
        }
        let rows = self
            .rows
            .iter()
            .map(|pauli| {
                pauli
                    .x
                    .iter()
                    .zip(&pauli.z)
                    .enumerate()
                    .filter(|&(_, (&x_mask, &z_mask))| x_mask != 0 || z_mask != 0)
                    .map(|(index, (&x_mask, &z_mask))| SupportWord {
                        index,
                        x_mask,
                        z_mask,
                    })
                    .collect()
            })
            .collect();
        let mut cache = self.support.borrow_mut();
        cache.rows = rows;
        cache.valid = true;
    }

    fn ensure_coordinate_columns(&self) {
        if self.coordinates.borrow().valid {
            return;
        }
        self.ensure_support_words();
        let row_words = self.rows.len().div_ceil(64);
        let mut x_columns = vec![0u64; self.nqubits * row_words];
        let mut z_columns = vec![0u64; self.nqubits * row_words];
        {
            let support = self.support.borrow();
            for (row, words) in support.rows.iter().enumerate() {
                let row_mask = bit_mask(row);
                let row_word = row >> 6;
                for word in words {
                    let mut x_bits = word.x_mask;
                    while x_bits != 0 {
                        let q = word.index * 64 + x_bits.trailing_zeros() as usize;
                        if q < self.nqubits {
                            x_columns[q * row_words + row_word] |= row_mask;
                        }
                        x_bits &= x_bits - 1;
                    }
                    let mut z_bits = word.z_mask;
                    while z_bits != 0 {
                        let q = word.index * 64 + z_bits.trailing_zeros() as usize;
                        if q < self.nqubits {
                            z_columns[q * row_words + row_word] |= row_mask;
                        }
                        z_bits &= z_bits - 1;
                    }
                }
            }
        }
        let mut cache = self.coordinates.borrow_mut();
        cache.x_columns = x_columns;
        cache.z_columns = z_columns;
        cache.valid = true;
    }
}

/// Multiplies `out` on the right by `row`, tracking the `i^2` carried by each
/// qubit where `out`'s Z meets `row`'s X.
///
/// `nwords` is the query Pauli's word count; when the row touches few enough
/// words, walking its support beats scanning every word.
fn multiply_row(out: &mut PauliString, row: &PauliString, support: &[SupportWord], nwords: usize) {
    let mut carry = 0u32;
    if support.len() * 2 <= nwords {
        for word in support {
            carry += (out.z[word.index] & word.x_mask).count_ones();
            out.x[word.index] ^= word.x_mask;
            out.z[word.index] ^= word.z_mask;
        }
    } else {
        for (i, (&row_x, &row_z)) in row.x.iter().zip(&row.z).enumerate() {
            carry += (out.z[i] & row_x).count_ones();
            out.x[i] ^= row_x;
            out.z[i] ^= row_z;
        }
    }
    out.set_phase(out.phase_exponent() + row.phase_exponent() + 2 * (carry as i32 & 1));
}

/// Returns `U† P U` for the Clifford `U` the frame represents.
pub fn preimage(frame: &CliffordFrame, pauli: &PauliString) -> PauliString {
    assert_eq!(
        pauli.nqubits, frame.nqubits,
        "Pauli string and Clifford frame have different numbers of qubits"
    );
    frame.ensure_support_words();
    let support = frame.support.borrow();

    let mut out = PauliString::new(frame.nqubits);
    out.set_phase(pauli.phase_exponent());
    let nwords = pauli.x.len();
    // Walking set bits keeps the ascending-q, X-row-before-Z-row multiplication
    // order that the accumulated phase depends on, without an O(n) scan.
    for (word, (&x_word, &z_word)) in pauli.x.iter().zip(&pauli.z).enumerate() {
        let mut bits = x_word | z_word;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            let q = word * 64 + bit as usize;
            if x_word & (1u64 << bit) != 0 {
                let row = frame.xrow(q);
                multiply_row(&mut out, &frame.rows[row], &support.rows[row], nwords);
            }
            if z_word & (1u64 << bit) != 0 {
                let row = frame.zrow(q);
                multiply_row(&mut out, &frame.rows[row], &support.rows[row], nwords);
            }
            bits &= bits - 1;
        }
    }
    out
}

/// Decomposes `pauli` over the frame's rows.
///
/// The result encodes which rows are needed: bit `q` of `x` selects `xrow(q)`
/// and bit `q` of `z` selects `zrow(q)`, so `preimage(frame, result) == pauli`
/// exactly, phase included.
///
/// Fails when the rows do not span the Pauli's body, which can only happen if
/// the tableau was corrupted by direct row writes.
pub fn coordinates_in_frame(frame: &CliffordFrame, pauli: &PauliString) -> Result<PauliString> {
    assert_eq!(
        pauli.nqubits, frame.nqubits,
        "Pauli string and Clifford frame have different numbers of qubits"
    );
    frame.ensure_coordinate_columns();
    let row_words = frame.rows.len().div_ceil(64);

    // Bit r of `parity` is the symplectic product of `pauli` with row r, i.e.
    // whether they anticommute. Because the rows are a symplectic basis, the
    // coefficient of xrow(q) is that product against its dual, zrow(q).
    let mut parity = vec![0u64; row_words];
    {
        let coordinates = frame.coordinates.borrow();
        for (word, (&x_word, &z_word)) in pauli.x.iter().zip(&pauli.z).enumerate() {
            let mut x_bits = x_word;
            while x_bits != 0 {
                let base = (word * 64 + x_bits.trailing_zeros() as usize) * row_words;
                for (slot, &column) in parity
                    .iter_mut()
                    .zip(&coordinates.z_columns[base..base + row_words])
                {
                    *slot ^= column;
                }
                x_bits &= x_bits - 1;
            }
            let mut z_bits = z_word;
            while z_bits != 0 {
                let base = (word * 64 + z_bits.trailing_zeros() as usize) * row_words;
                for (slot, &column) in parity
                    .iter_mut()
                    .zip(&coordinates.x_columns[base..base + row_words])
                {
                    *slot ^= column;
                }
                z_bits &= z_bits - 1;
            }
        }
    }

    // Rows 0..n land in `z` unshifted; rows n..2n land in `x`, which means
    // restitching them across word boundaries at offset n.
    let mut out = PauliString::new(frame.nqubits);
    let n_shift_words = frame.nqubits >> 6;
    let n_shift = (frame.nqubits & 63) as u32;
    let out_words = out.z.len();
    for word in 0..out_words {
        let mut low = parity[word];
        if word + 1 == out_words && n_shift != 0 {
            low &= (1u64 << n_shift) - 1;
        }
        out.z[word] = low;

        let mut high = parity[n_shift_words + word] >> n_shift;
        if n_shift != 0 && n_shift_words + word + 1 < parity.len() {
            high |= parity[n_shift_words + word + 1] << (64 - n_shift);
        }
        out.x[word] = high;
    }

    let reconstructed = preimage(frame, &out);
    if !reconstructed.same_body(pauli) {
        return Err(TicitError::new("frame rows do not span the Pauli body"));
    }
    out.set_phase(pauli.phase_exponent() - reconstructed.phase_exponent());
    Ok(out)
}

// ==============================================================================
// Row primitives shared by the gate implementations
// ==============================================================================

fn swap_rows(frame: &mut CliffordFrame, a: usize, b: usize) {
    if a != b {
        frame.rows.swap(a, b);
        frame.invalidate_support_cache();
    }
}

/// A phase change leaves every row's support untouched, so the caches stay valid.
fn add_row_phase(frame: &mut CliffordFrame, row: usize, delta: i32) {
    frame.rows[row].phase_shift(delta);
}

fn mul_rows(frame: &mut CliffordFrame, dst: usize, lhs: usize, rhs: usize, extra_phase: i32) {
    let mut out = &frame.rows[lhs] * &frame.rows[rhs];
    out.phase_shift(extra_phase);
    frame.rows[dst] = out;
    frame.invalidate_support_cache();
}

fn check_two_qubit_gate(frame: &CliffordFrame, a: usize, b: usize) {
    check_qubit(frame.nqubits, a);
    check_qubit(frame.nqubits, b);
    assert_ne!(a, b, "two-qubit Clifford gate requires distinct qubits");
}

/// Row indices `(xrow(a), zrow(a), xrow(b), zrow(b))` for a validated pair.
fn two_qubit_rows(frame: &CliffordFrame, a: usize, b: usize) -> (usize, usize, usize, usize) {
    check_two_qubit_gate(frame, a, b);
    (frame.xrow(a), frame.zrow(a), frame.xrow(b), frame.zrow(b))
}

// ==============================================================================
// Left-multiplication gates: U <- G U, i.e. the gate is applied after the frame
// ==============================================================================

pub fn left_h(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    swap_rows(frame, x, z);
}

pub fn left_h_nxy(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    mul_rows(frame, x, x, z, 3);
    add_row_phase(frame, z, 2);
}

pub fn left_h_nxz(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    swap_rows(frame, x, z);
    add_row_phase(frame, x, 2);
    add_row_phase(frame, z, 2);
}

pub fn left_h_nyz(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    mul_rows(frame, x, x, z, 3);
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 1);
}

pub fn left_h_xy(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    mul_rows(frame, x, x, z, 3);
    add_row_phase(frame, x, 2);
    add_row_phase(frame, z, 2);
}

pub fn left_h_yz(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    mul_rows(frame, x, x, z, 1);
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 3);
}

pub fn left_c_nxyz(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    mul_rows(frame, x, x, z, 3);
    swap_rows(frame, x, z);
    add_row_phase(frame, x, 2);
    add_row_phase(frame, z, 2);
}

pub fn left_c_nzyx(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 3);
    add_row_phase(frame, z, 2);
}

pub fn left_c_xnyz(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    mul_rows(frame, x, x, z, 3);
    swap_rows(frame, x, z);
}

pub fn left_c_xynz(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    mul_rows(frame, x, x, z, 3);
    swap_rows(frame, x, z);
    add_row_phase(frame, x, 2);
}

pub fn left_c_xyz(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    mul_rows(frame, x, x, z, 1);
    swap_rows(frame, x, z);
}

pub fn left_c_znyx(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 1);
}

pub fn left_c_zynx(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 3);
    add_row_phase(frame, x, 2);
    add_row_phase(frame, z, 2);
}

pub fn left_c_zyx(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 3);
}

pub fn left_s(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    mul_rows(frame, x, x, z, 3);
}

pub fn left_sdg(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    mul_rows(frame, x, x, z, 1);
}

pub fn left_sqrt_x(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 3);
    swap_rows(frame, x, z);
}

pub fn left_sqrt_x_dag(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 1);
    swap_rows(frame, x, z);
}

pub fn left_sqrt_y(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    swap_rows(frame, x, z);
    add_row_phase(frame, z, 2);
}

pub fn left_sqrt_y_dag(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    swap_rows(frame, x, z);
    add_row_phase(frame, x, 2);
}

pub fn left_x(frame: &mut CliffordFrame, q: usize) {
    let z = frame.zrow(q);
    add_row_phase(frame, z, 2);
}

pub fn left_y(frame: &mut CliffordFrame, q: usize) {
    let (x, z) = (frame.xrow(q), frame.zrow(q));
    add_row_phase(frame, x, 2);
    add_row_phase(frame, z, 2);
}

pub fn left_z(frame: &mut CliffordFrame, q: usize) {
    let x = frame.xrow(q);
    add_row_phase(frame, x, 2);
}

/// Apply an arbitrary Pauli to the represented state.
pub(crate) fn left_pauli(frame: &mut CliffordFrame, pauli: &PauliString) {
    assert_eq!(frame.nqubits, pauli.nqubits);
    for q in 0..frame.nqubits {
        if pauli.xbit(q) {
            add_row_phase(frame, frame.zrow(q), 2);
        }
        if pauli.zbit(q) {
            add_row_phase(frame, frame.xrow(q), 2);
        }
    }
}

/// Apply a Pauli-controlled Pauli gate using the frame's existing preimages.
pub(crate) fn left_controlled_pauli(
    frame: &mut CliffordFrame,
    control: &PauliString,
    target: &PauliString,
) {
    assert_eq!(frame.nqubits, control.nqubits);
    assert_eq!(frame.nqubits, target.nqubits);
    let target_preimage = preimage(frame, target);
    let control_preimage = preimage(frame, control);

    for q in 0..frame.nqubits {
        if control.xbit(q) {
            let row = frame.zrow(q);
            frame.rows[row] = &frame.rows[row] * &target_preimage;
        }
        if control.zbit(q) {
            let row = frame.xrow(q);
            frame.rows[row] = &frame.rows[row] * &target_preimage;
        }
    }
    for q in 0..frame.nqubits {
        if target.xbit(q) {
            let row = frame.zrow(q);
            frame.rows[row] = &control_preimage * &frame.rows[row];
        }
        if target.zbit(q) {
            let row = frame.xrow(q);
            frame.rows[row] = &control_preimage * &frame.rows[row];
        }
    }
    frame.invalidate_support_cache();
}

pub fn left_cx(frame: &mut CliffordFrame, control: usize, target: usize) {
    let (xc, zc, xt, zt) = two_qubit_rows(frame, control, target);
    mul_rows(frame, xc, xc, xt, 0);
    mul_rows(frame, zt, zc, zt, 0);
}

pub fn left_cy(frame: &mut CliffordFrame, control: usize, target: usize) {
    let (xc, zc, xt, zt) = two_qubit_rows(frame, control, target);
    mul_rows(frame, xc, xc, zc, 3);
    mul_rows(frame, xc, xc, zt, 0);
    mul_rows(frame, xt, zc, xt, 0);
    mul_rows(frame, xc, xc, xt, 0);
    mul_rows(frame, zt, zc, zt, 0);
}

pub fn left_cz(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, zb, 0);
    mul_rows(frame, xb, za, xb, 0);
}

pub fn left_swap(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    swap_rows(frame, xa, xb);
    swap_rows(frame, za, zb);
}

pub fn left_cxswap(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, xb, 0);
    mul_rows(frame, zb, za, zb, 0);
    swap_rows(frame, xa, xb);
    swap_rows(frame, za, zb);
}

pub fn left_czswap(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, zb, 0);
    mul_rows(frame, xb, za, xb, 0);
    swap_rows(frame, xa, xb);
    swap_rows(frame, za, zb);
}

pub fn left_iswap(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, za, 3);
    mul_rows(frame, xb, xb, zb, 3);
    mul_rows(frame, xa, xa, zb, 0);
    mul_rows(frame, xb, za, xb, 0);
    swap_rows(frame, xa, xb);
    swap_rows(frame, za, zb);
}

pub fn left_iswap_dag(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, za, 1);
    mul_rows(frame, xb, xb, zb, 1);
    mul_rows(frame, xa, xa, zb, 0);
    mul_rows(frame, xb, za, xb, 0);
    swap_rows(frame, xa, xb);
    swap_rows(frame, za, zb);
}

pub fn left_sqrt_xx(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, za, 1);
    mul_rows(frame, xa, xa, xb, 0);
    mul_rows(frame, zb, za, zb, 0);
    swap_rows(frame, xa, za);
    mul_rows(frame, xa, xa, za, 1);
    mul_rows(frame, xa, xa, xb, 0);
    mul_rows(frame, zb, za, zb, 0);
}

pub fn left_sqrt_xx_dag(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, za, 3);
    mul_rows(frame, xa, xa, xb, 0);
    mul_rows(frame, zb, za, zb, 0);
    swap_rows(frame, xa, za);
    mul_rows(frame, xa, xa, za, 3);
    mul_rows(frame, xa, xa, xb, 0);
    mul_rows(frame, zb, za, zb, 0);
}

pub fn left_sqrt_yy(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, za, 3);
    mul_rows(frame, xb, xb, xa, 0);
    mul_rows(frame, za, zb, za, 0);
    swap_rows(frame, xb, zb);
    add_row_phase(frame, xa, 2);
    mul_rows(frame, xb, xb, xa, 0);
    mul_rows(frame, za, zb, za, 0);
    mul_rows(frame, xa, xa, za, 3);
}

pub fn left_sqrt_yy_dag(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, za, 3);
    add_row_phase(frame, xb, 2);
    mul_rows(frame, xb, xb, xa, 0);
    mul_rows(frame, za, zb, za, 0);
    swap_rows(frame, xb, zb);
    mul_rows(frame, xb, xb, xa, 0);
    mul_rows(frame, za, zb, za, 0);
    mul_rows(frame, xa, xa, za, 1);
}

pub fn left_sqrt_zz(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, za, 3);
    mul_rows(frame, xb, xb, zb, 3);
    mul_rows(frame, xa, xa, zb, 0);
    mul_rows(frame, xb, za, xb, 0);
}

pub fn left_sqrt_zz_dag(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, za, 1);
    mul_rows(frame, xb, xb, zb, 1);
    mul_rows(frame, xa, xa, zb, 0);
    mul_rows(frame, xb, za, xb, 0);
}

pub fn left_swapcx(frame: &mut CliffordFrame, a: usize, b: usize) {
    let (xa, za, xb, zb) = two_qubit_rows(frame, a, b);
    mul_rows(frame, xa, xa, xb, 0);
    mul_rows(frame, zb, za, zb, 0);
    mul_rows(frame, xb, xb, xa, 0);
    mul_rows(frame, za, zb, za, 0);
}

pub fn left_xcx(frame: &mut CliffordFrame, control: usize, target: usize) {
    let (xc, zc, xt, zt) = two_qubit_rows(frame, control, target);
    swap_rows(frame, xc, zc);
    mul_rows(frame, xc, xc, xt, 0);
    mul_rows(frame, zt, zc, zt, 0);
    swap_rows(frame, xc, zc);
}

pub fn left_xcy(frame: &mut CliffordFrame, control: usize, target: usize) {
    let (xc, zc, xt, zt) = two_qubit_rows(frame, control, target);
    swap_rows(frame, xc, zc);
    mul_rows(frame, xc, xc, zc, 3);
    mul_rows(frame, xc, xc, zt, 0);
    mul_rows(frame, xt, zc, xt, 0);
    mul_rows(frame, xc, xc, xt, 0);
    mul_rows(frame, zt, zc, zt, 0);
    swap_rows(frame, xc, zc);
}

pub fn left_xcz(frame: &mut CliffordFrame, control: usize, target: usize) {
    let (xc, zc, xt, zt) = two_qubit_rows(frame, control, target);
    mul_rows(frame, xt, xt, xc, 0);
    mul_rows(frame, zc, zt, zc, 0);
}

pub fn left_ycx(frame: &mut CliffordFrame, control: usize, target: usize) {
    let (xc, zc, xt, zt) = two_qubit_rows(frame, control, target);
    swap_rows(frame, xt, zt);
    mul_rows(frame, xt, xt, zt, 3);
    mul_rows(frame, xc, xc, zt, 0);
    mul_rows(frame, xt, zc, xt, 0);
    mul_rows(frame, xt, xt, xc, 0);
    mul_rows(frame, zc, zt, zc, 0);
    swap_rows(frame, xt, zt);
}

pub fn left_ycy(frame: &mut CliffordFrame, control: usize, target: usize) {
    let (xc, zc, xt, zt) = two_qubit_rows(frame, control, target);
    swap_rows(frame, xc, zc);
    swap_rows(frame, xt, zt);
    mul_rows(frame, xc, xc, zc, 3);
    mul_rows(frame, xt, xt, xc, 0);
    mul_rows(frame, zc, zt, zc, 0);
    swap_rows(frame, xt, zt);
    mul_rows(frame, xt, xt, xc, 0);
    mul_rows(frame, zc, zt, zc, 0);
    mul_rows(frame, xc, xc, zc, 3);
}

pub fn left_ycz(frame: &mut CliffordFrame, control: usize, target: usize) {
    let (xc, zc, xt, zt) = two_qubit_rows(frame, control, target);
    mul_rows(frame, xt, xt, zt, 3);
    mul_rows(frame, xc, xc, zt, 0);
    mul_rows(frame, xt, zc, xt, 0);
    mul_rows(frame, xt, xt, xc, 0);
    mul_rows(frame, zc, zt, zc, 0);
}

// ==============================================================================
// Right-multiplication gates: U <- U G, i.e. the gate is applied before the frame
// ==============================================================================

/// Right-multiply by a Pauli, changing only tableau row signs.
pub(crate) fn right_pauli(frame: &mut CliffordFrame, pauli: &PauliString) {
    assert_eq!(frame.nqubits, pauli.nqubits);
    for row in &mut frame.rows {
        if pauli_anticommutes(row, pauli) {
            row.phase_shift(2);
        }
    }
}

/// Right-multiply by `exp(iπ/4·generator)`.
pub(crate) fn right_pauli_exp(frame: &mut CliffordFrame, generator: &PauliString) {
    assert_eq!(frame.nqubits, generator.nqubits);
    let mut changed = false;
    for row in &mut frame.rows {
        if pauli_anticommutes(row, generator) {
            *row = &*row * generator;
            row.phase_shift(1);
            changed = true;
        }
    }
    if changed {
        frame.invalidate_support_cache();
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for frame identities, coordinate transforms, `DormantState`, and
    //! both branches of sparse/dense row multiplication.

    use super::*;
    use crate::bits::nwords_for;
    use crate::pauli::{pauli_string, pauli_x, pauli_y, pauli_z};

    type SingleGate = fn(&mut CliffordFrame, usize);
    type TwoGate = fn(&mut CliffordFrame, usize, usize);

    fn parse(ops: &str) -> PauliString {
        pauli_string(ops).expect("test Pauli literals are valid")
    }

    /// Parses a preimage table entry: a leading `-` is a phase of `i^2`.
    fn expected_pauli_preimage(spec: &str) -> PauliString {
        match spec.strip_prefix('-') {
            Some(body) => {
                let mut out = parse(body);
                out.phase_shift(2);
                out
            }
            None => parse(spec),
        }
    }

    // ==============================================================================
    // CliffordFrame and preimage
    // ==============================================================================

    #[test]
    fn clifford_frame_preimages_and_composition_order() {
        let mut cf = CliffordFrame::new(2);
        left_cx(&mut cf, 0, 1);
        assert_eq!(preimage(&cf, &pauli_x(2, 0)), parse("XX"));
        assert_eq!(preimage(&cf, &pauli_z(2, 1)), parse("ZZ"));

        // left_* composes on the left, so this frame is S·H and not H·S.
        let mut h = CliffordFrame::new(1);
        left_h(&mut h, 0);
        left_s(&mut h, 0);
        assert_eq!(preimage(&h, &pauli_x(1, 0)), pauli_y(1, 0));
    }

    #[test]
    fn single_qubit_gate_preimage_table() {
        let cases: [(&str, SingleGate, &str, &str); 18] = [
            ("H_NXY", left_h_nxy, "-Y", "-Z"),
            ("H_NXZ", left_h_nxz, "-Z", "-X"),
            ("H_NYZ", left_h_nyz, "-X", "-Y"),
            ("H_XY", left_h_xy, "Y", "-Z"),
            ("H_YZ", left_h_yz, "-X", "Y"),
            ("C_NXYZ", left_c_nxyz, "-Z", "Y"),
            ("C_NZYX", left_c_nzyx, "Y", "-X"),
            ("C_XNYZ", left_c_xnyz, "Z", "-Y"),
            ("C_XYNZ", left_c_xynz, "-Z", "-Y"),
            ("C_XYZ", left_c_xyz, "Z", "Y"),
            ("C_ZNYX", left_c_znyx, "-Y", "X"),
            ("C_ZYNX", left_c_zynx, "-Y", "-X"),
            ("C_ZYX", left_c_zyx, "Y", "X"),
            ("SQRT_X", left_sqrt_x, "X", "Y"),
            ("SQRT_X_DAG", left_sqrt_x_dag, "X", "-Y"),
            ("SQRT_Y", left_sqrt_y, "Z", "-X"),
            ("SQRT_Y_DAG", left_sqrt_y_dag, "-Z", "X"),
            ("Y", left_y, "-X", "-Z"),
        ];

        for (name, apply, x_preimage, z_preimage) in cases {
            let mut frame = CliffordFrame::new(1);
            apply(&mut frame, 0);
            assert_eq!(
                preimage(&frame, &pauli_x(1, 0)),
                expected_pauli_preimage(x_preimage),
                "{name} X generator preimage"
            );
            assert_eq!(
                preimage(&frame, &pauli_z(1, 0)),
                expected_pauli_preimage(z_preimage),
                "{name} Z generator preimage"
            );
        }
    }

    #[test]
    fn two_qubit_gate_preimage_table() {
        let cases: [(&str, TwoGate, &str, &str, &str, &str); 18] = [
            ("CY", left_cy, "XY", "Z_", "ZX", "ZZ"),
            ("CXSWAP", left_cxswap, "_X", "ZZ", "XX", "Z_"),
            ("CZSWAP", left_czswap, "ZX", "_Z", "XZ", "Z_"),
            // ISWAP and ISWAP_DAG differ only in sign here: `preimage` is the
            // forward tableau of the daggered gate, the opposite of Stim's.
            ("ISWAP", left_iswap, "-ZY", "_Z", "-YZ", "Z_"),
            ("ISWAP_DAG", left_iswap_dag, "ZY", "_Z", "YZ", "Z_"),
            ("SQRT_XX", left_sqrt_xx, "X_", "YX", "_X", "XY"),
            ("SQRT_XX_DAG", left_sqrt_xx_dag, "X_", "-YX", "_X", "-XY"),
            ("SQRT_YY", left_sqrt_yy, "ZY", "-XY", "YZ", "-YX"),
            ("SQRT_YY_DAG", left_sqrt_yy_dag, "-ZY", "XY", "-YZ", "YX"),
            ("SQRT_ZZ", left_sqrt_zz, "-YZ", "Z_", "-ZY", "_Z"),
            ("SQRT_ZZ_DAG", left_sqrt_zz_dag, "YZ", "Z_", "ZY", "_Z"),
            ("SWAPCX", left_swapcx, "XX", "_Z", "X_", "ZZ"),
            ("XCX", left_xcx, "X_", "ZX", "_X", "XZ"),
            ("XCY", left_xcy, "X_", "ZY", "XX", "XZ"),
            ("XCZ", left_xcz, "X_", "ZZ", "XX", "_Z"),
            ("YCX", left_ycx, "XX", "ZX", "_X", "YZ"),
            ("YCY", left_ycy, "XY", "ZY", "YX", "YZ"),
            ("YCZ", left_ycz, "XZ", "ZZ", "YX", "_Z"),
        ];

        for (name, apply, xa, za, xb, zb) in cases {
            let mut frame = CliffordFrame::new(2);
            apply(&mut frame, 0, 1);
            for (query, expected, label) in [
                (pauli_x(2, 0), xa, "X_"),
                (pauli_z(2, 0), za, "Z_"),
                (pauli_x(2, 1), xb, "_X"),
                (pauli_z(2, 1), zb, "_Z"),
            ] {
                assert_eq!(
                    preimage(&frame, &query),
                    expected_pauli_preimage(expected),
                    "{name} {label} generator preimage"
                );
            }
        }
    }

    #[test]
    fn extended_gates_match_their_primitive_decompositions() {
        // The Clifford half of `test_extended_clifford_gate_directions`. This is the
        // only coverage `left_cz`, `left_swap` and `left_sdg` get.
        let parsed_match = |name: &str, native: &str, reference: &str| {
            let native = crate::circuit::parse_ticit_text(native).expect("native circuit parses");
            let reference =
                crate::circuit::parse_ticit_text(reference).expect("reference circuit parses");
            assert!(
                native.state.pending_operations.is_empty(),
                "{name} is Clifford-only"
            );
            assert!(
                reference.state.pending_operations.is_empty(),
                "{name} decomposition is Clifford-only"
            );
            assert_eq!(
                native.state.clifford, reference.state.clifford,
                "{name} parsed decomposition"
            );
        };
        let single: [(&str, SingleGate, &[SingleGate], &str); 10] = [
            (
                "C_NXYZ",
                left_c_nxyz,
                &[left_h, left_s, left_h, left_sdg],
                "H 0\nS 0\nH 0\nS_DAG 0\n",
            ),
            (
                "C_NZYX",
                left_c_nzyx,
                &[left_s, left_s, left_h, left_sdg],
                "S 0\nS 0\nH 0\nS_DAG 0\n",
            ),
            ("C_XNYZ", left_c_xnyz, &[left_s, left_h], "S 0\nH 0\n"),
            (
                "C_XYNZ",
                left_c_xynz,
                &[left_h, left_sdg, left_h, left_s],
                "H 0\nS_DAG 0\nH 0\nS 0\n",
            ),
            ("C_XYZ", left_c_xyz, &[left_sdg, left_h], "S_DAG 0\nH 0\n"),
            ("C_ZNYX", left_c_znyx, &[left_h, left_sdg], "H 0\nS_DAG 0\n"),
            (
                "C_ZYNX",
                left_c_zynx,
                &[left_s, left_h, left_sdg, left_h],
                "S 0\nH 0\nS_DAG 0\nH 0\n",
            ),
            ("C_ZYX", left_c_zyx, &[left_h, left_s], "H 0\nS 0\n"),
            (
                "SQRT_X",
                left_sqrt_x,
                &[left_h, left_s, left_h],
                "H 0\nS 0\nH 0\n",
            ),
            (
                "SQRT_X_DAG",
                left_sqrt_x_dag,
                &[left_h, left_sdg, left_h],
                "H 0\nS_DAG 0\nH 0\n",
            ),
        ];
        for (name, native, reference, reference_circuit) in single {
            let mut actual = CliffordFrame::new(1);
            native(&mut actual, 0);
            let mut expected = CliffordFrame::new(1);
            for gate in reference {
                gate(&mut expected, 0);
            }
            assert_eq!(actual, expected, "{name} decomposition");
            parsed_match(name, &format!("{name} 0\n"), reference_circuit);
        }

        let mut cxswap = CliffordFrame::new(2);
        left_cxswap(&mut cxswap, 0, 1);
        let mut cx_then_swap = CliffordFrame::new(2);
        left_cx(&mut cx_then_swap, 0, 1);
        left_swap(&mut cx_then_swap, 0, 1);
        assert_eq!(cxswap, cx_then_swap, "CXSWAP decomposition");
        parsed_match("CXSWAP", "CXSWAP 0 1\n", "CX 0 1\nSWAP 0 1\n");

        let mut swapcx = CliffordFrame::new(2);
        left_swapcx(&mut swapcx, 0, 1);
        let mut swap_then_cx = CliffordFrame::new(2);
        left_swap(&mut swap_then_cx, 0, 1);
        left_cx(&mut swap_then_cx, 0, 1);
        assert_eq!(swapcx, swap_then_cx, "SWAPCX decomposition");
        parsed_match("SWAPCX", "SWAPCX 0 1\n", "SWAP 0 1\nCX 0 1\n");

        let mut iswap = CliffordFrame::new(2);
        left_iswap(&mut iswap, 0, 1);
        let mut iswap_reference = CliffordFrame::new(2);
        left_cx(&mut iswap_reference, 1, 0);
        left_cx(&mut iswap_reference, 0, 1);
        left_cx(&mut iswap_reference, 1, 0);
        left_s(&mut iswap_reference, 0);
        left_h(&mut iswap_reference, 1);
        left_cx(&mut iswap_reference, 0, 1);
        left_h(&mut iswap_reference, 1);
        left_s(&mut iswap_reference, 1);
        assert_eq!(iswap, iswap_reference, "ISWAP decomposition");
        parsed_match(
            "ISWAP",
            "ISWAP 0 1\n",
            "CX 1 0\nCX 0 1\nCX 1 0\nS 0\nH 1\nCX 0 1\nH 1\nS 1\n",
        );

        let mut iswap_dag = CliffordFrame::new(2);
        left_iswap_dag(&mut iswap_dag, 0, 1);
        let mut iswap_dag_reference = CliffordFrame::new(2);
        left_sdg(&mut iswap_dag_reference, 1);
        left_h(&mut iswap_dag_reference, 1);
        left_cx(&mut iswap_dag_reference, 0, 1);
        left_h(&mut iswap_dag_reference, 1);
        left_sdg(&mut iswap_dag_reference, 0);
        left_cx(&mut iswap_dag_reference, 1, 0);
        left_cx(&mut iswap_dag_reference, 0, 1);
        left_cx(&mut iswap_dag_reference, 1, 0);
        assert_eq!(iswap_dag, iswap_dag_reference, "ISWAP_DAG decomposition");
        parsed_match(
            "ISWAP_DAG",
            "ISWAP_DAG 0 1\n",
            "S_DAG 1\nH 1\nCX 0 1\nH 1\nS_DAG 0\nCX 1 0\nCX 0 1\nCX 1 0\n",
        );
    }

    #[test]
    fn cz_and_swap_preimages() {
        // Neither gate appears in the C++ preimage tables.
        let mut cz = CliffordFrame::new(2);
        left_cz(&mut cz, 0, 1);
        assert_eq!(preimage(&cz, &pauli_x(2, 0)), parse("XZ"));
        assert_eq!(preimage(&cz, &pauli_z(2, 0)), parse("Z_"));
        assert_eq!(preimage(&cz, &pauli_x(2, 1)), parse("ZX"));
        assert_eq!(preimage(&cz, &pauli_z(2, 1)), parse("_Z"));

        let mut swap = CliffordFrame::new(2);
        left_swap(&mut swap, 0, 1);
        assert_eq!(preimage(&swap, &pauli_x(2, 0)), parse("_X"));
        assert_eq!(preimage(&swap, &pauli_z(2, 0)), parse("_Z"));
        assert_eq!(preimage(&swap, &pauli_x(2, 1)), parse("X_"));
        assert_eq!(preimage(&swap, &pauli_z(2, 1)), parse("Z_"));
    }

    #[test]
    fn pauli_gates_only_flip_row_phases() {
        // left_X/Y/Z are pure sign changes on the rows they anticommute with.
        let mut frame = CliffordFrame::new(1);
        left_x(&mut frame, 0);
        assert_eq!(preimage(&frame, &pauli_x(1, 0)), pauli_x(1, 0));
        assert_eq!(
            preimage(&frame, &pauli_z(1, 0)),
            expected_pauli_preimage("-Z")
        );

        let mut frame = CliffordFrame::new(1);
        left_z(&mut frame, 0);
        assert_eq!(
            preimage(&frame, &pauli_x(1, 0)),
            expected_pauli_preimage("-X")
        );
        assert_eq!(preimage(&frame, &pauli_z(1, 0)), pauli_z(1, 0));
    }

    /// Straightforward preimage: multiply the selected rows in ascending qubit
    /// order, X row before Z row. Used as an oracle for the bit-walking version.
    fn reference_preimage(frame: &CliffordFrame, pauli: &PauliString) -> PauliString {
        let mut out = PauliString::new(frame.nqubits);
        out.set_phase(pauli.phase_exponent());
        for q in 0..frame.nqubits {
            if pauli.xbit(q) {
                out = &out * &frame.rows[frame.xrow(q)];
            }
            if pauli.zbit(q) {
                out = &out * &frame.rows[frame.zrow(q)];
            }
        }
        out
    }

    /// Number of 64-bit words a row actually touches — the quantity `preimage`
    /// compares against the word count to pick its multiplication strategy.
    fn support_word_count(row: &PauliString) -> usize {
        row.x
            .iter()
            .zip(&row.z)
            .filter(|&(&x, &z)| x != 0 || z != 0)
            .count()
    }

    #[test]
    fn large_frame_preimage_takes_both_row_strategies() {
        const N: usize = 640;
        let nwords = nwords_for(N);
        let spread: [usize; 6] = [64, 128, 192, 256, 320, 384];

        let mut frame = CliffordFrame::new(N);
        // Each CX multiplies xrow(0) by X on the target, spreading its support
        // across words, and leaves zrow(target) touching two words.
        for target in spread {
            left_cx(&mut frame, 0, target);
        }
        left_s(&mut frame, 5);
        left_h(&mut frame, 600);

        let dense_row = &frame.rows[frame.xrow(0)];
        assert!(
            support_word_count(dense_row) * 2 > nwords,
            "xrow(0) should be dense enough to force the full-width scan"
        );
        let sparse_row = &frame.rows[frame.zrow(64)];
        assert!(
            support_word_count(sparse_row) * 2 <= nwords,
            "zrow(64) should stay sparse enough for the support walk"
        );

        let mut mixed = pauli_x(N, 0);
        mixed.set_zbit(64, true);
        mixed.set_xbit(600, true);
        mixed.set_zbit(639, true);
        mixed.set_phase(3);

        for query in [
            pauli_x(N, 0),
            pauli_z(N, 64),
            pauli_y(N, 5),
            pauli_x(N, 639),
            mixed,
        ] {
            assert_eq!(preimage(&frame, &query), reference_preimage(&frame, &query));
        }

        // The dense row is exactly the product of the X generators it absorbed.
        let expected = spread.iter().fold(pauli_x(N, 0), |acc, &target| {
            let mut out = acc;
            out.set_xbit(target, true);
            out
        });
        assert_eq!(preimage(&frame, &pauli_x(N, 0)), expected);
    }

    #[test]
    fn direct_row_writes_need_an_explicit_cache_invalidation() {
        let mut frame = CliffordFrame::new(3);
        left_cx(&mut frame, 0, 1);
        // Populate the lazy support cache.
        assert_eq!(preimage(&frame, &pauli_x(3, 0)), parse("XXI"));

        frame.copy_pauli_to_row(frame.xrow(0), &parse("XIX"));
        assert_eq!(preimage(&frame, &pauli_x(3, 0)), parse("XIX"));

        // Writing `rows` directly is allowed but the caller owns the invalidation.
        let row = frame.xrow(0);
        frame.rows[row] = parse("XXX");
        frame.invalidate_support_cache();
        assert_eq!(preimage(&frame, &pauli_x(3, 0)), parse("XXX"));
    }

    #[test]
    fn frame_equality_ignores_the_caches_and_respects_phases() {
        let mut left = CliffordFrame::new(2);
        let mut right = CliffordFrame::new(2);
        assert_eq!(left, right);

        left_h(&mut left, 0);
        assert_ne!(left, right);
        left_h(&mut left, 0);
        assert_eq!(left, right);

        // Populating one frame's cache must not affect equality.
        let _ = preimage(&left, &pauli_x(2, 0));
        assert_eq!(left, right);

        // A pure phase change is still a different Clifford.
        left_x(&mut right, 1);
        assert_ne!(left, right);
    }

    #[test]
    fn clifford_frame_can_move_between_threads() {
        // The lazy caches cost `Sync`, so a later multithreaded phase has to hand
        // each thread its own frame rather than share one. Pin `Send` so that stays
        // possible.
        fn assert_send<T: Send>() {}
        assert_send::<CliffordFrame>();

        let mut frame = CliffordFrame::new(2);
        left_cx(&mut frame, 0, 1);
        let moved = std::thread::spawn(move || preimage(&frame, &pauli_x(2, 0)))
            .join()
            .expect("worker thread");
        assert_eq!(moved, parse("XX"));
    }

    #[test]
    #[should_panic(expected = "Pauli string and Clifford frame have different numbers of qubits")]
    fn preimage_requires_a_matching_register() {
        preimage(&CliffordFrame::new(2), &pauli_x(3, 0));
    }

    #[test]
    #[should_panic(expected = "two-qubit Clifford gate requires distinct qubits")]
    fn two_qubit_gates_reject_a_repeated_qubit() {
        left_cx(&mut CliffordFrame::new(2), 1, 1);
    }

    // ==============================================================================
    // coordinates_in_frame
    // ==============================================================================

    #[test]
    fn coordinates_of_the_identity_frame_are_the_pauli_itself() {
        let frame = CliffordFrame::new(4);
        for query in [pauli_x(4, 0), pauli_z(4, 3), pauli_y(4, 2), parse("XYZI")] {
            assert_eq!(
                coordinates_in_frame(&frame, &query).expect("identity rows span everything"),
                query
            );
        }
    }

    #[test]
    fn coordinates_invert_preimage_across_word_boundaries() {
        // The row-occupancy bitset is indexed by row, so rows n..2n have to be
        // restitched across words at offset n. Sizes here straddle 64.
        for &n in &[1usize, 2, 63, 64, 65, 70, 128, 129] {
            let mut frame = CliffordFrame::new(n);
            left_h(&mut frame, 0);
            left_s(&mut frame, n / 2);
            left_x(&mut frame, n - 1);
            if n >= 2 {
                left_cx(&mut frame, 0, n - 1);
            }
            if n >= 3 {
                left_iswap(&mut frame, n / 2, n - 1);
            }

            let mut query = pauli_y(n, n - 1);
            query.set_xbit(0, true);
            query.set_zbit(n / 2, true);
            query.phase_shift(2);

            let coordinates =
                coordinates_in_frame(&frame, &query).expect("a Clifford tableau always spans");
            assert_eq!(
                preimage(&frame, &coordinates),
                query,
                "round trip at n = {n}"
            );
        }
    }

    #[test]
    fn coordinates_reject_a_tableau_that_no_longer_spans() {
        let mut frame = CliffordFrame::new(2);
        // Collapsing a row destroys the symplectic basis.
        frame.copy_pauli_to_row(frame.xrow(0), &parse("II"));

        let error = coordinates_in_frame(&frame, &pauli_x(2, 0)).expect_err("body is unreachable");
        assert_eq!(error.to_string(), "frame rows do not span the Pauli body");
    }

    // ==============================================================================
    // ActivePauliFrame
    // ==============================================================================

    #[test]
    fn active_pauli_frame_block_index() {
        // 70 terms with conditions 1..=70 (one X per qubit, wrapping at 65), then
        // the same correction twice under condition 100.
        let mut context = SymbolicContext::new();
        let mut frame = ActivePauliFrame::new(65);
        for term in 0..70i32 {
            frame.add_pauli(&pauli_x(65, term as usize % 65), term + 1, &mut context);
        }
        frame.add_pauli(&pauli_x(65, 0), 100, &mut context);
        frame.add_pauli(&pauli_x(65, 0), 100, &mut context);
        assert_eq!(frame.terms.len(), 72);

        let query = pauli_z(65, 0);
        let conjugated = conjugate_by(&frame, &query);
        assert_eq!(
            conjugated.pauli, query,
            "conjugation never touches the body"
        );
        // Terms with X on qubit 0 are 0 (s1), 65 (s66) and the two under s100; the
        // duplicated condition cancels.
        assert_eq!(conjugated.sign, SymbolicBool::new(false, vec![1, 66]));
    }

    #[test]
    fn active_pauli_frame_tracks_the_symbol_allocator() {
        let mut context = SymbolicContext::new();
        let mut frame = ActivePauliFrame::new(2);

        let term = frame.add_pauli_fresh(&pauli_x(2, 0), &mut context);
        assert_eq!(term.condition, 1);
        assert_eq!(context.next_condition, 2);

        frame.add_pauli(&pauli_z(2, 1), 40, &mut context);
        assert_eq!(
            context.next_condition, 41,
            "add_pauli bumps past its symbol"
        );
        assert_eq!(
            frame
                .add_pauli_fresh(&pauli_x(2, 1), &mut context)
                .condition,
            41
        );

        // A commuting query picks up no sign; an anticommuting one picks up both
        // corrections that touch it.
        assert!(
            conjugate_by(&frame, &pauli_x(2, 0))
                .sign
                .conditions
                .is_empty()
        );
        assert_eq!(conjugate_by(&frame, &pauli_z(2, 0)).sign, symbolic_bool(1));
        assert_eq!(conjugate_by(&frame, &pauli_x(2, 1)).sign, symbolic_bool(40));
    }

    #[test]
    fn empty_active_frame_contributes_no_sign() {
        let frame = ActivePauliFrame::new(3);
        let conjugated = conjugate_by(&frame, &pauli_y(3, 1));
        assert_eq!(conjugated.sign, SymbolicBool::default());
        assert_eq!(conjugated.pauli, pauli_y(3, 1));
    }

    #[test]
    fn zero_qubit_frames_are_degenerate_but_valid() {
        let empty = CliffordFrame::new(0);
        assert!(empty.rows.is_empty());
        let identity = pauli_string("").expect("the empty string parses");
        assert_eq!(preimage(&empty, &identity), identity);
        assert_eq!(
            coordinates_in_frame(&empty, &identity).expect("nothing to span"),
            identity
        );

        let mut context = SymbolicContext::new();
        let mut active = ActivePauliFrame::new(0);
        active.add_pauli(&identity, 1, &mut context);
        assert_eq!(
            conjugate_by(&active, &identity).sign,
            SymbolicBool::default()
        );
    }

    #[test]
    #[should_panic(expected = "condition id must be positive")]
    fn conditional_pauli_rejects_a_nonpositive_condition() {
        ConditionalPauliString::new(pauli_x(2, 0), 0);
    }

    #[test]
    #[should_panic(expected = "Pauli string dimension does not match active Pauli frame")]
    fn active_frame_rejects_a_mismatched_pauli() {
        let mut context = SymbolicContext::new();
        ActivePauliFrame::new(3).add_pauli(&pauli_x(2, 0), 1, &mut context);
    }

    // ==============================================================================
    // DormantState
    // ==============================================================================

    #[test]
    fn dormant_state_adopts_existing_bits() {
        let mut context = SymbolicContext::new();
        let dormant = DormantState::from_bits(
            vec![
                symbolic_bool(4),
                SymbolicBool::from(true),
                symbolic_bool(12),
            ],
            &mut context,
        );

        assert_eq!(dormant.d, 3);
        assert_eq!(dormant.bits.len(), 3);
        assert_eq!(context.next_condition, 13);
    }
}
