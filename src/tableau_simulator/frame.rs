//! Owned preimage tableau for the simulator's Clifford frame `R`.
//!
//! [`Frame`] holds exactly what `paulimer::CliffordUnitary` holds — the
//! preimages `R†·X_j·R` and `R†·Z_j·R` of the Pauli basis — but in a layout
//! built for the simulator's access pattern: whole-row word operations, no
//! pointer indirection, and no allocation on any hot path. Every sign
//! convention below is a deliberate replica of `paulimer` 0.2.3 (cited by file
//! and line), because the simulator's phases are pinned by cross-validation
//! against it.
//!
//! # Storage layout
//!
//! For `n` qubits and `W` words per row half the tableau is `2n` Pauli *rows*
//! in one contiguous `Vec<u64>`:
//!
//! ```text
//! row 2j     = R†·X_j·R      row 2j+1   = R†·Z_j·R
//! row stride = 2W words:  [ x-bits (W words) | z-bits (W words) ]
//! ```
//!
//! The `2j` / `2j+1` numbering is `paulimer`'s own phase-array convention
//! (`clifford_impl.rs:165`). Phases live in a parallel `Vec<u8>`, always
//! reduced mod 4, in the **xz** convention: a row is `i^k · X^a Z^b`.
//!
//! Keeping a row's x- and z-halves adjacent means one Pauli multiply touches
//! one cache line run, and the `H`/`SWAP` gates become physical word swaps of
//! `2W`-word blocks rather than the row-pointer permutation `paulimer` uses.
//! With `W ≤ 8` for the register sizes we compile, copying beats indirection.
//!
//! # Why `W` is rounded up
//!
//! `W` is *not* `⌈n/64⌉` but that rounded up to the next of `{1, 2, 4, 8}`
//! (see [`words_for`]). Three things fall out of it. The row kernels below are
//! generic over `W` and dispatched once per operation, so the inner loops run
//! on a compile-time trip count instead of a runtime one — worth having a
//! finite set of widths for. The stride `2W` becomes a power of two, turning
//! row addressing into a shift. And it matches the width the simulator's
//! labels use, so masks pass between the two without reslicing.
//!
//! Bits at positions `≥ n` — inside the last live word and across every
//! padding word — are always zero. Every operation preserves that (rows only
//! ever get XORed with other rows), and the parity computations below depend
//! on it.
//!
//! # Left multiplication
//!
//! Applying a gate `G` to the state is `R ← G·R`, which conjugates each row by
//! `G†…G`; the formulas are `paulimer`'s (`clifford/generic_algos.rs:113-173`).
//!
//! # Right multiplication
//!
//! The simulator's measurement path needs `R ← R·V` where `V` is built from a
//! Pauli it already holds *in the frame* (a preimage), which lets it skip
//! computing an image entirely. For `V = exp(iπ/4·G)` with `G` Hermitian
//! (`G² = I`), each row `ρ` becomes
//!
//! ```text
//! ρ' = V†·ρ·V = exp(−iπ/4·G)·ρ·exp(iπ/4·G)
//! ```
//!
//! which is `ρ` when `[ρ, G] = 0`, and when `{ρ, G} = 0` the anticommutation
//! moves the left factor through:
//!
//! ```text
//! ρ' = ρ·exp(iπ/2·G) = ρ·(cos(π/2)·I + i·sin(π/2)·G) = ρ·(i·G).
//! ```
//!
//! So the update is a right-multiply by `G` with one extra factor of `i` — the
//! same `+1` phase exponent `paulimer` adds in
//! `clifford_left_mul_eq_pauli_exp` (`generic_algos.rs:399`), and the rows it
//! touches are the same ones: `R†X_jR` anticommutes with `R†PR` exactly when
//! `X_j` anticommutes with `P`. That makes
//! `frame.right_pauli_exp(R.preimage(P))` and `R.left_mul_pauli_exp(P)` the
//! same tableau, which the differential unit tests check.
//!
//! For `V = Z_p` the same argument degenerates to a sign: `Z_p·ρ·Z_p = −ρ`
//! exactly when `ρ` has an x-bit at `p`.

use crate::PauliString;

#[cfg(test)]
use binar::{Bitwise, vec::AlignedBitVec};
#[cfg(test)]
use paulimer::{
    Clifford, CliffordUnitary, DensePauli, Pauli, PauliBinaryOps,
    clifford::{MutablePreImages, PreimageViews},
};

#[cfg(test)]
mod differential_tests;

/// `u64` words a register of `n` qubits actually occupies.
#[inline]
fn logical_words(n: usize) -> usize {
    n.div_ceil(64).max(1)
}

/// Words per tableau row half: [`logical_words`] rounded up to a width the row
/// kernels are specialized for. See the [module docs](self).
#[inline]
fn words_for(n: usize) -> usize {
    let words = logical_words(n);
    if words <= 8 {
        words.next_power_of_two()
    } else {
        words
    }
}

/// Trip count for a width-`W` kernel.
///
/// `W = 0` is the runtime-width arm, where the count comes from the data;
/// every other `W` folds to a constant, which is what lets the loops below
/// unroll into straight-line code.
#[inline(always)]
const fn width<const W: usize>(dynamic: usize) -> usize {
    if W == 0 { dynamic } else { W }
}

/// Run `$body` with `$w` bound to a `const` carrying the frame's row width,
/// usable as a const generic argument. The dispatch happens once per
/// operation, never per row.
macro_rules! by_width {
    ($words:expr, $w:ident => $body:expr) => {
        match $words {
            1 => {
                const $w: usize = 1;
                $body
            }
            2 => {
                const $w: usize = 2;
                $body
            }
            4 => {
                const $w: usize = 4;
                $body
            }
            8 => {
                const $w: usize = 8;
                $body
            }
            _ => {
                const $w: usize = 0;
                $body
            }
        }
    };
}

/// Whether the running CPU has the `popcnt` instruction.
///
/// Every parity fold in this file and in the amplitude engine bottoms out in
/// `u64::count_ones`. The workspace sets no target features — wasm is a
/// supported target and the x86 baseline is SSE2 — so that lowers to a
/// ~12-operation SWAR expansion instead of the single instruction every
/// x86-64 part since 2008 actually has. The kernels that are parity-bound
/// rather than hash-probe-bound therefore ship twice, and pick here.
///
/// `is_x86_feature_detected!` caches its CPUID answer in a static, so this is
/// a load and a test, and the branch is perfectly predicted.
///
/// Kept beside the parity kernels so every caller shares one cached feature
/// check.
#[inline]
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_popcnt() -> bool {
    std::arch::is_x86_feature_detected!("popcnt")
}

/// Row index of `R†·X_qubit·R`.
#[inline]
const fn px(qubit: usize) -> usize {
    2 * qubit
}

/// Row index of `R†·Z_qubit·R`.
#[inline]
const fn pz(qubit: usize) -> usize {
    2 * qubit + 1
}

/// Parity of `⟨a, b⟩` over whole words. Folding with XOR before a single
/// pop-count is valid because parity is linear per bit position.
#[inline]
fn dot_parity<const W: usize>(a: &[u64], b: &[u64]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let words = width::<W>(a.len());
    let mut acc = 0u64;
    for (&x, &y) in a[..words].iter().zip(&b[..words]) {
        acc ^= x & y;
    }
    acc.count_ones() & 1 == 1
}

#[inline]
fn xor_assign<const W: usize>(dst: &mut [u64], src: &[u64]) {
    debug_assert_eq!(dst.len(), src.len());
    let words = width::<W>(dst.len());
    for (d, &s) in dst[..words].iter_mut().zip(&src[..words]) {
        *d ^= s;
    }
}

/// `dst ← dst · src`, replicating `PauliBinaryOps::mul_assign_right` for the
/// xz-phase convention (`paulimer` `pauli/generic.rs:758`): commuting `X^a` in
/// `src` past `Z^b` in `dst` costs `(−1)^⟨dst.z, src.x⟩`.
#[inline]
fn mul_right<const W: usize>(
    dst_x: &mut [u64],
    dst_z: &mut [u64],
    dst_phase: &mut u8,
    src_x: &[u64],
    src_z: &[u64],
    src_phase: u8,
) {
    let cross = u8::from(dot_parity::<W>(dst_z, src_x)) << 1;
    xor_assign::<W>(dst_x, src_x);
    xor_assign::<W>(dst_z, src_z);
    *dst_phase = (*dst_phase + cross + src_phase) & 3;
}

/// `dst ← src · dst`, replicating `mul_assign_left` (`pauli/generic.rs:765`):
/// the cross term reads `dst`'s x-bits against `src`'s z-bits instead.
#[inline]
fn mul_left<const W: usize>(
    dst_x: &mut [u64],
    dst_z: &mut [u64],
    dst_phase: &mut u8,
    src_x: &[u64],
    src_z: &[u64],
    src_phase: u8,
) {
    let cross = u8::from(dot_parity::<W>(dst_x, src_z)) << 1;
    xor_assign::<W>(dst_x, src_x);
    xor_assign::<W>(dst_z, src_z);
    *dst_phase = (*dst_phase + cross + src_phase) & 3;
}

/// [`mul_right`] on `2W`-word rows.
#[inline]
fn mul_row_right<const W: usize>(dst: &mut [u64], dst_phase: &mut u8, src: &[u64], src_phase: u8) {
    let words = width::<W>(dst.len() / 2);
    let (dst_x, dst_z) = dst.split_at_mut(words);
    let (src_x, src_z) = src.split_at(words);
    mul_right::<W>(dst_x, dst_z, dst_phase, src_x, src_z, src_phase);
}

/// [`mul_left`] on `2W`-word rows.
#[inline]
fn mul_row_left<const W: usize>(dst: &mut [u64], dst_phase: &mut u8, src: &[u64], src_phase: u8) {
    let words = width::<W>(dst.len() / 2);
    let (dst_x, dst_z) = dst.split_at_mut(words);
    let (src_x, src_z) = src.split_at(words);
    mul_left::<W>(dst_x, dst_z, dst_phase, src_x, src_z, src_phase);
}

/// Two distinct rows of `data`, borrowed at once (returned in the order asked).
#[inline]
fn rows2_mut(
    data: &mut [u64],
    stride: usize,
    first: usize,
    second: usize,
) -> (&mut [u64], &mut [u64]) {
    debug_assert_ne!(first, second);
    if first < second {
        let (head, tail) = data.split_at_mut(second * stride);
        (&mut head[first * stride..][..stride], &mut tail[..stride])
    } else {
        let (head, tail) = data.split_at_mut(first * stride);
        (&mut tail[..stride], &mut head[second * stride..][..stride])
    }
}

/// The set bits of a word mask, in ascending index order.
#[inline]
fn set_bits(mask: &[u64]) -> impl Iterator<Item = usize> + '_ {
    mask.iter().enumerate().flat_map(|(word_index, &word)| {
        let mut rest = word;
        std::iter::from_fn(move || {
            (rest != 0).then(|| {
                let bit = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                word_index * 64 + bit
            })
        })
    })
}

/// The `X`-then-`Z` phase contributed by the string's Hermitian `Y = iXZ`
/// sites. The string's global phase is irrelevant to frame conjugation.
#[inline]
fn xz_phase_exponent(pauli: &PauliString) -> u8 {
    let y_sites: u32 = pauli
        .x_words()
        .iter()
        .zip(pauli.z_words())
        .map(|(&x, &z)| (x & z).count_ones())
        .sum();
    (y_sites & 3) as u8
}

/// A single-qubit Pauli axis, for the basis-preimage fast path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Axis {
    X,
    Y,
    Z,
}

/// A signed dense Pauli `i^phase · X^x · Z^z`, as raw word masks.
///
/// What the frame hands back where a caller wants a whole row rather than the
/// two masks a decomposition writes in place. Both masks are
/// [`logical_words`]`(n)` long — the register's own width, not the padded row
/// width the kernels prefer — so an index below `n` always lands inside them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowPauli {
    /// X part.
    pub(crate) x: Vec<u64>,
    /// Z part.
    pub(crate) z: Vec<u64>,
    /// Phase exponent of the X-then-Z normal form, reduced mod 4.
    pub(crate) phase: u8,
}

#[cfg(test)]
impl RowPauli {
    /// The equivalent `paulimer` Pauli, for the differential tests.
    pub(crate) fn to_dense(&self) -> DensePauli {
        DensePauli::from_bits(
            AlignedBitVec::from_words(&self.x),
            AlignedBitVec::from_words(&self.z),
            self.phase,
        )
    }
}

/// The Clifford frame `R`, stored as the preimages of the Pauli basis.
///
/// Left-multiplying operations (`left_*`) apply a gate to the state
/// (`R ← G·R`); right-multiplying ones (`right_*`) apply an operator already
/// expressed in the frame (`R ← R·V`). See the [module docs](self).
#[derive(Clone, Debug)]
pub(crate) struct Frame {
    n: usize,
    /// Words per row half, per [`words_for`].
    words: usize,
    /// `2n` rows of `2·words` words, x-half then z-half.
    data: Vec<u64>,
    /// One xz-phase exponent per row, reduced mod 4.
    phases: Vec<u8>,
    /// Row-shaped temporaries. Sized for two rows at construction so the
    /// two-preimage operations never allocate; [`Frame::left_clifford`] grows
    /// it once per support width and keeps the capacity.
    scratch: Vec<u64>,
    scratch_phases: Vec<u8>,
}

impl PartialEq for Frame {
    /// Scratch space is working memory, not state.
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.data == other.data && self.phases == other.phases
    }
}

impl Eq for Frame {}

impl Frame {
    /// The identity frame on `n` qubits.
    pub(crate) fn identity(n: usize) -> Self {
        let words = words_for(n);
        let mut frame = Frame {
            n,
            words,
            data: vec![0; 4 * n * words],
            phases: vec![0; 2 * n],
            scratch: vec![0; 4 * words],
            scratch_phases: vec![0; 2],
        };
        for qubit in 0..n {
            frame.set_identity_rows(qubit);
        }
        frame
    }

    pub(crate) fn num_qubits(&self) -> usize {
        self.n
    }

    /// Words per row half — the width every mask handed to this frame must have.
    pub(crate) fn words(&self) -> usize {
        self.words
    }

    #[inline]
    fn stride(&self) -> usize {
        2 * self.words
    }

    /// [`Frame::stride`] with the width known at compile time.
    #[inline(always)]
    fn stride_w<const W: usize>(&self) -> usize {
        2 * width::<W>(self.words)
    }

    #[inline]
    fn row(&self, index: usize) -> &[u64] {
        let stride = self.stride();
        &self.data[index * stride..][..stride]
    }

    /// Bit `index` of row `row`'s x-half.
    #[inline]
    fn x_bit(&self, row: usize, index: usize) -> bool {
        (self.data[row * self.stride() + (index >> 6)] >> (index & 63)) & 1 == 1
    }

    /// Bit `index` of row `row`'s z-half.
    #[inline]
    fn z_bit(&self, row: usize, index: usize) -> bool {
        (self.data[row * self.stride() + self.words + (index >> 6)] >> (index & 63)) & 1 == 1
    }

    /// Reset rows `2q`/`2q+1` to `X_q`/`Z_q`. Assumes they are zeroed.
    fn set_identity_rows(&mut self, qubit: usize) {
        let stride = self.stride();
        let (word, bit) = (qubit >> 6, 1u64 << (qubit & 63));
        self.data[px(qubit) * stride + word] = bit;
        self.data[pz(qubit) * stride + self.words + word] = bit;
        self.phases[px(qubit)] = 0;
        self.phases[pz(qubit)] = 0;
    }

    /// Grow or shrink to `new_n` qubits.
    ///
    /// Growth appends identity rows and re-strides existing ones word-wise
    /// (`paulimer`'s `resize` rebuilds through a tensor product that copies bit
    /// by bit). Shrinking requires the dropped qubits to be untouched, exactly
    /// as `shrink_clifford` (`generic_algos.rs:532`) does.
    ///
    /// # Panics
    ///
    /// Panics when shrinking past a qubit whose preimages are not `X_j`/`Z_j`.
    pub(crate) fn resize(&mut self, new_n: usize) {
        match new_n.cmp(&self.n) {
            std::cmp::Ordering::Equal => (),
            std::cmp::Ordering::Greater => self.grow(new_n),
            std::cmp::Ordering::Less => self.shrink(new_n),
        }
    }

    /// Copy `2n_rows` rows into a fresh `new_words`-wide tableau, taking `live`
    /// words of each half. The rest of the destination is padding and stays
    /// zero, which is the invariant every kernel here relies on.
    fn restride(&self, rows: usize, new_words: usize, live: usize) -> Vec<u64> {
        debug_assert!(live <= new_words && live <= self.words);
        let new_stride = 2 * new_words;
        let old_stride = self.stride();
        let mut data = vec![0u64; rows * new_stride];
        for row in 0..rows {
            let src = &self.data[row * old_stride..][..old_stride];
            let dst = &mut data[row * new_stride..][..new_stride];
            dst[..live].copy_from_slice(&src[..live]);
            dst[new_words..][..live].copy_from_slice(&src[self.words..][..live]);
        }
        data
    }

    fn grow(&mut self, new_n: usize) {
        let new_words = words_for(new_n);
        // Only the old register's live words carry anything; a wider register
        // is never narrower in live words, so this copies all of them.
        let live = logical_words(self.n);
        self.data = self.restride(2 * self.n, new_words, live);
        self.data.resize(4 * new_n * new_words, 0);
        self.phases.resize(2 * new_n, 0);
        self.words = new_words;
        let old_n = std::mem::replace(&mut self.n, new_n);
        for qubit in old_n..new_n {
            self.set_identity_rows(qubit);
        }
        self.reset_scratch();
    }

    fn shrink(&mut self, new_n: usize) {
        for qubit in new_n..self.n {
            assert!(
                self.is_basis_pair(qubit),
                "cannot drop qubit {qubit}: its preimages are not X/Z"
            );
        }
        // A retained preimage commutes with X_j and Z_j for every dropped j, so
        // it has no support there and truncation is lossless.
        let new_words = words_for(new_n);
        let live = logical_words(new_n);
        self.data = self.restride(2 * new_n, new_words, live);
        self.phases.truncate(2 * new_n);
        self.words = new_words;
        self.n = new_n;
        self.mask_tail();
        self.reset_scratch();
    }

    /// Clear the bits above `n` in each row's last live word. Words past it are
    /// padding that [`Frame::restride`] already left at zero.
    fn mask_tail(&mut self) {
        let tail = self.n & 63;
        if tail == 0 {
            return;
        }
        let mask = (1u64 << tail) - 1;
        let (words, live) = (self.words, logical_words(self.n));
        for row in self.data.chunks_exact_mut(2 * words) {
            debug_assert_eq!(row[live - 1] & !mask, 0);
            debug_assert_eq!(row[words + live - 1] & !mask, 0);
            row[live - 1] &= mask;
            row[words + live - 1] &= mask;
        }
    }

    /// Whether qubit's preimages are the untouched basis pair `X_q`, `Z_q`.
    fn is_basis_pair(&self, qubit: usize) -> bool {
        if self.phases[px(qubit)] != 0 || self.phases[pz(qubit)] != 0 {
            return false;
        }
        let (word, bit) = (qubit >> 6, 1u64 << (qubit & 63));
        // `unit_word` is the row offset the single set bit must sit at: the
        // x-half for an X preimage, the z-half for a Z preimage.
        let is_unit = |row: &[u64], unit_word: usize| {
            row.iter()
                .enumerate()
                .all(|(i, &w)| w == if i == unit_word { bit } else { 0 })
        };
        is_unit(self.row(px(qubit)), word) && is_unit(self.row(pz(qubit)), self.words + word)
    }

    fn reset_scratch(&mut self) {
        self.scratch.clear();
        self.scratch.resize(4 * self.words, 0);
        self.scratch_phases.clear();
        self.scratch_phases.resize(2, 0);
    }

    // --- Left multiplication: R ← G·R ---

    /// Hadamard: swap the two preimages of the qubit (`clifford_impl.rs:996`).
    pub(crate) fn left_h(&mut self, qubit: usize) {
        let stride = self.stride();
        let Frame { data, phases, .. } = self;
        let (x_row, z_row) = rows2_mut(data, stride, px(qubit), pz(qubit));
        x_row.swap_with_slice(z_row);
        phases.swap(px(qubit), pz(qubit));
    }

    /// `S = √Z` (`clifford_left_mul_eq_root_z`).
    pub(crate) fn left_s(&mut self, qubit: usize) {
        self.left_root_z(qubit, 1);
    }

    /// `S† = √Z†` (`clifford_left_mul_eq_root_z_inverse`).
    pub(crate) fn left_s_dag(&mut self, qubit: usize) {
        self.left_root_z(qubit, 3);
    }

    /// `√X` (`clifford_left_mul_eq_root_x`).
    pub(crate) fn left_sqrt_x(&mut self, qubit: usize) {
        self.left_root_x(qubit, 1);
    }

    /// `√X†` (`clifford_left_mul_eq_root_x_inverse`).
    pub(crate) fn left_sqrt_x_dag(&mut self, qubit: usize) {
        self.left_root_x(qubit, 3);
    }

    /// `√Y = Z` then `H` (`clifford_left_mul_eq_root_y`).
    pub(crate) fn left_sqrt_y(&mut self, qubit: usize) {
        self.left_z(qubit);
        self.left_h(qubit);
    }

    /// `√Y† = H` then `Z` (`clifford_left_mul_eq_root_y_inverse`).
    pub(crate) fn left_sqrt_y_dag(&mut self, qubit: usize) {
        self.left_h(qubit);
        self.left_z(qubit);
    }

    /// `x_row ← z_row · x_row`, then `i^delta`.
    fn left_root_z(&mut self, qubit: usize, delta: u8) {
        let stride = self.stride();
        let Frame { data, phases, .. } = self;
        let src_phase = phases[pz(qubit)];
        let (x_row, z_row) = rows2_mut(data, stride, px(qubit), pz(qubit));
        let dst_phase = &mut phases[px(qubit)];
        mul_row_left::<0>(x_row, dst_phase, z_row, src_phase);
        *dst_phase = (*dst_phase + delta) & 3;
    }

    /// `z_row ← x_row · z_row`, then `i^delta`.
    fn left_root_x(&mut self, qubit: usize, delta: u8) {
        let stride = self.stride();
        let Frame { data, phases, .. } = self;
        let src_phase = phases[px(qubit)];
        let (z_row, x_row) = rows2_mut(data, stride, pz(qubit), px(qubit));
        let dst_phase = &mut phases[pz(qubit)];
        mul_row_left::<0>(z_row, dst_phase, x_row, src_phase);
        *dst_phase = (*dst_phase + delta) & 3;
    }

    /// Pauli `X`: anticommutes with `Z_q` only (`clifford_left_mul_eq_x`).
    pub(crate) fn left_x(&mut self, qubit: usize) {
        self.phases[pz(qubit)] ^= 2;
    }

    /// Pauli `Y` (`clifford_left_mul_eq_y`).
    pub(crate) fn left_y(&mut self, qubit: usize) {
        self.phases[px(qubit)] ^= 2;
        self.phases[pz(qubit)] ^= 2;
    }

    /// Pauli `Z`: anticommutes with `X_q` only (`clifford_left_mul_eq_z`).
    pub(crate) fn left_z(&mut self, qubit: usize) {
        self.phases[px(qubit)] ^= 2;
    }

    /// CNOT (`clifford_left_mul_eq_cnot`): `x_c ← x_t·x_c`, `z_t ← z_c·z_t`.
    ///
    /// # Panics
    ///
    /// Panics if `control == target`.
    pub(crate) fn left_cx(&mut self, control: usize, target: usize) {
        assert_ne!(control, target, "cx needs distinct qubits");
        self.left_mul_row_by_row::<0>(px(control), px(target));
        self.left_mul_row_by_row::<0>(pz(target), pz(control));
    }

    /// CZ (`clifford_left_mul_eq_cz`): `x_c ← z_t·x_c`, `x_t ← z_c·x_t`.
    ///
    /// # Panics
    ///
    /// Panics if `first == second`.
    pub(crate) fn left_cz(&mut self, first: usize, second: usize) {
        assert_ne!(first, second, "cz needs distinct qubits");
        self.left_mul_row_by_row::<0>(px(first), pz(second));
        self.left_mul_row_by_row::<0>(px(second), pz(first));
    }

    /// SWAP: exchange both preimages of the two qubits (`clifford_impl.rs:1002`).
    pub(crate) fn left_swap(&mut self, first: usize, second: usize) {
        if first == second {
            return;
        }
        let stride = self.stride();
        let Frame { data, phases, .. } = self;
        let (a, b) = rows2_mut(data, stride, px(first), px(second));
        a.swap_with_slice(b);
        let (a, b) = rows2_mut(data, stride, pz(first), pz(second));
        a.swap_with_slice(b);
        phases.swap(px(first), px(second));
        phases.swap(pz(first), pz(second));
    }

    /// `dst ← src · dst` for two stored rows.
    fn left_mul_row_by_row<const W: usize>(&mut self, dst: usize, src: usize) {
        let stride = self.stride_w::<W>();
        let Frame { data, phases, .. } = self;
        let src_phase = phases[src];
        let (dst_row, src_row) = rows2_mut(data, stride, dst, src);
        mul_row_left::<W>(dst_row, &mut phases[dst], src_row, src_phase);
    }

    /// Apply a Pauli to the state (`R ← P·R`).
    ///
    /// Conjugation is blind to `P`'s global phase, so — like
    /// `left_mul_pauli` (`clifford_impl.rs:771`) — the operand's phase
    /// exponent is ignored and only sign flips remain.
    pub(crate) fn left_pauli(&mut self, pauli: &PauliString) {
        let phases = &mut self.phases;
        for qubit in set_bits(pauli.x_words()) {
            phases[pz(qubit)] ^= 2;
        }
        for qubit in set_bits(pauli.z_words()) {
            phases[px(qubit)] ^= 2;
        }
    }

    /// Apply `control`-conditioned `target`, both Paulis
    /// (`clifford_left_mul_eq_controlled_pauli`, `generic_algos.rs:408`).
    ///
    /// Both preimages are taken against the *unmodified* frame, and the
    /// right-multiplications by the target preimage all happen before the
    /// left-multiplications by the control preimage — the order matters when
    /// the two supports overlap.
    pub(crate) fn left_controlled_pauli(&mut self, control: &PauliString, target: &PauliString) {
        by_width!(self.words, W => self.left_controlled_pauli_w::<W>(control, target));
    }

    fn left_controlled_pauli_w<const W: usize>(
        &mut self,
        control: &PauliString,
        target: &PauliString,
    ) {
        let stride = self.stride_w::<W>();
        // Moving the buffers out of `self` (a pointer swap, not an allocation)
        // lets the preimage reads borrow `&self` while the temporaries are
        // written, and the row updates borrow `&mut self` while they are read.
        let mut scratch = std::mem::take(&mut self.scratch);
        let mut phases = std::mem::take(&mut self.scratch_phases);
        {
            let (target_row, rest) = scratch.split_at_mut(stride);
            let control_row = &mut rest[..stride];
            phases[0] = self.preimage_row::<W>(target, target_row);
            phases[1] = self.preimage_row::<W>(control, control_row);
        }
        let (target_row, rest) = scratch.split_at(stride);
        let control_row = &rest[..stride];

        for qubit in set_bits(control.x_words()) {
            self.mul_row_right_by::<W>(pz(qubit), target_row, phases[0]);
        }
        for qubit in set_bits(control.z_words()) {
            self.mul_row_right_by::<W>(px(qubit), target_row, phases[0]);
        }
        for qubit in set_bits(target.x_words()) {
            self.mul_row_left_by::<W>(pz(qubit), control_row, phases[1]);
        }
        for qubit in set_bits(target.z_words()) {
            self.mul_row_left_by::<W>(px(qubit), control_row, phases[1]);
        }

        self.scratch = scratch;
        self.scratch_phases = phases;
    }

    /// Apply an arbitrary Clifford to `support` (`support[i]` is `cl`'s qubit
    /// `i`), replicating `left_mul_clifford` (`clifford_impl.rs:1026`).
    ///
    /// The `paulimer` interop path, and the only place a raw tableau still
    /// reaches the frame. Nothing in production builds one — named gates and
    /// Pauli axes cover the engine's whole gate surface — so this survives as
    /// the oracle the `Gate1Q` compositions and the differential tests are
    /// checked against, and `paulimer` with it. Cost is `O(k · w · W)` for a
    /// `k`-qubit `cl` of preimage weight `w`.
    ///
    /// # Panics
    ///
    /// Panics if `support.len() != cl.num_qubits()`.
    #[cfg(test)]
    pub(crate) fn left_clifford(&mut self, cl: &CliffordUnitary, support: &[usize]) {
        assert_eq!(
            support.len(),
            cl.num_qubits(),
            "support width must match the Clifford's"
        );
        by_width!(self.words, W => self.left_clifford_w::<W>(cl, support));
    }

    #[cfg(test)]
    fn left_clifford_w<const W: usize>(&mut self, cl: &CliffordUnitary, support: &[usize]) {
        let k = cl.num_qubits();
        let stride = self.stride_w::<W>();
        let mut scratch = std::mem::take(&mut self.scratch);
        let mut phases = std::mem::take(&mut self.scratch_phases);
        if scratch.len() < 2 * k * stride {
            scratch.resize(2 * k * stride, 0);
            phases.resize(2 * k, 0);
        }

        for slot in 0..k {
            let x_preimage = cl.preimage_x_view(slot);
            let z_preimage = cl.preimage_z_view(slot);
            let (head, tail) = scratch.split_at_mut((2 * slot + 1) * stride);
            phases[2 * slot] = self.remapped_preimage_row::<W>(
                x_preimage.x_bits(),
                x_preimage.z_bits(),
                x_preimage.xz_phase_exponent(),
                support,
                &mut head[2 * slot * stride..],
            );
            phases[2 * slot + 1] = self.remapped_preimage_row::<W>(
                z_preimage.x_bits(),
                z_preimage.z_bits(),
                z_preimage.xz_phase_exponent(),
                support,
                &mut tail[..stride],
            );
        }

        for (slot, &qubit) in support.iter().enumerate() {
            self.assign_row(
                px(qubit),
                &scratch[2 * slot * stride..][..stride],
                phases[2 * slot],
            );
            self.assign_row(
                pz(qubit),
                &scratch[(2 * slot + 1) * stride..][..stride],
                phases[2 * slot + 1],
            );
        }

        self.scratch = scratch;
        self.scratch_phases = phases;
    }

    /// `R†·P·R` for a Pauli given on `cl`'s local indices, remapped through
    /// `support` (`sparse_pauli_on_support`, `clifford_impl.rs:820`).
    ///
    /// `paulimer` sorts the remapped indices (it collects them into an
    /// `IndexSet`) but the sort cannot change the answer: within the x group
    /// the factors are `R†X_iR`, which mutually commute, and likewise for the z
    /// group, so only the group order — x before z — fixes the phase.
    #[cfg(test)]
    fn remapped_preimage_row<const W: usize>(
        &self,
        x_bits: &impl Bitwise,
        z_bits: &impl Bitwise,
        pauli_phase: u8,
        support: &[usize],
        out: &mut [u64],
    ) -> u8 {
        let words = width::<W>(self.words);
        let (out_x, out_z) = out[..2 * words].split_at_mut(words);
        out_x.fill(0);
        out_z.fill(0);
        let mut phase = 0u8;
        // Tested bit by bit over `support`, not driven by `Bitwise::support()`:
        // these are *dense* views over an aligned bit vector, whose support
        // iterator walks every bit of the padded allocation — 512 of them for
        // the one-qubit tableau a gate hands over. Four of those iterations
        // measured at 1.1 µs per application, which was the entire cost of the
        // tableau path. `index` is O(1), and `support.len()` is the tableau's
        // own width.
        for (local, &qubit) in support.iter().enumerate() {
            if x_bits.index(local) {
                self.accumulate::<W>(px(qubit), out_x, out_z, &mut phase);
            }
        }
        for (local, &qubit) in support.iter().enumerate() {
            if z_bits.index(local) {
                self.accumulate::<W>(pz(qubit), out_x, out_z, &mut phase);
            }
        }
        (phase + pauli_phase) & 3
    }

    /// Overwrite a stored row.
    #[cfg(test)]
    fn assign_row(&mut self, row: usize, src: &[u64], phase: u8) {
        let stride = self.stride();
        self.data[row * stride..][..stride].copy_from_slice(src);
        self.phases[row] = phase;
    }

    fn mul_row_right_by<const W: usize>(&mut self, row: usize, src: &[u64], src_phase: u8) {
        let stride = self.stride_w::<W>();
        let Frame { data, phases, .. } = self;
        mul_row_right::<W>(
            &mut data[row * stride..][..stride],
            &mut phases[row],
            src,
            src_phase,
        );
    }

    fn mul_row_left_by<const W: usize>(&mut self, row: usize, src: &[u64], src_phase: u8) {
        let stride = self.stride_w::<W>();
        let Frame { data, phases, .. } = self;
        mul_row_left::<W>(
            &mut data[row * stride..][..stride],
            &mut phases[row],
            src,
            src_phase,
        );
    }

    // --- Frame decomposition ---

    /// `out ← out · ρ_row` for a stored row.
    #[inline]
    fn accumulate<const W: usize>(
        &self,
        row: usize,
        out_x: &mut [u64],
        out_z: &mut [u64],
        out_phase: &mut u8,
    ) {
        let words = width::<W>(self.words);
        let stride = 2 * words;
        let (src_x, src_z) = self.data[row * stride..][..stride].split_at(words);
        mul_right::<W>(out_x, out_z, out_phase, src_x, src_z, self.phases[row]);
    }

    /// Decompose `pauli` in the frame: `R†·P·R = i^k · X^out_x · Z^out_z`,
    /// returning `k`.
    ///
    /// This is `paulimer`'s `Clifford::preimage` without its two `DensePauli`
    /// allocations: the operand's own x/z words drive the row product, so it
    /// costs `O(weight · W)` and the caller gets word masks back instead of a
    /// dense Pauli it would have to scan bit by bit.
    ///
    /// The operand's contribution to `k` is
    /// [`xz_phase_exponent`] — one `i` per `Y` site, since the
    /// masks name Hermitian Paulis and carry no phase of their own.
    ///
    /// # Panics
    ///
    /// Panics unless both output slices are exactly [`Frame::words`] long, or
    /// if `pauli` has support at or beyond [`Frame::num_qubits`].
    pub(crate) fn preimage_into(
        &self,
        pauli: &PauliString,
        out_x: &mut [u64],
        out_z: &mut [u64],
    ) -> u8 {
        assert_eq!(
            out_x.len(),
            self.words,
            "output width must match the frame's"
        );
        assert_eq!(
            out_z.len(),
            self.words,
            "output width must match the frame's"
        );
        by_width!(self.words, W => self.preimage_into_w::<W>(pauli, out_x, out_z))
    }

    fn preimage_into_w<const W: usize>(
        &self,
        pauli: &PauliString,
        out_x: &mut [u64],
        out_z: &mut [u64],
    ) -> u8 {
        out_x.fill(0);
        out_z.fill(0);
        let mut phase = 0u8;
        for qubit in set_bits(pauli.x_words()) {
            self.accumulate::<W>(px(qubit), out_x, out_z, &mut phase);
        }
        for qubit in set_bits(pauli.z_words()) {
            self.accumulate::<W>(pz(qubit), out_x, out_z, &mut phase);
        }
        (phase + xz_phase_exponent(pauli)) & 3
    }

    /// [`Frame::preimage_into`] for a single-qubit basis axis, without the
    /// Pauli string the caller would otherwise have to build.
    ///
    /// `X_q` and `Z_q` are stored rows, so their preimage is a copy and a phase
    /// read; `Y_q = i·X_q·Z_q` costs one row multiply on top. The `T` and reset
    /// entry points measure and rotate about exactly these three axes, and they
    /// run once per physical gate — allocating a Pauli to describe `Z_q` was
    /// most of their cost.
    ///
    /// # Panics
    ///
    /// Panics if `qubit` is out of range or the output widths are not
    /// [`Frame::words`].
    pub(crate) fn preimage_basis_into(
        &self,
        axis: Axis,
        qubit: usize,
        out_x: &mut [u64],
        out_z: &mut [u64],
    ) -> u8 {
        assert!(
            qubit < self.n,
            "qubit {qubit} out of range for {} qubits",
            self.n
        );
        assert_eq!(
            out_x.len(),
            self.words,
            "output width must match the frame's"
        );
        assert_eq!(
            out_z.len(),
            self.words,
            "output width must match the frame's"
        );
        by_width!(self.words, W => self.preimage_basis_into_w::<W>(axis, qubit, out_x, out_z))
    }

    fn preimage_basis_into_w<const W: usize>(
        &self,
        axis: Axis,
        qubit: usize,
        out_x: &mut [u64],
        out_z: &mut [u64],
    ) -> u8 {
        let words = width::<W>(self.words);
        let row = if axis == Axis::Z {
            pz(qubit)
        } else {
            px(qubit)
        };
        let (src_x, src_z) = self.data[row * 2 * words..][..2 * words].split_at(words);
        out_x.copy_from_slice(src_x);
        out_z.copy_from_slice(src_z);
        let mut phase = self.phases[row];
        if axis == Axis::Y {
            // `Y = i·X·Z` in the xz normal form, so the extra `i` rides along
            // with the product of the two rows.
            self.accumulate::<W>(pz(qubit), out_x, out_z, &mut phase);
            phase += 1;
        }
        phase & 3
    }

    /// [`Frame::preimage_into`] writing into one `2W`-word row.
    fn preimage_row<const W: usize>(&self, pauli: &PauliString, out: &mut [u64]) -> u8 {
        let words = width::<W>(self.words);
        let (out_x, out_z) = out[..2 * words].split_at_mut(words);
        self.preimage_into_w::<W>(pauli, out_x, out_z)
    }

    // --- Right multiplication: R ← R·V ---

    /// `R ← R · exp(iπ/4·G)` for the Hermitian Pauli `G = i^phase·X^x·Z^z`
    /// already expressed in the frame.
    ///
    /// Rows commuting with `G` are untouched; the rest become `ρ·(i·G)` (see
    /// the [module docs](self)). One pass, two parity folds and two XORs per
    /// touched row, no allocation, no image computation.
    ///
    /// # Panics
    ///
    /// Panics unless both masks are exactly [`Frame::words`] long.
    pub(crate) fn right_pauli_exp(&mut self, g_x: &[u64], g_z: &[u64], g_phase: u8) {
        assert_eq!(g_x.len(), self.words, "mask width must match the frame's");
        assert_eq!(g_z.len(), self.words, "mask width must match the frame's");
        by_width!(self.words, W => self.right_pauli_exp_w::<W>(g_x, g_z, g_phase));
    }

    fn right_pauli_exp_w<const W: usize>(&mut self, g_x: &[u64], g_z: &[u64], g_phase: u8) {
        let words = width::<W>(self.words);
        let Frame { data, phases, .. } = self;
        for (row, phase) in data.chunks_exact_mut(2 * words).zip(phases.iter_mut()) {
            let (row_x, row_z) = row.split_at_mut(words);
            // The `⟨row.z, G.x⟩` fold doubles as `mul_right`'s cross term, so
            // it is computed before the XORs, as in `mul_assign_right`.
            let cross = dot_parity::<W>(row_z, g_x);
            if cross != dot_parity::<W>(row_x, g_z) {
                xor_assign::<W>(row_x, g_x);
                xor_assign::<W>(row_z, g_z);
                *phase = (*phase + (u8::from(cross) << 1) + g_phase + 1) & 3;
            }
        }
    }

    /// `R ← R · Z_pivot`: negate every row that anticommutes with `Z_pivot`.
    ///
    /// # Panics
    ///
    /// Panics if `pivot` is out of range.
    pub(crate) fn right_pauli_z(&mut self, pivot: usize) {
        assert!(
            pivot < self.n,
            "pivot {pivot} out of range for {} qubits",
            self.n
        );
        by_width!(self.words, W => self.right_pauli_z_w::<W>(pivot));
    }

    fn right_pauli_z_w<const W: usize>(&mut self, pivot: usize) {
        let (word, bit) = (pivot >> 6, 1u64 << (pivot & 63));
        let words = width::<W>(self.words);
        let Frame { data, phases, .. } = self;
        for (row, phase) in data.chunks_exact(2 * words).zip(phases.iter_mut()) {
            if row[word] & bit != 0 {
                *phase ^= 2;
            }
        }
    }

    // --- Cold paths: whole-row reads for tests and state reconstruction ---
    //
    // The simulator reads the frame through the masks above and needs a whole
    // row only to replay the state vector; the `paulimer` conversions at the
    // end exist purely for differential tests. Hence the `cfg(test)` gates.
    //
    // They all speak in `logical_words(n)`, not the padded row width: a
    // `DensePauli` handed to or taken from `paulimer` must be as wide as
    // `paulimer` would make it, not as wide as the row kernels prefer.

    /// `R†·X_qubit·R`.
    #[cfg(test)]
    pub(crate) fn preimage_x(&self, qubit: usize) -> RowPauli {
        self.dense_row(px(qubit))
    }

    /// `R†·Z_qubit·R`.
    #[cfg(test)]
    pub(crate) fn preimage_z(&self, qubit: usize) -> RowPauli {
        self.dense_row(pz(qubit))
    }

    #[cfg(test)]
    fn dense_row(&self, row: usize) -> RowPauli {
        let live = logical_words(self.n);
        let (x_bits, z_bits) = self.row(row).split_at(self.words);
        RowPauli {
            x: x_bits[..live].to_vec(),
            z: z_bits[..live].to_vec(),
            phase: self.phases[row],
        }
    }

    /// `R·X_qubit·R†`, the destabilizer `D_qubit`.
    ///
    /// The image bits are a column of the tableau — the symplectic transpose
    /// `projective_x_image_at` (`clifford_impl.rs:135`) reads — and the phase
    /// is recovered by pushing those bits back through the preimages
    /// (`clifford_image_with_phase`, `generic_algos.rs:363`). Per-bit work;
    /// this is a test and state-reconstruction path only.
    pub(crate) fn image_x(&self, qubit: usize) -> RowPauli {
        self.image(qubit, Frame::z_bit)
    }

    /// `R·Z_qubit·R†`, the stabilizer `S_qubit`.
    pub(crate) fn image_z(&self, qubit: usize) -> RowPauli {
        self.image(qubit, Frame::x_bit)
    }

    /// Shared image body: `read` picks the half of the tableau the image bits
    /// come from (z-halves for `image_x`, x-halves for `image_z`).
    fn image(&self, qubit: usize, read: impl Fn(&Self, usize, usize) -> bool) -> RowPauli {
        let live = logical_words(self.n);
        let mut x_bits = vec![0u64; live];
        let mut z_bits = vec![0u64; live];
        for j in 0..self.n {
            let (word, bit) = (j >> 6, 1u64 << (j & 63));
            if read(self, pz(j), qubit) {
                x_bits[word] |= bit;
            }
            if read(self, px(j), qubit) {
                z_bits[word] |= bit;
            }
        }

        let mut scratch_x = vec![0u64; self.words];
        let mut scratch_z = vec![0u64; self.words];
        let mut phase = 0u8;
        for j in set_bits(&x_bits) {
            self.accumulate::<0>(px(j), &mut scratch_x, &mut scratch_z, &mut phase);
        }
        for j in set_bits(&z_bits) {
            self.accumulate::<0>(pz(j), &mut scratch_x, &mut scratch_z, &mut phase);
        }

        RowPauli {
            x: x_bits,
            z: z_bits,
            // The image carries the conjugate of its preimage's phase.
            phase: (4 - phase) & 3,
        }
    }

    /// Rebuild an equivalent `paulimer` tableau.
    #[cfg(test)]
    pub(crate) fn to_clifford_unitary(&self) -> CliffordUnitary {
        let mut clifford = CliffordUnitary::identity(self.n);
        for qubit in 0..self.n {
            clifford
                .preimage_x_view_mut(qubit)
                .assign(&self.preimage_x(qubit).to_dense());
            clifford
                .preimage_z_view_mut(qubit)
                .assign(&self.preimage_z(qubit).to_dense());
        }
        clifford
    }

    /// Adopt a `paulimer` tableau.
    #[cfg(test)]
    pub(crate) fn from_clifford_unitary(clifford: &CliffordUnitary) -> Self {
        let mut frame = Frame::identity(clifford.num_qubits());
        for qubit in 0..frame.n {
            frame.assign_dense_row(px(qubit), &clifford.preimage_x(qubit));
            frame.assign_dense_row(pz(qubit), &clifford.preimage_z(qubit));
        }
        frame
    }

    #[cfg(test)]
    fn assign_dense_row(&mut self, row: usize, pauli: &DensePauli) {
        let words = self.words;
        let stride = self.stride();
        let live = logical_words(self.n);
        let target = &mut self.data[row * stride..][..stride];
        target.fill(0);
        target[..live].copy_from_slice(&pauli.x_bits().as_words()[..live]);
        target[words..][..live].copy_from_slice(&pauli.z_bits().as_words()[..live]);
        self.phases[row] = pauli.xz_phase_exponent() & 3;
    }
}
