//! Stabilizer-frame Clifford+T simulator.
//!
//! # State representation
//!
//! The engine tracks ticit's `CliffordFrame` `R` and a sparse complex amplitude
//! map over destabilizer-coset
//! labels. Writing
//! `S_i = R Z_i R† = image_z(i)` (stabilizers) and `D_i = R X_i R† = image_x(i)`
//! (destabilizers),
//!
//! ```text
//! |ψ⟩ = Σ_c xvec[c] · D^c |ψ0⟩,   |ψ0⟩ = R|0…0⟩,   D^c = ∏_{i: c_i=1} D_i.
//! ```
//!
//! The clean way to see the whole scheme: `D^c|ψ0⟩ = R|c⟩`, so
//! `|ψ⟩ = R|χ⟩` with `|χ⟩ = Σ_c xvec[c]|c⟩` the *rotated state* `R†|ψ⟩`. The
//! amplitude map **is** `|χ⟩` in the computational basis. This gives every
//! operation directly:
//!
//! * A Clifford gate `G` sends `|ψ⟩ → G|ψ⟩`, i.e. `R → G·R` (a `left_mul`),
//!   and leaves `|χ⟩` — the amplitude map — untouched.
//! * `T_P` and measurement act on `|ψ⟩` as `T`/projector about `P`, which in
//!   the rotated frame is the same operation about `Q = R†PR = preimage(P)`,
//!   acting on the computational-basis vector `|χ⟩`.
//!
//! # Pauli decomposition in the frame
//!
//! For a Pauli `P`, `Q = R†PR = i^k · X^a Z^b` with `a` the x-bits, `b` the
//! z-bits and `k` the phase exponent of the **X-then-Z** normal form (not
//! `xyz_phase_exponent`). Then on a basis term,
//!
//! ```text
//! Q|c⟩ = i^k · (−1)^{⟨b,c⟩} · |c ⊕ a⟩.
//! ```
//!
//! Everything that sweeps the amplitude map is generic over its packed label
//! type instead of dispatching per term.

use std::collections::{HashMap, hash_map::RandomState};
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::hash::{BuildHasher, Hasher};

use crate::frames::{self, CliffordFrame, coordinates_in_frame, preimage};
use crate::pauli::{PauliString, measurement_phase_sign, pauli_anticommutes};
use crate::random::rand_float;
use num_complex::Complex64;

mod error;
mod label;

pub use error::SimError;

use label::{Key, Label, LabelKey, Width};

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

/// Tolerance for "numerically deterministic" / "impossible to post-select"
/// decisions and internal reality checks.
const TOL: f64 = 1e-9;

/// Amplitudes this small are float-drift noise, not state. Sized for
/// correctness verification of small physical-circuit batches.
///
/// The pruning tests compare `norm_sqr()` against the square of this rather
/// than `norm()` against it: `Complex::norm` is a `hypot` call, which measured
/// at 5% of a profiled run, and squaring the threshold is exact enough here
/// (`1e-24` is nowhere near the subnormal range).
const DEFAULT_PRUNE_EPSILON: f64 = 1e-12;

/// Live-label ceiling: fails loudly long before memory is exhausted.
const DEFAULT_RANK_CAP: usize = 1 << 20;

/// Sign of the frame-compression rotation applied after a random measurement.
const PAULI_EXP_SIGN: f64 = -1.0;

/// Outcome of a Pauli measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureResult {
    /// Sampled (or forced) eigenvalue bit: `false` = `+1`, `true` = `−1`.
    pub outcome: bool,
    /// Probability the engine assigned to `outcome` in the pre-measurement state.
    pub probability: f64,
    /// Whether the outcome was forced by the state (probability `≈ 0` or `1`).
    pub deterministic: bool,
}

/// A stim-`TableauSimulator`-style procedural Clifford+T simulator.
///
/// Apply operations one at a time, read [`measure`](Self::measure) outcomes,
/// branch on them (feedforward), and continue. See the [module docs](self) for
/// the state representation.
///
/// # Naming
///
/// The method names are stim's wherever stim has one, so a circuit written
/// against `stim.TableauSimulator` transliterates: the single-qubit Cliffords
/// (`h`, `s`, `sqrt_x`, [`c_xyz`](Self::c_xyz), [`h_xy`](Self::h_xy), …), the
/// two-qubit `<A>C<B>` family ([`cx`](Self::cx)/[`cnot`](Self::cnot)/`zcx`,
/// `cy`/`zcy`, `cz`/`zcz`, `xcx` … `ycz`) plus `swap`/[`iswap`](Self::iswap),
/// [`measure`](Self::measure)/[`measure_observable`](Self::measure_observable),
/// the `postselect_*` family, `reset`/`reset_x`/`reset_y`/`reset_z`, and
/// [`peek_observable_expectation`](Self::peek_observable_expectation)/`peek_x`/
/// `peek_y`/`peek_z`. Two departures: the measurements return a
/// [`MeasureResult`] rather than a bare `bool` (the branch weight is computed
/// to sample from it, so reporting it is free), and observables are
/// [`PauliString`]s.
///
/// The Clifford+T extension keeps ticit's names — [`t`](Self::t),
/// [`t_dag`](Self::t_dag), [`t_pauli`](Self::t_pauli), [`ccz`](Self::ccz) and
/// [`rank`](Self::rank).
#[derive(Clone, Debug)]
pub struct TableauSimulator {
    core: Core,
    /// `|χ⟩ = R†|ψ⟩` as a sparse computational-basis vector, in whichever
    /// label-width specialization the register calls for.
    amps: Amps,
}

/// The simulator minus its amplitude storage.
///
/// Split out purely so the width-dispatched routines can hold a `&mut` to the
/// frame, the RNG and the thresholds while the amplitude map — a *different*
/// field, and the generic one — is borrowed alongside them.
#[derive(Clone, Debug)]
struct Core {
    /// Qubit count (grows on demand via [`TableauSimulator::ensure_qubits`]).
    n: usize,
    /// `u64` words per label, rounded to the selected label storage width.
    words: usize,
    /// The Clifford frame `R`.
    r: CliffordFrame,
    rng: u64,
    /// Amplitudes at or below this modulus are dropped after each `T` and
    /// measurement projection. Exact arithmetic preserves the norm, so this only
    /// removes terms that cancelled to within rounding.
    prune_epsilon: f64,
    /// Maximum live-label count. A `T`/measurement that would exceed it fails
    /// with [`SimError::RankOverflow`] rather than exhausting memory.
    rank_cap: usize,
}

/// The amplitude map and its staging buffers, at one label width.
#[derive(Clone, Debug)]
struct Terms<K: LabelKey> {
    map: HashMap<K, Complex64>,
    /// Staging buffers for the coset-pair rewrites (`T` and projection).
    rotation: RotationScratch<K>,
}

impl<K: LabelKey> PartialEq for Terms<K> {
    /// Staging buffers are working memory, not state.
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
    }
}

/// The live amplitude map, specialized to the register's label width.
///
/// Growing past a width boundary swaps the variant; see [`Amps::widen`].
#[derive(Clone, Debug, PartialEq)]
enum Amps {
    W1(Terms<Key<1>>),
    W2(Terms<Key<2>>),
    W4(Terms<Key<4>>),
    W8(Terms<Key<8>>),
    Wide(Terms<Label>),
}

/// Run `$body` for the amplitude storage's actual width, with `$terms` bound to
/// the monomorphized [`Terms`] and `$core` to everything else.
///
/// The body is duplicated across the variants at expansion, so it has to stay a
/// call into generic code — the algorithms live in `impl<K> Terms<K>`, not
/// here. Destructuring `*$sim` is what makes the two borrows disjoint.
macro_rules! with_terms {
    ($sim:expr, |$core:ident, $terms:ident| $body:expr) => {{
        let TableauSimulator {
            core: ref mut $core,
            amps: ref mut storage,
        } = *$sim;
        match storage {
            Amps::W1($terms) => $body,
            Amps::W2($terms) => $body,
            Amps::W4($terms) => $body,
            Amps::W8($terms) => $body,
            Amps::Wide($terms) => $body,
        }
    }};
}

/// [`with_terms`] for the read-only paths.
macro_rules! with_terms_ref {
    ($sim:expr, |$core:ident, $terms:ident| $body:expr) => {{
        let TableauSimulator {
            core: ref $core,
            amps: ref storage,
        } = *$sim;
        match storage {
            Amps::W1($terms) => $body,
            Amps::W2($terms) => $body,
            Amps::W4($terms) => $body,
            Amps::W8($terms) => $body,
            Amps::Wide($terms) => $body,
        }
    }};
}

/// Where a coset-pair rewrite stages its result before committing it.
///
/// The buffers are held on the amplitude map and reused: a long circuit pays
/// for them once. They are always left cleared, so cloning a simulator does not
/// copy them.
#[derive(Clone, Debug)]
struct RotationScratch<K> {
    /// New amplitude of each live label, in amplitude-map iteration order.
    values: Vec<Complex64>,
    /// Coset partners the rotation adds to the map, with their amplitudes.
    inserts: Vec<(K, Complex64)>,
    /// Each live label's coset partner amplitude, or `None` where the map has
    /// no such label — in the same iteration order. Written by the
    /// expectation pass a measurement already has to run, read by the
    /// projection that follows it, so the pair is located once instead of
    /// twice. Unused by `T`, which has no expectation pass to piggyback on.
    partners: Vec<Option<Complex64>>,
}

/// Hand-written rather than derived: the buffers start empty whatever `K` is,
/// and a derived `Default` would demand `K: Default` for no reason.
impl<K> Default for RotationScratch<K> {
    fn default() -> Self {
        RotationScratch {
            values: Vec::new(),
            inserts: Vec::new(),
            partners: Vec::new(),
        }
    }
}

impl<K> RotationScratch<K> {
    /// Empty the buffers, keeping their capacity for the next operation.
    fn clear(&mut self) {
        self.values.clear();
        self.inserts.clear();
        self.partners.clear();
    }
}

/// `i^k` as a complex number, `k` taken mod 4.
#[inline]
fn i_pow(k: u8) -> Complex64 {
    match k & 3 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

/// The `(a, b, ζ)` frame decomposition of a Pauli: `Q = ζ · X^a Z^b`, with `a`
/// and `b` as label-width word masks (the coset shift and the sign mask).
///
/// The masks carry the amplitude map's own key type, so the frame can write
/// them in place and the innermost loops — one `xor` and one `dot_parity`
/// against these per term — run at the same fixed width as the terms.
struct Decomp<K> {
    /// X part — the coset shift `c ↦ c ⊕ a`.
    a: K,
    /// Z part — the sign mask `(−1)^{⟨b,c⟩}`.
    b: K,
    /// The phase exponent `k`, kept alongside `zeta` because the measurement
    /// path does exponent arithmetic on it (`G = i^{k+3}·X^a Z^{b⊕e_p}`).
    phase: u8,
    /// `i^k`.
    zeta: Complex64,
}

/// The per-term constants of the random-measurement pass, hoisted out of the
/// loop: the projector, the `Z_p^s` fold and the frame compression, composed.
///
/// The derivation is in [`Terms::measure_random`]; what matters here is the
/// shape it leaves behind. All four scattered products land back on the pair
/// `{c, c ⊕ a}`, so the whole composite is a coset-pair rewrite and
/// [`Projection::rewrite_pair`] is its kernel.
struct Projection<'a, K> {
    d: &'a Decomp<K>,
    /// `G`'s z-mask, `b ⊕ e_pivot`.
    gb: &'a K,
    pivot: usize,
    /// The sampled outcome bit, `s`.
    s: bool,
    /// `(−1)^s`.
    ssign: f64,
    /// `i·PAULI_EXP_SIGN/√2·ζ_G`, the compression's off-diagonal weight.
    compress: Complex64,
    /// `(−1)^{⟨gb,a⟩}`: how the `gb` parity of `c ⊕ a` differs from `c`'s.
    shift_flip: f64,
}

impl<K: LabelKey> Projection<'_, K> {
    /// The pair `{c, c ⊕ a}`'s two new amplitudes from its two old ones, `x`
    /// at `c` and `y` at the partner (`y = 0` where the map has no partner).
    ///
    /// Everything the partner needs is derived from `c`'s own three bit tests
    /// rather than repeated on the partner's label, which is what makes this
    /// one kernel per *pair* instead of two per *member*:
    ///
    /// * the pivot is a set bit of `a`, so the partner's pivot bit — the whole
    ///   of its `Z_p^s` factor — is `c`'s flipped;
    /// * `⟨b, c ⊕ a⟩` and `⟨b, c⟩` differ by the constant `⟨b, a⟩`, and
    ///   likewise for `gb`.
    ///
    /// Those parities are the expensive part: the workspace sets no target
    /// features, so each is a ~12-operation SWAR pop-count, and this loop is
    /// the engine's hottest.
    ///
    /// `inline(always)`, not `inline`: the caller discards the second half on
    /// the common branch, and out of line that costs both the call and the
    /// complex multiplies inlining lets the optimizer delete. As a standalone
    /// symbol its predecessor measured at 30% of a profiled run.
    #[inline(always)]
    fn rewrite_pair(&self, c: &K, x: Complex64, y: Complex64) -> (Complex64, Complex64) {
        let zc = if self.s && c.get(self.pivot) {
            -1.0
        } else {
            1.0
        };
        // `Z_p^s` at the partner. Only the `s = true` fold sees the pivot bit
        // at all, and then the partner's is the opposite of `c`'s.
        let zp = if self.s { -zc } else { 1.0 };
        let sb = if c.dot_parity(&self.d.b) { -1.0 } else { 1.0 };
        let sg = if c.dot_parity(self.gb) { -1.0 } else { 1.0 };
        // `⟨b,a⟩` from `⟨gb,a⟩`: `gb = b ⊕ e_pivot` and the pivot bit of `a`
        // is set, so the two parities are always opposite.
        let shift_b = -self.shift_flip;

        // Projector and fold, gathered by destination rather than scattered by
        // source: `u` is everything landing on `c`, `v` everything landing on
        // the partner. Each member sends half of itself across.
        let u = 0.5 * zc * x + (0.5 * self.ssign * (sb * shift_b) * zc) * self.d.zeta * y;
        let v = (0.5 * self.ssign * sb * zp) * self.d.zeta * x + 0.5 * zp * y;

        // The compression `(I ± iG)/√2` then mixes the pair once more.
        (
            FRAC_1_SQRT_2 * u + self.compress * (sg * self.shift_flip) * v,
            self.compress * sg * u + FRAC_1_SQRT_2 * v,
        )
    }
}

// ==============================================================================
// Public surface
// ==============================================================================

impl TableauSimulator {
    // --- Construction ---

    /// A fresh `|0…0⟩` simulator on `num_qubits` qubits with an OS-seeded RNG.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_seed(num_qubits, RandomState::new().build_hasher().finish())
    }

    /// A fresh `|0…0⟩` simulator with a fixed RNG seed (reproducible sampling).
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let r = CliffordFrame::new(num_qubits);
        let words = storage_words(num_qubits.div_ceil(64));
        TableauSimulator {
            core: Core {
                n: num_qubits,
                words,
                r,
                rng: seed,
                prune_epsilon: DEFAULT_PRUNE_EPSILON,
                rank_cap: DEFAULT_RANK_CAP,
            },
            amps: Amps::unit(words),
        }
    }

    /// Lowers the rank cap so a test can provoke [`SimError::RankOverflow`]
    /// without building a genuinely magic-heavy circuit.
    #[doc(hidden)]
    pub fn set_rank_cap(&mut self, cap: usize) {
        self.core.rank_cap = cap;
    }

    /// Raises the pruning threshold so a test can provoke
    /// [`SimError::EmptyStateAfterPruning`], which the default never reaches.
    #[doc(hidden)]
    pub fn set_prune_epsilon(&mut self, epsilon: f64) {
        self.core.prune_epsilon = epsilon;
    }

    /// Reseed the outcome RNG in place, leaving the quantum state alone.
    ///
    /// A `RepeatUntilSuccess` retry replays the same ops from the same state;
    /// without a fresh seed every attempt would sample the identical outcomes
    /// and the loop could never make progress.
    pub fn reseed_rng(&mut self, seed: u64) {
        self.core.rng = seed;
    }

    /// Adopt `snapshot`'s RNG position. Rolling a failed attempt's state back
    /// must not also roll the RNG back, or the retry repeats the same outcomes.
    pub fn restore_rng_from(&mut self, snapshot: &Self) {
        self.core.rng = snapshot.core.rng;
    }

    /// Current qubit count.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.core.n
    }

    /// Number of live amplitude terms (the stabilizer rank).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.amps.len()
    }

    /// Ensure at least `need` qubits exist, growing `R` and widening labels.
    fn ensure_qubits(&mut self, need: usize) {
        if need <= self.core.n {
            return;
        }
        self.core.r.grow_to(need);
        let new_words = storage_words(need.div_ceil(64));
        if new_words > self.core.words {
            self.amps.widen(new_words);
            self.core.words = new_words;
        }
        self.core.n = need;
        debug_assert_eq!(self.core.n, self.core.r.nqubits, "frame tracks register");
        debug_assert_eq!(
            self.core.words,
            storage_words(self.core.n.div_ceil(64)),
            "label width tracks register",
        );
    }

    fn ensure_for(&mut self, pauli: &PauliString) {
        if let Some(max) = max_support(pauli) {
            self.ensure_qubits(max + 1);
        }
    }

    // --- Clifford gates — mutate R only, xvec untouched. ---

    /// Hadamard.
    pub fn h(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_h(&mut self.core.r, q);
    }
    /// Phase gate `S = √Z`.
    pub fn s(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_s(&mut self.core.r, q);
    }
    /// `S† = √Z†`.
    pub fn s_dag(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_sdg(&mut self.core.r, q);
    }
    /// Pauli `X`.
    pub fn x(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_x(&mut self.core.r, q);
    }
    /// Pauli `Y`.
    pub fn y(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_y(&mut self.core.r, q);
    }
    /// Pauli `Z`.
    pub fn z(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_z(&mut self.core.r, q);
    }
    /// `√X`.
    pub fn sqrt_x(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_sqrt_x(&mut self.core.r, q);
    }
    /// `√X†`.
    pub fn sqrt_x_dag(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_sqrt_x_dag(&mut self.core.r, q);
    }
    /// `√Y`.
    pub fn sqrt_y(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_sqrt_y(&mut self.core.r, q);
    }
    /// `√Y†`.
    pub fn sqrt_y_dag(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_sqrt_y_dag(&mut self.core.r, q);
    }
    /// CNOT with `control`, `target`.
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn cx(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        if control == target {
            return Err(SimError::RepeatedQubit(control));
        }
        self.ensure_qubits(control.max(target) + 1);
        frames::left_cx(&mut self.core.r, control, target);
        Ok(())
    }
    /// Controlled-Z (symmetric).
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn cz(&mut self, a: usize, b: usize) -> Result<(), SimError> {
        if a == b {
            return Err(SimError::RepeatedQubit(a));
        }
        self.ensure_qubits(a.max(b) + 1);
        frames::left_cz(&mut self.core.r, a, b);
        Ok(())
    }
    /// Swap.
    pub fn swap(&mut self, a: usize, b: usize) {
        self.ensure_qubits(a.max(b) + 1);
        frames::left_swap(&mut self.core.r, a, b);
    }

    /// `ISWAP`: swap `a` and `b`, with an `i` on the two odd-parity terms.
    ///
    /// `ISWAP = SWAP·CZ·(S⊗S)`. All three factors are symmetric and either
    /// diagonal or a permutation, so they commute and the order is free; the
    /// `CZ` goes first only because it is the fallible step, which keeps a
    /// rejected call from leaving half a gate behind.
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn iswap(&mut self, a: usize, b: usize) -> Result<(), SimError> {
        self.cz(a, b)?;
        self.s(a);
        self.s(b);
        self.swap(a, b);
        Ok(())
    }

    /// `ISWAP†`.
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn iswap_dag(&mut self, a: usize, b: usize) -> Result<(), SimError> {
        self.cz(a, b)?;
        self.s_dag(a);
        self.s_dag(b);
        self.swap(a, b);
        Ok(())
    }

    /// `C_XYZ`: the order-three Pauli cycle `X → Y → Z → X`.
    pub fn c_xyz(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_c_xyz(&mut self.core.r, q);
    }

    /// `C_ZYX`: the inverse cycle `X → Z → Y → X`.
    pub fn c_zyx(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_c_zyx(&mut self.core.r, q);
    }

    /// `H_XY`: the Hadamard-like exchange of `X` and `Y`.
    pub fn h_xy(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_h_xy(&mut self.core.r, q);
    }

    /// `H_YZ`: the Hadamard-like exchange of `Y` and `Z`.
    pub fn h_yz(&mut self, q: usize) {
        self.ensure_qubits(q + 1);
        frames::left_h_yz(&mut self.core.r, q);
    }

    // --- The `<A>C<B>` family ---
    //
    // Nine one-liners over `gate2`, plus the two aliases stim also spells
    // without a control letter. They exist so a stim circuit transliterates
    // without the reader having to translate `zcy` into an axis pair; the work
    // is `gate2`'s, which reaches `CZ` conjugated by basis rotations.

    /// CNOT — stim's spelling of [`cx`](Self::cx).
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        self.cx(control, target)
    }

    /// Controlled-`Y`, i.e. [`zcy`](Self::zcy).
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn cy(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        self.zcy(control, target)
    }

    /// `XCX`: apply `X` to `target` when `control` is in the `−1` eigenstate
    /// of `X`.
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn xcx(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        self.apply_two_qubit(control, target, frames::left_xcx)
    }

    /// `XCY`.
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn xcy(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        self.apply_two_qubit(control, target, frames::left_xcy)
    }

    /// `XCZ`.
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn xcz(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        self.apply_two_qubit(control, target, frames::left_xcz)
    }

    /// `YCX`.
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn ycx(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        self.apply_two_qubit(control, target, frames::left_ycx)
    }

    /// `YCY`.
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn ycy(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        self.apply_two_qubit(control, target, frames::left_ycy)
    }

    /// `YCZ`.
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn ycz(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        self.apply_two_qubit(control, target, frames::left_ycz)
    }

    /// `ZCX`, i.e. [`cx`](Self::cx).
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn zcx(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        self.cx(control, target)
    }

    /// `ZCY`, i.e. [`cy`](Self::cy).
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn zcy(&mut self, control: usize, target: usize) -> Result<(), SimError> {
        self.apply_two_qubit(control, target, frames::left_cy)
    }

    /// `ZCZ`, i.e. [`cz`](Self::cz).
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if both operands name the same qubit.
    pub fn zcz(&mut self, a: usize, b: usize) -> Result<(), SimError> {
        self.cz(a, b)
    }

    /// Apply a Pauli `P` to the state (`R → P·R`).
    pub fn pauli(&mut self, p: &PauliString) {
        self.ensure_for(p);
        let p = pauli_on_register(p, self.core.n).expect("register covers Pauli support");
        frames::left_pauli(&mut self.core.r, &p);
    }

    /// Apply `control`-conditioned `target` (both Paulis).
    ///
    /// # Errors
    /// [`SimError::NonCommutingControlledPaulis`] if the axes anticommute.
    pub fn controlled_pauli(
        &mut self,
        control: &PauliString,
        target: &PauliString,
    ) -> Result<(), SimError> {
        let nqubits = [max_support(control), max_support(target)]
            .into_iter()
            .flatten()
            .max()
            .map_or(self.core.n, |q| self.core.n.max(q + 1));
        let control = pauli_on_register(control, nqubits)?;
        let target = pauli_on_register(target, nqubits)?;
        if pauli_anticommutes(&control, &target) {
            return Err(SimError::NonCommutingControlledPaulis);
        }
        if measurement_phase_sign(&control).ok() != Some(false)
            || measurement_phase_sign(&target).ok() != Some(false)
        {
            return Err(SimError::InvalidControlledPauli);
        }
        self.ensure_qubits(nqubits);
        frames::left_controlled_pauli(&mut self.core.r, &control, &target);
        Ok(())
    }

    fn apply_two_qubit(
        &mut self,
        a: usize,
        b: usize,
        gate: fn(&mut CliffordFrame, usize, usize),
    ) -> Result<(), SimError> {
        if a == b {
            return Err(SimError::RepeatedQubit(a));
        }
        self.ensure_qubits(a.max(b) + 1);
        gate(&mut self.core.r, a, b);
        Ok(())
    }

    // --- T gate ---

    /// Apply `T_P(±) = cos(π/8)·I ∓ i·sin(π/8)·P` about the Pauli axis `axis`
    /// (`adjoint = true` selects the `+` sign, i.e. `T†`).
    ///
    /// The register grows to cover the axis support. Axes with an imaginary
    /// coefficient are rejected.
    ///
    /// # Errors
    ///
    /// [`SimError::RankOverflow`] if the term count exceeds the cap, or
    /// [`SimError::EmptyStateAfterPruning`] if pruning erases every term.
    ///
    /// # Examples
    ///
    /// ```
    /// use ticit::{TableauSimulator, pauli_string};
    ///
    /// // T about Z on |+⟩ leaves ⟨X⟩ = ⟨Y⟩ = 1/√2.
    /// let mut sim = TableauSimulator::with_seed(1, 0);
    /// sim.h(0);
    /// sim.t_pauli(&pauli_string("Z")?, false)?;
    /// let x = sim.peek_observable_expectation(&pauli_string("X")?)?;
    /// assert!((x - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn t_pauli(&mut self, axis: &PauliString, adjoint: bool) -> Result<(), SimError> {
        measurement_phase_sign(axis).map_err(|_| SimError::NonHermitianPauli)?;
        self.ensure_for(axis);
        with_terms!(self, |core, terms| {
            let d = core.decompose(axis)?;
            terms.t_decomposed(core, &d, adjoint)
        })
    }

    /// `T = T_Z` on qubit `q`.
    ///
    /// # Errors
    /// Propagates [`t_pauli`](Self::t_pauli) errors.
    pub fn t(&mut self, q: usize) -> Result<(), SimError> {
        self.t_about(Axis::Z, q, false)
    }

    /// `T† = T_Z†` on qubit `q`.
    ///
    /// # Errors
    /// Propagates [`t_pauli`](Self::t_pauli) errors.
    pub fn t_dag(&mut self, q: usize) -> Result<(), SimError> {
        self.t_about(Axis::Z, q, true)
    }

    /// `T_axis(±)` on a single qubit, decomposing the basis axis straight out
    /// of the frame instead of building a `PauliString` to describe it.
    fn t_about(&mut self, axis: Axis, q: usize, adjoint: bool) -> Result<(), SimError> {
        self.ensure_qubits(q + 1);
        with_terms!(self, |core, terms| {
            let d = core.decompose_basis(axis, q);
            terms.t_decomposed(core, &d, adjoint)
        })
    }

    /// Apply a `CCZ` on qubits `a`, `b`, `c` (symmetric) via seven `π/8`
    /// rotations. `CCZ` has the phase-polynomial decomposition
    /// `exp(iπ/8·[−Z_a − Z_b − Z_c + Z_aZ_b + Z_aZ_c + Z_bZ_c − Z_aZ_bZ_c])`
    /// (global phase `e^{iπ/8}` unobservable):
    /// `4abc = a + b + c − (a⊕b) − (a⊕c) −
    /// (b⊕c) + (a⊕b⊕c)`, i.e. 7 `T`s + 6 `CX`s. In the stabilizer frame the
    /// `CX` conjugations only retarget the `T` axes, so we apply the seven
    /// rotations directly on multi-qubit `Z` axes — no `CX`s needed:
    ///
    /// * `T` on `Z_a`, `Z_b`, `Z_c` (singles, `−` sign → non-adjoint),
    /// * `T†` on `Z_aZ_b`, `Z_aZ_c`, `Z_bZ_c` (pairs, `+` sign → adjoint),
    /// * `T` on `Z_aZ_bZ_c` (triple, `−` sign → non-adjoint).
    ///
    /// # Errors
    /// [`SimError::RepeatedQubit`] if `a`, `b`, `c` are not distinct;
    /// propagates [`t_pauli`](Self::t_pauli) errors (e.g. rank overflow — the
    /// seven axes lift the rank by at most `2^7` transiently before pruning).
    /// Errors leave the simulator unchanged.
    pub fn ccz(&mut self, a: usize, b: usize, c: usize) -> Result<(), SimError> {
        if a == b || a == c {
            return Err(SimError::RepeatedQubit(a));
        }
        if b == c {
            return Err(SimError::RepeatedQubit(b));
        }
        // Rolling a failure back by discarding a clone costs a full copy of the
        // frame and the amplitude map, so skip it when neither failure is
        // reachable. Seven `T`s at most double the rank each, so `2^7` of
        // headroom rules out [`SimError::RankOverflow`]; and at the default
        // pruning threshold nothing can be emptied either — a normalized state
        // of rank ≤ 2^20 has an amplitude of modulus ≥ 2^-10, and each `T`
        // preserves its coset pair's norm, so after seven of them the largest
        // surviving amplitude is still ≥ 2^-13.5 ≈ 9e-5, eight orders of
        // magnitude clear of `1e-12`.
        let safe = self.rank() <= self.core.rank_cap >> 7
            && self.core.prune_epsilon <= DEFAULT_PRUNE_EPSILON;
        if safe {
            return self.ccz_rotations(a, b, c);
        }
        let mut next = self.clone();
        next.ccz_rotations(a, b, c)?;
        *self = next;
        Ok(())
    }

    /// The seven `π/8` rotations of a `CCZ`, applied in place.
    ///
    /// The operands are distinct, so each axis is just `Z` on a subset of
    /// `{a, b, c}` — no Pauli product, and no phase to track. One buffer is
    /// rewritten between rotations rather than seven strings allocated: a
    /// `PauliString` is dense, so building each axis from scratch would be a
    /// larger share of a `CCZ` than the rotations themselves.
    fn ccz_rotations(&mut self, a: usize, b: usize, c: usize) -> Result<(), SimError> {
        self.ensure_qubits(a.max(b).max(c) + 1);
        let mut axis = PauliString::new(self.core.n);
        for (operands, adjoint) in [
            (&[a][..], false),
            (&[b][..], false),
            (&[c][..], false),
            (&[a, b][..], true),
            (&[a, c][..], true),
            (&[b, c][..], true),
            (&[a, b, c][..], false),
        ] {
            axis.z.fill(0);
            for &site in operands {
                axis.set_zbit(site, true);
            }
            self.t_pauli(&axis, adjoint)?;
        }
        Ok(())
    }

    // --- Measurement ---

    /// Measure qubit `q` in the `Z` basis, sampling the outcome and collapsing
    /// the state onto it. The register grows to cover `q`.
    ///
    /// The single-qubit case never builds an observable: `Z_q`'s preimage is a
    /// stored frame row, so this decomposes it directly. Use
    /// [`measure_observable`](Self::measure_observable) for a Pauli product.
    ///
    /// # Errors
    /// [`SimError::RankOverflow`] if the projection exceeds the rank cap.
    ///
    /// # Examples
    ///
    /// ```
    /// use ticit::TableauSimulator;
    ///
    /// let mut sim = TableauSimulator::with_seed(2, 7);
    /// sim.h(0);
    /// sim.cx(0, 1)?;
    /// let first = sim.measure(0)?;
    /// let second = sim.measure(1)?;
    /// assert_eq!(first.outcome, second.outcome, "a Bell pair agrees");
    /// assert!(second.deterministic, "the partner is pinned by the first read");
    /// # Ok::<(), ticit::SimError>(())
    /// ```
    pub fn measure(&mut self, q: usize) -> Result<MeasureResult, SimError> {
        self.measure_axis(Axis::Z, q, None)
    }

    /// Measure the Pauli observable `observable`, sampling the outcome. The
    /// register grows to cover its support.
    ///
    /// Observables with an imaginary coefficient are rejected.
    ///
    /// # Errors
    /// [`SimError::RankOverflow`] if the projection exceeds the rank cap.
    ///
    /// # Examples
    ///
    /// ```
    /// use ticit::{TableauSimulator, pauli_string};
    ///
    /// let mut sim = TableauSimulator::with_seed(2, 1);
    /// sim.h(0);
    /// sim.cx(0, 1)?;
    /// // `ZZ` stabilizes a Bell pair, so reading it disturbs nothing.
    /// let result = sim.measure_observable(&pauli_string("ZZ")?)?;
    /// assert!(result.deterministic && !result.outcome);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn measure_observable(
        &mut self,
        observable: &PauliString,
    ) -> Result<MeasureResult, SimError> {
        measurement_phase_sign(observable).map_err(|_| SimError::NonHermitianPauli)?;
        self.ensure_for(observable);
        with_terms!(self, |core, terms| {
            let d = core.decompose(observable)?;
            terms.measure_decomposed(core, &d, None)
        })
    }

    /// Force `observable`'s measurement to `desired_value` (`false` = `+1`)
    /// instead of sampling it, projecting the state onto that eigenspace.
    ///
    /// The returned [`MeasureResult::probability`] is the weight the branch had
    /// before the projection — what a caller weights the post-selected shot by.
    ///
    /// # Errors
    /// [`SimError::PostselectImpossible`] if `desired_value` has probability
    /// `≈ 0`, plus the
    /// [`measure_observable`](Self::measure_observable) errors.
    pub fn postselect_observable(
        &mut self,
        observable: &PauliString,
        desired_value: bool,
    ) -> Result<MeasureResult, SimError> {
        measurement_phase_sign(observable).map_err(|_| SimError::NonHermitianPauli)?;
        self.ensure_for(observable);
        with_terms!(self, |core, terms| {
            let d = core.decompose(observable)?;
            terms.measure_decomposed(core, &d, Some(desired_value))
        })
    }

    /// Force qubit `q`'s `Z` measurement to `desired_value`: `false` selects
    /// `|0⟩`, `true` selects `|1⟩`.
    ///
    /// # Errors
    /// As [`postselect_observable`](Self::postselect_observable).
    pub fn postselect_z(
        &mut self,
        q: usize,
        desired_value: bool,
    ) -> Result<MeasureResult, SimError> {
        self.measure_axis(Axis::Z, q, Some(desired_value))
    }

    /// Force qubit `q`'s `X` measurement: `false` selects `|+⟩`, `true` `|−⟩`.
    ///
    /// # Errors
    /// As [`postselect_observable`](Self::postselect_observable).
    pub fn postselect_x(
        &mut self,
        q: usize,
        desired_value: bool,
    ) -> Result<MeasureResult, SimError> {
        self.measure_axis(Axis::X, q, Some(desired_value))
    }

    /// Force qubit `q`'s `Y` measurement: `false` selects `|i⟩`, `true` `|−i⟩`.
    ///
    /// # Errors
    /// As [`postselect_observable`](Self::postselect_observable).
    pub fn postselect_y(
        &mut self,
        q: usize,
        desired_value: bool,
    ) -> Result<MeasureResult, SimError> {
        self.measure_axis(Axis::Y, q, Some(desired_value))
    }

    /// Measure a single-qubit basis axis, sampling or forcing the outcome.
    ///
    /// The one body behind `measure`, the three `postselect_*` wrappers and the
    /// three resets: all six decompose a stored frame row rather than a Pauli.
    fn measure_axis(
        &mut self,
        axis: Axis,
        q: usize,
        forced: Option<bool>,
    ) -> Result<MeasureResult, SimError> {
        self.ensure_qubits(q + 1);
        with_terms!(self, |core, terms| {
            let d = core.decompose_basis(axis, q);
            terms.measure_decomposed(core, &d, forced)
        })
    }

    // --- Non-collapsing reads ---

    /// Non-collapsing expectation value `⟨P⟩ ∈ [−1, 1]` of `observable`,
    /// leaving both the state and the RNG untouched.
    ///
    /// This is the same `⟨Q⟩` the measurement path derives its outcome
    /// probability from (`p₊ = (1 + ⟨Q⟩)/2`), read out without projecting. On
    /// an eigenstate it is exactly `±1`; off an eigenstate it is the true
    /// expectation (e.g. `⟨X⟩ = ⟨Y⟩ = 1/√2` on `T|+⟩`).
    ///
    /// `observable`'s support must lie within the allocated qubits — being
    /// `&self`, this cannot grow the register.
    ///
    /// # Errors
    /// [`SimError::QubitIndexOutOfRange`] if the support exceeds the live
    /// register.
    ///
    /// # Examples
    ///
    /// ```
    /// use ticit::{TableauSimulator, pauli_string};
    ///
    /// let mut sim = TableauSimulator::with_seed(2, 0);
    /// sim.h(0);
    /// sim.cx(0, 1)?;
    /// let xx = pauli_string("XX")?;
    /// assert!((sim.peek_observable_expectation(&xx)? - 1.0).abs() < 1e-9);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn peek_observable_expectation(&self, observable: &PauliString) -> Result<f64, SimError> {
        with_terms_ref!(self, |core, terms| {
            let d = core.decompose(observable)?;
            Ok(terms.expectation_of(&d).clamp(-1.0, 1.0))
        })
    }

    /// `⟨Z_q⟩`, without measuring.
    ///
    /// # Errors
    /// [`SimError::QubitIndexOutOfRange`] if `q` is outside the live register.
    pub fn peek_z(&self, q: usize) -> Result<f64, SimError> {
        self.peek_axis(Axis::Z, q)
    }

    /// `⟨X_q⟩`, without measuring.
    ///
    /// # Errors
    /// [`SimError::QubitIndexOutOfRange`] if `q` is outside the live register.
    pub fn peek_x(&self, q: usize) -> Result<f64, SimError> {
        self.peek_axis(Axis::X, q)
    }

    /// `⟨Y_q⟩`, without measuring.
    ///
    /// # Errors
    /// [`SimError::QubitIndexOutOfRange`] if `q` is outside the live register.
    pub fn peek_y(&self, q: usize) -> Result<f64, SimError> {
        self.peek_axis(Axis::Y, q)
    }

    /// [`peek_observable_expectation`](Self::peek_observable_expectation) for a
    /// single-qubit basis axis. The range check is explicit because
    /// `decompose_basis` is infallible and asserts instead.
    fn peek_axis(&self, axis: Axis, q: usize) -> Result<f64, SimError> {
        if q >= self.core.n {
            return Err(SimError::QubitIndexOutOfRange {
                index: q,
                num_qubits: self.core.n,
            });
        }
        with_terms_ref!(self, |core, terms| {
            let d = core.decompose_basis(axis, q);
            Ok(terms.expectation_of(&d).clamp(-1.0, 1.0))
        })
    }

    // --- Reset ---

    /// Reset qubit `q` to `|0⟩` — stim's spelling of
    /// [`reset_z`](Self::reset_z).
    ///
    /// # Errors
    /// Propagates measurement errors.
    pub fn reset(&mut self, q: usize) -> Result<(), SimError> {
        self.reset_z(q)
    }

    /// Reset qubit `q` to `|0⟩`: measure `Z_q`, then apply `X_q` if the outcome
    /// was `−1`.
    ///
    /// # Errors
    /// Propagates measurement errors.
    pub fn reset_z(&mut self, q: usize) -> Result<(), SimError> {
        self.reset_about(Axis::Z, q, Axis::X)
    }

    /// Reset qubit `q` to `|+⟩`: measure `X_q`, correct with `Z_q` if `−1`.
    ///
    /// # Errors
    /// Propagates measurement errors.
    pub fn reset_x(&mut self, q: usize) -> Result<(), SimError> {
        self.reset_about(Axis::X, q, Axis::Z)
    }

    /// Reset qubit `q` to `|+i⟩`: measure `Y_q`, correct with `Z_q` if `−1`.
    ///
    /// # Errors
    /// Propagates measurement errors.
    pub fn reset_y(&mut self, q: usize) -> Result<(), SimError> {
        self.reset_about(Axis::Y, q, Axis::Z)
    }

    /// Measure a basis axis and, on a `−1` outcome, apply the Pauli that maps
    /// that eigenstate onto the `+1` one. One dispatch site for all three
    /// resets, which is also three fewer copies of the measurement body.
    fn reset_about(&mut self, axis: Axis, q: usize, correction: Axis) -> Result<(), SimError> {
        if self.measure_axis(axis, q, None)?.outcome {
            match correction {
                Axis::X => frames::left_x(&mut self.core.r, q),
                Axis::Y => frames::left_y(&mut self.core.r, q),
                Axis::Z => frames::left_z(&mut self.core.r, q),
            }
        }
        Ok(())
    }

    // --- Test/inspection hook ---

    /// Reconstruct the dense state vector `|ψ⟩` (length `2^n`).
    ///
    /// Replays the frame `R` (via its `image_x`/`image_z` images) against the
    /// amplitude map. This is `O(rank · n · 2^n)` and intended for testing and
    /// small `n` only.
    ///
    /// # Panics
    ///
    /// Panics unless `n < 64`, where the length stops fitting a `usize`. The
    /// check is an assertion rather than an error because the bound it enforces
    /// is nowhere near the one that matters: `2^n` amplitudes are 16 GiB by
    /// `n = 30`, so every caller that gets an answer at all is far below it.
    /// Without the check the shift wraps, and a release build hands back a
    /// vector of the wrong length instead of failing.
    #[must_use]
    pub fn state_vector(&self) -> Vec<Complex64> {
        let n = self.core.n;
        assert!(
            n < usize::BITS as usize,
            "state-vector reconstruction needs a 2^{n} length that fits a usize"
        );
        let dim = 1usize << n;

        // |ψ0⟩ is the stabilizer state with stabilizers S_i = image_z(i). Build
        // it by projecting a computational-basis fiducial through ∏(I+S_i)/2 =
        // |ψ0⟩⟨ψ0|; any fiducial with nonzero overlap works (global phase is
        // irrelevant to the up-to-phase comparison callers make).
        let stabs: Vec<PauliString> = (0..n)
            .map(|i| {
                coordinates_in_frame(&self.core.r, &crate::pauli::pauli_z(n, i))
                    .expect("valid Clifford frame")
            })
            .collect();
        let mut psi0 = vec![Complex64::new(0.0, 0.0); dim];
        for fiducial in 0..dim {
            let mut v = vec![Complex64::new(0.0, 0.0); dim];
            v[fiducial] = Complex64::new(1.0, 0.0);
            for s in &stabs {
                let projected = apply_pauli_dense(&v, s);
                for (dst, add) in v.iter_mut().zip(projected) {
                    *dst = (*dst + add) * 0.5;
                }
            }
            let norm: f64 = v.iter().map(num_complex::Complex::norm_sqr).sum();
            if norm > TOL {
                let scale = norm.sqrt().recip();
                for (dst, src) in psi0.iter_mut().zip(v) {
                    *dst = src * scale;
                }
                break;
            }
        }

        // |ψ⟩ = Σ_c xvec[c] · D^c|ψ0⟩, D_i = image_x(i), applied per set bit.
        let destabs: Vec<PauliString> = (0..n)
            .map(|i| {
                coordinates_in_frame(&self.core.r, &crate::pauli::pauli_x(n, i))
                    .expect("valid Clifford frame")
            })
            .collect();
        let mut out = vec![Complex64::new(0.0, 0.0); dim];
        match &self.amps {
            Amps::W1(t) => replay_terms(&t.map, &psi0, &destabs, &mut out),
            Amps::W2(t) => replay_terms(&t.map, &psi0, &destabs, &mut out),
            Amps::W4(t) => replay_terms(&t.map, &psi0, &destabs, &mut out),
            Amps::W8(t) => replay_terms(&t.map, &psi0, &destabs, &mut out),
            Amps::Wide(t) => replay_terms(&t.map, &psi0, &destabs, &mut out),
        }
        out
    }
}

// ==============================================================================
// Frame decomposition and outcome sampling
// ==============================================================================

impl Core {
    /// Decompose Pauli `p` in the frame: `Q = R†PR = ζ X^a Z^b`.
    ///
    /// The frame writes the two masks straight into the label words, so the
    /// only cost beyond the row product is the two (inline) labels.
    fn decompose<K: LabelKey>(&self, p: &PauliString) -> Result<Decomp<K>, SimError> {
        measurement_phase_sign(p).map_err(|_| SimError::NonHermitianPauli)?;
        let p = pauli_on_register(p, self.n)?;
        let transformed = preimage(&self.r, &p);
        let mut a = K::zeros(self.words);
        let mut b = K::zeros(self.words);
        a.as_mut_slice()[..transformed.x.len()].copy_from_slice(&transformed.x);
        b.as_mut_slice()[..transformed.z.len()].copy_from_slice(&transformed.z);
        let phase = transformed.phase_exponent() as u8;
        Ok(Decomp {
            a,
            b,
            phase,
            zeta: i_pow(phase),
        })
    }

    /// [`decompose`](Self::decompose) for a single-qubit basis axis. `T`, the
    /// single-qubit measurements, the post-selections and the resets all go
    /// through here, which is what keeps them off the `PauliString` path: a
    /// basis preimage is a stored row.
    ///
    /// The caller must have grown the register past `qubit`; a basis axis
    /// cannot be non-Hermitian or out of range, so this is infallible.
    fn decompose_basis<K: LabelKey>(&self, axis: Axis, qubit: usize) -> Decomp<K> {
        let pauli = match axis {
            Axis::X => crate::pauli::pauli_x(self.n, qubit),
            Axis::Y => crate::pauli::pauli_y(self.n, qubit),
            Axis::Z => crate::pauli::pauli_z(self.n, qubit),
        };
        self.decompose(&pauli)
            .expect("single-qubit Pauli is Hermitian and in range")
    }

    /// Sample or force an outcome. `p0` is the probability of the `+1` (`false`)
    /// outcome; the RNG draw matches `SOFT`'s `rand() >= p0 ? 1 : 0` (`>=` keeps
    /// `p0 = 0` deterministic — a `+1` of probability zero never samples).
    fn choose(&mut self, forced: Option<bool>, p0: f64) -> bool {
        match forced {
            Some(o) => o,
            None => {
                let r = rand_float(&mut self.rng);
                r >= p0
            }
        }
    }

    /// Reject a projected label count that exceeds the cap. Always called
    /// before the state is touched, so the failure is transactional.
    fn check_cap(&self, rank: usize) -> Result<(), SimError> {
        if rank > self.rank_cap {
            return Err(SimError::RankOverflow {
                rank,
                cap: self.rank_cap,
            });
        }
        Ok(())
    }
}

// ==============================================================================
// Amplitude storage — width selection
// ==============================================================================

impl Amps {
    /// The single-term map of a fresh `|0…0⟩` state.
    fn unit(words: usize) -> Self {
        match Width::for_words(words) {
            Width::W1 => Amps::W1(Terms::unit(words)),
            Width::W2 => Amps::W2(Terms::unit(words)),
            Width::W4 => Amps::W4(Terms::unit(words)),
            Width::W8 => Amps::W8(Terms::unit(words)),
            Width::Wide => Amps::Wide(Terms::unit(words)),
        }
    }

    fn width(&self) -> Width {
        match self {
            Amps::W1(_) => Width::W1,
            Amps::W2(_) => Width::W2,
            Amps::W4(_) => Width::W4,
            Amps::W8(_) => Width::W8,
            Amps::Wide(_) => Width::Wide,
        }
    }

    fn len(&self) -> usize {
        match self {
            Amps::W1(t) => t.map.len(),
            Amps::W2(t) => t.map.len(),
            Amps::W4(t) => t.map.len(),
            Amps::W8(t) => t.map.len(),
            Amps::Wide(t) => t.map.len(),
        }
    }

    /// Re-key the live terms for a register that has grown to `words` words.
    ///
    /// Inside one width class there is nothing to do: a fixed-width key's
    /// surplus words were already zero and stay zero, so the existing keys are
    /// still valid at the wider register. Only crossing into a wider class — or
    /// widening a runtime-width label, whose length *is* its width — changes the
    /// key, and then the map is rebuilt. Cold either way: growth only happens
    /// when an operation names a qubit past the current register.
    fn widen(&mut self, words: usize) {
        let target = Width::for_words(words);
        if target == self.width() && target != Width::Wide {
            return;
        }
        let live = self.drain_terms();
        *self = match target {
            Width::W1 => Amps::W1(Terms::rekeyed(live, words)),
            Width::W2 => Amps::W2(Terms::rekeyed(live, words)),
            Width::W4 => Amps::W4(Terms::rekeyed(live, words)),
            Width::W8 => Amps::W8(Terms::rekeyed(live, words)),
            Width::Wide => Amps::Wide(Terms::rekeyed(live, words)),
        };
    }

    /// Every live term as raw words plus its amplitude.
    fn drain_terms(&mut self) -> Vec<(Vec<u64>, Complex64)> {
        fn collect<K: LabelKey>(map: &mut HashMap<K, Complex64>) -> Vec<(Vec<u64>, Complex64)> {
            map.drain()
                .map(|(key, value)| (key.as_slice().to_vec(), value))
                .collect()
        }
        match self {
            Amps::W1(t) => collect(&mut t.map),
            Amps::W2(t) => collect(&mut t.map),
            Amps::W4(t) => collect(&mut t.map),
            Amps::W8(t) => collect(&mut t.map),
            Amps::Wide(t) => collect(&mut t.map),
        }
    }
}

// ==============================================================================
// The amplitude engine, monomorphized per label width
// ==============================================================================

impl<K: LabelKey> Terms<K> {
    /// The single-term map of a fresh `|0…0⟩` state.
    fn unit(words: usize) -> Self {
        let mut map = HashMap::new();
        map.insert(K::zeros(words), Complex64::new(1.0, 0.0));
        Terms {
            map,
            rotation: RotationScratch::default(),
        }
    }

    /// Adopt terms drained from a narrower width.
    fn rekeyed(live: Vec<(Vec<u64>, Complex64)>, words: usize) -> Self {
        Terms {
            map: live
                .into_iter()
                .map(|(bits, value)| (K::from_words(&bits, words), value))
                .collect(),
            rotation: RotationScratch::default(),
        }
    }

    // --- T ---

    /// `T_P(±)` about an axis already decomposed in the frame — the one place
    /// the rotation lives, shared by [`TableauSimulator::t_pauli`] and the basis-axis
    /// entry points that never build a `PauliString`.
    fn t_decomposed(&mut self, core: &Core, d: &Decomp<K>, adjoint: bool) -> Result<(), SimError> {
        self.t_decomposed_inner(core, d, adjoint)
    }

    #[inline(always)]
    fn t_decomposed_inner(
        &mut self,
        core: &Core,
        d: &Decomp<K>,
        adjoint: bool,
    ) -> Result<(), SimError> {
        let cos = (PI / 8.0).cos();
        let sin = (PI / 8.0).sin();
        // Non-adjoint uses `−i·sin`; adjoint flips to `+i·sin`.
        let branch = Complex64::new(0.0, if adjoint { sin } else { -sin });

        if d.a.is_zero() {
            return self.t_diagonal(core, d, cos, branch);
        }
        // The staging buffers live on `self`, so they have to be moved out for
        // the duration; every exit path hands them back, cleared but with their
        // capacity intact.
        let mut scratch = std::mem::take(&mut self.rotation);
        let result = self.t_paired(core, d, cos, branch, &mut scratch);
        scratch.clear();
        self.rotation = scratch;
        result
    }

    /// `T` about a frame-diagonal axis (`a = 0`), where both branches land on
    /// the same label and the rotation collapses to a per-label factor
    /// `cos ∓ i·sin·ζ·(−1)^{⟨b,c⟩}`.
    ///
    /// `ζ` is real (`±1`) here — a Hermitian `Q = ζ Z^b` forces an even phase
    /// exponent — so every factor is `cos ± i·sin`, of modulus one. Labels,
    /// individual moduli and the total norm therefore all survive untouched,
    /// which is what lets this path skip the map rebuild, the pruning and the
    /// renormalization that the general case needs.
    fn t_diagonal(
        &mut self,
        core: &Core,
        d: &Decomp<K>,
        cos: f64,
        branch: Complex64,
    ) -> Result<(), SimError> {
        debug_assert!(
            d.zeta.im.abs() < TOL,
            "diagonal T axis has a non-real phase"
        );
        // The rank cannot grow here, but a cap lowered under the live rank must
        // still be reported, and reported before anything is touched.
        core.check_cap(self.map.len())?;
        let plus = Complex64::new(cos, branch.im * d.zeta.re);
        let minus = plus.conj();
        for (c, value) in &mut self.map {
            *value *= if c.dot_parity(&d.b) { minus } else { plus };
        }
        Ok(())
    }

    /// `T` about an off-diagonal axis (`a ≠ 0`). Labels split into cosets
    /// `{c, c ⊕ a}` that the rotation mixes only with themselves, so every live
    /// label keeps its slot: its new amplitude is written back in place and the
    /// map only grows by the coset partners it was missing.
    fn t_paired(
        &mut self,
        core: &Core,
        d: &Decomp<K>,
        cos: f64,
        branch: Complex64,
        scratch: &mut RotationScratch<K>,
    ) -> Result<(), SimError> {
        let (removals, norm) = self.stage_t_rotation(core, d, cos, branch, scratch);
        // `T_P` is unitary, so each pair's norm is preserved exactly and only a
        // pruned amplitude can cost the state its normalization.
        self.commit_pair_rewrite(core, scratch, removals, norm)
    }

    /// Install a staged coset-pair rewrite, rescaled to unit norm by `norm`.
    ///
    /// Both operations that mix a label with its coset partner — the `T`
    /// rotation and the random-measurement projection — recompute every live
    /// label's amplitude from its own and its partner's, so both stage
    /// positionally into `scratch` and commit the same way. `removals` is how
    /// many staged amplitudes pruned away to nothing, and `norm` the `Σ|x|²`
    /// the survivors carry — `None` where the operation is norm-preserving and
    /// nothing was pruned, so there is nothing to rescale.
    ///
    /// One sweep does all three of writing back, pruning and rescaling.
    /// Rescaling used to be its own pass (measure the norm, then divide),
    /// which is redundant once the staging pass hands its own sum over: a
    /// projection is precisely the operation whose norm cannot be assumed, and
    /// it is also the one that has just computed every surviving modulus.
    ///
    /// The rank is vetted before a single slot is written, which is what makes
    /// a [`SimError::RankOverflow`] transactional: the caller can still be
    /// holding a frame update it has not applied yet.
    fn commit_pair_rewrite(
        &mut self,
        core: &Core,
        scratch: &mut RotationScratch<K>,
        removals: usize,
        norm: Option<f64>,
    ) -> Result<(), SimError> {
        let rank = self.map.len() + scratch.inserts.len() - removals;
        if rank == 0 {
            return Err(SimError::EmptyStateAfterPruning {
                epsilon: core.prune_epsilon,
            });
        }
        core.check_cap(rank)?;

        let scale = match norm {
            Some(total) if total > 0.0 => total.sqrt().recip(),
            _ => 1.0,
        };
        debug_assert_eq!(scratch.values.len(), self.map.len());
        if removals == 0 {
            // Nothing has touched the map since the staging pass, so
            // `values_mut` walks the same slots in the same order the staging
            // `iter` did.
            for (slot, &value) in self.map.values_mut().zip(&scratch.values) {
                *slot = value * scale;
            }
        } else {
            // `retain` walks that same order, and hashbrown erases in place
            // without moving a survivor, so the writeback and the prune are
            // one sweep. The keep test is spelled against the *staged*
            // modulus, not the rescaled one it just stored, so it agrees with
            // `removals` bit for bit rather than merely algebraically.
            let eps_sq = core.prune_epsilon.powi(2);
            let mut staged = scratch.values.iter();
            self.map.retain(|_, slot| {
                let &value = staged
                    .next()
                    .expect("one staged amplitude per live label, in map order");
                *slot = value * scale;
                value.norm_sqr() > eps_sq
            });
        }
        self.map.reserve(scratch.inserts.len());
        for (label, value) in scratch.inserts.drain(..) {
            self.map.insert(label, value * scale);
        }
        // A projection routinely halves the rank, and the table it collapsed
        // out of would otherwise be walked in full by every later pass.
        if removals > 0 {
            shrink_if_sparse(&mut self.map, rank);
        }
        Ok(())
    }

    /// Compute every live label's post-`T` amplitude and the partners the
    /// rotation adds. Returns the number of live labels that prune away to
    /// nothing, and the norm to rescale by — `None` unless something was
    /// pruned, since the rotation is otherwise exactly norm-preserving (a
    /// partner dropped below the threshold costs norm without removing a
    /// label, so the two are not the same condition).
    ///
    /// Read-only, so the caller can vet the resulting rank before committing
    /// any of it. The staged amplitudes are positional — the applying pass
    /// relies on them lining up with the amplitude map's iteration order.
    fn stage_t_rotation(
        &self,
        core: &Core,
        d: &Decomp<K>,
        cos: f64,
        branch: Complex64,
        scratch: &mut RotationScratch<K>,
    ) -> (usize, Option<f64>) {
        // On the pair `{c, p = c ⊕ a}` the rotation acts as
        //
        //   new_c = cos·x_c + g·sign_p·x_p,   g = ∓i·sin·ζ,  sign_w = (−1)^{⟨b,w⟩},
        //
        // so a live label's new amplitude needs nothing but its partner's old
        // one. The two signs differ by the constant `(−1)^{⟨b,a⟩}`, which saves
        // a parity per term. A missing partner counts as `x_p = 0`, and the
        // pair then gains it at `g·sign_c·x_c`.
        let g = branch * d.zeta;
        let flip = if d.a.dot_parity(&d.b) { -1.0 } else { 1.0 };
        let eps_sq = core.prune_epsilon.powi(2);
        let mut removals = 0;
        let mut pruned = false;
        let mut norm = 0.0;

        scratch.values.clear();
        scratch.inserts.clear();
        scratch.values.reserve(self.map.len());
        for (c, &x) in &self.map {
            let sign_c = if c.dot_parity(&d.b) { -1.0 } else { 1.0 };
            let partner = c.xor(&d.a);
            let value = match self.map.get(&partner) {
                Some(&y) => cos * x + g * (sign_c * flip) * y,
                None => {
                    let added = g * sign_c * x;
                    let weight = added.norm_sqr();
                    if weight > eps_sq {
                        scratch.inserts.push((partner, added));
                        norm += weight;
                    } else {
                        pruned = true;
                    }
                    cos * x
                }
            };
            let weight = value.norm_sqr();
            if weight <= eps_sq {
                removals += 1;
                pruned = true;
            } else {
                norm += weight;
            }
            scratch.values.push(value);
        }
        (removals, pruned.then_some(norm))
    }

    // --- Measurement ---

    /// Measure an observable already decomposed in the frame. The branch is
    /// decided by `a`: zero means every term is an eigenstate, non-zero means
    /// the outcome is genuinely random. Shared with the reset entry points,
    /// which decompose their basis axis directly.
    fn measure_decomposed(
        &mut self,
        core: &mut Core,
        d: &Decomp<K>,
        forced: Option<bool>,
    ) -> Result<MeasureResult, SimError> {
        self.measure_decomposed_inner(core, d, forced)
    }

    #[inline(always)]
    fn measure_decomposed_inner(
        &mut self,
        core: &mut Core,
        d: &Decomp<K>,
        forced: Option<bool>,
    ) -> Result<MeasureResult, SimError> {
        if d.a.is_zero() {
            self.measure_frame_deterministic(core, d, forced)
        } else {
            self.measure_random(core, d, forced)
        }
    }

    /// Case A: `a = 0`, every term is a `±1` eigenstate. `R` is
    /// unchanged; we split by eigenvalue and keep the winning class.
    fn measure_frame_deterministic(
        &mut self,
        core: &mut Core,
        d: &Decomp<K>,
        forced: Option<bool>,
    ) -> Result<MeasureResult, SimError> {
        // ζ = i^k is real (±1) because a Hermitian diagonal Pauli has even k.
        debug_assert!(
            d.zeta.im.abs() < TOL,
            "diagonal observable has non-real phase"
        );
        let zsign = d.zeta.re;

        // p₊ = (1 + ⟨Q⟩)/2 with ⟨Q⟩ the (real) diagonal expectation shared
        // with [`TableauSimulator::expectation`].
        let p_plus = ((1.0 + self.expectation_of(d)) / 2.0).clamp(0.0, 1.0);
        let deterministic = !(TOL..=1.0 - TOL).contains(&p_plus);

        let outcome = core.choose(forced, p_plus);
        let probability = if outcome { 1.0 - p_plus } else { p_plus };
        if forced.is_some() && probability < TOL {
            return Err(SimError::PostselectImpossible {
                outcome,
                probability,
            });
        }

        // Projection is a filter on the existing labels, so it runs in place.
        // The survivors are counted (and their norm accumulated for the
        // rescale) in a read-only pass first: an empty result is an error, and
        // a failed projection must leave the state exactly as it was.
        let keep_plus = !outcome;
        let eps_sq = core.prune_epsilon.powi(2);
        let keep = |label: &K, amplitude: &Complex64| {
            eig_plus(label, &d.b, zsign) == keep_plus && amplitude.norm_sqr() > eps_sq
        };
        let mut live = 0;
        let mut total = 0.0;
        for (label, amplitude) in &self.map {
            if keep(label, amplitude) {
                live += 1;
                total += amplitude.norm_sqr();
            }
        }
        if live == 0 {
            return Err(SimError::EmptyStateAfterPruning {
                epsilon: core.prune_epsilon,
            });
        }
        core.check_cap(live)?;
        // Measuring an eigenstate — the common case in verification circuits —
        // keeps every label, and then there is nothing to walk the table for.
        if live < self.map.len() {
            self.map.retain(|label, amplitude| keep(label, amplitude));
            shrink_if_sparse(&mut self.map, live);
        }
        self.rescale(total);

        Ok(MeasureResult {
            outcome,
            probability,
            deterministic,
        })
    }

    /// Case B: `a ≠ 0`, a genuinely random outcome. The amplitude map is
    /// projected onto the `Q`-eigenspace and the frame re-compressed:
    ///
    /// ```text
    /// |χ⟩ ← exp(−iπ/4·G) · Z_p^s · Π_s^Q |χ⟩,   R ← R · Z_p^s · exp(iπ/4·G).
    /// ```
    ///
    /// The two rotations are inverses of one another and the two `Z_p^s` sit on
    /// opposite sides of them, so `R|χ⟩` telescopes back to `Π_s^P|ψ⟩` — which
    /// is the whole correctness condition, and the reason both sides must read
    /// `G` from the same place. (`exp(−iπ/4·G) = (I − iG)/√2` is where
    /// [`PAULI_EXP_SIGN`] comes from.)
    ///
    /// # Where `G` comes from
    ///
    /// `pauliverse` updates the frame from the *left*, as
    /// `R ← S_p^s · exp(iπ/4·pa)·R` with `S_p = R Z_p R†` the stabilizer at the
    /// pivot and `pa = −i·P·S_p`. Both factors move to the right of `R`, which
    /// is what lets this path skip `S_p` — an image, i.e. a column gather over
    /// the whole tableau plus a phase solve — altogether:
    ///
    /// ```text
    /// exp(iπ/4·pa)·R = R·exp(iπ/4·G)   with G = R†·pa·R,
    /// S_p·(R·V)      = R·Z_p·V         for any V, since R†S_pR = Z_p exactly.
    /// ```
    ///
    /// Mind the order: `S_p` was applied *after* the rotation on the left, so
    /// `Z_p` lands *before* it on the right. That is not a free choice — the
    /// pivot is a set bit of `a`, so `Z_p` anticommutes with `G` and the two
    /// orders differ by `exp(iπ/2·G) = iG`, which would leave the state off by
    /// a Pauli.
    ///
    /// `G` itself needs no frame work at all. With `Q = R†PR = i^k·X^a Z^b`
    /// already in hand from `d`,
    ///
    /// ```text
    /// G = R†(−i·P·S_p)R = −i·Q·Z_p = i^{k+3}·X^a·Z^{b ⊕ e_p},
    /// ```
    ///
    /// because `Z^b·Z_p = Z^{b⊕e_p}` costs no phase. The `−i` is what makes `G`
    /// Hermitian: `P` anticommutes with `S_p` here, so their product is
    /// anti-Hermitian. `+i` would serve as well — the amplitude side derives
    /// its rotation from the same `G`, so the two sign choices cancel. One
    /// source of truth for `G` keeps the two sides consistent.
    fn measure_random(
        &mut self,
        core: &mut Core,
        d: &Decomp<K>,
        forced: Option<bool>,
    ) -> Result<MeasureResult, SimError> {
        // The staging buffers live on `self`, so they are moved out for the
        // duration and handed back cleared, capacity intact.
        let mut scratch = std::mem::take(&mut self.rotation);
        let result = self.measure_random_staged(core, d, forced, &mut scratch);
        scratch.clear();
        self.rotation = scratch;
        result
    }

    /// [`measure_random`](Self::measure_random) with its staging buffers in
    /// hand, so every exit path hands them back through one place.
    fn measure_random_staged(
        &mut self,
        core: &mut Core,
        d: &Decomp<K>,
        forced: Option<bool>,
        scratch: &mut RotationScratch<K>,
    ) -> Result<MeasureResult, SimError> {
        let pivot = d.a.first_set_bit().expect("random branch has nonzero a");

        // ⟨Q⟩ = Σ_c Re( ζ·(−1)^{⟨b,c⟩}·xvec[c]·conj(xvec[c⊕a]) ), the
        // off-diagonal branch of [`TableauSimulator::expectation`] — run here in the
        // variant that records where it found each pair, because the
        // projection below needs exactly the same pairing and the map does not
        // move between the two passes.
        let p0 = ((1.0 + self.expectation_paired(d, &mut scratch.partners)) / 2.0).clamp(0.0, 1.0);

        let outcome = core.choose(forced, p0);
        let probability = if outcome { 1.0 - p0 } else { p0 };
        if forced.is_some() && probability < TOL {
            return Err(SimError::PostselectImpossible {
                outcome,
                probability,
            });
        }
        let s = outcome;

        // `G = i^{k+3}·X^a·Z^{b⊕e_p}`, derived above. `Z_p` contributes no `X`
        // part, so `G` and `Q` share their coset shift `a` by construction —
        // that is what keeps the compression on the same pair of labels the
        // projection uses, below.
        let g_phase = (d.phase + 3) & 3;
        let gzeta = i_pow(g_phase);
        let mut gb = d.b.clone();
        gb.flip(pivot);

        // Projection `Π_s^Q = ½(I + (−1)^s Q)`, the `Z_p^s` fold and the frame
        // compression `(I ± iG)/√2` all in one pass. As scatters, the first two
        // send an amplitude `x` at label `c` to
        //
        //   y0 = ½·z(c)·x                              at c,
        //   y1 = ½·(−1)^s·ζ·(−1)^{⟨b,c⟩}·z(c⊕a)·x      at c ⊕ a,
        //
        // writing `z(k) = (−1)^{s·k_p}` for `Z_p^s`, and the third sends `y` at
        // `k` to `y/√2` at `k` and `w·(−1)^{⟨gb,k⟩}·y` at `k ⊕ a`, with
        // `w = i·PAULI_EXP_SIGN/√2·ζ_g`. Composing them, all four products land
        // back on `{c, c⊕a}` — which is the whole reason this can run in place,
        // exactly like the `T` rotation: a label's new amplitude is a function
        // of its own and its coset partner's, so it keeps its slot and the map
        // is never rebuilt.
        let projection = Projection {
            d,
            gb: &gb,
            pivot,
            s,
            ssign: if s { -1.0 } else { 1.0 },
            compress: Complex64::new(0.0, PAULI_EXP_SIGN * FRAC_1_SQRT_2) * gzeta,
            shift_flip: if gb.dot_parity(&d.a) { -1.0 } else { 1.0 },
        };
        debug_assert_ne!(
            gb.dot_parity(&d.a),
            d.b.dot_parity(&d.a),
            "the pivot bit of `a` is set, so `gb` and `b` differ in their `a` parity"
        );

        // A projection is not norm-preserving — what it removes is the weight
        // of the eigenspace it rejected — so the staged norm always feeds the
        // commit's rescale.
        let (removals, norm) = self.stage_projection(core, &projection, scratch);
        // Commit the amplitude map before touching `R`: the commit is fallible
        // (`RankOverflow`) and returns without modifying the map on that path,
        // so deferring the `R` update keeps the two atomic — a frame advanced
        // past an un-committed map would be unrecoverable.
        self.commit_pair_rewrite(core, scratch, removals, Some(norm))?;

        // Update R to match the committed map: `R ← R·Z_p^s·exp(iπ/4·G)`, the
        // right-multiplied form of the frame update (see above).
        if s {
            let z = crate::pauli::pauli_z(core.n, pivot);
            frames::right_pauli(&mut core.r, &z);
        }
        let mut generator = PauliString::new(core.n);
        let words = generator.x.len();
        generator.x.copy_from_slice(&d.a.as_slice()[..words]);
        generator.z.copy_from_slice(&gb.as_slice()[..words]);
        generator.set_phase(i32::from(g_phase));
        frames::right_pauli_exp(&mut core.r, &generator);

        // A frame-random observable can still be a state eigenvalue (e.g. X on
        // |+⟩): report determinism from the probability, not the branch.
        let deterministic = !(TOL..=1.0 - TOL).contains(&p0);
        Ok(MeasureResult {
            outcome,
            probability,
            deterministic,
        })
    }

    /// Compute every live label's post-projection amplitude and the partners
    /// the projection brings into the map, returning how many live labels
    /// prune away to nothing and the `Σ|x|²` the survivors carry.
    ///
    /// The mirror of [`stage_t_rotation`](Self::stage_t_rotation): read-only
    /// and positional. Unlike it, this pass does not probe the map at all —
    /// [`expectation_paired`](Self::expectation_paired) has just located every
    /// coset partner for it.
    fn stage_projection(
        &self,
        core: &Core,
        projection: &Projection<'_, K>,
        scratch: &mut RotationScratch<K>,
    ) -> (usize, f64) {
        let eps_sq = core.prune_epsilon.powi(2);
        let mut removals = 0;
        let mut norm = 0.0;

        let RotationScratch {
            values,
            inserts,
            partners,
        } = scratch;
        values.clear();
        inserts.clear();
        values.reserve(self.map.len());
        debug_assert_eq!(partners.len(), self.map.len());
        for ((c, &x), partner) in self.map.iter().zip(partners.iter()) {
            let value = match *partner {
                Some(y) => projection.rewrite_pair(c, x, y).0,
                None => {
                    // A missing partner contributes nothing to `c` and gains
                    // the whole of what `c` sends across.
                    let (kept, sent) = projection.rewrite_pair(c, x, Complex64::new(0.0, 0.0));
                    let weight = sent.norm_sqr();
                    if weight > eps_sq {
                        inserts.push((c.xor(&projection.d.a), sent));
                        norm += weight;
                    }
                    kept
                }
            };
            let weight = value.norm_sqr();
            if weight <= eps_sq {
                removals += 1;
            } else {
                norm += weight;
            }
            values.push(value);
        }
        (removals, norm)
    }

    // --- Shared helpers ---

    /// The (real) expectation `⟨Q⟩` of a frame-decomposed observable. Diagonal
    /// terms (`a = 0`) are `±1` eigenstates weighted by `|xvec[c]|²`;
    /// off-diagonal terms (`a ≠ 0`) pair each label with its coset partner
    /// `c ⊕ a`. Shared by [`TableauSimulator::measure`] and [`TableauSimulator::expectation`]
    /// — the one place the algebra lives.
    fn expectation_of(&self, d: &Decomp<K>) -> f64 {
        self.expectation_of_inner(d)
    }

    #[inline(always)]
    fn expectation_of_inner(&self, d: &Decomp<K>) -> f64 {
        if d.a.is_zero() {
            let zsign = d.zeta.re;
            self.map
                .iter()
                .map(|(c, &x)| {
                    let signed = if c.dot_parity(&d.b) { -zsign } else { zsign };
                    signed * x.norm_sqr()
                })
                .sum()
        } else {
            // Hermiticity of `Q = ζ X^a Z^b` forces `ζ̄·(−1)^{⟨a,b⟩} = ζ`, which
            // makes a pair's two terms equal, so half the probes could be
            // skipped by visiting only the member with the pivot bit clear and
            // doubling. Benchmarked at rank 4096: that trades a well-predicted
            // probe for a coin-flip branch and lands ~6% slower, so both
            // members are visited.
            let mut ev = 0.0;
            for (c, &x) in &self.map {
                if let Some(&y) = self.map.get(&c.xor(&d.a)) {
                    let sign = if c.dot_parity(&d.b) { -1.0 } else { 1.0 };
                    ev += (d.zeta * sign * x * y.conj()).re;
                }
            }
            ev
        }
    }

    /// [`expectation_of`](Self::expectation_of)'s off-diagonal branch, writing
    /// down where it found each term's coset partner.
    ///
    /// Every random measurement runs this expectation to get its outcome
    /// probability and then projects onto the same cosets, so without the
    /// record the projection would locate the identical pairing a second time
    /// — a full hash probe per term. `partners` is positional and the map is
    /// not touched between the two passes, which is what makes a plain `Vec`
    /// (rather than anything holding into the table) the right handle.
    ///
    /// No `popcnt` twin, unlike its sibling: this branch is probe-bound, and
    /// the feature measured at +0.8% — noise — on `expectation/off-diagonal`.
    fn expectation_paired(&self, d: &Decomp<K>, partners: &mut Vec<Option<Complex64>>) -> f64 {
        partners.clear();
        partners.reserve(self.map.len());
        let mut ev = 0.0;
        for (c, &x) in &self.map {
            let partner = self.map.get(&c.xor(&d.a)).copied();
            if let Some(y) = partner {
                let sign = if c.dot_parity(&d.b) { -1.0 } else { 1.0 };
                ev += (d.zeta * sign * x * y.conj()).re;
            }
            partners.push(partner);
        }
        ev
    }

    /// Rescale the amplitude map to unit norm from an already-measured `Σ|x|²`.
    fn rescale(&mut self, total: f64) {
        if total > 0.0 {
            let scale = total.sqrt().recip();
            for v in self.map.values_mut() {
                *v *= scale;
            }
        }
    }
}

// ==============================================================================
// Free helpers
// ==============================================================================

fn storage_words(words: usize) -> usize {
    match words {
        0 | 1 => 1,
        2 => 2,
        3 | 4 => 4,
        5..=8 => 8,
        _ => words,
    }
}

fn max_support(pauli: &PauliString) -> Option<usize> {
    pauli
        .x
        .iter()
        .zip(&pauli.z)
        .enumerate()
        .rev()
        .find_map(|(word, (&x, &z))| {
            let bits = x | z;
            (bits != 0).then(|| word * 64 + 63 - bits.leading_zeros() as usize)
        })
}

fn pauli_on_register(pauli: &PauliString, nqubits: usize) -> Result<PauliString, SimError> {
    if let Some(index) = max_support(pauli).filter(|&index| index >= nqubits) {
        return Err(SimError::QubitIndexOutOfRange {
            index,
            num_qubits: nqubits,
        });
    }
    let mut out = PauliString::new(nqubits);
    let words = out.x.len().min(pauli.x.len());
    out.x[..words].copy_from_slice(&pauli.x[..words]);
    out.z[..words].copy_from_slice(&pauli.z[..words]);
    out.set_phase(pauli.phase_exponent());
    Ok(out)
}

/// Hand capacity back once a map has outgrown its contents fourfold.
///
/// The amplitude map and its staging buffer are reused rather than rebuilt, and
/// a hash map never shrinks on its own. Since iteration walks the whole bucket
/// array, a map that peaked at a million labels would keep charging that on
/// every pass long after a measurement collapsed the rank back to one. The 4×
/// hysteresis keeps ordinary growth off the reallocation path.
fn shrink_if_sparse<K: LabelKey>(map: &mut HashMap<K, Complex64>, live: usize) {
    let target = live.max(16);
    if map.capacity() > 4 * target {
        map.shrink_to(2 * target);
    }
}

/// Eigenvalue-`+1` test for a diagonal frame term: `ζ·(−1)^{⟨b,c⟩} > 0`.
#[inline]
fn eig_plus<K: LabelKey>(c: &K, b: &K, zsign: f64) -> bool {
    let signed = if c.dot_parity(b) { -zsign } else { zsign };
    signed > 0.0
}

/// Accumulate `Σ_c xvec[c]·D^c|ψ0⟩` into `out` — the amplitude-map half of
/// [`TableauSimulator::state_vector`], split out so it can be monomorphized per width
/// like everything else that reads a label.
fn replay_terms<K: LabelKey>(
    map: &HashMap<K, Complex64>,
    psi0: &[Complex64],
    destabs: &[PauliString],
    out: &mut [Complex64],
) {
    for (c, &amp) in map {
        let mut term = psi0.to_vec();
        for (i, d_i) in destabs.iter().enumerate() {
            if c.get(i) {
                term = apply_pauli_dense(&term, d_i);
            }
        }
        for (dst, t) in out.iter_mut().zip(term) {
            *dst += amp * t;
        }
    }
}

/// Apply a signed dense Pauli `P = i^k X^a Z^b` to a state vector:
/// `(P v)[y ⊕ a] = i^k·(−1)^{⟨b,y⟩}·v[y]`. Small-`n` test support.
fn apply_pauli_dense(v: &[Complex64], p: &PauliString) -> Vec<Complex64> {
    /// A frame row's mask as a state-vector index mask. Reconstruction is
    /// `O(2^n)`, so `n` never reaches the second word.
    fn index_mask(mask: &[u64]) -> usize {
        debug_assert!(
            mask[1..].iter().all(|&word| word == 0),
            "state-vector reconstruction is unreachable past 64 qubits"
        );
        mask[0] as usize
    }

    let zeta = i_pow(p.phase_exponent() as u8);
    let amask = index_mask(&p.x);
    let bmask = index_mask(&p.z);
    let mut out = vec![Complex64::new(0.0, 0.0); v.len()];
    for (y, &val) in v.iter().enumerate() {
        let sign = if (y & bmask).count_ones() & 1 == 1 {
            -1.0
        } else {
            1.0
        };
        out[y ^ amask] += zeta * sign * val;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::{neg, pauli_string, pauli_x, pauli_y, pauli_z};

    fn assert_same_state(actual: &[Complex64], expected: &[Complex64]) {
        let (&a, &b) = actual
            .iter()
            .zip(expected)
            .find(|(_, b)| b.norm_sqr() > TOL)
            .expect("state has a nonzero amplitude");
        let phase = a / b;
        for (&a, &b) in actual.iter().zip(expected) {
            assert!((a - phase * b).norm() < TOL, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn measurement_can_drive_external_control_flow() {
        let mut sim = TableauSimulator::with_seed(2, 7);
        sim.h(0);
        sim.cx(0, 1).expect("distinct qubits");

        if sim.measure(0).expect("measurement succeeds").outcome {
            sim.x(1);
        }

        let second = sim.measure(1).expect("measurement succeeds");
        assert!(!second.outcome);
        assert!(second.deterministic);
    }

    #[test]
    fn pauli_rotation_and_expectation_use_existing_pauli_strings() {
        let mut sim = TableauSimulator::with_seed(1, 0);
        sim.h(0);
        sim.t_pauli(&pauli_z(1, 0), false)
            .expect("rotation succeeds");
        let x = sim
            .peek_observable_expectation(&pauli_string("X").expect("valid Pauli"))
            .expect("observable is in range");
        assert!((x - FRAC_1_SQRT_2).abs() < TOL);
    }

    #[test]
    fn gates_grow_the_shared_tableau_engine() {
        let mut sim = TableauSimulator::with_seed(1, 0);
        sim.h(65);
        assert_eq!(sim.num_qubits(), 66);
        assert_eq!(sim.core.r.nqubits, 66);
        assert!((sim.peek_x(65).expect("qubit exists") - 1.0).abs() < TOL);
    }

    #[test]
    fn state_vector_matches_hadamard_and_t() {
        let mut sim = TableauSimulator::with_seed(1, 0);
        sim.h(0);
        sim.t(0).expect("rotation succeeds");
        let phase = Complex64::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        assert_same_state(
            &sim.state_vector(),
            &[Complex64::new(FRAC_1_SQRT_2, 0.0), FRAC_1_SQRT_2 * phase],
        );
    }

    #[test]
    fn generic_controlled_paulis_match_all_named_axis_pairs() {
        type AxisPauli = fn(usize, usize) -> PauliString;
        type NamedGate = fn(&mut TableauSimulator, usize, usize) -> Result<(), SimError>;
        let cases: [(AxisPauli, AxisPauli, NamedGate); 9] = [
            (pauli_x, pauli_x, TableauSimulator::xcx),
            (pauli_x, pauli_y, TableauSimulator::xcy),
            (pauli_x, pauli_z, TableauSimulator::xcz),
            (pauli_y, pauli_x, TableauSimulator::ycx),
            (pauli_y, pauli_y, TableauSimulator::ycy),
            (pauli_y, pauli_z, TableauSimulator::ycz),
            (pauli_z, pauli_x, TableauSimulator::zcx),
            (pauli_z, pauli_y, TableauSimulator::zcy),
            (pauli_z, pauli_z, TableauSimulator::zcz),
        ];

        for (control, target, named_gate) in cases {
            let mut generic = TableauSimulator::with_seed(2, 0);
            generic.h(0);
            generic.s(0);
            generic.h(1);
            generic.sqrt_x(1);
            let mut named = generic.clone();

            generic
                .controlled_pauli(&control(2, 0), &target(2, 1))
                .expect("axes on distinct qubits commute");
            named_gate(&mut named, 0, 1).expect("qubits are distinct");

            assert_same_state(&generic.state_vector(), &named.state_vector());
        }
    }

    #[test]
    fn ccz_flips_only_the_all_one_amplitude() {
        let mut sim = TableauSimulator::with_seed(3, 0);
        for q in 0..3 {
            sim.h(q);
        }
        sim.ccz(0, 1, 2).expect("distinct qubits");

        let mut expected = vec![Complex64::new(FRAC_1_SQRT_2.powi(3), 0.0); 8];
        expected[7] = -expected[7];
        assert_same_state(&sim.state_vector(), &expected);
    }

    #[test]
    fn signed_observables_work_and_non_hermitian_inputs_do_not_grow() {
        let mut sim = TableauSimulator::with_seed(1, 0);
        let measured = sim
            .measure_observable(&neg(pauli_z(1, 0)))
            .expect("negative Z is Hermitian");
        assert!(measured.outcome && measured.deterministic);

        let mut invalid = pauli_x(5, 4);
        invalid.phase_shift(1);
        assert_eq!(
            sim.measure_observable(&invalid),
            Err(SimError::NonHermitianPauli)
        );
        assert_eq!(sim.num_qubits(), 1);
    }

    #[test]
    fn postselection_recompresses_to_the_selected_state() {
        for outcome in [false, true] {
            let mut sim = TableauSimulator::with_seed(1, 0);
            sim.h(0);
            sim.postselect_z(0, outcome)
                .expect("both Hadamard branches are reachable");
            let expected = if outcome {
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
            } else {
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
            };
            assert_same_state(&sim.state_vector(), &expected);
        }
    }
}
