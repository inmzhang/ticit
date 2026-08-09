//! Compact per-Pauli kernels baked into sampler instructions by the planner.
//!
//! # Basis-index layout
//!
//! Amplitude `basis` of a `k`-qubit active state is the computational basis
//! state whose qubit `q` is bit `q` of `basis`. Because a basis index must fit
//! in a machine word, `k` is capped at 62 by [`active_length`] and Pauli widths
//! at 63 by [`ActivePauliAction::new`]; both limits are load-bearing, since the
//! whole kernel layer reduces a Pauli to the single-word masks `x[0]` / `z[0]`.
//!
//! # Kernel size contract
//!
//! Both precomputed kernels are O(1)-sized: per-basis coefficients are *derived*
//! from the masks during execution rather than tabulated, which is what keeps a
//! plan's memory independent of `2^k`. The `size_of` assertions below keep
//! both kernels within two cache lines.

use std::mem::size_of;

use num_complex::Complex64;

use crate::bits::is_odd_popcount;
use crate::errors::{Result, TicitError};
use crate::pauli::{PauliString, pauli_squares_to_identity};

/// `1 / sqrt(2)`. Bit-identical to the C++ literal, so the two ports round the
/// same way; re-exported here because the kernels read better with this name.
pub const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Number of amplitudes in a `k`-qubit active state.
///
/// Fails past 62 qubits: basis indices are machine words, and the kernels need
/// one spare bit for [`insert_zero_bit`].
pub fn active_length(k: usize) -> Result<usize> {
    if k >= 62 {
        return Err(TicitError::new(
            "active qubit count is too large for machine basis indices",
        ));
    }
    Ok(1usize << k)
}

/// `i^phase`, for a phase exponent taken modulo 4.
pub fn phase_factor(phase: i32) -> Complex64 {
    match phase & 3 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

/// Widens `packed` by opening a zero bit at position `bit`.
///
/// This is the inverse of dropping qubit `bit` from a basis index: measurement
/// kernels enumerate the `2^(k-1)` output states and reconstruct the two input
/// states each of them draws from.
#[inline]
pub fn insert_zero_bit(packed: usize, bit: usize) -> usize {
    let low_mask = (1usize << bit) - 1;
    (packed & low_mask) | ((packed & !low_mask) << 1)
}

fn is_near_zero(value: f64) -> bool {
    value.abs() < 1e-14
}

fn active_mask_x(pauli: &PauliString) -> u64 {
    pauli.x.first().copied().unwrap_or(0)
}

fn active_mask_z(pauli: &PauliString) -> u64 {
    pauli.z.first().copied().unwrap_or(0)
}

// ==============================================================================
// Compact Pauli descriptor
// ==============================================================================

/// A Hermitian Pauli reduced to the single-word masks the kernels consume.
///
/// `even_phase` is the operator's coefficient `i^phase`; a basis index whose
/// overlap with `zmask` has odd parity picks up `odd_phase == -even_phase`
/// instead. `xz_overlap_odd` says whether the two members of an X-pair see
/// opposite parities, which is what makes the general pair rotation
/// antisymmetric.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActivePauliAction {
    pub nqubits: usize,
    pub xmask: u64,
    pub zmask: u64,
    pub even_phase: Complex64,
    pub odd_phase: Complex64,
    pub xz_overlap_odd: bool,
}

impl Default for ActivePauliAction {
    fn default() -> Self {
        Self {
            nqubits: 0,
            xmask: 0,
            zmask: 0,
            even_phase: Complex64::new(1.0, 0.0),
            odd_phase: Complex64::new(-1.0, 0.0),
            xz_overlap_odd: false,
        }
    }
}

impl ActivePauliAction {
    /// Fails for a Pauli too wide for single-word masks, or one that does not
    /// square to the identity (a rotation generator must be Hermitian).
    pub fn new(pauli: &PauliString) -> Result<Self> {
        if pauli.nqubits >= 63 {
            return Err(TicitError::new(
                "Pauli string has too many qubits for active-state basis indexing",
            ));
        }
        if !pauli_squares_to_identity(pauli) {
            return Err(TicitError::new("Pauli rotation requires P^2 == I"));
        }
        let xmask = active_mask_x(pauli);
        let zmask = active_mask_z(pauli);
        let even_phase = phase_factor(pauli.phase_exponent());
        Ok(Self {
            nqubits: pauli.nqubits,
            xmask,
            zmask,
            even_phase,
            odd_phase: -even_phase,
            xz_overlap_odd: is_odd_popcount(xmask & zmask),
        })
    }

    /// Whether basis index `basis` sits in the `-1` eigenspace of the Z part.
    #[inline]
    pub fn phase_odd(&self, basis: usize) -> bool {
        is_odd_popcount(basis as u64 & self.zmask)
    }
}

/// True when every pair rotation can use a *real* per-pair coefficient with an
/// antisymmetric sign, which is a cheaper kernel than the general complex case.
fn can_rotate_real_pair_flip(action: &ActivePauliAction) -> bool {
    if action.zmask == 0 || !action.xz_overlap_odd {
        return false;
    }
    is_near_zero(action.even_phase.re) && !is_near_zero(action.even_phase.im)
}

// ==============================================================================
// Precomputed rotation kernel
// ==============================================================================

/// Everything `exp(-i * kernel_angle * P)` needs, in O(1) memory.
///
/// `pair_bit` is the **highest** set bit of `xmask`. That choice is load-bearing
/// for the dense kernels: it makes `i ^ xmask` map the low half of every
/// `2^(pair_bit+1)`-sized block onto the high half of the same block, so pairs
/// are addressable by block walking instead of gathers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecomputedActivePauliRotationKernel {
    pub action: ActivePauliAction,
    pub is_diagonal: bool,
    pub uniform_imag_pairs: bool,
    pub real_pair_flip: bool,
    pub pair_bit: u32,
    pub pair_count: usize,
    /// The internal angle `phi` of `exp(-i phi P)`. A frontend `R_P(theta)`
    /// passes `phi = theta / 2`.
    pub kernel_angle: f64,
    pub cos_kernel_angle: f64,
    pub sin_kernel_angle: f64,
    pub minus_even_coefficient: Complex64,
}

impl Default for PrecomputedActivePauliRotationKernel {
    fn default() -> Self {
        Self {
            action: ActivePauliAction::default(),
            is_diagonal: true,
            uniform_imag_pairs: false,
            real_pair_flip: false,
            pair_bit: 0,
            pair_count: 0,
            kernel_angle: 0.0,
            cos_kernel_angle: 1.0,
            sin_kernel_angle: 0.0,
            minus_even_coefficient: Complex64::new(0.0, 0.0),
        }
    }
}

impl PrecomputedActivePauliRotationKernel {
    pub fn new(action: &ActivePauliAction, kernel_angle: f64) -> Result<Self> {
        let dim = active_length(action.nqubits)?;
        let sin_kernel_angle = kernel_angle.sin();
        let mut kernel = Self {
            action: *action,
            is_diagonal: action.xmask == 0,
            uniform_imag_pairs: false,
            real_pair_flip: false,
            pair_bit: 0,
            pair_count: 0,
            kernel_angle,
            cos_kernel_angle: kernel_angle.cos(),
            sin_kernel_angle,
            minus_even_coefficient: Complex64::new(0.0, -sin_kernel_angle) * action.even_phase,
        };
        if kernel.is_diagonal {
            return Ok(kernel);
        }
        kernel.uniform_imag_pairs = action.zmask == 0;
        kernel.real_pair_flip = can_rotate_real_pair_flip(action);
        kernel.pair_bit = 63 - action.xmask.leading_zeros();
        kernel.pair_count = dim >> 1;
        Ok(kernel)
    }

    /// The `-i sin(phi)` coefficient at basis index `source`, with the runtime
    /// `sign` folded in. Flipping `sign` is exactly negating the angle.
    #[inline]
    pub fn coefficient(&self, source: usize, sign: bool) -> Complex64 {
        if sign != self.action.phase_odd(source) {
            -self.minus_even_coefficient
        } else {
            self.minus_even_coefficient
        }
    }
}

// ==============================================================================
// Precomputed measurement kernel
// ==============================================================================

/// Everything a Born-rule sample and projection of `P` needs, in O(1) memory.
///
/// `pivot` is the highest set bit of `xmask`, or of `zmask` for a diagonal
/// Pauli. It names the coordinate the measurement consumes: output index `idx`
/// is the input index with bit `pivot` deleted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecomputedActivePauliMeasurementKernel {
    pub action: ActivePauliAction,
    pub pivot: usize,
    pub is_diagonal: bool,
    /// For a diagonal Pauli: 1 when the operator's coefficient is `-1`, so that
    /// the pivot bit selecting a given branch flips.
    pub diagonal_phase_bit: i32,
    pub z_without_pivot: u64,
    pub out_dim: usize,
    pub nondiagonal_coefficient1_even: Complex64,
}

impl Default for PrecomputedActivePauliMeasurementKernel {
    fn default() -> Self {
        Self {
            action: ActivePauliAction::default(),
            pivot: 0,
            is_diagonal: true,
            diagonal_phase_bit: 0,
            z_without_pivot: 0,
            out_dim: 0,
            nondiagonal_coefficient1_even: Complex64::new(0.0, 0.0),
        }
    }
}

impl PrecomputedActivePauliMeasurementKernel {
    pub fn from_pauli(pauli: &PauliString) -> Result<Self> {
        Self::from_action(&ActivePauliAction::new(pauli)?)
    }

    /// Picks the pivot: highest X qubit, else highest Z qubit.
    pub fn from_action(action: &ActivePauliAction) -> Result<Self> {
        let pivot = if action.xmask != 0 {
            63 - action.xmask.leading_zeros()
        } else if action.zmask != 0 {
            63 - action.zmask.leading_zeros()
        } else {
            return Err(TicitError::new(
                "cannot build an active measurement kernel for identity Pauli",
            ));
        };
        Self::with_pivot(action, pivot as usize)
    }

    /// Builds the kernel around an explicitly chosen pivot, which the component
    /// planner needs because a component's local coordinate order does not
    /// preserve which qubit was globally highest.
    pub fn with_pivot(action: &ActivePauliAction, pivot: usize) -> Result<Self> {
        if action.nqubits == 0 {
            return Err(TicitError::new(
                "cannot build an active measurement kernel for k == 0",
            ));
        }
        if pivot >= action.nqubits {
            return Err(TicitError::new("active measurement pivot is out of range"));
        }
        let mut kernel = Self {
            action: *action,
            pivot,
            out_dim: active_length(action.nqubits)? >> 1,
            nondiagonal_coefficient1_even: action.even_phase.conj() * INV_SQRT2,
            ..Self::default()
        };
        let pivot_bit = 1u64 << pivot;
        if action.xmask != 0 {
            if action.xmask & pivot_bit == 0 {
                return Err(TicitError::new(
                    "nondiagonal active measurement pivot must have an X component",
                ));
            }
            kernel.is_diagonal = false;
        } else if action.zmask != 0 {
            if action.zmask & pivot_bit == 0 {
                return Err(TicitError::new(
                    "diagonal active measurement pivot must have a Z component",
                ));
            }
            kernel.is_diagonal = true;
            // A diagonal measurement just reads a parity, so its eigenvalues
            // have to be real; +-i would leave no bit to record.
            let negative_phase =
                (action.even_phase.re + 1.0).abs() < 1e-12 && action.even_phase.im.abs() < 1e-12;
            let positive_phase =
                (action.even_phase.re - 1.0).abs() < 1e-12 && action.even_phase.im.abs() < 1e-12;
            if !negative_phase && !positive_phase {
                return Err(TicitError::new(
                    "diagonal active measurement Pauli must have real eigenvalues",
                ));
            }
            kernel.diagonal_phase_bit = i32::from(negative_phase);
            kernel.z_without_pivot = action.zmask & !pivot_bit;
        } else {
            return Err(TicitError::new(
                "cannot build an active measurement kernel for identity Pauli",
            ));
        }
        Ok(kernel)
    }

    /// Input index that output index `packed` reads, for a diagonal Pauli.
    #[inline]
    pub fn diagonal_source(&self, packed: usize, branch: bool) -> usize {
        let without_pivot = insert_zero_bit(packed, self.pivot);
        let parity = if self.z_without_pivot == 0 {
            0
        } else {
            (without_pivot as u64 & self.z_without_pivot).count_ones() as i32 & 1
        };
        let pivot_value = self.diagonal_phase_bit ^ parity ^ i32::from(branch);
        without_pivot | ((pivot_value as usize) << self.pivot)
    }

    /// First of the two input indices output index `packed` mixes.
    #[inline]
    pub fn nondiagonal_source0(&self, packed: usize) -> usize {
        insert_zero_bit(packed, self.pivot)
    }

    /// Partner of [`nondiagonal_source0`](Self::nondiagonal_source0).
    #[inline]
    pub fn nondiagonal_source1(&self, packed: usize) -> usize {
        self.nondiagonal_source0(packed) ^ self.action.xmask as usize
    }

    /// Weight of the partner amplitude in the projected output.
    #[inline]
    pub fn nondiagonal_coefficient1(&self, packed: usize, branch: bool) -> Complex64 {
        let odd = self.action.phase_odd(self.nondiagonal_source0(packed));
        if branch != odd {
            -self.nondiagonal_coefficient1_even
        } else {
            self.nondiagonal_coefficient1_even
        }
    }
}

// Plan memory must stay independent of 2^k; these mirror the C++ static_asserts.
const _: () = assert!(size_of::<PrecomputedActivePauliRotationKernel>() <= 128);
const _: () = assert!(size_of::<PrecomputedActivePauliMeasurementKernel>() <= 128);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::{pauli_x, pauli_z};

    #[test]
    fn active_length_rejects_oversized_registers() {
        assert_eq!(active_length(0), Ok(1));
        assert_eq!(active_length(61), Ok(1usize << 61));
        assert!(active_length(62).is_err());
    }

    #[test]
    fn insert_zero_bit_opens_a_gap() {
        assert_eq!(insert_zero_bit(0b1011, 0), 0b10110);
        assert_eq!(insert_zero_bit(0b1011, 2), 0b10011);
        assert_eq!(insert_zero_bit(0b1011, 4), 0b01011);
    }

    #[test]
    fn action_rejects_non_hermitian_paulis() {
        let mut pauli = pauli_x(1, 0);
        pauli.set_phase(1);
        assert!(ActivePauliAction::new(&pauli).is_err());
    }

    #[test]
    fn measurement_kernel_picks_the_highest_pivot() {
        let diagonal = &pauli_z(4, 0) * &pauli_z(4, 3);
        let kernel = PrecomputedActivePauliMeasurementKernel::from_pauli(&diagonal)
            .expect("Z0*Z3 is a valid diagonal measurement");
        assert!(kernel.is_diagonal);
        assert_eq!(kernel.pivot, 3);
        assert_eq!(kernel.z_without_pivot, 0b0001);

        let nondiagonal = &(&pauli_x(4, 1) * &pauli_z(4, 2)) * &pauli_x(4, 3);
        let kernel = PrecomputedActivePauliMeasurementKernel::from_pauli(&nondiagonal)
            .expect("X1*Z2*X3 is a valid measurement");
        assert!(!kernel.is_diagonal);
        assert_eq!(kernel.pivot, 3);
    }

    #[test]
    fn identity_has_no_measurement_kernel() {
        let action = ActivePauliAction::new(&PauliString::new(3)).expect("identity is Hermitian");
        assert!(PrecomputedActivePauliMeasurementKernel::from_action(&action).is_err());
    }

    #[test]
    fn diagonal_measurement_needs_real_eigenvalues() {
        // Unreachable through `ActivePauliAction::new` — a diagonal Hermitian
        // Pauli always has an even phase exponent — but the component planner
        // builds actions by hand, so the guard still has to hold.
        let action = ActivePauliAction {
            nqubits: 2,
            zmask: 0b10,
            even_phase: Complex64::new(0.0, 1.0),
            odd_phase: Complex64::new(0.0, -1.0),
            ..ActivePauliAction::default()
        };
        assert!(PrecomputedActivePauliMeasurementKernel::from_action(&action).is_err());
    }
}
