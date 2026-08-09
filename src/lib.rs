//! Exact batch sampling for noisy, adaptive Clifford-dominated quantum circuits.
//!
//! Parse a [`.ticit` circuit](Circuit::from_file), [`compile`](Circuit::compile)
//! a [`Sampler`], then reuse it across any number of sampling calls. `DISCARD`
//! declarations and [`SamplerOptions::postselection_mask`] are combined during
//! compilation.

#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod active;
mod bits;
mod circuit;
mod component_plan;
mod contiguous;
mod errors;
mod factored;
mod frames;
#[cfg(feature = "gpu")]
#[doc(hidden)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub mod gpu;
mod pauli;
mod pending_optimizer;
mod planner;
mod random;
mod sampler;
mod symbolic;
pub mod tableau_simulator;

pub use crate::circuit::Circuit;
pub use crate::errors::{Result, TicitError};
pub use crate::pauli::{PauliString, neg, pauli_identity, pauli_string, pauli_x, pauli_y, pauli_z};
pub use crate::sampler::prepared::{
    SampleCounts, SampleResult, Sampler, SamplerInfo, SamplerOptions, SamplingTiming,
};
pub use crate::tableau_simulator::{MeasureResult, SimError, TableauSimulator};

pub(crate) use crate::sampler::{batch, exogenous, presampled_expression};

#[cfg(test)]
mod test_support;
