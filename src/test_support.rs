//! Fixtures shared by the crate's unit tests.

use std::path::PathBuf;

use crate::factored::{PendingFactoredState, PendingOperation};

fn circuit_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/circuits")
        .join(name)
}

pub fn soft_benchmark_circuits() -> PathBuf {
    circuit_dir("soft")
}

pub fn ccz_nontels_circuits() -> PathBuf {
    circuit_dir("ccz")
}

pub fn msc_d3_circuit() -> PathBuf {
    soft_benchmark_circuits().join("msc_d3_inject_cultivate_p1e-3.stim")
}

// ==============================================================================
// Causality
// ==============================================================================

/// Asserts that no queued operation consumes a measurement's record condition
/// before the operation that produces it.
///
/// This is the invariant the optimizer's movement rules exist to protect: a
/// symbol must be assigned before it is used, or the sampler reads garbage.
pub fn require_pending_record_conditions_are_causal(state: &PendingFactoredState, context: &str) {
    let producers: Vec<(i32, usize)> = state
        .pending_operations
        .iter()
        .enumerate()
        .filter_map(|(index, operation)| {
            operation
                .record_condition()
                .map(|condition| (condition, index))
        })
        .collect();

    for (use_index, operation) in state.pending_operations.iter().enumerate() {
        let expression = match operation {
            PendingOperation::ClassicalRecord(record) => &record.outcome,
            PendingOperation::PauliRotation(rotation) => &rotation.pauli.sign,
            PendingOperation::PauliMeasurement(measurement) => &measurement.pauli.sign,
        };
        for condition in &expression.conditions {
            for &(producer, producer_index) in &producers {
                assert!(
                    producer != *condition || producer_index < use_index,
                    "{context} uses measurement condition {condition} before it is assigned"
                );
            }
        }
    }
}
