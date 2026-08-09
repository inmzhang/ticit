//! Peephole optimization of the pending-operation queue, run once before
//! planning starts.
//!
//! Two transformations, both aimed at removing quantum work rather than at
//! tidying the schedule:
//!
//! * **Rotation fusion** — two rotations about the same Pauli separated only by
//!   operations that commute with it are one rotation. Their angles add (or
//!   subtract, when their symbolic signs are exact opposites), and an exactly
//!   zero result deletes both.
//! * **Measurement left-movement** — sliding a measurement earlier over
//!   commuting rotations shortens the lifetime of the active coordinate it will
//!   consume. This is only worth the churn when fusion already proved the
//!   segment has slack, so the unconditional form is gated on that; otherwise a
//!   measurement moves only as far as a rotation about its own Pauli body,
//!   where the rotation degenerates into a branch phase the planner can drop.
//!
//! # What must not move
//!
//! Measurements never cross other measurements or classical records: record
//! order is observable, and a symbol must be assigned before it is used.
//! Rotations may cross classical records freely — fusion moves the *earlier*
//! rotation later, which cannot make a use precede its assignment.
//!
//! # Segmentation
//!
//! Detector anchors (`preserved_prefixes`) cut the queue into independently
//! optimized segments, so a detector's operation-index still means what the
//! frontend thinks it means. [`PendingOptimizationStats::prefix_remap`] reports
//! where each survived.

use crate::errors::{Result, TicitError};
use crate::factored::{
    PendingFactoredState, PendingOperation, PendingOptimizationStats, PendingPauliRotation,
};
use crate::frames::SymbolicPauliString;
use crate::pauli::{
    PauliString, measurement_phase_sign, pauli_anticommutes, pauli_body_y_count,
    pauli_squares_to_identity,
};
use crate::symbolic::{SymbolicBool, xor_bool_constant};

/// A rotation with its `+-` stripped out of the Pauli and into the sign, and its
/// body phase normalized. Two rotations fuse only if they agree here.
struct CanonicalRotation {
    body: PauliString,
    sign: SymbolicBool,
}

fn canonical_rotation(rotation: &PendingPauliRotation) -> Result<CanonicalRotation> {
    if !pauli_squares_to_identity(&rotation.pauli.pauli) {
        return Err(TicitError::new(
            "pending Pauli rotation must have a Hermitian generator",
        ));
    }
    let mut body = rotation.pauli.pauli.clone();
    let negative =
        measurement_phase_sign(&body).expect("the body was just checked to square to the identity");
    let sign = xor_bool_constant(&rotation.pauli.sign, negative);
    body.set_phase(pauli_body_y_count(&body));
    Ok(CanonicalRotation { body, sign })
}

fn pauli_bodies_commute(lhs: &PauliString, rhs: &PauliString) -> bool {
    !pauli_anticommutes(lhs, rhs)
}

/// Whether the earlier rotation may be moved past `operation`.
fn rotation_can_cross(rotation: &CanonicalRotation, operation: &PendingOperation) -> bool {
    match operation {
        PendingOperation::PauliRotation(other) => {
            pauli_bodies_commute(&rotation.body, &other.pauli.pauli)
        }
        PendingOperation::PauliMeasurement(measurement) => {
            // An expectation probe observes the state as it stands, so nothing
            // may be reordered across it.
            measurement.exp_val.is_none()
                && pauli_bodies_commute(&rotation.body, &measurement.pauli.pauli)
        }
        PendingOperation::ClassicalRecord(_) => true,
    }
}

/// Fuses `earlier` into `later`, returning the replacement and whether it
/// cancelled to nothing.
fn try_fuse_rotations(
    earlier: &PendingPauliRotation,
    later: &PendingPauliRotation,
) -> Result<Option<(PendingPauliRotation, bool)>> {
    let a = canonical_rotation(earlier)?;
    let b = canonical_rotation(later)?;
    if !a.body.same_body(&b.body) || a.sign.conditions != b.sign.conditions {
        return Ok(None);
    }

    // Signs that differ only in their constant are exact opposites, so the
    // earlier rotation contributes its angle with the opposite sense.
    let earlier_direction = if a.sign.constant == b.sign.constant {
        1.0
    } else {
        -1.0
    };
    let angle = later.kernel_angle + earlier_direction * earlier.kernel_angle;
    let cancelled = angle == 0.0;
    Ok(Some((
        PendingPauliRotation {
            kernel_angle: angle,
            pauli: SymbolicPauliString::with_sign(b.body, b.sign),
        },
        cancelled,
    )))
}

fn fuse_commuting_rotations(
    operations: &mut Vec<PendingOperation>,
    stats: &mut PendingOptimizationStats,
) -> Result<()> {
    let mut deleted = vec![false; operations.len()];
    for i in 0..operations.len() {
        if deleted[i] {
            continue;
        }
        let PendingOperation::PauliRotation(earlier) = &operations[i] else {
            continue;
        };
        let earlier = earlier.clone();
        let earlier_canonical = canonical_rotation(&earlier)?;
        for j in i + 1..operations.len() {
            if deleted[j] {
                continue;
            }
            if !rotation_can_cross(&earlier_canonical, &operations[j]) {
                break;
            }
            let PendingOperation::PauliRotation(later) = &operations[j] else {
                continue;
            };
            let Some((fused, cancelled)) = try_fuse_rotations(&earlier, later)? else {
                continue;
            };
            deleted[i] = true;
            stats.fused_rotations += 1;
            if cancelled {
                deleted[j] = true;
                stats.cancelled_rotations += 1;
            } else {
                operations[j] = fused.into();
            }
            break;
        }
    }

    let mut kept = Vec::with_capacity(operations.len());
    for (operation, &deleted) in std::mem::take(operations).into_iter().zip(&deleted) {
        if !deleted {
            kept.push(operation);
        }
    }
    *operations = kept;
    Ok(())
}

fn move_measurements_earlier(
    operations: &mut [PendingOperation],
    stats: &mut PendingOptimizationStats,
    allow_all_commuting: bool,
) {
    for i in 1..operations.len() {
        let PendingOperation::PauliMeasurement(measurement) = &operations[i] else {
            continue;
        };
        let measured_body = measurement.pauli.pauli.clone();

        let mut target = i;
        for cursor in (1..=i).rev() {
            let PendingOperation::PauliRotation(rotation) = &operations[cursor - 1] else {
                break;
            };
            if !pauli_bodies_commute(&measured_body, &rotation.pauli.pauli) {
                break;
            }
            if allow_all_commuting {
                target = cursor - 1;
                continue;
            }
            // Reaching a rotation about the same body is the payoff case: the
            // measurement pins that rotation to a branch phase.
            if measured_body.same_body(&rotation.pauli.pauli) {
                target = cursor - 1;
                break;
            }
        }

        let mut current = i;
        while current > target {
            operations.swap(current - 1, current);
            current -= 1;
            stats.measurement_left_swaps += 1;
        }
    }
}

fn optimize_segment(
    operations: &mut Vec<PendingOperation>,
    stats: &mut PendingOptimizationStats,
) -> Result<()> {
    let fused_before = stats.fused_rotations;
    fuse_commuting_rotations(operations, stats)?;
    move_measurements_earlier(operations, stats, stats.fused_rotations != fused_before);
    Ok(())
}

/// Optimizes `state.pending_operations` in place.
///
/// `preserved_prefixes` are operation-count boundaries that must survive as
/// boundaries; each becomes a segment edge.
pub fn optimize_pending_operations(
    state: &mut PendingFactoredState,
    preserved_prefixes: &[usize],
) -> Result<PendingOptimizationStats> {
    if !state.instructions.is_empty() || !state.pending_prefix_instruction_indices.is_empty() {
        return Err(TicitError::new(
            "pending-operation optimization must run before planning",
        ));
    }

    let mut stats = PendingOptimizationStats::default();
    let operation_count = state.pending_operations.len();
    stats.input_operations = operation_count;
    let has_expectation = state.pending_operations.iter().any(|operation| {
        matches!(operation, PendingOperation::PauliMeasurement(measurement) if measurement.exp_val.is_some())
    });
    state.has_expectation = has_expectation;
    if has_expectation {
        // Expectation probes pin the whole schedule, so there is nothing to do
        // but report an identity remap.
        // TODO(perf): a linear commute pass could still fuse between probes.
        stats.prefix_remap = (0..=operation_count as i32).collect();
        state.pending_operations_optimized = true;
        stats.output_operations = operation_count;
        return Ok(stats);
    }
    stats.prefix_remap = vec![-1; operation_count + 1];
    stats.prefix_remap[0] = 0;

    let mut segment_ends = Vec::with_capacity(preserved_prefixes.len() + 1);
    for &prefix in preserved_prefixes {
        if prefix > operation_count {
            return Err(TicitError::new(
                "preserved pending-operation prefix is out of range",
            ));
        }
        if prefix > 0 {
            segment_ends.push(prefix);
        }
    }
    segment_ends.push(operation_count);
    segment_ends.sort_unstable();
    segment_ends.dedup();

    let mut input = std::mem::take(&mut state.pending_operations).into_iter();
    let mut output: Vec<PendingOperation> = Vec::with_capacity(operation_count);
    let mut segment_start = 0;
    for segment_end in segment_ends {
        let mut segment: Vec<PendingOperation> =
            input.by_ref().take(segment_end - segment_start).collect();
        optimize_segment(&mut segment, &mut stats)?;
        output.append(&mut segment);
        stats.prefix_remap[segment_end] = output.len() as i32;
        segment_start = segment_end;
    }

    state.pending_operations = output;
    state.pending_operations_optimized = true;
    stats.output_operations = state.pending_operations.len();
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factored::{PendingClassicalRecord, PendingPauliMeasurement};
    use crate::pauli::{pauli_x, pauli_y, pauli_z};

    fn rotation(pauli: PauliString, angle: f64) -> PendingOperation {
        PendingPauliRotation {
            kernel_angle: angle,
            pauli: SymbolicPauliString::new(pauli),
        }
        .into()
    }

    #[test]
    fn canonicalization_strips_the_sign_out_of_the_body() {
        // -Y is Hermitian with a negative coefficient; the sign belongs in the
        // symbolic sign, not in the stored phase.
        let mut body = pauli_y(1, 0);
        body.phase_shift(2);
        let canonical = canonical_rotation(&PendingPauliRotation {
            kernel_angle: 0.5,
            pauli: SymbolicPauliString::new(body),
        })
        .expect("-Y is Hermitian");
        assert!(canonical.sign.constant);
        assert_eq!(canonical.body, pauli_y(1, 0));
    }

    #[test]
    fn non_hermitian_generators_are_rejected() {
        let mut body = pauli_x(1, 0);
        body.set_phase(1);
        let mut state = PendingFactoredState::new(1, 0).expect("k <= n");
        state.pending_operations.push(rotation(body, 0.5));
        state.pending_operations.push(rotation(pauli_x(1, 0), 0.5));
        assert!(optimize_pending_operations(&mut state, &[]).is_err());
    }

    #[test]
    fn optimizing_after_planning_started_is_rejected() {
        let mut state = PendingFactoredState::new(1, 0).expect("k <= n");
        state.pending_prefix_instruction_indices.push(0);
        assert!(optimize_pending_operations(&mut state, &[]).is_err());
    }

    #[test]
    fn out_of_range_preserved_prefixes_are_rejected() {
        let mut state = PendingFactoredState::new(1, 0).expect("k <= n");
        state.pending_operations.push(rotation(pauli_x(1, 0), 0.5));
        assert!(optimize_pending_operations(&mut state, &[2]).is_err());
    }

    #[test]
    fn expectation_probes_disable_the_optimizer() {
        let mut state = PendingFactoredState::new(1, 0).expect("k <= n");
        state.pending_operations.push(rotation(pauli_x(1, 0), 0.1));
        state.pending_operations.push(rotation(pauli_x(1, 0), 0.2));
        state.pending_operations.push(
            PendingPauliMeasurement {
                pauli: SymbolicPauliString::new(pauli_x(1, 0)),
                exp_val: Some(0),
                ..PendingPauliMeasurement::default()
            }
            .into(),
        );
        let stats = optimize_pending_operations(&mut state, &[]).expect("no planning yet");
        assert!(state.has_expectation);
        assert_eq!(stats.fused_rotations, 0);
        assert_eq!(stats.prefix_remap, vec![0, 1, 2, 3]);
        assert_eq!(state.pending_operations.len(), 3);
    }

    fn signed_rotation(pauli: PauliString, angle: f64, sign: SymbolicBool) -> PendingOperation {
        PendingPauliRotation {
            kernel_angle: angle,
            pauli: SymbolicPauliString::with_sign(pauli, sign),
        }
        .into()
    }

    fn measurement(pauli: PauliString, record: i32, record_condition: i32) -> PendingOperation {
        PendingPauliMeasurement {
            pauli: SymbolicPauliString::new(pauli),
            record: Some(record),
            record_condition: Some(record_condition),
            exp_val: None,
        }
        .into()
    }

    fn state_with(n: usize, operations: Vec<PendingOperation>) -> PendingFactoredState {
        let mut state = PendingFactoredState::new(n, 0).expect("k <= n");
        state.pending_operations = operations;
        state
    }

    #[test]
    fn commuting_rotations_fuse_and_let_the_measurement_move_up() {
        let sign = SymbolicBool::new(false, vec![1]);
        let mut state = state_with(
            2,
            vec![
                signed_rotation(pauli_x(2, 0), 0.1, sign.clone()),
                signed_rotation(pauli_z(2, 1), 0.2, SymbolicBool::default()),
                signed_rotation(pauli_x(2, 0), 0.3, sign.clone()),
                measurement(pauli_x(2, 0), 1, 2),
            ],
        );

        let stats = optimize_pending_operations(&mut state, &[]).expect("not planned");
        assert_eq!(stats.input_operations, 4);
        assert_eq!(stats.output_operations, 3);
        assert_eq!(stats.fused_rotations, 1);
        assert_eq!(stats.cancelled_rotations, 0);
        assert_eq!(stats.measurement_left_swaps, 2);
        assert!(matches!(
            state.pending_operations[0],
            PendingOperation::PauliMeasurement(_)
        ));

        let fused = state
            .pending_operations
            .iter()
            .find_map(|operation| match operation {
                PendingOperation::PauliRotation(rotation)
                    if rotation.pauli.pauli.same_body(&pauli_x(2, 0)) =>
                {
                    Some(rotation)
                }
                _ => None,
            })
            .expect("fused rotation");
        assert!((fused.kernel_angle - 0.4).abs() < 1e-12);
        assert_eq!(fused.pauli.sign, sign);
    }

    #[test]
    fn an_anticommuting_rotation_blocks_fusion() {
        let mut state = state_with(
            1,
            vec![
                signed_rotation(pauli_x(1, 0), 0.1, SymbolicBool::default()),
                signed_rotation(pauli_z(1, 0), 0.2, SymbolicBool::default()),
                signed_rotation(pauli_x(1, 0), 0.3, SymbolicBool::default()),
            ],
        );
        let stats = optimize_pending_operations(&mut state, &[]).expect("not planned");
        assert_eq!(stats.fused_rotations, 0);
        assert_eq!(state.pending_operations.len(), 3);
    }

    #[test]
    fn different_symbolic_signs_do_not_fuse() {
        let mut state = state_with(
            1,
            vec![
                signed_rotation(pauli_x(1, 0), 0.1, SymbolicBool::new(false, vec![1])),
                signed_rotation(pauli_x(1, 0), 0.2, SymbolicBool::new(false, vec![2])),
                measurement(pauli_z(1, 0), 1, 3),
            ],
        );
        let stats = optimize_pending_operations(&mut state, &[]).expect("not planned");
        assert_eq!(stats.fused_rotations, 0);
        assert_eq!(stats.measurement_left_swaps, 0);
        assert!(matches!(
            state.pending_operations[0],
            PendingOperation::PauliRotation(_)
        ));
    }

    #[test]
    fn unrelated_commuting_measurement_keeps_the_schedule() {
        let mut state = state_with(
            2,
            vec![
                signed_rotation(pauli_x(2, 0), 0.1, SymbolicBool::default()),
                measurement(pauli_z(2, 1), 1, 2),
            ],
        );
        let stats = optimize_pending_operations(&mut state, &[]).expect("not planned");
        assert_eq!(stats.fused_rotations, 0);
        assert_eq!(stats.measurement_left_swaps, 0);
        assert!(matches!(
            state.pending_operations[0],
            PendingOperation::PauliRotation(_)
        ));
    }

    #[test]
    fn opposite_symbolic_signs_subtract_angles() {
        let sign = SymbolicBool::new(false, vec![1]);
        let mut state = state_with(
            1,
            vec![
                signed_rotation(pauli_x(1, 0), 0.2, sign.clone()),
                signed_rotation(pauli_x(1, 0), 0.3, !sign.clone()),
            ],
        );
        let stats = optimize_pending_operations(&mut state, &[]).expect("not planned");
        assert_eq!(stats.fused_rotations, 1);
        let PendingOperation::PauliRotation(fused) = &state.pending_operations[0] else {
            panic!("expected rotation");
        };
        assert!((fused.kernel_angle - 0.1).abs() < 1e-12);
        assert_eq!(fused.pauli.sign, !sign);
    }

    #[test]
    fn exact_inverse_rotations_cancel() {
        let mut state = state_with(
            1,
            vec![
                signed_rotation(pauli_y(1, 0), 0.25, SymbolicBool::default()),
                signed_rotation(pauli_y(1, 0), -0.25, SymbolicBool::default()),
            ],
        );
        let stats = optimize_pending_operations(&mut state, &[]).expect("not planned");
        assert_eq!(stats.fused_rotations, 1);
        assert_eq!(stats.cancelled_rotations, 1);
        assert!(state.pending_operations.is_empty());
    }

    #[test]
    fn preserved_prefix_blocks_cross_detector_fusion() {
        let mut state = state_with(
            1,
            vec![
                signed_rotation(pauli_x(1, 0), 0.1, SymbolicBool::default()),
                signed_rotation(pauli_x(1, 0), 0.2, SymbolicBool::default()),
            ],
        );
        let stats = optimize_pending_operations(&mut state, &[1]).expect("not planned");
        assert_eq!(stats.fused_rotations, 0);
        assert_eq!(stats.prefix_remap, vec![0, 1, 2]);
    }

    #[test]
    fn detector_prefix_stops_measurement_movement() {
        let mut state = state_with(
            1,
            vec![
                signed_rotation(pauli_x(1, 0), 0.1, SymbolicBool::default()),
                measurement(pauli_x(1, 0), 1, 2),
            ],
        );
        let stats = optimize_pending_operations(&mut state, &[1]).expect("not planned");
        assert_eq!(stats.measurement_left_swaps, 0);
        assert!(matches!(
            state.pending_operations[0],
            PendingOperation::PauliRotation(_)
        ));
    }

    #[test]
    fn measurement_movement_preserves_classical_record_order() {
        let mut state = state_with(
            1,
            vec![
                signed_rotation(pauli_x(1, 0), 0.1, SymbolicBool::default()),
                PendingClassicalRecord {
                    outcome: SymbolicBool::default(),
                    record: Some(1),
                    record_condition: None,
                }
                .into(),
                measurement(pauli_x(1, 0), 2, 3),
            ],
        );
        let stats = optimize_pending_operations(&mut state, &[]).expect("not planned");
        assert_eq!(stats.measurement_left_swaps, 0);
        assert!(matches!(
            state.pending_operations[1],
            PendingOperation::ClassicalRecord(_)
        ));
    }
}
