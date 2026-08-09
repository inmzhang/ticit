//! Plan-time factorization of the active state into independent tensor
//! components.
//!
//! # The idea
//!
//! A dense $2^k$ amplitude vector is wasteful when the active qubits are not
//! actually entangled with each other. This pass simulates *connectivity* only:
//! it walks the planned instruction stream tracking which coordinates have been
//! linked by some operation, keeps each connected group as its own small vector,
//! and merges groups only when an instruction spans them.
//!
//! Nothing here is required for correctness — the runtime can always ignore the
//! plan and run dense. It is enabled only when a deliberately conservative cost
//! model says it pays off, because the per-instruction dispatch overhead is real
//! and small active states are already cache-resident.
//!
//! # Cost model
//!
//! Dense work is $2^k$ per rotation and $2 \cdot 2^k$ per promotion or measurement.
//! Component work is the merges it has to perform ($2 \cdot 2^{k\_\mathrm{merged}}$
//! each) plus $2^{k\_\mathrm{component}}$ for the operation itself plus a fixed dispatch charge. Four
//! gates then have to pass before [`ActiveComponentPlan::selected`] is set: a
//! wide enough active state, enough quantum instructions, a large absolute
//! *and* relative saving, and a cap on the extra allocation.

use crate::active::{
    ActivePauliAction, PrecomputedActivePauliMeasurementKernel,
    PrecomputedActivePauliRotationKernel, active_length,
};
use crate::bits::is_odd_popcount;
use crate::errors::{Result, TicitError};
use crate::factored::{FactoredInstruction, FactoredInstructionProgram};

/// Fixed per-instruction cost charged to the component path, in units of one
/// amplitude update. Pays for the dispatch and merge bookkeeping.
const COMPONENT_DISPATCH_WORK: f64 = 32.0;
/// Absolute floor on the predicted saving, so tiny wins do not pay the dispatch.
const MINIMUM_SAVED_WORK: f64 = 8192.0;
/// Dense work must exceed component work by at least this factor.
const REQUIRED_WORK_RATIO: f64 = 1.8;
/// Components may allocate at most this multiple of the dense peak.
const MAXIMUM_ALLOCATION_RATIO: f64 = 1.25;

/// What the runtime should do for one planned instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ActiveComponentStepKind {
    /// Not part of the component path; run it densely.
    #[default]
    None,
    /// A rotation about the identity on a `k == 0` component: pure global phase.
    IgnoredGlobalPhase,
    Rotation,
    Promotion,
    Measurement,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActiveComponentStepRef {
    pub kind: ActiveComponentStepKind,
    pub payload: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveComponentRotationStep {
    pub component: usize,
    /// Slice of [`ActiveComponentPlan::merge_components`] to merge first; its
    /// first entry is the target the rest merge into.
    pub merge_offset: usize,
    pub merge_count: usize,
    pub kernel: PrecomputedActivePauliRotationKernel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActiveComponentPromotionStep {
    pub component: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveComponentMeasurementStep {
    pub component: usize,
    pub merge_offset: usize,
    pub merge_count: usize,
    pub kernel: PrecomputedActivePauliMeasurementKernel,
    /// The measurement consumed the component's last coordinate.
    pub deactivate_after: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ActiveComponentPlan {
    pub selected: bool,
    pub initial_components: usize,
    pub component_count: usize,
    pub component_max_k: Vec<usize>,
    pub instruction_steps: Vec<ActiveComponentStepRef>,
    pub merge_components: Vec<usize>,
    pub rotations: Vec<ActiveComponentRotationStep>,
    pub promotions: Vec<ActiveComponentPromotionStep>,
    pub measurements: Vec<ActiveComponentMeasurementStep>,

    pub estimated_dense_work: f64,
    pub estimated_component_work: f64,
    pub dense_peak_dimension: usize,
    pub component_peak_live_dimension: usize,
    pub component_allocated_dimension: usize,
}

/// One connected group of active coordinates during the simulation.
#[derive(Clone, Debug)]
struct PlanningComponent {
    active: bool,
    coordinates: Vec<usize>,
}

fn live_component_dimension(components: &[PlanningComponent]) -> Result<usize> {
    let mut total = 0usize;
    for component in components {
        if component.active && !component.coordinates.is_empty() {
            total = total.saturating_add(active_length(component.coordinates.len())?);
        }
    }
    Ok(total)
}

/// Rebuilds the coordinate-to-global-position map after the active order moved.
fn refresh_coordinate_positions(
    active_order: &[usize],
    coordinate_positions: &mut [Option<usize>],
) -> Result<()> {
    coordinate_positions.fill(None);
    for (position, &coordinate) in active_order.iter().enumerate() {
        if coordinate >= coordinate_positions.len() {
            return Err(TicitError::new(
                "component planner encountered an invalid active coordinate",
            ));
        }
        coordinate_positions[coordinate] = Some(position);
    }
    Ok(())
}

/// Rewrites a global-coordinate action into one component's local bit order.
fn remap_action(
    action: &ActivePauliAction,
    component_coordinates: &[usize],
    coordinate_positions: &[Option<usize>],
) -> Result<ActivePauliAction> {
    let mut local = *action;
    local.nqubits = component_coordinates.len();
    local.xmask = 0;
    local.zmask = 0;
    for (local_position, &coordinate) in component_coordinates.iter().enumerate() {
        let Some(global_position) = coordinate_positions[coordinate] else {
            return Err(TicitError::new(
                "component planner tried to remap an inactive coordinate",
            ));
        };
        let global_bit = 1u64 << global_position;
        let local_bit = 1u64 << local_position;
        if action.xmask & global_bit != 0 {
            local.xmask |= local_bit;
        }
        if action.zmask & global_bit != 0 {
            local.zmask |= local_bit;
        }
    }
    local.xz_overlap_odd = is_odd_popcount(local.xmask & local.zmask);
    Ok(local)
}

/// Components an action's support reaches, in first-touched order.
fn touched_components(
    action: &ActivePauliAction,
    active_order: &[usize],
    coordinate_components: &[Option<usize>],
) -> Result<Vec<usize>> {
    let mut touched = Vec::new();
    let support = action.xmask | action.zmask;
    for (position, &coordinate) in active_order.iter().enumerate() {
        if support & (1u64 << position) == 0 {
            continue;
        }
        let Some(component) = coordinate_components[coordinate] else {
            return Err(TicitError::new(
                "component planner found an active coordinate without a component",
            ));
        };
        if !touched.contains(&component) {
            touched.push(component);
        }
    }
    // An identity action is only a global phase, but pinning it to some live
    // component keeps the dense and factored states literally equal rather than
    // equal-up-to-phase.
    if touched.is_empty()
        && let Some(component) = active_order
            .first()
            .and_then(|&first| coordinate_components[first])
    {
        touched.push(component);
    }
    Ok(touched)
}

/// Merging into the largest component moves the least data; ties go to the
/// lowest index, which keeps the choice deterministic.
fn select_merge_target(touched: &[usize], components: &[PlanningComponent]) -> Option<usize> {
    touched.iter().copied().reduce(|best, candidate| {
        let best_size = components[best].coordinates.len();
        let candidate_size = components[candidate].coordinates.len();
        let take_candidate = if candidate_size != best_size {
            best_size < candidate_size
        } else {
            best > candidate
        };
        if take_candidate { candidate } else { best }
    })
}

/// The merge list for one instruction: target first, then the rest ascending.
fn ordered_merge_sources(target: usize, mut touched: Vec<usize>) -> Vec<usize> {
    touched.sort_unstable();
    touched.dedup();
    touched.retain(|&component| component != target);
    touched.insert(0, target);
    touched
}

/// Folds `sources[1..]` into `sources[0]`, returning the amplitude work.
fn merge_planning_components(
    sources: &[usize],
    components: &mut [PlanningComponent],
    coordinate_components: &mut [Option<usize>],
    component_max_k: &mut [usize],
) -> Result<f64> {
    let Some((&target, rest)) = sources.split_first() else {
        return Ok(0.0);
    };
    if !components[target].active {
        return Err(TicitError::new("component merge target is inactive"));
    }
    let mut work = 0.0;
    let mut target_k = components[target].coordinates.len();
    for &source in rest {
        if !components[source].active {
            return Err(TicitError::new("component merge source is inactive"));
        }
        target_k += components[source].coordinates.len();
        // Every merge rewrites the whole grown component: read plus write.
        work += 2.0 * (target_k as f64).exp2();
        let moved = std::mem::take(&mut components[source].coordinates);
        for &coordinate in &moved {
            coordinate_components[coordinate] = Some(target);
        }
        components[target].coordinates.extend_from_slice(&moved);
        components[source].active = false;
    }
    component_max_k[target] = component_max_k[target].max(target_k);
    Ok(work)
}

fn append_merge_sources(plan: &mut ActiveComponentPlan, sources: &[usize]) -> (usize, usize) {
    let offset = plan.merge_components.len();
    plan.merge_components.extend_from_slice(sources);
    (offset, sources.len())
}

fn update_peak_live_dimension(
    plan: &mut ActiveComponentPlan,
    components: &[PlanningComponent],
) -> Result<()> {
    plan.component_peak_live_dimension = plan
        .component_peak_live_dimension
        .max(live_component_dimension(components)?);
    Ok(())
}

/// All four gates must pass before the runtime is allowed to use components.
///
/// The comparisons are written negated on purpose: a NaN work estimate must
/// fail the gate, which `!(a >= b)` gives and `a < b` does not.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn should_select_component_plan(
    program: &FactoredInstructionProgram,
    plan: &ActiveComponentPlan,
    quantum_instruction_count: usize,
) -> bool {
    if program.max_k < 8 || quantum_instruction_count < 4 {
        return false;
    }
    if !(plan.estimated_dense_work >= REQUIRED_WORK_RATIO * plan.estimated_component_work) {
        return false;
    }
    if !(plan.estimated_dense_work - plan.estimated_component_work >= MINIMUM_SAVED_WORK) {
        return false;
    }
    let allocation_limit = MAXIMUM_ALLOCATION_RATIO * plan.dense_peak_dimension as f64;
    plan.component_allocated_dimension as f64 <= allocation_limit
}

/// Simulates component connectivity over the planned instruction stream.
pub fn build_active_component_plan(
    program: &FactoredInstructionProgram,
) -> Result<ActiveComponentPlan> {
    let mut plan = ActiveComponentPlan {
        initial_components: program.initial_k,
        dense_peak_dimension: active_length(program.max_k)?,
        ..ActiveComponentPlan::default()
    };
    // Small active states are already cache-resident and cannot repay dispatch;
    // expectation probes read the whole state, which components cannot serve.
    if program.max_k < 8 || program.nexpvals != 0 {
        return Ok(plan);
    }
    plan.instruction_steps = vec![ActiveComponentStepRef::default(); program.instructions.len()];

    let mut components: Vec<PlanningComponent> = Vec::new();
    let mut active_order: Vec<usize> = Vec::new();
    let mut coordinate_components: Vec<Option<usize>> = Vec::new();
    let mut coordinate_positions: Vec<Option<usize>> = Vec::new();

    for coordinate in 0..program.initial_k {
        components.push(PlanningComponent {
            active: true,
            coordinates: vec![coordinate],
        });
        active_order.push(coordinate);
        coordinate_components.push(Some(components.len() - 1));
        coordinate_positions.push(Some(coordinate));
        plan.component_max_k.push(1);
    }
    update_peak_live_dimension(&mut plan, &components)?;

    let mut quantum_instruction_count = 0;
    for (instruction_index, instruction) in program.instructions.iter().enumerate() {
        let global_k = active_order.len();
        refresh_coordinate_positions(&active_order, &mut coordinate_positions)?;
        let dense_dim = (global_k as f64).exp2();

        match instruction {
            FactoredInstruction::ApplyPrecomputedActivePauliRotation(rotation) => {
                if rotation.rotation_kernel.action.nqubits != global_k {
                    return Err(TicitError::new(
                        "component planner saw a rotation with the wrong active width",
                    ));
                }
                quantum_instruction_count += 1;
                plan.estimated_dense_work += dense_dim;
                let touched = touched_components(
                    &rotation.rotation_kernel.action,
                    &active_order,
                    &coordinate_components,
                )?;
                let Some(target) = select_merge_target(&touched, &components) else {
                    // A phase on the k == 0 scalar has no observable effect.
                    plan.instruction_steps[instruction_index] = ActiveComponentStepRef {
                        kind: ActiveComponentStepKind::IgnoredGlobalPhase,
                        payload: 0,
                    };
                    continue;
                };
                let sources = ordered_merge_sources(target, touched);
                plan.estimated_component_work += merge_planning_components(
                    &sources,
                    &mut components,
                    &mut coordinate_components,
                    &mut plan.component_max_k,
                )?;
                let (merge_offset, merge_count) = append_merge_sources(&mut plan, &sources);
                let local_action = remap_action(
                    &rotation.rotation_kernel.action,
                    &components[target].coordinates,
                    &coordinate_positions,
                )?;
                plan.rotations.push(ActiveComponentRotationStep {
                    component: target,
                    merge_offset,
                    merge_count,
                    kernel: PrecomputedActivePauliRotationKernel::new(
                        &local_action,
                        rotation.rotation_kernel.kernel_angle,
                    )?,
                });
                plan.instruction_steps[instruction_index] = ActiveComponentStepRef {
                    kind: ActiveComponentStepKind::Rotation,
                    payload: plan.rotations.len() - 1,
                };
                plan.estimated_component_work +=
                    (components[target].coordinates.len() as f64).exp2() + COMPONENT_DISPATCH_WORK;
                update_peak_live_dimension(&mut plan, &components)?;
            }

            FactoredInstruction::PromoteDormantRotation(_) => {
                quantum_instruction_count += 1;
                plan.estimated_dense_work += 2.0 * dense_dim;
                // A promoted qubit starts unentangled, so it is its own
                // component until something merges it.
                let coordinate = coordinate_components.len();
                let component = components.len();
                components.push(PlanningComponent {
                    active: true,
                    coordinates: vec![coordinate],
                });
                active_order.push(coordinate);
                coordinate_components.push(Some(component));
                coordinate_positions.push(Some(active_order.len() - 1));
                plan.component_max_k.push(1);
                plan.promotions
                    .push(ActiveComponentPromotionStep { component });
                plan.instruction_steps[instruction_index] = ActiveComponentStepRef {
                    kind: ActiveComponentStepKind::Promotion,
                    payload: plan.promotions.len() - 1,
                };
                plan.estimated_component_work += 2.0 + COMPONENT_DISPATCH_WORK;
                update_peak_live_dimension(&mut plan, &components)?;
            }

            FactoredInstruction::MeasurePrecomputedActivePauli(measurement) => {
                if measurement.exp_val.is_some() {
                    // Expectations are read-only barriers; leave them dense so
                    // no component coordinates get collapsed.
                    plan.instruction_steps[instruction_index] = ActiveComponentStepRef {
                        kind: ActiveComponentStepKind::None,
                        payload: 0,
                    };
                    continue;
                }
                if measurement.kernel.action.nqubits != global_k {
                    return Err(TicitError::new(
                        "component planner saw a measurement with the wrong active width",
                    ));
                }
                quantum_instruction_count += 1;
                plan.estimated_dense_work += 2.0 * dense_dim;
                let touched = touched_components(
                    &measurement.kernel.action,
                    &active_order,
                    &coordinate_components,
                )?;
                if touched.is_empty() {
                    return Err(TicitError::new(
                        "active measurement has no component support",
                    ));
                }
                let target = select_merge_target(&touched, &components)
                    .expect("a non-empty touched set has a target");
                let sources = ordered_merge_sources(target, touched);
                plan.estimated_component_work += merge_planning_components(
                    &sources,
                    &mut components,
                    &mut coordinate_components,
                    &mut plan.component_max_k,
                )?;
                update_peak_live_dimension(&mut plan, &components)?;
                let (merge_offset, merge_count) = append_merge_sources(&mut plan, &sources);
                refresh_coordinate_positions(&active_order, &mut coordinate_positions)?;
                let local_action = remap_action(
                    &measurement.kernel.action,
                    &components[target].coordinates,
                    &coordinate_positions,
                )?;
                // The globally highest pivot need not be the component's local
                // highest, so the local pivot is passed explicitly.
                let global_pivot = measurement.kernel.pivot;
                if global_pivot >= active_order.len() {
                    return Err(TicitError::new(
                        "component planner saw an invalid measurement pivot",
                    ));
                }
                let pivot_coordinate = active_order[global_pivot];
                let Some(local_pivot) = components[target]
                    .coordinates
                    .iter()
                    .position(|&coordinate| coordinate == pivot_coordinate)
                else {
                    return Err(TicitError::new(
                        "component measurement pivot is not in its target component",
                    ));
                };
                let kernel = PrecomputedActivePauliMeasurementKernel::with_pivot(
                    &local_action,
                    local_pivot,
                )?;
                plan.estimated_component_work += 2.0
                    * (components[target].coordinates.len() as f64).exp2()
                    + COMPONENT_DISPATCH_WORK;

                components[target].coordinates.remove(local_pivot);
                active_order.remove(global_pivot);
                coordinate_components[pivot_coordinate] = None;
                let deactivate_after = components[target].coordinates.is_empty();
                if deactivate_after {
                    components[target].active = false;
                }
                plan.measurements.push(ActiveComponentMeasurementStep {
                    component: target,
                    merge_offset,
                    merge_count,
                    kernel,
                    deactivate_after,
                });
                plan.instruction_steps[instruction_index] = ActiveComponentStepRef {
                    kind: ActiveComponentStepKind::Measurement,
                    payload: plan.measurements.len() - 1,
                };
                update_peak_live_dimension(&mut plan, &components)?;
            }

            FactoredInstruction::RecordMeasurement(_)
            | FactoredInstruction::RecordDetector(_)
            | FactoredInstruction::IntroduceDormantMeasurementBranch(_) => {}
        }
    }

    plan.component_count = components.len();
    for &max_k in &plan.component_max_k {
        plan.component_allocated_dimension = plan
            .component_allocated_dimension
            .saturating_add(active_length(max_k)?);
    }
    plan.selected = should_select_component_plan(program, &plan, quantum_instruction_count);
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factored::{
        ApplyPrecomputedActivePauliRotation, MeasurePrecomputedActivePauli, PromoteDormantRotation,
        RecordDetector,
    };
    use crate::pauli::PauliString;
    use crate::symbolic::{SymbolicBool, SymbolicBoolEvaluationPlan, symbolic_bool};

    fn rotation_instruction(pauli: PauliString, angle: f64) -> FactoredInstruction {
        let action = ActivePauliAction::new(&pauli).expect("Hermitian generator");
        let sign = SymbolicBool::from(false);
        ApplyPrecomputedActivePauliRotation {
            rotation_kernel: PrecomputedActivePauliRotationKernel::new(&action, angle)
                .expect("k < 62"),
            sign_plan: SymbolicBoolEvaluationPlan::new(&sign),
            sign,
        }
        .into()
    }

    fn promotion_instruction(angle: f64) -> FactoredInstruction {
        let sign = SymbolicBool::from(false);
        PromoteDormantRotation {
            kernel_angle: angle,
            sign_plan: SymbolicBoolEvaluationPlan::new(&sign),
            sign,
        }
        .into()
    }

    fn measurement_instruction(
        pauli: PauliString,
        branch: i32,
        record: i32,
    ) -> FactoredInstruction {
        let outcome = symbolic_bool(branch);
        MeasurePrecomputedActivePauli {
            kernel: PrecomputedActivePauliMeasurementKernel::from_pauli(&pauli)
                .expect("non-identity Pauli"),
            branch,
            outcome_plan: SymbolicBoolEvaluationPlan::new(&outcome),
            outcome,
            record: Some(record),
            record_condition: None,
            exp_val: None,
        }
        .into()
    }

    fn component_test_program() -> FactoredInstructionProgram {
        let mut merge_late = PauliString::new(3);
        merge_late.set_xbit(1, true);
        merge_late.set_xbit(2, true);

        let mut merge_early_into_late = PauliString::new(3);
        merge_early_into_late.set_xbit(0, true);
        merge_early_into_late.set_xbit(1, true);

        let mut reversed_single = PauliString::new(2);
        reversed_single.set_zbit(1, true);
        let mut final_x = PauliString::new(1);
        final_x.set_xbit(0, true);

        let detector_sign = SymbolicBool::from(false);
        let instructions = vec![
            rotation_instruction(PauliString::new(0), 0.19),
            promotion_instruction(0.17),
            promotion_instruction(-0.29),
            promotion_instruction(0.41),
            rotation_instruction(merge_late, 0.23),
            rotation_instruction(merge_early_into_late.clone(), -0.31),
            measurement_instruction(merge_early_into_late, 1, 1),
            RecordDetector {
                outcome_plan: SymbolicBoolEvaluationPlan::new(&detector_sign),
                outcome: detector_sign,
                records: vec![1],
                detector: 1,
                postselect: false,
            }
            .into(),
            measurement_instruction(reversed_single, 2, 2),
            measurement_instruction(final_x, 3, 3),
        ];
        FactoredInstructionProgram::new(8, 0, instructions, 8).expect("valid dimensions")
    }

    #[test]
    fn small_active_states_are_never_factored() {
        let program = FactoredInstructionProgram::new(4, 4, Vec::new(), 4).expect("valid dims");
        let plan = build_active_component_plan(&program).expect("k < 62");
        assert!(!plan.selected);
        assert!(plan.instruction_steps.is_empty());
        assert_eq!(plan.dense_peak_dimension, 16);
    }

    #[test]
    fn merge_target_is_the_largest_component() {
        let components = vec![
            PlanningComponent {
                active: true,
                coordinates: vec![0],
            },
            PlanningComponent {
                active: true,
                coordinates: vec![1, 2],
            },
            PlanningComponent {
                active: true,
                coordinates: vec![3],
            },
        ];
        assert_eq!(select_merge_target(&[0, 1, 2], &components), Some(1));
        // Ties break towards the lowest index.
        assert_eq!(select_merge_target(&[2, 0], &components), Some(0));
        assert_eq!(select_merge_target(&[], &components), None);
    }

    #[test]
    fn merge_sources_put_the_target_first() {
        assert_eq!(ordered_merge_sources(2, vec![3, 1, 2, 1]), vec![2, 1, 3]);
    }

    #[test]
    fn component_planner_tracks_merges_and_local_pivots() {
        let program = component_test_program();
        let plan = build_active_component_plan(&program).expect("k < 62");

        assert_eq!(plan.initial_components, 0);
        assert_eq!(plan.promotions.len(), 3);
        assert_eq!(plan.rotations.len(), 2);
        assert_eq!(plan.measurements.len(), 3);
        assert_eq!(
            plan.instruction_steps[0].kind,
            ActiveComponentStepKind::IgnoredGlobalPhase
        );
        assert_eq!(
            plan.instruction_steps[7].kind,
            ActiveComponentStepKind::None
        );

        let merge_late = &plan.rotations[0];
        assert_eq!(merge_late.component, 1);
        assert_eq!(
            &plan.merge_components[merge_late.merge_offset..][..merge_late.merge_count],
            &[1, 2]
        );
        let merge_early = &plan.rotations[1];
        assert_eq!(merge_early.component, 1);
        assert_eq!(
            &plan.merge_components[merge_early.merge_offset..][..merge_early.merge_count],
            &[1, 0]
        );

        let first_measurement = &plan.measurements[0];
        assert_eq!(first_measurement.component, 1);
        assert_eq!(
            &plan.merge_components[first_measurement.merge_offset..]
                [..first_measurement.merge_count],
            &[1]
        );
        assert_eq!(first_measurement.kernel.pivot, 0);
        assert!(plan.measurements[2].deactivate_after);
        assert_eq!(plan.component_max_k, vec![1, 3, 1]);
        assert!(!plan.selected);

        let mut component = program.clone();
        component.use_active_components = true;
        let mut dense = program;
        dense.use_active_components = false;
        assert_eq!(
            crate::sampler::batch::sample_measurements_batch(&component, 256, 0, 43)
                .expect("component samples"),
            crate::sampler::batch::sample_measurements_batch(&dense, 256, 0, 43)
                .expect("dense samples")
        );
    }

    #[test]
    fn fully_separable_program_selects_the_component_plan() {
        let k = 12;
        let instructions: Vec<FactoredInstruction> = (0..40)
            .map(|step| {
                let mut pauli = PauliString::new(k);
                pauli.set_zbit(step % k, true);
                rotation_instruction(pauli, 0.05 * (step as f64 + 1.0))
            })
            .collect();
        let program =
            FactoredInstructionProgram::new(k, k, instructions, k).expect("valid dimensions");
        let plan = build_active_component_plan(&program).expect("k < 62");

        assert!(plan.selected);
        assert_eq!(plan.initial_components, k);
        assert_eq!(plan.component_max_k, vec![1; k]);
        assert_eq!(plan.dense_peak_dimension, 1 << k);
        assert_eq!(plan.component_allocated_dimension, 2 * k);
        assert!(plan.estimated_dense_work > plan.estimated_component_work);
        assert!(program.use_active_components);
    }

    #[test]
    fn expectation_probes_disable_the_component_plan() {
        let mut program = component_test_program();
        program.nexpvals = 1;
        let plan = build_active_component_plan(&program).expect("k < 62");
        assert!(plan.instruction_steps.is_empty());
        assert!(!plan.selected);
    }
}
