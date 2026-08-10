//! Lowering from ticit's factored program to GPU kernel metadata.

use std::collections::HashMap;

use crate::bits::{packed_bit, set_packed_bit};
use crate::factored::{FactoredInstruction, FactoredInstructionProgram, RecordMeasurement};
use crate::sampler::exogenous::exogenous_assigned_words;
use crate::sampler::presampled_expression::{
    PresampledExpressionPlan, prepare_presampled_expression_plan_from_words,
};
use crate::symbolic::{SymbolicBool, symbolic_bool, xor_bool};
use anyhow::{Result, bail, ensure};

pub const META_WORDS: usize = 12;
pub const PARAM_WORDS: usize = 4;
pub const CONTROL_WORDS: usize = 1;
pub const MAX_BRANCHES: usize = 256;
pub const MAX_GPU_K: usize = 28;

pub const OP_ROTATE: u64 = 0;
pub const OP_PROMOTE: u64 = 1;
pub const OP_MEASURE: u64 = 2;
pub const OP_DORMANT_BRANCH: u64 = 3;
pub const OP_DETECTOR: u64 = 4;
pub const OP_EXPECTATION_RECORD: u64 = 5;
pub const OP_EXPECTATION_ACTIVE: u64 = 6;

#[derive(Clone, Debug)]
pub struct GpuExpression {
    pub block: usize,
    pub branch_masks: [u64; 4],
}

#[derive(Clone, Debug)]
pub struct GpuInstruction {
    pub opcode: u64,
    pub expression: GpuExpression,
    pub xmask: u64,
    pub zmask: u64,
    pub pivot: usize,
    pub diagonal_phase: bool,
    pub z_without_pivot: u64,
    pub branch: Option<usize>,
    pub expectation: Option<usize>,
    pub detector: Option<usize>,
    pub params: [f32; PARAM_WORDS],
}

#[derive(Clone, Debug)]
pub struct GpuPlan {
    pub instructions: Vec<GpuInstruction>,
    pub detector_start: usize,
    pub logical: GpuExpression,
    pub expression_plan: PresampledExpressionPlan,
    pub exogenous_plan: GpuExogenousPlan,
    pub branch_count: usize,
}

#[derive(Clone, Debug, Default)]
pub struct GpuExogenousPlan {
    pub draw_count: usize,
    pub mask_words: usize,
    pub constant_masks: Vec<u64>,
    pub draw_transition_offsets: Vec<i32>,
    pub draw_base_masks: Vec<u64>,
    pub transition_upper: Vec<f32>,
    pub transition_masks: Vec<u64>,
    pub sparse_group_metadata: Vec<i32>,
    pub sparse_group_keys: Vec<u64>,
    pub sparse_gap_thresholds: Vec<u64>,
    pub sparse_transition_upper: Vec<f32>,
    pub sparse_base_masks: Vec<u64>,
    pub sparse_transition_masks: Vec<u64>,
}

#[derive(Clone, Debug)]
struct DrawSource {
    conditions: Vec<i32>,
    assignments: Vec<Vec<u64>>,
    probabilities: Vec<f64>,
}

#[derive(Clone, Debug)]
struct SparseGroupSource {
    event_probability: f64,
    outcome_probabilities: Vec<f64>,
    outcome_masks: Vec<Vec<Vec<u64>>>,
}

fn remap_active_mask(mask: u64, physical_bits: &[usize]) -> Result<u64> {
    ensure!(
        mask >> physical_bits.len() == 0,
        "active mask {mask:#x} exceeds k {}",
        physical_bits.len()
    );
    Ok(physical_bits
        .iter()
        .enumerate()
        .fold(0u64, |mapped, (logical, &physical)| {
            mapped | (((mask >> logical) & 1) << physical)
        }))
}

// Selected rotations carry the active-slot mask in `pivot`; the measurement
// fields mark the first/last item because those fields are otherwise unused.
fn mark_x_basis_rotation_runs(instructions: &mut [GpuInstruction]) {
    let mut begin = 0;
    while begin < instructions.len() {
        if instructions[begin].opcode != OP_ROTATE {
            begin += 1;
            continue;
        }
        if instructions[begin].zmask != 0 {
            instructions[begin].pivot = 0;
            begin += 1;
            continue;
        }
        let active_mask = instructions[begin].pivot as u64;
        let mut end = begin;
        let mut last_rotation = begin;
        let mut basis_mask = 0u64;
        let mut xor_bits = 0;
        while end < instructions.len() {
            let instruction = &instructions[end];
            if instruction.opcode == OP_ROTATE
                && instruction.zmask == 0
                && instruction.pivot as u64 == active_mask
            {
                basis_mask |= instruction.xmask;
                xor_bits += instruction.xmask.count_ones();
                last_rotation = end;
            } else if instruction.opcode != OP_DORMANT_BRANCH && instruction.opcode != OP_DETECTOR {
                break;
            }
            end += 1;
        }
        let basis_bits = basis_mask.count_ones();
        if basis_bits != 0 && xor_bits > 2 * basis_bits {
            let scale = 2.0f32.powf(-0.5 * basis_bits as f32);
            for instruction in &mut instructions[begin..=last_rotation] {
                if instruction.opcode == OP_ROTATE {
                    instruction.pivot = basis_mask as usize;
                    instruction.params[3] = scale;
                }
            }
            instructions[begin].diagonal_phase = true;
            instructions[last_rotation].z_without_pivot = 1;
        } else {
            for instruction in &mut instructions[begin..=last_rotation] {
                if instruction.opcode == OP_ROTATE {
                    instruction.pivot = 0;
                }
            }
        }
        begin = end;
    }
}

impl GpuPlan {
    pub fn build(
        program: &FactoredInstructionProgram,
        logical_records: &[Vec<i32>],
    ) -> Result<Self> {
        ensure!(
            program.max_k <= MAX_GPU_K,
            "the GPU plan supports max_k <= {MAX_GPU_K}, got {}",
            program.max_k
        );
        let resident = program.max_k <= 12;

        let adaptive_branches = program
            .instructions
            .iter()
            .filter(|instruction| {
                instruction.exp_val().is_none()
                    && matches!(
                        instruction,
                        FactoredInstruction::MeasurePrecomputedActivePauli(_)
                            | FactoredInstruction::IntroduceDormantMeasurementBranch(_)
                    )
            })
            .count();
        let presampled_dormant_branches: Vec<i32> = if adaptive_branches > MAX_BRANCHES {
            program
                .instructions
                .iter()
                .filter_map(|instruction| match instruction {
                    FactoredInstruction::IntroduceDormantMeasurementBranch(inst)
                        if inst.exp_val.is_none() =>
                    {
                        Some(inst.branch)
                    }
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut assigned = exogenous_assigned_words(program)?;
        for &condition in &presampled_dormant_branches {
            ensure!(
                condition > 0 && condition as usize <= program.nsymbols,
                "dormant branch condition {condition} is out of range"
            );
            set_packed_bit(&mut assigned, condition as usize - 1, true);
        }
        let mut definitions = vec![None; program.nsymbols + 1];
        for (condition, definition) in definitions.iter_mut().enumerate().skip(1) {
            let bit = condition - 1;
            if assigned[bit >> 6] & (1u64 << (bit & 63)) != 0 {
                *definition = Some(symbolic_bool(condition as i32));
            }
        }

        let mut records = vec![None; program.nrecords + 1];
        let mut branch_slots = HashMap::new();
        let mut expressions = Vec::new();
        let mut instructions = Vec::new();
        // Measurements leave their physical state bit at zero. Reusing that
        // slot for a later promotion avoids permuting the whole state vector.
        let mut physical_bits: Vec<usize> = (0..program.initial_k).collect();
        let mut free_physical_bits: Vec<usize> = (program.initial_k..program.max_k).rev().collect();
        let mut active_k = program.initial_k;

        for instruction in &program.instructions {
            match instruction {
                FactoredInstruction::ApplyPrecomputedActivePauliRotation(inst) => {
                    let expression = expand(&inst.sign, &definitions)?;
                    let kernel = inst.rotation_kernel;
                    let (xmask, zmask, pivot) = if resident {
                        (
                            remap_active_mask(kernel.action.xmask, &physical_bits)?,
                            remap_active_mask(kernel.action.zmask, &physical_bits)?,
                            physical_bits
                                .iter()
                                .fold(0usize, |mask, &physical| mask | (1usize << physical)),
                        )
                    } else {
                        (
                            kernel.action.xmask,
                            kernel.action.zmask,
                            kernel.pair_bit as usize,
                        )
                    };
                    expressions.push(expression);
                    instructions.push(GpuInstruction {
                        opcode: OP_ROTATE,
                        expression: placeholder_expression(),
                        xmask,
                        zmask,
                        pivot,
                        diagonal_phase: false,
                        z_without_pivot: 0,
                        branch: None,
                        expectation: None,
                        detector: None,
                        params: [
                            kernel.cos_kernel_angle as f32,
                            kernel.minus_even_coefficient.re as f32,
                            kernel.minus_even_coefficient.im as f32,
                            0.0,
                        ],
                    });
                }
                FactoredInstruction::PromoteDormantRotation(inst) => {
                    let expression = expand(&inst.sign, &definitions)?;
                    let physical_bit = if resident {
                        let bit = free_physical_bits
                            .pop()
                            .ok_or_else(|| anyhow::anyhow!("GPU promotion exceeds max_k"))?;
                        physical_bits.push(bit);
                        bit
                    } else {
                        active_k
                    };
                    active_k += 1;
                    expressions.push(expression);
                    instructions.push(GpuInstruction {
                        opcode: OP_PROMOTE,
                        expression: placeholder_expression(),
                        xmask: 0,
                        zmask: 0,
                        pivot: physical_bit,
                        diagonal_phase: false,
                        z_without_pivot: 0,
                        branch: None,
                        expectation: None,
                        detector: None,
                        params: [
                            inst.kernel_angle.cos() as f32,
                            inst.kernel_angle.sin() as f32,
                            0.0,
                            0.0,
                        ],
                    });
                }
                FactoredInstruction::RecordMeasurement(inst) => {
                    let outcome = expand(&inst.outcome, &definitions)?;
                    if let Some(exp_val) = inst.exp_val {
                        ensure!(exp_val >= 0, "negative GPU expectation index");
                        expressions.push(outcome);
                        instructions.push(GpuInstruction {
                            opcode: OP_EXPECTATION_RECORD,
                            expression: placeholder_expression(),
                            xmask: 0,
                            zmask: 0,
                            pivot: 0,
                            diagonal_phase: false,
                            z_without_pivot: 0,
                            branch: None,
                            expectation: Some(exp_val as usize),
                            detector: None,
                            params: [0.0; PARAM_WORDS],
                        });
                    } else {
                        assign_record(&mut records, inst.record, &outcome)?;
                        assign_definition(&mut definitions, inst.record_condition, &outcome)?;
                    }
                }
                FactoredInstruction::RecordDetector(inst) => {
                    let outcome = if inst.records.is_empty() {
                        expand(&inst.outcome, &definitions)?
                    } else {
                        records_expression(&records, &inst.records)?
                    };
                    expressions.push(outcome);
                    instructions.push(GpuInstruction {
                        opcode: OP_DETECTOR,
                        expression: placeholder_expression(),
                        xmask: 0,
                        zmask: 0,
                        pivot: 0,
                        diagonal_phase: inst.postselect,
                        z_without_pivot: 0,
                        branch: None,
                        expectation: None,
                        detector: Some((inst.detector - 1) as usize),
                        params: [0.0; PARAM_WORDS],
                    });
                }
                FactoredInstruction::MeasurePrecomputedActivePauli(inst) => {
                    let expectation = inst
                        .exp_val
                        .map(|exp_val| {
                            ensure!(exp_val >= 0, "negative GPU expectation index");
                            Ok(exp_val as usize)
                        })
                        .transpose()?;
                    let branch = if expectation.is_none() {
                        Some(register_branch(
                            &mut branch_slots,
                            &mut definitions,
                            inst.branch,
                        )?)
                    } else {
                        None
                    };
                    let outcome = expand(&inst.outcome, &definitions)?;
                    if expectation.is_none() {
                        assign_record(&mut records, inst.record, &outcome)?;
                        assign_definition(&mut definitions, inst.record_condition, &outcome)?;
                    }
                    let kernel = inst.kernel;
                    let physical_pivot = if resident {
                        *physical_bits.get(kernel.pivot).ok_or_else(|| {
                            anyhow::anyhow!(
                                "GPU measurement pivot {} exceeds active k {}",
                                kernel.pivot,
                                physical_bits.len()
                            )
                        })?
                    } else {
                        kernel.pivot
                    };
                    let (xmask, zmask, z_without_pivot) = if resident {
                        (
                            remap_active_mask(kernel.action.xmask, &physical_bits)?,
                            remap_active_mask(kernel.action.zmask, &physical_bits)?,
                            remap_active_mask(kernel.z_without_pivot, &physical_bits)?,
                        )
                    } else {
                        (
                            kernel.action.xmask,
                            kernel.action.zmask,
                            kernel.z_without_pivot,
                        )
                    };
                    expressions.push(outcome);
                    instructions.push(GpuInstruction {
                        opcode: if expectation.is_some() {
                            OP_EXPECTATION_ACTIVE
                        } else {
                            OP_MEASURE
                        },
                        expression: placeholder_expression(),
                        xmask,
                        zmask,
                        pivot: physical_pivot,
                        diagonal_phase: kernel.diagonal_phase_bit != 0,
                        z_without_pivot,
                        branch,
                        expectation,
                        detector: None,
                        params: [
                            kernel.nondiagonal_coefficient1_even.re as f32,
                            kernel.nondiagonal_coefficient1_even.im as f32,
                            0.0,
                            0.0,
                        ],
                    });
                    if expectation.is_none() {
                        active_k -= 1;
                        if resident {
                            physical_bits.remove(kernel.pivot);
                            free_physical_bits.push(physical_pivot);
                        }
                    }
                }
                FactoredInstruction::IntroduceDormantMeasurementBranch(inst) => {
                    if inst.exp_val.is_none() {
                        let presampled = adaptive_branches > MAX_BRANCHES;
                        let branch = if presampled {
                            None
                        } else {
                            Some(register_branch(
                                &mut branch_slots,
                                &mut definitions,
                                inst.branch,
                            )?)
                        };
                        let outcome = expand(&inst.outcome, &definitions)?;
                        assign_record(&mut records, inst.record, &outcome)?;
                        assign_definition(&mut definitions, inst.record_condition, &outcome)?;
                        if let Some(branch) = branch {
                            expressions.push(outcome);
                            instructions.push(GpuInstruction {
                                opcode: OP_DORMANT_BRANCH,
                                expression: placeholder_expression(),
                                xmask: 0,
                                zmask: 0,
                                pivot: 0,
                                diagonal_phase: false,
                                z_without_pivot: 0,
                                branch: Some(branch),
                                expectation: None,
                                detector: None,
                                params: [0.0; PARAM_WORDS],
                            });
                        }
                    }
                }
            }
        }
        if resident {
            mark_x_basis_rotation_runs(&mut instructions);
        }

        let logical = logical_expression(&records, logical_records)?;
        expressions.push(logical);

        let mut expression_program = program.clone();
        expression_program.instructions = expressions
            .into_iter()
            .map(|outcome| {
                FactoredInstruction::RecordMeasurement(RecordMeasurement {
                    outcome_plan: crate::symbolic::SymbolicBoolEvaluationPlan::new(&outcome),
                    outcome,
                    ..RecordMeasurement::default()
                })
            })
            .collect();

        let mut expression_plan = PresampledExpressionPlan::default();
        prepare_presampled_expression_plan_from_words(
            &mut expression_plan,
            &expression_program,
            &assigned,
        );

        for (index, instruction) in instructions.iter_mut().enumerate() {
            instruction.expression = compile_expression(
                &expression_plan.instruction_expressions[index],
                &branch_slots,
            )?;
        }
        let detector_start = instructions.len();
        let logical = compile_expression(
            expression_plan
                .instruction_expressions
                .last()
                .expect("the logical expression was appended"),
            &branch_slots,
        )?;
        let exogenous_plan =
            GpuExogenousPlan::build(program, &expression_plan, &presampled_dormant_branches)?;

        Ok(Self {
            instructions,
            detector_start,
            logical,
            expression_plan,
            exogenous_plan,
            branch_count: branch_slots.len(),
        })
    }

    pub fn encode(&self) -> (Vec<u64>, Vec<f32>, Vec<i32>, Vec<i32>) {
        let mut metadata = Vec::with_capacity(self.instructions.len() * META_WORDS);
        let mut parameters = Vec::with_capacity(self.instructions.len() * PARAM_WORDS);
        let mut controls = Vec::with_capacity(self.instructions.len() * CONTROL_WORDS);
        let mut expectations = Vec::with_capacity(self.instructions.len());
        for instruction in &self.instructions {
            let branch_or_expression = instruction
                .branch
                .map_or((instruction.expression.block >> 6) as u64, |slot| {
                    slot as u64
                });
            let branch_bit = instruction.branch.map_or_else(
                || 1u64 << (instruction.expression.block & 63),
                |slot| 1u64 << (slot & 63),
            );
            metadata.extend_from_slice(&[
                instruction.opcode,
                instruction.expression.branch_masks[0],
                instruction.expression.branch_masks[1],
                instruction.expression.branch_masks[2],
                instruction.expression.branch_masks[3],
                instruction.xmask,
                instruction.zmask,
                instruction.pivot as u64,
                instruction.diagonal_phase as u64,
                instruction.z_without_pivot,
                branch_or_expression,
                branch_bit,
            ]);
            parameters.extend_from_slice(&instruction.params);
            controls.push(
                instruction
                    .branch
                    .map_or((instruction.expression.block >> 6) as i32, |slot| {
                        slot as i32
                    }),
            );
            expectations.push(
                instruction
                    .expectation
                    .or(instruction.detector)
                    .map_or(-1, |index| index as i32),
            );
        }
        (metadata, parameters, controls, expectations)
    }
}

impl GpuExogenousPlan {
    fn build(
        program: &FactoredInstructionProgram,
        expressions: &PresampledExpressionPlan,
        presampled_dormant_branches: &[i32],
    ) -> Result<Self> {
        let mask_words = expressions.block_expressions.len().div_ceil(64).max(1);
        let mut condition_masks = vec![vec![0u64; mask_words]; program.nsymbols + 1];
        let mut constant_masks = vec![0u64; mask_words];
        for (expression_index, expression) in expressions.block_expressions.iter().enumerate() {
            let word = expression_index >> 6;
            let expression_mask = 1u64 << (expression_index & 63);
            if expression.constant {
                constant_masks[word] ^= expression_mask;
            }
            for &condition in &expression.exogenous_conditions {
                ensure!(
                    condition > 0,
                    "exogenous condition {condition} is not positive"
                );
                let mask = condition_masks.get_mut(condition as usize).ok_or_else(|| {
                    anyhow::anyhow!("exogenous condition {condition} is out of range")
                })?;
                mask[word] ^= expression_mask;
            }
        }

        let mut assigned = vec![false; program.nsymbols + 1];
        let mut draws = Vec::new();
        let mut sparse_groups = Vec::new();

        for distribution in &program.sampled_categorical_distributions {
            register_draw(
                &mut draws,
                &mut assigned,
                &distribution.conditions,
                &distribution.assignments,
                &distribution.probabilities,
            )?;
        }
        for group in &program.sampled_rare_categorical_groups {
            ensure!(
                group.event_probability.is_finite()
                    && (0.0..=1.0).contains(&group.event_probability),
                "invalid rare categorical event probability {}",
                group.event_probability
            );
            ensure!(
                group.event_rows.len() == group.event_probabilities.len()
                    && (!group.event_rows.is_empty() || group.event_probability == 0.0),
                "rare categorical group has mismatched event rows"
            );
            let event_total: f64 = group.event_probabilities.iter().sum();
            ensure!(
                group
                    .event_probabilities
                    .iter()
                    .all(|probability| probability.is_finite() && *probability >= 0.0)
                    && (group.event_probability == 0.0 || (event_total - 1.0).abs() <= 1.0e-9),
                "rare categorical event probabilities do not sum to one"
            );
            let mut outcomes: Vec<usize> = (0..group.event_rows.len()).collect();
            outcomes.sort_by(|&lhs, &rhs| {
                group.event_probabilities[lhs].total_cmp(&group.event_probabilities[rhs])
            });
            let mut outcome_masks = Vec::new();
            for conditions in &group.conditions {
                ensure!(
                    conditions.len() == group.nbits,
                    "rare categorical condition count does not match nbits"
                );
                register_conditions(&mut assigned, conditions)?;
                let masks: Vec<Vec<u64>> = outcomes
                    .iter()
                    .map(|&outcome| {
                        let row = group.event_rows[outcome];
                        let assignment = group.assignments.get(row).ok_or_else(|| {
                            anyhow::anyhow!("rare categorical event row is out of range")
                        })?;
                        assignment_mask(conditions, assignment, &condition_masks)
                    })
                    .collect::<Result<_>>()?;
                if masks.iter().flatten().any(|&mask| mask != 0) {
                    outcome_masks.push(masks);
                }
            }
            if group.event_probability > 0.0 && !outcome_masks.is_empty() {
                sparse_groups.push(SparseGroupSource {
                    event_probability: group.event_probability,
                    outcome_probabilities: outcomes
                        .iter()
                        .map(|&outcome| group.event_probabilities[outcome])
                        .collect(),
                    outcome_masks,
                });
            }
        }
        ensure!(
            program.sampled_bernoulli_conditions.len()
                == program.sampled_bernoulli_probabilities.len(),
            "Bernoulli sampling plan has mismatched conditions and probabilities"
        );
        for (&condition, &probability) in program
            .sampled_bernoulli_conditions
            .iter()
            .zip(&program.sampled_bernoulli_probabilities)
        {
            register_bernoulli_draw(&mut draws, &mut assigned, condition, probability)?;
        }
        for &condition in presampled_dormant_branches {
            register_bernoulli_draw(&mut draws, &mut assigned, condition, 0.5)?;
        }
        for group in &program.sampled_low_probability_bernoulli_groups {
            ensure!(
                group.probability.is_finite() && (0.0..=1.0).contains(&group.probability),
                "invalid low-probability Bernoulli probability {}",
                group.probability
            );
            let mut outcome_masks = Vec::new();
            for &condition in &group.conditions {
                register_conditions(&mut assigned, &[condition])?;
                let mask = condition_masks[condition as usize].clone();
                if mask.iter().any(|&word| word != 0) {
                    outcome_masks.push(vec![mask]);
                }
            }
            if group.probability > 0.0 && !outcome_masks.is_empty() {
                sparse_groups.push(SparseGroupSource {
                    event_probability: group.probability,
                    outcome_probabilities: vec![1.0],
                    outcome_masks,
                });
            }
        }

        for expression in &expressions.block_expressions {
            for &condition in &expression.exogenous_conditions {
                ensure!(
                    assigned[condition as usize],
                    "expression condition {condition} has no exogenous draw"
                );
            }
        }

        let mut out = Self {
            mask_words,
            constant_masks,
            draw_transition_offsets: vec![0],
            ..Self::default()
        };
        for source in &draws {
            let masks: Vec<Vec<u64>> = source
                .assignments
                .iter()
                .map(|assignment| assignment_mask(&source.conditions, assignment, &condition_masks))
                .collect::<Result<_>>()?;
            let mut probability_by_mask = HashMap::<Vec<u64>, f64>::new();
            for (mask, &probability) in masks.iter().zip(&source.probabilities) {
                *probability_by_mask.entry(mask.clone()).or_default() += probability;
            }
            let mut outcomes: Vec<_> = probability_by_mask.into_iter().collect();
            outcomes.sort_by(
                |(left_mask, left_probability), (right_mask, right_probability)| {
                    left_probability
                        .total_cmp(right_probability)
                        .then_with(|| left_mask.cmp(right_mask))
                },
            );
            if outcomes.len() == 1 {
                for (constant, mask) in out.constant_masks.iter_mut().zip(&outcomes[0].0) {
                    *constant ^= mask;
                }
                continue;
            }
            out.draw_base_masks
                .extend_from_slice(&outcomes.last().expect("draw has outcomes").0);
            let mut cumulative = 0.0;
            for outcome in 0..outcomes.len() - 1 {
                cumulative += outcomes[outcome].1;
                out.transition_upper.push(cumulative as f32);
                out.transition_masks.extend(
                    outcomes[outcome]
                        .0
                        .iter()
                        .zip(&outcomes[outcome + 1].0)
                        .map(|(left, right)| left ^ right),
                );
            }
            out.draw_count += 1;
            out.draw_transition_offsets
                .push(out.transition_upper.len() as i32);
        }
        for source in &sparse_groups {
            push_sparse_group(&mut out, source)?;
        }
        ensure!(
            out.draw_count <= i32::MAX as usize
                && out.transition_upper.len() <= i32::MAX as usize
                && out.sparse_group_keys.len() <= i32::MAX as usize
                && out.sparse_gap_thresholds.len() <= i32::MAX as usize
                && out.sparse_base_masks.len() <= i32::MAX as usize
                && out.sparse_transition_upper.len() <= i32::MAX as usize
                && out.sparse_transition_masks.len() <= i32::MAX as usize,
            "GPU exogenous mask plan exceeds i32 indexing"
        );
        Ok(out)
    }

    pub fn sparse_group_count(&self) -> usize {
        self.sparse_group_keys.len()
    }

    pub fn sparse_set_count(&self) -> usize {
        self.sparse_group_metadata
            .chunks_exact(7)
            .map(|metadata| metadata[1] as usize)
            .sum()
    }

    pub fn sparse_mask(&self, seed: u64, shot: u64, word: usize) -> u64 {
        let mut value = 0u64;
        for group in 0..self.sparse_group_count() {
            let metadata = &self.sparse_group_metadata[group * 7..group * 7 + 7];
            let base_offset = metadata[0] as usize;
            let set_count = metadata[1] as usize;
            let upper_offset = metadata[2] as usize;
            let transition_count = metadata[3] as usize;
            let transition_mask_offset = metadata[4] as usize;
            let gap_threshold_offset = metadata[5] as usize;
            let mut state = mix_u64(seed ^ self.sparse_group_keys[group] ^ shot);
            let mut set = 0usize;
            while set < set_count {
                let numerator = u64::from(next_sparse_bits(&mut state)) * 2 + 1;
                let thresholds = &self.sparse_gap_thresholds
                    [gap_threshold_offset..gap_threshold_offset + set_count];
                let gap = thresholds.partition_point(|&threshold| numerator <= threshold);
                let remaining = set_count - set;
                if gap >= remaining {
                    break;
                }
                set += gap;
                let outcome_uniform = next_sparse_uniform(&mut state);
                let mut mask = self.sparse_base_masks[(base_offset + set) * self.mask_words + word];
                for transition in 0..transition_count {
                    if outcome_uniform <= self.sparse_transition_upper[upper_offset + transition] {
                        mask ^= self.sparse_transition_masks[(transition_mask_offset
                            + set * transition_count
                            + transition)
                            * self.mask_words
                            + word];
                    }
                }
                value ^= mask;
                set += 1;
            }
        }
        value
    }
}

fn register_bernoulli_draw(
    draws: &mut Vec<DrawSource>,
    assigned: &mut [bool],
    condition: i32,
    probability: f64,
) -> Result<()> {
    ensure!(
        probability.is_finite() && (0.0..=1.0).contains(&probability),
        "invalid Bernoulli probability {probability}"
    );
    register_draw(
        draws,
        assigned,
        &[condition],
        &[vec![0], vec![1]],
        &[1.0 - probability, probability],
    )
}

fn register_draw(
    draws: &mut Vec<DrawSource>,
    assigned: &mut [bool],
    conditions: &[i32],
    assignments: &[Vec<u64>],
    probabilities: &[f64],
) -> Result<()> {
    ensure!(
        assignments.len() == probabilities.len() && !assignments.is_empty(),
        "categorical draw has mismatched or empty rows"
    );
    let total: f64 = probabilities.iter().sum();
    ensure!(
        probabilities
            .iter()
            .all(|probability| probability.is_finite() && *probability >= 0.0)
            && (total - 1.0).abs() <= 1.0e-9,
        "categorical probabilities do not sum to one"
    );
    register_conditions(assigned, conditions)?;

    // Put rare rows first. Besides preserving the distribution, this avoids
    // losing small event intervals to f32 cancellation near cumulative 1.0.
    let mut rows: Vec<usize> = (0..probabilities.len()).collect();
    rows.sort_by(|&lhs, &rhs| probabilities[lhs].total_cmp(&probabilities[rhs]));
    draws.push(DrawSource {
        conditions: conditions.to_vec(),
        assignments: rows.iter().map(|&row| assignments[row].clone()).collect(),
        probabilities: rows.iter().map(|&row| probabilities[row]).collect(),
    });
    Ok(())
}

fn register_conditions(assigned: &mut [bool], conditions: &[i32]) -> Result<()> {
    for &condition in conditions {
        ensure!(
            condition > 0,
            "exogenous condition {condition} is not positive"
        );
        let slot = assigned
            .get_mut(condition as usize)
            .ok_or_else(|| anyhow::anyhow!("exogenous condition {condition} is out of range"))?;
        ensure!(
            !*slot,
            "exogenous condition {condition} belongs to multiple draws"
        );
        *slot = true;
    }
    Ok(())
}

fn assignment_mask(
    conditions: &[i32],
    assignment: &[u64],
    condition_masks: &[Vec<u64>],
) -> Result<Vec<u64>> {
    let mask_words = condition_masks.first().map_or(0, Vec::len);
    let mut mask = vec![0u64; mask_words];
    for (bit, &condition) in conditions.iter().enumerate() {
        if packed_bit(assignment, bit) {
            let condition_mask = condition_masks.get(condition as usize).ok_or_else(|| {
                anyhow::anyhow!("exogenous condition {condition} is out of range")
            })?;
            for (word, condition_word) in mask.iter_mut().zip(condition_mask) {
                *word ^= condition_word;
            }
        }
    }
    Ok(mask)
}

fn push_sparse_group(out: &mut GpuExogenousPlan, source: &SparseGroupSource) -> Result<()> {
    ensure!(
        source.event_probability > 0.0 && source.event_probability < 1.0,
        "sparse event probability must be between zero and one"
    );
    ensure!(
        !source.outcome_probabilities.is_empty()
            && source.outcome_masks.iter().all(|masks| {
                masks.len() == source.outcome_probabilities.len()
                    && masks.iter().all(|mask| mask.len() == out.mask_words)
            }),
        "sparse group has mismatched outcomes"
    );
    let probability_total: f64 = source.outcome_probabilities.iter().sum();
    ensure!(
        source
            .outcome_probabilities
            .iter()
            .all(|probability| probability.is_finite() && *probability >= 0.0)
            && (probability_total - 1.0).abs() <= 1.0e-9,
        "sparse outcome probabilities do not sum to one"
    );

    let base_offset = out.sparse_base_masks.len() / out.mask_words;
    let upper_offset = out.sparse_transition_upper.len();
    let transition_count = source.outcome_probabilities.len() - 1;
    let transition_mask_offset = out.sparse_transition_masks.len() / out.mask_words;
    let gap_threshold_offset = out.sparse_gap_thresholds.len();
    ensure!(
        base_offset <= i32::MAX as usize
            && source.outcome_masks.len() <= i32::MAX as usize
            && upper_offset <= i32::MAX as usize
            && transition_count <= i32::MAX as usize
            && transition_mask_offset <= i32::MAX as usize
            && gap_threshold_offset <= i32::MAX as usize,
        "sparse GPU plan exceeds i32 indexing"
    );
    out.sparse_group_metadata.extend_from_slice(&[
        base_offset as i32,
        source.outcome_masks.len() as i32,
        upper_offset as i32,
        transition_count as i32,
        transition_mask_offset as i32,
        gap_threshold_offset as i32,
        (source.outcome_masks.len().ilog2() + 1) as i32,
    ]);
    let group = out.sparse_group_keys.len() as u64 + 1;
    out.sparse_group_keys
        .push(mix_u64(group.wrapping_mul(0xd2b7_4407_b1ce_6e93)));
    let survival_probability = 1.0 - source.event_probability;
    let mut survival = survival_probability;
    for _ in &source.outcome_masks {
        out.sparse_gap_thresholds
            .push((survival * 33_554_432.0).floor() as u64);
        survival *= survival_probability;
    }

    let mut cumulative = 0.0;
    for &probability in &source.outcome_probabilities[..transition_count] {
        cumulative += probability;
        out.sparse_transition_upper.push(cumulative as f32);
    }
    for masks in &source.outcome_masks {
        out.sparse_base_masks
            .extend_from_slice(masks.last().expect("sparse outcomes are nonempty"));
        for transition in 0..transition_count {
            out.sparse_transition_masks.extend(
                masks[transition]
                    .iter()
                    .zip(&masks[transition + 1])
                    .map(|(left, right)| left ^ right),
            );
        }
    }
    Ok(())
}

fn mix_u64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn next_sparse_bits(state: &mut u64) -> u32 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    (mix_u64(*state) >> 40) as u32
}

fn next_sparse_uniform(state: &mut u64) -> f32 {
    let bits = next_sparse_bits(state);
    (bits as f32 + 0.5) * (1.0 / 16_777_216.0)
}

fn placeholder_expression() -> GpuExpression {
    GpuExpression {
        block: 0,
        branch_masks: [0; 4],
    }
}

fn expand(expression: &SymbolicBool, definitions: &[Option<SymbolicBool>]) -> Result<SymbolicBool> {
    let mut expanded = SymbolicBool::from(expression.constant);
    for &condition in &expression.conditions {
        let definition = definitions
            .get(condition as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| anyhow::anyhow!("condition {condition} is read before assignment"))?;
        expanded = xor_bool(&expanded, definition);
    }
    Ok(expanded)
}

fn assign_definition(
    definitions: &mut [Option<SymbolicBool>],
    condition: Option<i32>,
    value: &SymbolicBool,
) -> Result<()> {
    let Some(condition) = condition else {
        return Ok(());
    };
    let slot = definitions
        .get_mut(condition as usize)
        .ok_or_else(|| anyhow::anyhow!("condition {condition} is out of range"))?;
    if let Some(previous) = slot {
        if previous != value {
            bail!("condition {condition} has inconsistent assignments");
        }
    } else {
        *slot = Some(value.clone());
    }
    Ok(())
}

fn assign_record(
    records: &mut [Option<SymbolicBool>],
    record: Option<i32>,
    value: &SymbolicBool,
) -> Result<()> {
    let Some(record) = record else {
        return Ok(());
    };
    let slot = records
        .get_mut(record as usize)
        .ok_or_else(|| anyhow::anyhow!("record {record} is out of range"))?;
    if slot.is_some() {
        bail!("record {record} is written more than once");
    }
    *slot = Some(value.clone());
    Ok(())
}

fn records_expression(records: &[Option<SymbolicBool>], ids: &[i32]) -> Result<SymbolicBool> {
    let mut expression = SymbolicBool::default();
    for &record in ids {
        let value = records
            .get(record as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| anyhow::anyhow!("record {record} is read before it is written"))?;
        expression = xor_bool(&expression, value);
    }
    Ok(expression)
}

fn logical_expression(
    records: &[Option<SymbolicBool>],
    logical_records: &[Vec<i32>],
) -> Result<SymbolicBool> {
    let mut expression = SymbolicBool::default();
    for group in logical_records {
        expression = xor_bool(&expression, &records_expression(records, group)?);
    }
    Ok(expression)
}

fn register_branch(
    slots: &mut HashMap<i32, usize>,
    definitions: &mut [Option<SymbolicBool>],
    condition: i32,
) -> Result<usize> {
    if let Some(&slot) = slots.get(&condition) {
        return Ok(slot);
    }
    if slots.len() == MAX_BRANCHES {
        bail!("GPU plan needs more than {MAX_BRANCHES} adaptive branch bits");
    }
    let slot = slots.len();
    let atom = symbolic_bool(condition);
    assign_definition(definitions, Some(condition), &atom)?;
    slots.insert(condition, slot);
    Ok(slot)
}

fn compile_expression(
    expression: &crate::sampler::presampled_expression::PresampledExpression,
    branch_slots: &HashMap<i32, usize>,
) -> Result<GpuExpression> {
    let mut branch_masks = [0u64; 4];
    for &condition in &expression.residual_plan.conditions {
        let slot = *branch_slots
            .get(&condition)
            .ok_or_else(|| anyhow::anyhow!("residual condition {condition} is not a branch"))?;
        branch_masks[slot >> 6] ^= 1u64 << (slot & 63);
    }
    Ok(GpuExpression {
        block: expression.block_expression_index,
        branch_masks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, plan_circuit};
    use crate::factored::{
        BernoulliSampleGroup, IntroduceDormantMeasurementBranch, RecordDetector,
    };
    use crate::sampler::prepared::logical_records_for_observable;
    use crate::symbolic::{SymbolicBoolEvaluationPlan, SymbolicCategoricalDistribution};

    #[test]
    fn active_masks_follow_physical_slots() {
        assert_eq!(remap_active_mask(0b101, &[2, 0, 3]).unwrap(), 0b1100);
        assert!(remap_active_mask(0b1000, &[2, 0, 3]).is_err());
    }

    #[test]
    fn only_profitable_x_rotation_runs_use_the_x_basis() {
        let rotation = |xmask, zmask| GpuInstruction {
            opcode: OP_ROTATE,
            expression: placeholder_expression(),
            xmask,
            zmask,
            pivot: 0b1111,
            diagonal_phase: false,
            z_without_pivot: 0,
            branch: None,
            expectation: None,
            detector: None,
            params: [0.0; PARAM_WORDS],
        };
        let mut detector = rotation(0, 0);
        detector.opcode = OP_DETECTOR;
        detector.pivot = usize::MAX;
        let mut profitable = vec![
            rotation(0b1111, 0),
            detector,
            rotation(0b1111, 0),
            rotation(0b1111, 0),
        ];
        mark_x_basis_rotation_runs(&mut profitable);
        assert!(profitable[0].diagonal_phase);
        assert_eq!(profitable[3].z_without_pivot, 1);
        assert_eq!(profitable[1].pivot, usize::MAX);
        assert_eq!(profitable[2].pivot, 0b1111);
        assert_eq!(profitable[2].params[3], 0.25);

        let mut sparse_support = vec![rotation(0b0011, 0); 3];
        mark_x_basis_rotation_runs(&mut sparse_support);
        assert_eq!(sparse_support[1].pivot, 0b0011);
        assert_eq!(sparse_support[1].params[3], 0.5);

        let mut short = vec![rotation(0b1111, 0), rotation(0b0011, 1)];
        mark_x_basis_rotation_runs(&mut short);
        assert_eq!(short[0].pivot, 0);
        assert_eq!(short[1].pivot, 0);
    }

    #[test]
    fn adaptive_records_are_inlined_into_detectors() {
        let branch = 1;
        let recorded = 2;
        let branch_outcome = SymbolicBool::new(true, vec![branch]);
        let instructions = vec![
            FactoredInstruction::IntroduceDormantMeasurementBranch(
                IntroduceDormantMeasurementBranch {
                    branch,
                    outcome: branch_outcome.clone(),
                    outcome_plan: SymbolicBoolEvaluationPlan::new(&branch_outcome),
                    record: Some(1),
                    record_condition: Some(recorded),
                    exp_val: None,
                },
            ),
            FactoredInstruction::RecordDetector(RecordDetector {
                records: vec![1],
                detector: 1,
                postselect: true,
                ..RecordDetector::default()
            }),
        ];
        let mut program =
            FactoredInstructionProgram::new(1, 0, instructions, 0).expect("valid test program");
        program.nrecords = 1;
        program.ndetectors = 1;

        let plan = GpuPlan::build(&program, &[vec![1]]).expect("GPU plan");
        assert_eq!(plan.branch_count, 1);
        assert_eq!(plan.instructions.len(), 2);
        assert_eq!(plan.detector_start, 2);
        assert_eq!(plan.instructions[1].opcode, OP_DETECTOR);
        assert!(plan.instructions[1].diagonal_phase);
        assert_eq!(plan.instructions[1].detector, Some(0));
        assert_eq!(plan.instructions[1].expression.branch_masks[0], 1);
        assert_eq!(plan.logical.branch_masks[0], 1);

        let (metadata, parameters, controls, expectations) = plan.encode();
        assert_eq!(metadata.len(), 2 * META_WORDS);
        assert_eq!(parameters.len(), 2 * PARAM_WORDS);
        assert_eq!(controls.len(), 2 * CONTROL_WORDS);
        assert_eq!(expectations, [-1, 0]);
        assert_eq!(controls, [0, 0]);
        assert_eq!(metadata[10], 0);
        assert_eq!(metadata[11], 1);
    }

    #[test]
    fn detector_plan_preserves_postselection_flag() {
        for postselect in [false, true] {
            let program = FactoredInstructionProgram::new(
                1,
                0,
                vec![FactoredInstruction::RecordDetector(RecordDetector {
                    detector: 1,
                    postselect,
                    ..RecordDetector::default()
                })],
                0,
            )
            .expect("valid test program");

            let plan = GpuPlan::build(&program, &[]).expect("GPU plan");
            assert_eq!(plan.instructions[0].diagonal_phase, postselect);
        }
    }

    #[test]
    fn excess_dormant_branches_are_presampled() {
        let instructions = (1..=MAX_BRANCHES as i32 + 1)
            .map(|branch| {
                let outcome = symbolic_bool(branch);
                FactoredInstruction::IntroduceDormantMeasurementBranch(
                    IntroduceDormantMeasurementBranch {
                        branch,
                        outcome: outcome.clone(),
                        outcome_plan: SymbolicBoolEvaluationPlan::new(&outcome),
                        record: Some(branch),
                        record_condition: None,
                        exp_val: None,
                    },
                )
            })
            .collect();
        let program =
            FactoredInstructionProgram::new(1, 0, instructions, 0).expect("valid test program");
        let logical_records = vec![(1..=MAX_BRANCHES as i32 + 1).collect()];

        let plan = GpuPlan::build(&program, &logical_records).expect("GPU plan");

        assert_eq!(plan.branch_count, 0);
        assert!(plan.instructions.is_empty());
        assert_eq!(plan.exogenous_plan.draw_count, MAX_BRANCHES + 1);
    }

    #[test]
    fn counts_plan_keeps_expectation_work() {
        let instructions = vec![FactoredInstruction::RecordMeasurement(RecordMeasurement {
            exp_val: Some(0),
            ..RecordMeasurement::default()
        })];
        let program =
            FactoredInstructionProgram::new(1, 0, instructions, 0).expect("valid test program");

        assert_eq!(program.nexpvals, 1);
        let plan = GpuPlan::build(&program, &[]).expect("GPU counts plan");
        assert_eq!(plan.instructions.len(), 1);
        assert_eq!(plan.instructions[0].opcode, OP_EXPECTATION_RECORD);
        assert_eq!(plan.instructions[0].expectation, Some(0));
        assert_eq!(plan.branch_count, 0);
    }

    #[test]
    fn exogenous_plan_reuses_one_categorical_draw() {
        let program = FactoredInstructionProgram {
            nsymbols: 3,
            sampled_categorical_distributions: vec![SymbolicCategoricalDistribution {
                nbits: 2,
                conditions: vec![1, 2],
                assignments: vec![vec![0], vec![1], vec![2], vec![3]],
                probabilities: vec![0.4, 0.1, 0.2, 0.3],
            }],
            sampled_bernoulli_conditions: vec![3],
            sampled_bernoulli_probabilities: vec![0.25],
            ..FactoredInstructionProgram::default()
        };
        let expressions = PresampledExpressionPlan {
            block_expressions: vec![
                crate::sampler::presampled_expression::PresampledExpression {
                    exogenous_conditions: vec![1, 2, 3],
                    ..Default::default()
                },
                crate::sampler::presampled_expression::PresampledExpression {
                    constant: true,
                    exogenous_conditions: vec![2, 3],
                    parent_block_expression_index: Some(0),
                    parent_delta_constant: true,
                    parent_delta_exogenous_conditions: vec![1],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let plan =
            GpuExogenousPlan::build(&program, &expressions, &[]).expect("GPU exogenous plan");

        assert_eq!(plan.draw_count, 2);
        assert_eq!(plan.constant_masks, [2]);
        assert_eq!(plan.draw_transition_offsets, [0, 3, 4]);
        assert_eq!(plan.draw_base_masks, [0, 0]);
        assert_eq!(plan.transition_upper, [0.1, 0.3, 0.6, 0.25]);
        assert_eq!(plan.transition_masks, [2, 1, 2, 3]);
    }

    #[test]
    fn exogenous_plan_packs_more_than_one_mask_word() {
        let program = FactoredInstructionProgram {
            nsymbols: 1,
            sampled_bernoulli_conditions: vec![1],
            sampled_bernoulli_probabilities: vec![0.25],
            ..FactoredInstructionProgram::default()
        };
        let expressions = PresampledExpressionPlan {
            block_expressions: (0..65)
                .map(
                    |_| crate::sampler::presampled_expression::PresampledExpression {
                        exogenous_conditions: vec![1],
                        ..Default::default()
                    },
                )
                .collect(),
            ..Default::default()
        };

        let plan =
            GpuExogenousPlan::build(&program, &expressions, &[]).expect("GPU exogenous plan");

        assert_eq!(plan.mask_words, 2);
        assert_eq!(plan.draw_base_masks, [0, 0]);
        assert_eq!(plan.transition_upper, [0.25]);
        assert_eq!(plan.transition_masks, [u64::MAX, 1]);
    }

    #[test]
    fn sparse_bernoulli_group_uses_geometric_gaps() {
        let program = FactoredInstructionProgram {
            nsymbols: 4,
            sampled_low_probability_bernoulli_groups: vec![BernoulliSampleGroup {
                probability: 0.125,
                conditions: vec![1, 2, 3, 4],
            }],
            ..FactoredInstructionProgram::default()
        };
        let expressions = PresampledExpressionPlan {
            block_expressions: (1..=4)
                .map(
                    |condition| crate::sampler::presampled_expression::PresampledExpression {
                        exogenous_conditions: vec![condition],
                        ..Default::default()
                    },
                )
                .collect(),
            ..Default::default()
        };

        let plan =
            GpuExogenousPlan::build(&program, &expressions, &[]).expect("GPU exogenous plan");

        assert_eq!(plan.draw_count, 0);
        assert_eq!(plan.sparse_group_metadata, [0, 4, 0, 0, 0, 0, 3]);
        assert_eq!(plan.sparse_base_masks, [1, 2, 4, 8]);
        assert!(
            plan.sparse_gap_thresholds
                .windows(2)
                .all(|pair| pair[0] > pair[1])
        );
        let mut hits = [0usize; 4];
        for shot in 0..20_000 {
            let mask = plan.sparse_mask(123, shot, 0);
            for (bit, count) in hits.iter_mut().enumerate() {
                *count += ((mask >> bit) & 1) as usize;
            }
        }
        for count in hits {
            assert!((2_300..=2_700).contains(&count), "biased hit count {count}");
        }
    }

    #[test]
    fn deterministic_zero_has_false_detector_expression() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/circuits/gpu/deterministic_zero.stim");
        let parsed = Circuit::from_file(&path).expect("parse fixture");
        let program = plan_circuit(&parsed, &[]).expect("plan fixture");
        let logical = logical_records_for_observable(&parsed.observables, 0);
        let plan = GpuPlan::build(&program, &logical).expect("GPU plan");

        assert_eq!(plan.exogenous_plan.constant_masks, [0]);
        assert_eq!(plan.exogenous_plan.draw_transition_offsets, [0]);
        assert_eq!(plan.instructions[0].expression.block, 0);
        assert_eq!(plan.logical.block, 0);
    }

    #[test]
    fn wide_plan_uses_compact_active_coordinates() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/circuits/gpu/wide_compact.stim");
        let parsed = Circuit::from_file(&path).expect("parse fixture");
        let program = plan_circuit(&parsed, &[]).expect("plan fixture");
        let logical = logical_records_for_observable(&parsed.observables, 0);
        let plan = GpuPlan::build(&program, &logical).expect("wide GPU plan");

        assert_eq!(program.max_k, 13);
        assert!(plan.instructions.iter().all(|instruction| {
            instruction.xmask < 1u64 << 13 && instruction.zmask < 1u64 << 13
        }));
        assert!(
            plan.instructions
                .iter()
                .filter(|instruction| instruction.opcode == OP_ROTATE)
                .all(|instruction| instruction.params[3] == 0.0)
        );
    }
}
