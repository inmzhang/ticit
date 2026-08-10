//! Experimental cuTile batch-sampling backend.

mod kernel;
mod plan;

use std::collections::HashSet;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::circuit::{Circuit, has_postselection, plan_circuit};
use crate::factored::FactoredInstructionProgram;
use crate::random::block_seed;
use crate::sampler::prepared::{
    SampleCounts, SampleResult, SamplingTiming, logical_records_for_observable,
    make_circuit_sampling_input,
};
use crate::symbolic::SymbolicCategoricalDistribution;
use anyhow::{Context, Result, bail, ensure};
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;

const EXOGENOUS_SEED_BASE: u64 = 0x7eed_0000;
const BRANCH_SEED_BASE: u64 = 0x5eed_1234;
const DRAWS_PER_GROUP: usize = 16;
const SPARSE_RNG_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
const SPARSE_RNG_MULTIPLIER0: u64 = 0xbf58_476d_1ce4_e5b9;
const SPARSE_RNG_MULTIPLIER1: u64 = 0x94d0_49bb_1331_11eb;
const EXOGENOUS_TILE_SHOTS: usize = 64;
const COUNT_REDUCTION_BLOCKS: usize = 256;
const WIDE_WORKSPACE_BUDGET_BYTES: usize = 16 * 1024 * 1024 * 1024;
const WIDE_CHUNK_SHOTS: usize = 2048;
const WIDE_LARGE_STATE_CHUNK_SHOTS: usize = 256;

fn condition_exact_k(program: &mut FactoredInstructionProgram, k: usize) -> Result<()> {
    ensure!(
        program.sampled_rare_categorical_groups.is_empty()
            && program.sampled_low_probability_bernoulli_groups.is_empty(),
        "exact-k sampling does not support sparse or multi-outcome sources"
    );
    ensure!(
        program.sampled_bernoulli_conditions.len() == program.sampled_bernoulli_probabilities.len(),
        "Bernoulli sampling plan has mismatched conditions and probabilities"
    );
    let mut conditions = std::mem::take(&mut program.sampled_bernoulli_conditions);
    for distribution in std::mem::take(&mut program.sampled_categorical_distributions) {
        ensure!(
            distribution.nbits == 1
                && distribution.conditions.len() == 1
                && distribution.assignments.len() == 2
                && distribution.assignments.iter().all(|row| row.len() == 1)
                && distribution.assignments.iter().any(|row| row[0] & 1 == 0)
                && distribution.assignments.iter().any(|row| row[0] & 1 == 1),
            "exact-k sampling requires independent binary sources"
        );
        conditions.push(distribution.conditions[0]);
    }
    let n = conditions.len();
    ensure!(n != 0, "exact-k sampling found no Bernoulli sources");
    // ponytail: exhaustive masks keep the exact-k path tiny; use a
    // combinatorial generator when a validation circuit exceeds 20 sources.
    ensure!(
        n <= 20,
        "exact-k sampling currently supports at most 20 sources"
    );
    ensure!(
        k <= n,
        "exact-k value {k} exceeds the {n} Bernoulli sources"
    );
    let assignments: Vec<Vec<u64>> = (0..1u64 << n)
        .filter(|mask| mask.count_ones() as usize == k)
        .map(|mask| vec![mask])
        .collect();
    let probability = 1.0 / assignments.len() as f64;
    program.sampled_bernoulli_probabilities.clear();
    program
        .sampled_categorical_distributions
        .push(SymbolicCategoricalDistribution {
            nbits: n,
            conditions,
            probabilities: vec![probability; assignments.len()],
            assignments,
        });
    Ok(())
}

fn wide_workspace_budget() -> Result<usize> {
    let (mut free, mut total) = (0usize, 0usize);
    // Safety: Device::new has made a valid CUDA context current on this thread.
    let status = unsafe { cutile::cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) };
    if status != cutile::cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
        bail!("failed to query free GPU memory (CUDA error {status})");
    }
    Ok(WIDE_WORKSPACE_BUDGET_BYTES.min(free.saturating_mul(2) / 3))
}

/// Options for the experimental GPU sampler.
pub struct GpuOptions {
    /// Stim circuit to sample.
    pub circuit: PathBuf,

    /// Number of attempted shots.
    pub shots: u64,

    /// Seed for deterministic sampling.
    pub seed: u64,

    /// Shots presampled and uploaded per launch group.
    pub chunk_shots: NonZeroUsize,

    /// Flat detector flags; any nonzero entry enables postselection.
    pub postselection_mask: Vec<u8>,

    /// Compute a noiseless reference on the CPU before GPU sampling.
    pub normalize_syndromes: bool,
}

struct WideBuffers {
    primary_re: Tensor<f32>,
    primary_im: Tensor<f32>,
    scratch_re: Tensor<f32>,
    scratch_im: Tensor<f32>,
    branches: Tensor<u64>,
    discarded: Tensor<u64>,
    dimension: usize,
    scratch_dimension: usize,
}

fn apply_detector_reference(plan: &mut plan::GpuPlan, expected_detectors: &[u8]) {
    for instruction in &mut plan.instructions {
        if let Some(detector) = instruction.detector {
            instruction.z_without_pivot = u64::from(
                expected_detectors
                    .get(detector)
                    .is_some_and(|&bit| bit != 0),
            );
        }
    }
}

fn normalized_logical_errors(
    shots: u64,
    discarded: u64,
    raw_logical_errors: u64,
    expected: bool,
) -> u64 {
    if expected {
        shots - discarded - raw_logical_errors
    } else {
        raw_logical_errors
    }
}

/// Runs the experimental GPU sampler.
pub fn run(args: &GpuOptions) -> Result<()> {
    let parse_start = Instant::now();
    let parsed = Circuit::from_file(&args.circuit)
        .with_context(|| format!("failed to parse {}", args.circuit.display()))?;
    let parse_s = parse_start.elapsed().as_secs_f64();
    let reference = if args.normalize_syndromes {
        parsed.reference_sample()?
    } else {
        crate::ReferenceSample::default()
    };

    sample_circuit_impl(
        &parsed,
        args.shots,
        args.seed,
        args.chunk_shots,
        &args.postselection_mask,
        0,
        parse_s,
        Some(&args.circuit),
        false,
        None,
        &reference.detectors,
        &reference.observables,
    )?;
    Ok(())
}

/// Samples a parsed circuit on CUDA and returns the same aggregate counters as
/// the CPU [`crate::Sampler`].
///
/// Nonzero `postselection_mask` entries reject shots at the corresponding
/// detectors. `observable` selects the observable index counted as a logical
/// error.
///
/// # Examples
///
/// ```no_run
/// use std::num::NonZeroUsize;
/// use ticit::Circuit;
///
/// let circuit = Circuit::from_text("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")?;
/// let result = ticit::gpu::sample_circuit(
///     &circuit,
///     1_000,
///     5,
///     NonZeroUsize::new(1_000).unwrap(),
///     &[],
///     0,
/// )?;
/// assert_eq!(result.counts.shots, 1_000);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns an error if planning fails, CUDA is unavailable, the circuit
/// exceeds a GPU indexing limit, or the requested workspace does not fit.
pub fn sample_circuit(
    circuit: &Circuit,
    shots: u64,
    seed: u64,
    chunk_shots: NonZeroUsize,
    postselection_mask: &[u8],
    observable: usize,
) -> Result<SampleResult> {
    sample_circuit_impl(
        circuit,
        shots,
        seed,
        chunk_shots,
        postselection_mask,
        observable,
        0.0,
        None,
        false,
        None,
        &[],
        &[],
    )
}

/// Samples on CUDA while XOR-normalizing against explicit reference vectors.
///
/// The reference vectors are normally produced once by
/// [`Circuit::reference_sample`] on the CPU.
///
/// # Errors
///
/// Returns the same errors as [`sample_circuit`], plus an error when either
/// reference vector has the wrong length.
#[allow(clippy::too_many_arguments)]
pub fn sample_circuit_with_reference(
    circuit: &Circuit,
    shots: u64,
    seed: u64,
    chunk_shots: NonZeroUsize,
    postselection_mask: &[u8],
    observable: usize,
    expected_detectors: &[u8],
    expected_observables: &[u8],
) -> Result<SampleResult> {
    sample_circuit_impl(
        circuit,
        shots,
        seed,
        chunk_shots,
        postselection_mask,
        observable,
        0.0,
        None,
        false,
        None,
        expected_detectors,
        expected_observables,
    )
}

/// Samples a parsed circuit on CUDA and retains row-major detector outcomes
/// and expectation values for every shot.
///
/// When `exact_k` is set, all ordinary Bernoulli sources are conditioned on
/// exactly that many faults. Record capture currently requires a circuit with
/// no active detector postselection.
///
/// # Errors
///
/// Returns an error under the same conditions as [`sample_circuit`], when the
/// exact-k source plan is unsupported, or when postselection is active.
pub fn sample_circuit_records(
    circuit: &Circuit,
    shots: u64,
    seed: u64,
    chunk_shots: NonZeroUsize,
    observable: usize,
    exact_k: Option<usize>,
) -> Result<SampleResult> {
    sample_circuit_records_with_reference(
        circuit,
        shots,
        seed,
        chunk_shots,
        observable,
        exact_k,
        &[],
        &[],
    )
}

/// Retains GPU records while XOR-normalizing against explicit references.
///
/// # Errors
///
/// Returns the same errors as [`sample_circuit_records`], plus an error when
/// either reference vector has the wrong length.
#[allow(clippy::too_many_arguments)]
pub fn sample_circuit_records_with_reference(
    circuit: &Circuit,
    shots: u64,
    seed: u64,
    chunk_shots: NonZeroUsize,
    observable: usize,
    exact_k: Option<usize>,
    expected_detectors: &[u8],
    expected_observables: &[u8],
) -> Result<SampleResult> {
    sample_circuit_impl(
        circuit,
        shots,
        seed,
        chunk_shots,
        &[],
        observable,
        0.0,
        None,
        true,
        exact_k,
        expected_detectors,
        expected_observables,
    )
}

#[allow(clippy::too_many_arguments)]
fn sample_circuit_impl(
    parsed: &Circuit,
    shots: u64,
    seed: u64,
    chunk_shots: NonZeroUsize,
    postselection_mask: &[u8],
    observable: usize,
    parse_s: f64,
    report_path: Option<&Path>,
    keep_records: bool,
    exact_k: Option<usize>,
    expected_detectors: &[u8],
    expected_observables: &[u8],
) -> Result<SampleResult> {
    if shots == 0 {
        bail!("shots must be positive");
    }
    if !expected_detectors.is_empty() && expected_detectors.len() != parsed.detector_count() {
        bail!(
            "expected_detectors has length {}, expected {}",
            expected_detectors.len(),
            parsed.detector_count(),
        );
    }
    if !expected_observables.is_empty() && expected_observables.len() != parsed.observable_count() {
        bail!(
            "expected_observables has length {}, expected {}",
            expected_observables.len(),
            parsed.observable_count(),
        );
    }
    let expected_logical = expected_observables
        .get(observable)
        .is_some_and(|&bit| bit != 0);

    let plan_start = Instant::now();
    let mut program = plan_circuit(parsed, postselection_mask).context("failed to plan circuit")?;
    if let Some(k) = exact_k {
        condition_exact_k(&mut program, k)?;
    }
    if keep_records && has_postselection(&program) {
        bail!("GPU record capture does not support active detector postselection");
    }
    let logical_records = logical_records_for_observable(&parsed.observables, observable);
    let input = make_circuit_sampling_input(
        program,
        logical_records,
        observable,
        parsed.observable_count(),
        SamplingTiming::default(),
    );
    let mut gpu_plan = plan::GpuPlan::build(&input.program, &input.logical_records)
        .context("failed to lower the GPU plan")?;
    apply_detector_reference(&mut gpu_plan, expected_detectors);
    // Detector postselection is carried by each `OP_DETECTOR`'s own postselect
    // bit, which planning derives from `postselection_mask`, so every kernel
    // runs the whole instruction stream.
    let instruction_count = gpu_plan.instructions.len();
    let sample_tile_shots = if !keep_records
        && input.program.max_k <= 4
        && input.program.nexpvals == 0
        && gpu_plan.exogenous_plan.mask_words == 1
    {
        64
    } else {
        1
    };
    let plan_s = plan_start.elapsed().as_secs_f64();

    let device = Device::new(0)?;
    // cuRAND's public wrapper uses CUDA's default stream. Use that same stream
    // for allocations and kernels so a persistent generator stays ordered.
    let stream = unsafe { cutile::cuda_core::Stream::borrow_raw(std::ptr::null_mut(), &device) };
    let requested_chunk_limit = chunk_shots.get().min(i32::MAX as usize);
    let wide_dimension = (input.program.max_k > 12).then(|| 1usize << input.program.max_k);
    if keep_records && wide_dimension.is_some() {
        bail!("GPU record capture currently requires max_k <= 12");
    }
    let wide_chunk_limit = if let Some(dimension) = wide_dimension {
        let workspace_budget = wide_workspace_budget()?;
        let bytes_per_shot = dimension
            .checked_mul(3 * size_of::<f32>())
            .context("wide GPU state size overflow")?;
        if bytes_per_shot > workspace_budget {
            bail!("one wide GPU shot exceeds the available workspace budget");
        }
        // One full-GPU wave wins for k=22; smaller states amortize launches.
        let launch_cap = if input.program.max_k >= 22 {
            WIDE_LARGE_STATE_CHUNK_SHOTS
        } else {
            WIDE_CHUNK_SHOTS
        };
        let tensor_chunk_limit = (i32::MAX as usize) / dimension;
        if tensor_chunk_limit == 0 {
            bail!("one wide GPU state exceeds cuTile's tensor dimension limit");
        }
        let chunk_limit = (workspace_budget / bytes_per_shot)
            .min(tensor_chunk_limit)
            .min(launch_cap);
        // Power-of-two groups avoid a second JIT specialization for the
        // remainder of the common power-of-two benchmark batches.
        1usize << chunk_limit.ilog2()
    } else {
        requested_chunk_limit
    };
    let chunk_limit = requested_chunk_limit.min(wide_chunk_limit) as u64;
    let max_chunk = chunk_limit.min(shots) as usize;
    if keep_records {
        let rows = usize::try_from(shots).context("GPU record row count exceeds usize")?;
        rows.checked_mul(input.program.ndetectors)
            .context("GPU detector output is too large")?;
        rows.checked_mul(input.program.nexpvals)
            .context("GPU expectation output is too large")?;
    }
    if gpu_plan.exogenous_plan.draw_count > i32::MAX as usize
        || gpu_plan.expression_plan.block_expressions.len() > i32::MAX as usize
    {
        bail!("the GPU exogenous plan exceeds cuTile's i32 indexing limit");
    }
    let draw_group_count = gpu_plan.exogenous_plan.draw_count.div_ceil(DRAWS_PER_GROUP);
    let draw_group_grid = u32::try_from(draw_group_count)
        .context("the GPU exogenous draw-group count exceeds cuTile's grid limit")?;
    let sparse_group_grid = u32::try_from(gpu_plan.exogenous_plan.sparse_group_count())
        .context("the GPU sparse-group count exceeds cuTile's grid limit")?;
    let mask_word_grid = u32::try_from(gpu_plan.exogenous_plan.mask_words)
        .context("the GPU expression-mask word count exceeds cuTile's grid limit")?;
    let rng_setup_start = Instant::now();
    let exogenous_seed = block_seed(EXOGENOUS_SEED_BASE, seed, 0);
    let exogenous_rng = unsafe { cutile::cuda_core::curand::RNG::new(Some(exogenous_seed)) };
    let branch_rng =
        unsafe { cutile::cuda_core::curand::RNG::new(Some(block_seed(BRANCH_SEED_BASE, seed, 0))) };
    // ponytail: materialize one cuRAND row per independent draw; stream or
    // counter-generate when this buffer limits representative wide circuits.
    let exogenous_random_capacity = if gpu_plan.exogenous_plan.draw_count == 0 {
        1
    } else {
        gpu_plan
            .exogenous_plan
            .draw_count
            .checked_mul(max_chunk)
            .context("GPU exogenous random buffer is too large")?
    };
    let branch_random_capacity = if gpu_plan.branch_count == 0 {
        1
    } else {
        gpu_plan
            .branch_count
            .checked_mul(max_chunk)
            .context("GPU branch random buffer is too large")?
    };
    let exogenous_randoms: Tensor<f32> =
        cutile::api::zeros(&[exogenous_random_capacity]).sync_on(&stream)?;
    let branch_randoms: Tensor<f32> =
        cutile::api::zeros(&[branch_random_capacity]).sync_on(&stream)?;
    let expression_value_capacity = gpu_plan
        .exogenous_plan
        .mask_words
        .checked_mul(max_chunk)
        .context("GPU expression-value buffer is too large")?;
    let expression_values: Tensor<u64> =
        cutile::api::zeros(&[expression_value_capacity]).sync_on(&stream)?;
    let max_exogenous_blocks = max_chunk.div_ceil(EXOGENOUS_TILE_SHOTS);
    let max_sample_blocks = max_chunk.div_ceil(sample_tile_shots);
    let shot_block_offsets: Vec<u64> = (0..max_exogenous_blocks)
        .map(|block| (block * EXOGENOUS_TILE_SHOTS) as u64)
        .collect();
    let shot_block_offsets: Tensor<u64> =
        cutile::api::copy_host_vec_to_device(&Arc::new(shot_block_offsets)).sync_on(&stream)?;
    let expression_partial_capacity = draw_group_count
        .max(1)
        .checked_mul(gpu_plan.exogenous_plan.mask_words)
        .context("GPU exogenous partial buffer is too large")?
        .checked_mul(max_chunk)
        .context("GPU exogenous partial buffer is too large")?;
    let expression_partials: Tensor<u64> =
        cutile::api::zeros(&[expression_partial_capacity]).sync_on(&stream)?;
    let block_counts: Arc<Tensor<u64>> =
        Arc::new(cutile::api::zeros(&[max_sample_blocks * 2]).sync_on(&stream)?);
    let expectations: Arc<Tensor<f32>> = Arc::new(
        cutile::api::zeros(&[input
            .program
            .nexpvals
            .max(1)
            .checked_mul(max_chunk)
            .context("GPU expectation buffer is too large")?])
        .sync_on(&stream)?,
    );
    let detector_values: Arc<Tensor<u64>> = Arc::new(
        cutile::api::zeros(&[input
            .program
            .ndetectors
            .max(1)
            .checked_mul(max_chunk)
            .context("GPU detector buffer is too large")?])
        .sync_on(&stream)?,
    );
    let count_partials: Arc<Tensor<u64>> = Arc::new(
        cutile::api::zeros(&[max_sample_blocks.div_ceil(COUNT_REDUCTION_BLOCKS) * 2])
            .sync_on(&stream)?,
    );
    let wide_buffers = if let Some(dimension) = wide_dimension {
        let scratch_dimension = dimension / 2;
        let primary_capacity = dimension
            .checked_mul(max_chunk)
            .context("wide GPU primary state is too large")?;
        let scratch_capacity = scratch_dimension
            .checked_mul(max_chunk)
            .context("wide GPU scratch state is too large")?;
        Some(WideBuffers {
            primary_re: cutile::api::zeros(&[primary_capacity]).sync_on(&stream)?,
            primary_im: cutile::api::zeros(&[primary_capacity]).sync_on(&stream)?,
            scratch_re: cutile::api::zeros(&[scratch_capacity]).sync_on(&stream)?,
            scratch_im: cutile::api::zeros(&[scratch_capacity]).sync_on(&stream)?,
            branches: cutile::api::zeros(&[4 * max_chunk]).sync_on(&stream)?,
            discarded: cutile::api::zeros(&[max_chunk]).sync_on(&stream)?,
            dimension,
            scratch_dimension,
        })
    } else {
        None
    };
    // cuRAND initializes lazily on its first generation; keep that cost out of
    // steady-state sampling and overwrite the warmup value before use.
    unsafe {
        exogenous_rng.generate_uniform_f32(exogenous_randoms.device_pointer().cu_deviceptr(), 1);
        branch_rng.generate_uniform_f32(branch_randoms.device_pointer().cu_deviceptr(), 1);
        stream.synchronize()?;
    }
    let rng_setup_s = rng_setup_start.elapsed().as_secs_f64();
    let (metadata, parameters, controls, expectation_indices) = gpu_plan.encode();
    let metadata: Tensor<u64> =
        cutile::api::copy_host_vec_to_device(&Arc::new(metadata)).sync_on(&stream)?;
    let parameters: Tensor<f32> =
        cutile::api::copy_host_vec_to_device(&Arc::new(parameters)).sync_on(&stream)?;
    let controls: Tensor<i32> =
        cutile::api::copy_host_vec_to_device(&Arc::new(controls)).sync_on(&stream)?;
    let expectation_indices: Tensor<i32> =
        cutile::api::copy_host_vec_to_device(&Arc::new(expectation_indices)).sync_on(&stream)?;

    let exogenous = &gpu_plan.exogenous_plan;
    let constant_masks: Tensor<u64> =
        cutile::api::copy_host_vec_to_device(&Arc::new(exogenous.constant_masks.clone()))
            .sync_on(&stream)?;
    let draw_transition_offsets: Tensor<i32> =
        cutile::api::copy_host_vec_to_device(&Arc::new(exogenous.draw_transition_offsets.clone()))
            .sync_on(&stream)?;
    let mut draw_base_masks = exogenous.draw_base_masks.clone();
    let mut transition_upper = exogenous.transition_upper.clone();
    let mut transition_masks = exogenous.transition_masks.clone();
    draw_base_masks.resize(draw_base_masks.len().max(1), 0);
    transition_upper.resize(transition_upper.len().max(1), 0.0);
    transition_masks.resize(transition_masks.len().max(1), 0);
    let draw_base_masks: Tensor<u64> =
        cutile::api::copy_host_vec_to_device(&Arc::new(draw_base_masks)).sync_on(&stream)?;
    let transition_upper: Tensor<f32> =
        cutile::api::copy_host_vec_to_device(&Arc::new(transition_upper)).sync_on(&stream)?;
    let transition_masks: Tensor<u64> =
        cutile::api::copy_host_vec_to_device(&Arc::new(transition_masks)).sync_on(&stream)?;
    let mut sparse_group_metadata = exogenous.sparse_group_metadata.clone();
    let mut sparse_group_keys = exogenous.sparse_group_keys.clone();
    let mut sparse_gap_thresholds = exogenous.sparse_gap_thresholds.clone();
    let mut sparse_transition_upper = exogenous.sparse_transition_upper.clone();
    let mut sparse_base_masks = exogenous.sparse_base_masks.clone();
    let mut sparse_transition_masks = exogenous.sparse_transition_masks.clone();
    sparse_group_metadata.resize(sparse_group_metadata.len().max(1), 0);
    sparse_group_keys.resize(sparse_group_keys.len().max(1), 0);
    sparse_gap_thresholds.resize(sparse_gap_thresholds.len().max(1), 0);
    sparse_transition_upper.resize(sparse_transition_upper.len().max(1), 0.0);
    sparse_base_masks.resize(sparse_base_masks.len().max(1), 0);
    sparse_transition_masks.resize(sparse_transition_masks.len().max(1), 0);
    let sparse_group_metadata: Tensor<i32> =
        cutile::api::copy_host_vec_to_device(&Arc::new(sparse_group_metadata)).sync_on(&stream)?;
    let sparse_group_keys: Tensor<u64> =
        cutile::api::copy_host_vec_to_device(&Arc::new(sparse_group_keys)).sync_on(&stream)?;
    let sparse_gap_thresholds: Tensor<u64> =
        cutile::api::copy_host_vec_to_device(&Arc::new(sparse_gap_thresholds)).sync_on(&stream)?;
    let sparse_transition_upper: Tensor<f32> =
        cutile::api::copy_host_vec_to_device(&Arc::new(sparse_transition_upper))
            .sync_on(&stream)?;
    let sparse_base_masks: Tensor<u64> =
        cutile::api::copy_host_vec_to_device(&Arc::new(sparse_base_masks)).sync_on(&stream)?;
    let sparse_transition_masks: Tensor<u64> =
        cutile::api::copy_host_vec_to_device(&Arc::new(sparse_transition_masks))
            .sync_on(&stream)?;

    let mut discarded = 0u64;
    let mut logical_errors = 0u64;
    let record_rows = if keep_records { shots as usize } else { 0 };
    let mut recorded_detectors = Vec::with_capacity(record_rows * input.program.ndetectors);
    let mut recorded_expectations = Vec::with_capacity(record_rows * input.program.nexpvals);
    let mut warmup_s = 0.0;
    let mut exogenous_rng_s = 0.0;
    let mut exogenous_kernel_s = 0.0;
    let mut rng_s = 0.0;
    let mut kernel_s = 0.0;
    let mut copy_s = 0.0;
    let mut warmed_exogenous_grids = HashSet::new();
    let mut warmed_sample_grids = HashSet::new();
    let sample_start = Instant::now();

    let chunks = shots.div_ceil(chunk_limit);
    for chunk_index in 0..chunks {
        let offset = chunk_index * chunk_limit;
        let chunk = chunk_limit.min(shots - offset) as usize;
        let exogenous_blocks = chunk.div_ceil(EXOGENOUS_TILE_SHOTS) as u32;
        let sample_blocks = chunk.div_ceil(sample_tile_shots) as u32;
        let count_partial_blocks = (sample_blocks as usize).div_ceil(COUNT_REDUCTION_BLOCKS) as u32;

        let exogenous_rng_start = Instant::now();
        let exogenous_random_len = gpu_plan.exogenous_plan.draw_count * chunk;
        if exogenous_random_len != 0 {
            // Safety: the allocation holds one contiguous f32 row per draw.
            unsafe {
                exogenous_rng.generate_uniform_f32(
                    exogenous_randoms.device_pointer().cu_deviceptr(),
                    exogenous_random_len,
                );
                stream.synchronize()?;
            }
        }
        exogenous_rng_s += exogenous_rng_start.elapsed().as_secs_f64();

        // Safety: all metadata arrays were built from the same expression
        // plan, and every allocation remains live through the launch.
        let launch_exogenous_partials = || {
            unsafe {
                kernel::evaluate_exogenous_partials(
                    expression_partials.device_pointer(),
                    exogenous_randoms.device_pointer(),
                    draw_transition_offsets.device_pointer(),
                    draw_base_masks.device_pointer(),
                    transition_upper.device_pointer(),
                    transition_masks.device_pointer(),
                    exogenous.draw_count as i32,
                    exogenous.mask_words as i32,
                    chunk as i32,
                    chunk as i32,
                )
            }
            .grid((exogenous_blocks, draw_group_grid, mask_word_grid))
        };
        let launch_exogenous_reduction = || {
            unsafe {
                kernel::reduce_exogenous_partials(
                    expression_values.device_pointer(),
                    expression_partials.device_pointer(),
                    constant_masks.device_pointer(),
                    draw_group_count as i32,
                    chunk as i32,
                    chunk as i32,
                )
            }
            .grid((exogenous_blocks, mask_word_grid, 1))
        };
        let launch_sparse_exogenous = || {
            unsafe {
                kernel::apply_sparse_exogenous(
                    expression_values.device_pointer(),
                    shot_block_offsets.device_pointer(),
                    sparse_group_metadata.device_pointer(),
                    sparse_group_keys.device_pointer(),
                    sparse_gap_thresholds.device_pointer(),
                    sparse_transition_upper.device_pointer(),
                    sparse_base_masks.device_pointer(),
                    sparse_transition_masks.device_pointer(),
                    exogenous_seed,
                    offset,
                    SPARSE_RNG_GAMMA,
                    SPARSE_RNG_MULTIPLIER0,
                    SPARSE_RNG_MULTIPLIER1,
                    exogenous.sparse_group_count() as i32,
                    exogenous.mask_words as i32,
                    chunk as i32,
                    chunk as i32,
                )
            }
            .grid((exogenous_blocks, mask_word_grid, 1))
        };
        if warmed_exogenous_grids.insert((
            exogenous_blocks,
            draw_group_grid,
            sparse_group_grid,
            mask_word_grid,
        )) {
            let warmup_start = Instant::now();
            if draw_group_count != 0 {
                launch_exogenous_partials().sync_on(&stream)?;
            }
            launch_exogenous_reduction().sync_on(&stream)?;
            if sparse_group_grid != 0 {
                launch_sparse_exogenous().sync_on(&stream)?;
            }
            warmup_s += warmup_start.elapsed().as_secs_f64();
        }
        let exogenous_kernel_start = Instant::now();
        if draw_group_count != 0 {
            launch_exogenous_partials().sync_on(&stream)?;
        }
        launch_exogenous_reduction().sync_on(&stream)?;
        let dense_device_values = if shots <= 1024 {
            let mut values = vec![0u64; exogenous.mask_words * chunk];
            unsafe {
                cutile::cuda_core::memcpy_dtoh_async(
                    values.as_mut_ptr(),
                    expression_values.device_pointer().cu_deviceptr(),
                    values.len(),
                    &stream,
                );
                stream.synchronize()?;
            }
            Some(values)
        } else {
            None
        };
        if sparse_group_grid != 0 {
            launch_sparse_exogenous().sync_on(&stream)?;
        }
        exogenous_kernel_s += exogenous_kernel_start.elapsed().as_secs_f64();
        if shots <= 1024 {
            let mut random_probe = vec![0.0f32; exogenous_random_len];
            let mut device_values = vec![0u64; exogenous.mask_words * chunk];
            unsafe {
                if !random_probe.is_empty() {
                    cutile::cuda_core::memcpy_dtoh_async(
                        random_probe.as_mut_ptr(),
                        exogenous_randoms.device_pointer().cu_deviceptr(),
                        random_probe.len(),
                        &stream,
                    );
                }
                cutile::cuda_core::memcpy_dtoh_async(
                    device_values.as_mut_ptr(),
                    expression_values.device_pointer().cu_deviceptr(),
                    device_values.len(),
                    &stream,
                );
                stream.synchronize()?;
            }
            let mut expected = vec![0u64; exogenous.mask_words * chunk];
            for word in 0..exogenous.mask_words {
                expected[word * chunk..(word + 1) * chunk].fill(exogenous.constant_masks[word]);
            }
            for shot in 0..chunk {
                for draw in 0..exogenous.draw_count {
                    let uniform = random_probe[draw * chunk + shot];
                    for word in 0..exogenous.mask_words {
                        expected[word * chunk + shot] ^=
                            exogenous.draw_base_masks[draw * exogenous.mask_words + word];
                        for transition in exogenous.draw_transition_offsets[draw] as usize
                            ..exogenous.draw_transition_offsets[draw + 1] as usize
                        {
                            if uniform <= exogenous.transition_upper[transition] {
                                expected[word * chunk + shot] ^= exogenous.transition_masks
                                    [transition * exogenous.mask_words + word];
                            }
                        }
                    }
                }
            }
            if let Some(index) = dense_device_values
                .as_ref()
                .expect("small runs copy dense exogenous values")
                .iter()
                .zip(&expected)
                .position(|(actual, expected)| actual != expected)
            {
                let word = index / chunk;
                let shot = index % chunk;
                bail!(
                    "GPU dense exogenous mismatch at word {word}, shot {shot}: expected {:#018x}, got {:#018x}",
                    expected[index],
                    dense_device_values.as_ref().expect("dense values")[index],
                );
            }
            for shot in 0..chunk {
                for word in 0..exogenous.mask_words {
                    expected[word * chunk + shot] ^=
                        exogenous.sparse_mask(exogenous_seed, offset + shot as u64, word);
                }
            }
            if let Some(index) = expected
                .iter()
                .zip(&device_values)
                .position(|(expected, actual)| expected != actual)
            {
                let word = index / chunk;
                let shot = index % chunk;
                bail!(
                    "GPU exogenous evaluator mismatch at word {word}, shot {shot}: expected {:#018x}, got {:#018x}",
                    expected[index],
                    device_values[index],
                );
            }
        }

        let setup_start = Instant::now();
        let random_shots = if gpu_plan.branch_count == 0 { 1 } else { chunk };
        let random_len = gpu_plan.branch_count.max(1) * random_shots;
        // Safety: `branch_randoms` owns at least `random_len` contiguous f32
        // slots and remains live through every kernel read.
        unsafe {
            branch_rng
                .generate_uniform_f32(branch_randoms.device_pointer().cu_deviceptr(), random_len);
            stream.synchronize()?;
        }
        rng_s += setup_start.elapsed().as_secs_f64();

        // Safety: every raw pointer comes from a live cuTile allocation. The
        // fixed metadata strides and output mask bound all device accesses.
        let launch_small = || {
            unsafe {
                kernel::sample16(
                    block_counts.device_pointer(),
                    metadata.device_pointer(),
                    controls.device_pointer(),
                    parameters.device_pointer(),
                    expression_values.device_pointer(),
                    branch_randoms.device_pointer(),
                    instruction_count as i32,
                    chunk as i32,
                    chunk as i32,
                    1u64 << (gpu_plan.logical.block & 63),
                    gpu_plan.logical.branch_masks[0],
                    gpu_plan.logical.branch_masks[1],
                    gpu_plan.logical.branch_masks[2],
                    gpu_plan.logical.branch_masks[3],
                )
            }
            .grid((sample_blocks, 1, 1))
        };
        let launch_medium = || {
            unsafe {
                kernel::sample1024(
                    block_counts.device_pointer(),
                    metadata.device_pointer(),
                    controls.device_pointer(),
                    expectation_indices.device_pointer(),
                    parameters.device_pointer(),
                    expression_values.device_pointer(),
                    expectations.device_pointer(),
                    detector_values.device_pointer(),
                    branch_randoms.device_pointer(),
                    instruction_count as i32,
                    chunk as i32,
                    chunk as i32,
                    keep_records as i32,
                    (gpu_plan.logical.block >> 6) as i32,
                    1u64 << (gpu_plan.logical.block & 63),
                    gpu_plan.logical.branch_masks[0],
                    gpu_plan.logical.branch_masks[1],
                    gpu_plan.logical.branch_masks[2],
                    gpu_plan.logical.branch_masks[3],
                )
            }
            .compile_options(CompileOptions::default().occupancy(4))
            .grid((sample_blocks, 1, 1))
        };
        let launch_compact = || {
            unsafe {
                kernel::sample128(
                    block_counts.device_pointer(),
                    metadata.device_pointer(),
                    controls.device_pointer(),
                    expectation_indices.device_pointer(),
                    parameters.device_pointer(),
                    expression_values.device_pointer(),
                    expectations.device_pointer(),
                    detector_values.device_pointer(),
                    branch_randoms.device_pointer(),
                    instruction_count as i32,
                    chunk as i32,
                    chunk as i32,
                    keep_records as i32,
                    (gpu_plan.logical.block >> 6) as i32,
                    1u64 << (gpu_plan.logical.block & 63),
                    gpu_plan.logical.branch_masks[0],
                    gpu_plan.logical.branch_masks[1],
                    gpu_plan.logical.branch_masks[2],
                    gpu_plan.logical.branch_masks[3],
                )
            }
            .grid((sample_blocks, 1, 1))
        };
        let launch_large = || {
            unsafe {
                kernel::sample4096(
                    block_counts.device_pointer(),
                    metadata.device_pointer(),
                    controls.device_pointer(),
                    expectation_indices.device_pointer(),
                    parameters.device_pointer(),
                    expression_values.device_pointer(),
                    expectations.device_pointer(),
                    detector_values.device_pointer(),
                    branch_randoms.device_pointer(),
                    instruction_count as i32,
                    chunk as i32,
                    chunk as i32,
                    keep_records as i32,
                    (gpu_plan.logical.block >> 6) as i32,
                    1u64 << (gpu_plan.logical.block & 63),
                    gpu_plan.logical.branch_masks[0],
                    gpu_plan.logical.branch_masks[1],
                    gpu_plan.logical.branch_masks[2],
                    gpu_plan.logical.branch_masks[3],
                )
            }
            .compile_options(CompileOptions::default().occupancy(2))
            .grid((sample_blocks, 1, 1))
        };
        let launch_wide = || -> Result<()> {
            let wide = wide_buffers
                .as_ref()
                .expect("wide buffers exist for the wide launch path");
            // Safety: every pointer targets a live allocation, and all launches
            // are ordered on `stream` before the final synchronization.
            unsafe {
                kernel::wide_sample_init(
                    wide.primary_re.device_pointer(),
                    wide.primary_im.device_pointer(),
                    wide.branches.device_pointer(),
                    wide.discarded.device_pointer(),
                    wide.dimension as i32,
                    (1usize << input.program.initial_k) as i32,
                    chunk as i32,
                )
                .grid((sample_blocks, 1, 1))
                .async_on(&stream)?;
            }

            let mut active_k = input.program.initial_k;
            let mut primary = true;
            for (instruction_index, instruction) in gpu_plan
                .instructions
                .iter()
                .take(instruction_count)
                .enumerate()
            {
                let (input_re, input_im, input_stride) = if primary {
                    (
                        wide.primary_re.device_pointer(),
                        wide.primary_im.device_pointer(),
                        wide.dimension,
                    )
                } else {
                    (
                        wide.scratch_re.device_pointer(),
                        wide.scratch_im.device_pointer(),
                        wide.scratch_dimension,
                    )
                };
                let (output_re, output_im, output_stride, next_primary) =
                    if instruction.opcode == plan::OP_MEASURE {
                        if primary {
                            (
                                wide.scratch_re.device_pointer(),
                                wide.scratch_im.device_pointer(),
                                wide.scratch_dimension,
                                false,
                            )
                        } else {
                            (
                                wide.primary_re.device_pointer(),
                                wide.primary_im.device_pointer(),
                                wide.dimension,
                                true,
                            )
                        }
                    } else if instruction.opcode == plan::OP_PROMOTE {
                        (
                            wide.primary_re.device_pointer(),
                            wide.primary_im.device_pointer(),
                            wide.dimension,
                            true,
                        )
                    } else {
                        (input_re, input_im, input_stride, primary)
                    };
                unsafe {
                    kernel::wide_sample_step(
                        input_re,
                        input_im,
                        output_re,
                        output_im,
                        wide.branches.device_pointer(),
                        wide.discarded.device_pointer(),
                        metadata.device_pointer(),
                        controls.device_pointer(),
                        parameters.device_pointer(),
                        expression_values.device_pointer(),
                        branch_randoms.device_pointer(),
                        instruction_index as i32,
                        active_k as i32,
                        input_stride as i32,
                        output_stride as i32,
                        chunk as i32,
                    )
                    .grid((sample_blocks, 1, 1))
                    .async_on(&stream)?;
                }
                if instruction.opcode == plan::OP_PROMOTE {
                    active_k += 1;
                } else if instruction.opcode == plan::OP_MEASURE {
                    active_k -= 1;
                }
                primary = next_primary;
            }
            unsafe {
                kernel::wide_sample_finalize(
                    block_counts.device_pointer(),
                    wide.branches.device_pointer(),
                    wide.discarded.device_pointer(),
                    expression_values.device_pointer(),
                    chunk as i32,
                    (gpu_plan.logical.block >> 6) as i32,
                    1u64 << (gpu_plan.logical.block & 63),
                    gpu_plan.logical.branch_masks[0],
                    gpu_plan.logical.branch_masks[1],
                    gpu_plan.logical.branch_masks[2],
                    gpu_plan.logical.branch_masks[3],
                )
                .grid((sample_blocks, 1, 1))
                .async_on(&stream)?;
            }
            Ok(())
        };
        let launch_sample = || -> Result<()> {
            if wide_buffers.is_some() {
                launch_wide()?;
            } else {
                // Safety: the launch arguments own every allocation until the
                // following count reduction synchronizes this stream.
                unsafe {
                    if sample_tile_shots == 64 {
                        launch_small().async_on(&stream)?;
                    } else if input.program.max_k <= 7 {
                        launch_compact().async_on(&stream)?;
                    } else if input.program.max_k <= 10 {
                        launch_medium().async_on(&stream)?;
                    } else {
                        launch_large().async_on(&stream)?;
                    }
                }
            }
            Ok(())
        };
        let launch_count_reduction = || {
            unsafe {
                kernel::reduce_block_counts(
                    count_partials.device_pointer(),
                    block_counts.device_pointer(),
                    sample_blocks as i32,
                )
            }
            .grid((count_partial_blocks, 1, 1))
        };
        if warmed_sample_grids.insert((sample_blocks, sample_tile_shots as u32)) {
            let warmup_start = Instant::now();
            launch_sample()?;
            launch_count_reduction().sync_on(&stream)?;
            warmup_s += warmup_start.elapsed().as_secs_f64();
        }

        let kernel_start = Instant::now();
        launch_sample()?;
        launch_count_reduction().sync_on(&stream)?;
        kernel_s += kernel_start.elapsed().as_secs_f64();

        let copy_start = Instant::now();
        let count_partials = (&count_partials).to_host_vec().sync_on(&stream)?;
        let detector_chunk = keep_records
            .then(|| (&detector_values).to_host_vec().sync_on(&stream))
            .transpose()?;
        let expectation_chunk = keep_records
            .then(|| (&expectations).to_host_vec().sync_on(&stream))
            .transpose()?;
        copy_s += copy_start.elapsed().as_secs_f64();

        let mut chunk_discarded = 0u64;
        let mut chunk_logical_errors = 0u64;
        for counts in count_partials
            .chunks_exact(2)
            .take(count_partial_blocks as usize)
        {
            chunk_discarded += counts[0];
            chunk_logical_errors += counts[1];
        }
        discarded += chunk_discarded;
        logical_errors += normalized_logical_errors(
            chunk as u64,
            chunk_discarded,
            chunk_logical_errors,
            expected_logical,
        );
        if let (Some(detector_chunk), Some(expectation_chunk)) = (detector_chunk, expectation_chunk)
        {
            for shot in 0..chunk {
                for detector in 0..input.program.ndetectors {
                    recorded_detectors.push(detector_chunk[detector * chunk + shot] as u8);
                }
                for expectation in 0..input.program.nexpvals {
                    recorded_expectations
                        .push(expectation_chunk[expectation * chunk + shot] as f64);
                }
            }
        }
    }

    let accepted = shots - discarded;
    let sample_s = sample_start.elapsed().as_secs_f64() - warmup_s;
    if let Some(report_path) = report_path {
        println!("sampler cutile");
        println!("file {}", report_path.display());
        println!("qubits {}", input.program.n);
        println!("records {}", input.program.nrecords);
        println!("max_active_qubits {}", input.program.max_k);
        println!("sample_tile_shots {sample_tile_shots}");
        println!("chunk_shots {chunk_limit}");
        println!("gpu_instructions {}", gpu_plan.instructions.len());
        let instruction_count = |opcode| {
            gpu_plan
                .instructions
                .iter()
                .filter(|instruction| instruction.opcode == opcode)
                .count()
        };
        let rotations = gpu_plan
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode == plan::OP_ROTATE);
        let uniform_rotations = rotations
            .clone()
            .filter(|instruction| instruction.zmask == 0)
            .count();
        let unique_rotation_xmasks = rotations
            .clone()
            .map(|instruction| instruction.xmask)
            .collect::<HashSet<_>>()
            .len();
        let rotation_xor_bits: u32 = rotations
            .map(|instruction| instruction.xmask.count_ones())
            .sum();
        let x_basis_rotations = gpu_plan
            .instructions
            .iter()
            .filter(|instruction| {
                instruction.opcode == plan::OP_ROTATE && instruction.params[3] != 0.0
            })
            .count();
        let x_basis_runs = gpu_plan
            .instructions
            .iter()
            .filter(|instruction| {
                instruction.opcode == plan::OP_ROTATE && instruction.diagonal_phase
            })
            .count();
        let x_basis_support_bits: u32 = gpu_plan
            .instructions
            .iter()
            .filter(|instruction| {
                instruction.opcode == plan::OP_ROTATE && instruction.diagonal_phase
            })
            .map(|instruction| instruction.pivot.count_ones())
            .sum();
        let repeated_rotation_xmask_pairs = gpu_plan
            .instructions
            .windows(2)
            .filter(|pair| {
                pair[0].opcode == plan::OP_ROTATE
                    && pair[1].opcode == plan::OP_ROTATE
                    && pair[0].xmask == pair[1].xmask
            })
            .count();
        let uniform_run_lengths: Vec<usize> = gpu_plan
            .instructions
            .split(|instruction| instruction.opcode != plan::OP_ROTATE || instruction.zmask != 0)
            .map(<[_]>::len)
            .filter(|&len| len != 0)
            .collect();
        let max_uniform_run = uniform_run_lengths.iter().copied().max().unwrap_or(0);
        let measurement_xor_bits: u32 = gpu_plan
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode == plan::OP_MEASURE)
            .map(|instruction| instruction.xmask.count_ones())
            .sum();
        println!(
            "instruction_mix rotations={} unique_rotation_xmasks={unique_rotation_xmasks} repeated_rotation_xmask_pairs={repeated_rotation_xmask_pairs} uniform_rotations={uniform_rotations} uniform_runs={} max_uniform_run={max_uniform_run} x_basis_rotations={x_basis_rotations} x_basis_runs={x_basis_runs} x_basis_support_bits={x_basis_support_bits} rotation_xor_bits={rotation_xor_bits} promotions={} measurements={} measurement_xor_bits={measurement_xor_bits} dormant_branches={} detectors={}",
            instruction_count(plan::OP_ROTATE),
            uniform_run_lengths.len(),
            instruction_count(plan::OP_PROMOTE),
            instruction_count(plan::OP_MEASURE),
            instruction_count(plan::OP_DORMANT_BRANCH),
            instruction_count(plan::OP_DETECTOR),
        );
        println!("adaptive_branches {}", gpu_plan.branch_count);
        println!(
            "exogenous_sources dense_draws={} dense_groups={} sparse_groups={} sparse_sets={} categorical={} rare={} bernoulli={} low_probability={} expression_rows={} mask_words={} dense_transitions={}",
            gpu_plan.exogenous_plan.draw_count,
            draw_group_count,
            gpu_plan.exogenous_plan.sparse_group_count(),
            gpu_plan.exogenous_plan.sparse_set_count(),
            input.program.sampled_categorical_distributions.len(),
            input
                .program
                .sampled_rare_categorical_groups
                .iter()
                .map(|group| group.conditions.len())
                .sum::<usize>(),
            input.program.sampled_bernoulli_conditions.len(),
            input
                .program
                .sampled_low_probability_bernoulli_groups
                .iter()
                .map(|group| group.conditions.len())
                .sum::<usize>(),
            gpu_plan.expression_plan.block_expressions.len(),
            gpu_plan.exogenous_plan.mask_words,
            gpu_plan.exogenous_plan.transition_upper.len(),
        );
        println!("shots {shots}");
        println!("discarded {discarded}");
        println!("accepted {accepted}");
        println!("logical_errors {logical_errors}");
        println!(
            "detector_postselection {}",
            has_postselection(&input.program)
        );
        println!("parse_s {parse_s}");
        println!("plan_s {plan_s}");
        println!("rng_setup_s {rng_setup_s}");
        println!("warmup_s {warmup_s}");
        println!("exogenous_rng_s {exogenous_rng_s}");
        println!("exogenous_kernel_s {exogenous_kernel_s}");
        println!("rng_s {rng_s}");
        println!("kernel_s {kernel_s}");
        println!("copy_s {copy_s}");
        println!(
            "execute_s {}",
            exogenous_rng_s + exogenous_kernel_s + rng_s + kernel_s + copy_s
        );
        println!("sample_s {sample_s}");
    }
    let mut observable_ones = vec![0; parsed.observable_count()];
    if let Some(count) = observable_ones.get_mut(observable) {
        *count = logical_errors;
    }
    Ok(SampleResult {
        counts: SampleCounts {
            shots,
            discarded,
            accepted,
            logical_errors,
        },
        timing: SamplingTiming {
            compile_s: parse_s + plan_s + rng_setup_s + warmup_s,
            presample_s: exogenous_rng_s + exogenous_kernel_s,
            execute_s: rng_s + kernel_s + copy_s,
            sample_s,
        },
        active_threads: 1,
        record_rows,
        detectors: recorded_detectors,
        exp_vals: recorded_expectations,
        observable_ones,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_reduction_applies_cpu_reference_bits() {
        let circuit = Circuit::from_text("X 0\nM 0\nDETECTOR rec[-1]\n").expect("parses");
        let program = plan_circuit(&circuit, &[1]).expect("plans");
        let mut gpu_plan = plan::GpuPlan::build(&program, &[]).expect("lowers");
        apply_detector_reference(&mut gpu_plan, &[1]);
        let detector = gpu_plan
            .instructions
            .iter()
            .find(|instruction| instruction.detector == Some(0))
            .expect("detector instruction");
        assert_eq!(detector.z_without_pivot, 1);
        assert_eq!(normalized_logical_errors(10, 3, 2, true), 5);
    }

    #[test]
    fn exact_k_replaces_independent_draws_with_uniform_weight_k_rows() {
        let mut program = FactoredInstructionProgram {
            sampled_bernoulli_conditions: vec![1, 2],
            sampled_bernoulli_probabilities: vec![0.125; 2],
            sampled_categorical_distributions: vec![
                SymbolicCategoricalDistribution {
                    nbits: 1,
                    conditions: vec![3],
                    assignments: vec![vec![0], vec![1]],
                    probabilities: vec![0.875, 0.125],
                },
                SymbolicCategoricalDistribution {
                    nbits: 1,
                    conditions: vec![4],
                    assignments: vec![vec![1], vec![0]],
                    probabilities: vec![0.125, 0.875],
                },
            ],
            ..FactoredInstructionProgram::default()
        };
        condition_exact_k(&mut program, 2).expect("valid exact-k plan");
        assert!(program.sampled_bernoulli_conditions.is_empty());
        let distribution = &program.sampled_categorical_distributions[0];
        assert_eq!(distribution.conditions, [1, 2, 3, 4]);
        assert_eq!(distribution.assignments.len(), 6);
        assert!(
            distribution
                .assignments
                .iter()
                .all(|row| row[0].count_ones() == 2)
        );
        assert!(
            distribution
                .probabilities
                .iter()
                .all(|&probability| (probability - 1.0 / 6.0).abs() < f64::EPSILON)
        );
    }
}
