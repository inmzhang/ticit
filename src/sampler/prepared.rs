//! Prepared count samplers — the public high-throughput API.
//!
//! A prepared sampler owns the planned program, the interned expression plan
//! and per-worker buffers; `sample_with_seed(shots, seed, bit_packed)` returns
//! per-shot records and aggregate counts. Seeding is chunk/block-indexed via
//! [`block_seed`] so results depend only on `(seed, chunk_index)` — never on
//! which worker ran a chunk — keeping multithreaded scheduling bit-identical.
//!
//! Batch threads use scoped standard-library threads over independent worker
//! states.

use std::{
    hash::{BuildHasher, Hasher, RandomState},
    thread,
    time::Instant,
};

use crate::batch::{
    BatchDetectorPostselectionOptions, BatchDetectorPostselectionScratch,
    BatchFactoredExecutorState, default_batch_count, execute_batch_in_place_expressions,
    execute_batch_postselected_in_place, prepare_batch_detector_postselection_scratch_for_program,
    reset_batch_executor,
};
use crate::circuit::ir::CircuitObservableInclude;
use crate::circuit::{Circuit, has_postselection, plan_circuit};
use crate::errors::{Result, TicitError};
use crate::exogenous::{
    PackedPresampledExogenous, prepare_presampled_exogenous_packed,
    resample_prepared_exogenous_packed_in_place,
};
use crate::factored::FactoredInstructionProgram;
use crate::presampled_expression::{
    PresampledExpressionBlock, PresampledExpressionPlan, evaluate_presampled_expression_block,
    prepare_presampled_expression_plan,
};
use crate::random::block_seed;
const EXOGENOUS_SEED_BASE: u64 = 0x7eed_0000;
const BRANCH_SEED_BASE: u64 = 0x5eed_1234;
const DEFAULT_SAMPLE_CHUNK_SHOTS: usize = 2048;
const POSTSELECTION_COMPACTION_DENOMINATOR: usize = 2;

/// Aggregate outcomes from a batch sampling call.
///
/// Counts satisfy `shots == discarded + accepted`. `logical_errors` counts
/// accepted shots where the selected observable parity is one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SampleCounts {
    /// Total attempted shots.
    pub shots: u64,
    /// Shots rejected by detector postselection.
    pub discarded: u64,
    /// Shots retained after detector postselection.
    pub accepted: u64,
    /// Accepted shots where [`SamplerOptions::observable`] was one.
    pub logical_errors: u64,
}

impl SampleCounts {
    /// Returns `discarded / shots`, or NaN when no shots were attempted.
    #[must_use]
    pub fn discard_rate(&self) -> f64 {
        if self.shots == 0 {
            f64::NAN
        } else {
            self.discarded as f64 / self.shots as f64
        }
    }

    /// Returns `logical_errors / accepted`, or NaN when no shots were accepted.
    #[must_use]
    pub fn logical_error_rate(&self) -> f64 {
        if self.accepted == 0 {
            f64::NAN
        } else {
            self.logical_errors as f64 / self.accepted as f64
        }
    }
}

/// Wall and phase timings reported by preparation and sampling.
///
/// [`Sampler::preprocessing_timing`] carries `compile_s`. A
/// [`SampleResult`] carries the other fields; its `compile_s` is zero because
/// compilation is not repeated on each call. With multiple workers,
/// `presample_s` and `execute_s` are sums of worker time and can exceed the
/// wall-clock `sample_s`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplingTiming {
    /// Circuit planning time in seconds.
    pub compile_s: f64,
    /// Exogenous-noise generation and expression evaluation time in seconds.
    pub presample_s: f64,
    /// Factored-program execution and result reduction time in seconds.
    pub execute_s: f64,
    /// Wall-clock time for the complete sampling call in seconds.
    pub sample_s: f64,
}

/// Configuration used when compiling a reusable [`Sampler`].
///
/// Zero selects the documented automatic value for `sample_chunk_shots` and
/// `batch_size`. A zero `threads` value is normalized to one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplerOptions {
    /// Observable index whose one parity is counted as a logical error.
    pub observable: usize,
    /// Flat detector flags; any nonzero entry enables postselection.
    pub postselection_mask: Vec<u8>,
    /// Compute and XOR a noiseless detector/observable reference sample.
    pub normalize_syndromes: bool,
    /// Explicit detector reference bits; empty leaves detector parity raw.
    pub expected_detectors: Vec<u8>,
    /// Explicit observable reference bits; empty leaves observable parity raw.
    pub expected_observables: Vec<u8>,
    /// 0 selects the default (`max(2048, batch_size)` for the batch sampler).
    pub sample_chunk_shots: usize,
    /// 0 selects a batch size based on the circuit's peak active width.
    pub batch_size: usize,
    /// Maximum CPU worker count.
    pub threads: usize,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            observable: 0,
            postselection_mask: Vec::new(),
            normalize_syndromes: false,
            expected_detectors: Vec::new(),
            expected_observables: Vec::new(),
            sample_chunk_shots: 0,
            batch_size: 0,
            threads: 1,
        }
    }
}

/// Read-only metadata for a prepared [`Sampler`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplerInfo {
    /// Circuit qubit count.
    pub qubits: usize,
    /// Circuit measurement-record count.
    pub measurement_records: usize,
    /// Circuit detector count.
    pub detectors: usize,
    /// Circuit observable-index count.
    pub observables: usize,
    /// Circuit expectation-value count.
    pub expectation_values: usize,
    /// Selected observable index.
    pub observable: usize,
    /// Peak active-state width in qubits.
    pub max_active_qubits: usize,
    /// Shots executed together in one bit-packed batch.
    pub batch_size: usize,
    /// Shots assigned to one scheduling and presampling chunk.
    pub sample_chunk_shots: usize,
    /// Configured maximum worker count.
    pub threads: usize,
    /// Whether the planner selected component-local active-state execution.
    pub active_components: bool,
    /// Whether any source or caller-selected detector postselection is active.
    pub detector_postselection: bool,
    /// Runtime-selected CPU SIMD backend name.
    pub cpu_backend: &'static str,
}

/// Row-major shot records, aggregate counts, and timing from one [`Sampler`] call.
///
/// Each record vector has `record_rows` rows. Column counts are available from
/// the sampler's [`SamplerInfo`]. With `bit_packed`, each row has
/// `ceil(column_count / 8)` bytes in little bit order. Normal sampling stores
/// every accepted row; the count-only methods set `record_rows` to zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SampleResult {
    /// Attempted, discarded, accepted, and logical-error counts.
    pub counts: SampleCounts,
    /// Sampling phase timings; compilation timing lives on the sampler.
    pub timing: SamplingTiming,
    /// Workers that received at least one chunk.
    pub active_threads: usize,
    /// Rows stored in each per-shot output vector.
    pub record_rows: usize,
    /// Whether measurement, detector, and observable rows pack eight bits per byte.
    pub bit_packed: bool,
    /// Row-major measurement bits.
    pub measurements: Vec<u8>,
    /// Row-major detector bits.
    pub detectors: Vec<u8>,
    /// Row-major observable bits.
    pub observables: Vec<u8>,
    /// Number of one bits in each observable column.
    pub observable_ones: Vec<u64>,
    /// Row-major expectation values.
    pub exp_vals: Vec<f64>,
}

/// Full noiseless detector and observable parity vectors for a circuit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceSample {
    /// One bit per detector declaration.
    pub detectors: Vec<u8>,
    /// One bit per observable index, including unused gaps.
    pub observables: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SamplingInput {
    pub program: FactoredInstructionProgram,
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub logical_records: Vec<Vec<i32>>,
    pub observable_records: Vec<Vec<Vec<i32>>>,
    pub observable: usize,
    pub observable_includes: usize,
    pub preprocessing_timing: SamplingTiming,
}

/// Record groups of every `OBSERVABLE_INCLUDE` whose index matches.
pub(crate) fn logical_records_for_observable(
    observables: &[CircuitObservableInclude],
    observable: usize,
) -> Vec<Vec<i32>> {
    observables
        .iter()
        .filter(|include| include.index == observable)
        .map(|include| {
            include
                .records
                .iter()
                .map(|&record| record as i32)
                .collect()
        })
        .collect()
}

fn logical_records_by_observable(
    observables: &[CircuitObservableInclude],
    observable_count: usize,
) -> Vec<Vec<Vec<i32>>> {
    (0..observable_count)
        .map(|observable| logical_records_for_observable(observables, observable))
        .collect()
}

#[cfg(any(test, feature = "gpu"))]
pub(crate) fn make_circuit_sampling_input(
    program: FactoredInstructionProgram,
    logical_records: Vec<Vec<i32>>,
    observable: usize,
    observable_includes: usize,
    preprocessing_timing: SamplingTiming,
) -> SamplingInput {
    let mut observable_records = vec![Vec::new(); observable_includes];
    if let Some(records) = observable_records.get_mut(observable) {
        *records = logical_records.clone();
    }
    SamplingInput {
        program,
        logical_records,
        observable_records,
        observable,
        observable_includes,
        preprocessing_timing,
    }
}

fn seconds_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64()
}

fn random_seed() -> u64 {
    RandomState::new().build_hasher().finish()
}

fn checked_output_len(shots: u64, columns: usize) -> Result<usize> {
    usize::try_from(shots)
        .ok()
        .and_then(|shots| shots.checked_mul(columns))
        .ok_or_else(|| TicitError::new("sampling output size exceeds usize"))
}

fn output_columns(columns: usize, bit_packed: bool) -> usize {
    if bit_packed {
        columns.div_ceil(8)
    } else {
        columns
    }
}

fn sample_chunk_or_default(requested: usize, min_auto_shots: usize) -> usize {
    if requested > 0 {
        requested
    } else {
        DEFAULT_SAMPLE_CHUNK_SHOTS.max(min_auto_shots)
    }
}

fn make_info(
    program: &FactoredInstructionProgram,
    observable: usize,
    observable_includes: usize,
    options: &SamplerOptions,
    batch_size: usize,
    postselection: bool,
) -> SamplerInfo {
    SamplerInfo {
        qubits: program.n,
        measurement_records: program.nrecords,
        detectors: program.ndetectors,
        observables: observable_includes,
        expectation_values: program.nexpvals,
        observable,
        max_active_qubits: program.max_k,
        batch_size,
        sample_chunk_shots: options.sample_chunk_shots,
        threads: options.threads,
        active_components: program.use_active_components,
        detector_postselection: postselection,
        cpu_backend: crate::contiguous::backend_name(),
    }
}

fn prepare_expression_plan(
    expression_plan: &mut PresampledExpressionPlan,
    program: &FactoredInstructionProgram,
) -> Result<()> {
    let mut samples = PackedPresampledExogenous::default();
    prepare_presampled_exogenous_packed(&mut samples, program)?;
    prepare_presampled_expression_plan(expression_plan, program, &samples)
}

fn reference_sample_for_program(
    program: &FactoredInstructionProgram,
    observable_records: &[Vec<Vec<i32>>],
) -> Result<ReferenceSample> {
    if program.ndetectors == 0 && observable_records.is_empty() {
        return Ok(ReferenceSample::default());
    }
    let mut samples = PackedPresampledExogenous::default();
    prepare_presampled_exogenous_packed(&mut samples, program)?;
    samples.nshots = 1;
    samples.shot_words = 1;
    samples.value_words.resize(program.nsymbols, 0);
    samples.value_words.fill(0);
    samples.sparse_condition_words.fill(0);
    samples.sparse_hit_offsets.resize(program.nsymbols + 1, 0);
    samples.sparse_hit_offsets.fill(0);

    let mut expression_plan = PresampledExpressionPlan::default();
    prepare_presampled_expression_plan(&mut expression_plan, program, &samples)?;
    let mut expression_block = PresampledExpressionBlock::default();
    evaluate_presampled_expression_block(&mut expression_block, &expression_plan, &samples)?;

    let mut runtime = BatchFactoredExecutorState::new(program, 1, 1)?;
    runtime.dense_shot_major_active = true;
    runtime.store_detector_records = true;
    reset_batch_executor(&mut runtime, program, 1)?;
    execute_batch_in_place_expressions(
        &mut runtime,
        program,
        &expression_plan,
        &expression_block,
        0,
    )?;

    let detectors = (0..program.ndetectors)
        .map(|detector| u8::from(runtime.detector_words[detector * runtime.batch_words] & 1 != 0))
        .collect();
    let mut observable_words = Vec::new();
    fill_observable_words(&mut observable_words, &runtime, observable_records)?;
    let observables = (0..observable_records.len())
        .map(|observable| u8::from(observable_words[observable * runtime.batch_words] & 1 != 0))
        .collect();
    Ok(ReferenceSample {
        detectors,
        observables,
    })
}

pub(crate) fn circuit_reference_sample(circuit: &Circuit) -> Result<ReferenceSample> {
    let program = plan_circuit(circuit, &[])?;
    let observable_records =
        logical_records_by_observable(&circuit.observables, circuit.observable_count());
    reference_sample_for_program(&program, &observable_records)
}

fn ceil_div_u64(numerator: u64, denominator: u64) -> Result<u64> {
    if denominator == 0 {
        return Err(TicitError::new("division by zero in chunk sizing"));
    }
    Ok(numerator.div_ceil(denominator))
}

fn active_worker_count(requested: usize, nchunks: u64) -> usize {
    nchunks.max(1).min(requested as u64) as usize
}

// ==============================================================================
// Batch prepared sampler
// ==============================================================================

struct BatchWorker {
    counts: SampleCounts,
    timing: SamplingTiming,
    samples: PackedPresampledExogenous,
    expression_block: PresampledExpressionBlock,
    runtime: BatchFactoredExecutorState,
    observable_words: Vec<u64>,
    measurements: Vec<u8>,
    detectors: Vec<u8>,
    observables: Vec<u8>,
    observable_ones: Vec<u64>,
    exp_vals: Vec<f64>,
    postselection_scratch: BatchDetectorPostselectionScratch,
}

/// A circuit compiled and buffered for repeated batch sampling.
///
/// Compilation performs all circuit planning and allocates one independent
/// worker state per configured thread. Calls mutate those reusable buffers, so
/// a sampler is intentionally used through `&mut self`.
///
/// # Examples
///
/// ```
/// use ticit::{Circuit, SamplerOptions};
///
/// let circuit = Circuit::from_text(
///     "H 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]",
/// )?;
/// let mut sampler = circuit.compile(
///     SamplerOptions { threads: 1, ..Default::default() },
/// )?;
/// let result = sampler.sample_with_seed(100, 7, false)?;
/// assert_eq!(result.counts.shots, 100);
/// assert_eq!(result.counts.accepted, 100);
/// assert_eq!(result.record_rows, 100);
/// # Ok::<(), ticit::TicitError>(())
/// ```
pub struct Sampler {
    options: SamplerOptions,
    postselection: bool,
    program: FactoredInstructionProgram,
    observable_records: Vec<Vec<Vec<i32>>>,
    retained_observable_records: Vec<Vec<i32>>,
    retained_output_records: Vec<Vec<i32>>,
    info: SamplerInfo,
    preprocessing_timing: SamplingTiming,
    expression_plan: PresampledExpressionPlan,
    workers: Vec<BatchWorker>,
}

impl Sampler {
    /// Plans `circuit`, applies detector postselection options, and allocates
    /// reusable sampling workers.
    ///
    /// Source `DISCARD` declarations are unioned with
    /// [`SamplerOptions::postselection_mask`]. A nonzero mask entry enables
    /// the corresponding detector; missing entries are treated as zero.
    ///
    /// # Errors
    ///
    /// Returns an error if the circuit cannot be planned, an option overflows
    /// an internal index, or the required active state is unsupported.
    pub(crate) fn new(circuit: &Circuit, options: SamplerOptions) -> Result<Self> {
        let compile_start = Instant::now();
        let program = plan_circuit(circuit, &options.postselection_mask)?;
        let observable_includes = circuit.observable_count();
        let observable_records =
            logical_records_by_observable(&circuit.observables, observable_includes);
        let input = SamplingInput {
            program,
            logical_records: observable_records
                .get(options.observable)
                .cloned()
                .unwrap_or_default(),
            observable_records,
            observable: options.observable,
            observable_includes,
            preprocessing_timing: SamplingTiming {
                compile_s: seconds_since(compile_start),
                ..Default::default()
            },
        };
        Self::from_input(input, options)
    }

    pub(crate) fn from_input(input: SamplingInput, mut options: SamplerOptions) -> Result<Self> {
        if options.normalize_syndromes
            && (!options.expected_detectors.is_empty() || !options.expected_observables.is_empty())
        {
            return Err(TicitError::new(
                "normalize_syndromes cannot be combined with expected_detectors or expected_observables",
            ));
        }
        if !options.expected_detectors.is_empty()
            && options.expected_detectors.len() != input.program.ndetectors
        {
            return Err(TicitError::new(format!(
                "expected_detectors has length {}, expected {}",
                options.expected_detectors.len(),
                input.program.ndetectors,
            )));
        }
        if !options.expected_observables.is_empty()
            && options.expected_observables.len() != input.observable_includes
        {
            return Err(TicitError::new(format!(
                "expected_observables has length {}, expected {}",
                options.expected_observables.len(),
                input.observable_includes,
            )));
        }
        if options.normalize_syndromes {
            let reference =
                reference_sample_for_program(&input.program, &input.observable_records)?;
            options.expected_detectors = reference.detectors;
            options.expected_observables = reference.observables;
        }
        options.batch_size = if options.batch_size > 0 {
            options.batch_size
        } else {
            default_batch_count(input.program.max_k)?
        };
        options.sample_chunk_shots =
            sample_chunk_or_default(options.sample_chunk_shots, options.batch_size);
        options.threads = options.threads.max(1);
        let postselection = has_postselection(&input.program);
        let info = make_info(
            &input.program,
            input.observable,
            input.observable_includes,
            &options,
            options.batch_size,
            postselection,
        );
        let mut expression_plan = PresampledExpressionPlan::default();
        prepare_expression_plan(&mut expression_plan, &input.program)?;
        let retained_observable_records = input
            .observable_records
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        let retained_output_records = vec![(1..=input.program.nrecords as i32).collect()];
        let mut workers = Vec::with_capacity(options.threads);
        for _ in 0..options.threads {
            let mut runtime =
                BatchFactoredExecutorState::new(&input.program, options.batch_size, 1)?;
            runtime.dense_shot_major_active = true;
            let mut samples = PackedPresampledExogenous::default();
            prepare_presampled_exogenous_packed(&mut samples, &input.program)?;
            let mut worker = BatchWorker {
                counts: SampleCounts::default(),
                timing: SamplingTiming::default(),
                samples,
                expression_block: PresampledExpressionBlock::default(),
                observable_words: vec![0; input.observable_includes * runtime.batch_words],
                measurements: Vec::new(),
                detectors: Vec::new(),
                observables: Vec::new(),
                observable_ones: vec![0; input.observable_includes],
                exp_vals: Vec::new(),
                postselection_scratch: BatchDetectorPostselectionScratch::default(),
                runtime,
            };
            if postselection {
                let postselection_options = BatchDetectorPostselectionOptions {
                    mask_dead_shots_min_fraction_denominator: POSTSELECTION_COMPACTION_DENOMINATOR,
                    retained_record_uses: Some(&retained_output_records),
                    expected_detectors: &options.expected_detectors,
                };
                prepare_batch_detector_postselection_scratch_for_program(
                    &mut worker.postselection_scratch,
                    &worker.runtime,
                    &input.program,
                    &postselection_options,
                )?;
            }
            workers.push(worker);
        }
        Ok(Self {
            options,
            postselection,
            observable_records: input.observable_records,
            retained_observable_records,
            retained_output_records,
            info,
            preprocessing_timing: input.preprocessing_timing,
            expression_plan,
            workers,
            program: input.program,
        })
    }

    /// Returns immutable circuit, layout, thread, and backend metadata.
    #[must_use]
    pub fn info(&self) -> &SamplerInfo {
        &self.info
    }

    /// Returns the planning timing recorded by [`Circuit::compile`].
    #[must_use]
    pub fn preprocessing_timing(&self) -> &SamplingTiming {
        &self.preprocessing_timing
    }

    /// Samples `shots` using an internally generated seed.
    ///
    /// When `bit_packed` is true, each row stores eight little-endian bits per
    /// byte, matching `numpy.packbits(..., bitorder="little", axis=1)`.
    ///
    /// # Errors
    ///
    /// Returns an error if noise presampling or factored execution fails, or
    /// if a worker panics.
    pub fn sample(&mut self, shots: u64, bit_packed: bool) -> Result<SampleResult> {
        self.sample_impl(shots, random_seed(), true, bit_packed)
    }

    /// Samples `shots` using `seed`.
    ///
    /// Results depend on the seed and logical chunk indices, not thread
    /// scheduling, so the same sampler configuration and seed are reproducible
    /// across runs.
    ///
    /// # Errors
    ///
    /// Returns an error if noise presampling or factored execution fails, or
    /// if a worker panics.
    pub fn sample_with_seed(
        &mut self,
        shots: u64,
        seed: u64,
        bit_packed: bool,
    ) -> Result<SampleResult> {
        self.sample_impl(shots, seed, true, bit_packed)
    }

    /// Samples aggregate counters without retaining per-shot rows.
    ///
    /// # Errors
    ///
    /// Returns an error if noise presampling or factored execution fails, or
    /// if a worker panics.
    pub fn sample_counts(&mut self, shots: u64) -> Result<SampleResult> {
        self.sample_impl(shots, random_seed(), false, false)
    }

    /// Samples aggregate counters with `seed` without retaining per-shot rows.
    ///
    /// # Errors
    ///
    /// Returns an error if noise presampling or factored execution fails, or
    /// if a worker panics.
    pub fn sample_counts_with_seed(&mut self, shots: u64, seed: u64) -> Result<SampleResult> {
        self.sample_impl(shots, seed, false, false)
    }

    fn sample_impl(
        &mut self,
        shots: u64,
        seed: u64,
        keep_records: bool,
        bit_packed: bool,
    ) -> Result<SampleResult> {
        if keep_records {
            checked_output_len(
                shots,
                output_columns(self.info.measurement_records, bit_packed),
            )?;
            checked_output_len(shots, output_columns(self.info.detectors, bit_packed))?;
            checked_output_len(shots, output_columns(self.info.observables, bit_packed))?;
            checked_output_len(shots, self.info.expectation_values)?;
        }
        let chunk_shots = self.options.sample_chunk_shots as u64;
        let batch_size = self.options.batch_size;
        let nchunks = ceil_div_u64(shots, chunk_shots)?;
        let blocks_per_chunk = ceil_div_u64(chunk_shots, batch_size as u64)?;
        let active_threads = active_worker_count(self.options.threads, nchunks);
        let mut result = SampleResult {
            active_threads,
            bit_packed,
            observable_ones: vec![0; self.info.observables],
            ..Default::default()
        };
        let postselection_options = BatchDetectorPostselectionOptions {
            mask_dead_shots_min_fraction_denominator: POSTSELECTION_COMPACTION_DENOMINATOR,
            retained_record_uses: Some(if keep_records {
                &self.retained_output_records
            } else {
                &self.retained_observable_records
            }),
            expected_detectors: &self.options.expected_detectors,
        };

        let sample_start = Instant::now();
        let postselection = self.postselection;
        let program = &self.program;
        let expression_plan = &self.expression_plan;
        let observable_records = &self.observable_records;
        let expected_detectors = &self.options.expected_detectors;
        let expected_observables = &self.options.expected_observables;
        let selected_observable = self.options.observable;
        let workers = &mut self.workers[..active_threads];
        let run_worker = |worker_id: usize, worker: &mut BatchWorker| -> Result<()> {
            worker.counts = SampleCounts::default();
            worker.timing = SamplingTiming::default();
            worker.measurements.clear();
            worker.detectors.clear();
            worker.observables.clear();
            worker.observable_ones.fill(0);
            worker.exp_vals.clear();
            worker.runtime.store_detector_records = keep_records;
            let mut chunk_index = worker_id as u64;
            while chunk_index < nchunks {
                let chunk_offset = chunk_index * chunk_shots;
                let chunk_shots_here = chunk_shots.min(shots - chunk_offset) as usize;

                let presample_start = Instant::now();
                resample_prepared_exogenous_packed_in_place(
                    &mut worker.samples,
                    program,
                    chunk_shots_here,
                    block_seed(EXOGENOUS_SEED_BASE, seed, chunk_index),
                )?;
                evaluate_presampled_expression_block(
                    &mut worker.expression_block,
                    expression_plan,
                    &worker.samples,
                )?;
                worker.timing.presample_s += seconds_since(presample_start);

                let execute_start = Instant::now();
                let mut chunk_local_offset = 0usize;
                let mut local_block_index = 0u64;
                while chunk_local_offset < chunk_shots_here {
                    let block = batch_size.min(chunk_shots_here - chunk_local_offset);
                    let block_index = chunk_index * blocks_per_chunk + local_block_index;
                    // Expression-mode execution never reads a symbol value row
                    // before fully writing it (branch symbols are assigned by
                    // whole-row copies; exogenous symbols live in the
                    // expression block), so neither caller clears the value
                    // table.
                    reset_batch_executor(&mut worker.runtime, program, block)?;
                    worker.runtime.rng_state = block_seed(BRANCH_SEED_BASE, seed, block_index);
                    if postselection {
                        let postselection_result = execute_batch_postselected_in_place(
                            &mut worker.runtime,
                            program,
                            expression_plan,
                            &worker.expression_block,
                            chunk_local_offset,
                            &mut worker.postselection_scratch,
                            &postselection_options,
                        )?;
                        worker.counts.discarded += postselection_result.discarded as u64;
                    } else {
                        execute_batch_in_place_expressions(
                            &mut worker.runtime,
                            program,
                            expression_plan,
                            &worker.expression_block,
                            chunk_local_offset,
                        )?;
                    }
                    append_block_outputs(
                        worker,
                        observable_records,
                        expected_detectors,
                        expected_observables,
                        selected_observable,
                        keep_records,
                        bit_packed,
                    )?;
                    worker.counts.shots += block as u64;
                    chunk_local_offset += batch_size;
                    local_block_index += 1;
                }
                worker.timing.execute_s += seconds_since(execute_start);
                chunk_index += active_threads as u64;
            }
            Ok(())
        };

        if active_threads == 1 {
            run_worker(0, &mut workers[0])?;
        } else {
            thread::scope(|scope| -> Result<()> {
                let mut handles = Vec::with_capacity(active_threads);
                for (worker_id, worker) in workers.iter_mut().enumerate() {
                    let run_worker = &run_worker;
                    handles.push(scope.spawn(move || run_worker(worker_id, worker)));
                }
                for handle in handles {
                    handle.join().map_err(|_| TicitError::WorkerPanic)??;
                }
                Ok(())
            })?;
        }

        for worker in workers {
            result.counts.shots += worker.counts.shots;
            result.counts.discarded += worker.counts.discarded;
            result.counts.accepted += worker.counts.accepted;
            result.counts.logical_errors += worker.counts.logical_errors;
            result.timing.presample_s += worker.timing.presample_s;
            result.timing.execute_s += worker.timing.execute_s;
            for (total, count) in result
                .observable_ones
                .iter_mut()
                .zip(&worker.observable_ones)
            {
                *total += count;
            }
            result.measurements.append(&mut worker.measurements);
            result.detectors.append(&mut worker.detectors);
            result.observables.append(&mut worker.observables);
            result.exp_vals.append(&mut worker.exp_vals);
        }
        if keep_records {
            result.record_rows = usize::try_from(result.counts.accepted)
                .map_err(|_| TicitError::new("sample row count exceeds usize"))?;
        }
        result.timing.sample_s = seconds_since(sample_start);
        Ok(result)
    }
}

// ==============================================================================
// Batch accumulation
// ==============================================================================

fn batch_word_count(shots: usize) -> usize {
    shots.div_ceil(64)
}

fn live_word_mask(shots: usize, word: usize) -> u64 {
    let remaining = shots as i64 - ((word as i64) << 6);
    if remaining <= 0 {
        0
    } else if remaining >= 64 {
        u64::MAX
    } else {
        (1u64 << remaining) - 1
    }
}

fn fill_observable_words(
    out: &mut Vec<u64>,
    runtime: &BatchFactoredExecutorState,
    observable_records: &[Vec<Vec<i32>>],
) -> Result<()> {
    let stride_words = runtime.batch_words;
    let nwords = batch_word_count(runtime.active_shots);
    out.resize(observable_records.len() * stride_words, 0);
    out.fill(0);
    for (observable, includes) in observable_records.iter().enumerate() {
        let out_base = observable * stride_words;
        for records in includes {
            for &record in records {
                if record <= 0 || record as usize > runtime.nrecords {
                    return Err(TicitError::new(
                        "observable references an out-of-range measurement record",
                    ));
                }
                let record_base = (record - 1) as usize * stride_words;
                for word in 0..nwords {
                    out[out_base + word] ^= runtime.measurement_words[record_base + word];
                }
            }
        }
    }
    Ok(())
}

fn append_bit_rows(
    out: &mut Vec<u8>,
    columns: &[u64],
    expected: &[u8],
    column_count: usize,
    stride_words: usize,
    rows: usize,
    bit_packed: bool,
) {
    let output_columns = output_columns(column_count, bit_packed);
    out.reserve(rows * output_columns);
    for shot in 0..rows {
        let word = shot >> 6;
        let mask = 1u64 << (shot & 63);
        if bit_packed {
            let row_start = out.len();
            out.resize(row_start + output_columns, 0);
            for column in 0..column_count {
                let bit = (columns[column * stride_words + word] & mask != 0)
                    ^ expected.get(column).is_some_and(|&value| value != 0);
                if bit {
                    out[row_start + column / 8] |= 1 << (column % 8);
                }
            }
        } else {
            for column in 0..column_count {
                let bit = (columns[column * stride_words + word] & mask != 0)
                    ^ expected.get(column).is_some_and(|&value| value != 0);
                out.push(u8::from(bit));
            }
        }
    }
}

fn append_block_outputs(
    worker: &mut BatchWorker,
    observable_records: &[Vec<Vec<i32>>],
    expected_detectors: &[u8],
    expected_observables: &[u8],
    selected_observable: usize,
    keep_records: bool,
    bit_packed: bool,
) -> Result<()> {
    let BatchWorker {
        counts,
        runtime,
        observable_words,
        measurements,
        detectors,
        observables,
        observable_ones,
        exp_vals,
        ..
    } = worker;
    let rows = runtime.active_shots;
    fill_observable_words(observable_words, runtime, observable_records)?;
    for (observable, &expected) in expected_observables.iter().enumerate() {
        if expected == 0 {
            continue;
        }
        let base = observable * runtime.batch_words;
        for word in 0..batch_word_count(rows) {
            observable_words[base + word] ^= live_word_mask(rows, word);
        }
    }
    counts.accepted += rows as u64;
    let nwords = batch_word_count(rows);
    for (observable, total) in observable_ones
        .iter_mut()
        .enumerate()
        .take(observable_records.len())
    {
        let base = observable * runtime.batch_words;
        let ones = (0..nwords)
            .map(|word| {
                (observable_words[base + word] & live_word_mask(rows, word)).count_ones() as u64
            })
            .sum::<u64>();
        *total += ones;
        if observable == selected_observable {
            counts.logical_errors += ones;
        }
    }
    if keep_records {
        append_bit_rows(
            measurements,
            &runtime.measurement_words,
            &[],
            runtime.nrecords,
            runtime.batch_words,
            rows,
            bit_packed,
        );
        append_bit_rows(
            detectors,
            &runtime.detector_words,
            expected_detectors,
            runtime.ndetectors,
            runtime.batch_words,
            rows,
            bit_packed,
        );
        append_bit_rows(
            observables,
            observable_words,
            &[],
            observable_records.len(),
            runtime.batch_words,
            rows,
            bit_packed,
        );
        exp_vals.reserve(rows * runtime.nexpvals);
        for shot in 0..rows {
            for exp_val in 0..runtime.nexpvals {
                exp_vals.push(runtime.exp_values[exp_val * runtime.batches + shot]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn estimate_logical_error_rate(
    circuit: &Circuit,
    shots: u64,
    seed: u64,
) -> Result<SampleCounts> {
    let options = SamplerOptions {
        postselection_mask: vec![1; circuit.detector_count()],
        ..Default::default()
    };
    Ok(Sampler::new(circuit, options)?
        .sample_with_seed(shots, seed, false)?
        .counts)
}

#[cfg(test)]
pub(crate) fn discard_rate(counts: &SampleCounts) -> f64 {
    counts.discard_rate()
}

#[cfg(test)]
pub(crate) fn logical_error_rate(counts: &SampleCounts) -> f64 {
    counts.logical_error_rate()
}

#[cfg(test)]
mod tests {
    //! Prepared-sampler tests: counts semantics, defaults, and seed reproducibility.

    use super::*;
    use crate::circuit::parse_ticit_text;
    use crate::test_support::ccz_nontels_circuits;

    fn sampler_input(text: &str, options: &SamplerOptions) -> SamplingInput {
        let parsed = parse_ticit_text(text).expect("test circuit parses");
        let program = crate::circuit::plan_circuit(&parsed, &options.postselection_mask)
            .expect("test circuit plans");
        let logical_records =
            logical_records_for_observable(&parsed.observables, options.observable);
        make_circuit_sampling_input(
            program,
            logical_records,
            options.observable,
            parsed.observables.len(),
            Default::default(),
        )
    }

    #[test]
    #[ignore = "slow end-to-end CCZ fixture check"]
    fn direct_t_fixture_has_deterministic_detectors_without_noise() {
        let path = ccz_nontels_circuits().join("d05_p1e-3.clifft");
        let mut text = std::fs::read_to_string(path).expect("reads CCZ fixture");
        assert_eq!(text.matches("E(0.125)").count(), 8);
        for (noise, zero) in [
            ("E(0.125)", "E(0)"),
            ("DEPOLARIZE1(0.001)", "DEPOLARIZE1(0)"),
            ("DEPOLARIZE2(0.001)", "DEPOLARIZE2(0)"),
            ("X_ERROR(0.001)", "X_ERROR(0)"),
            ("Z_ERROR(0.001)", "Z_ERROR(0)"),
            ("M(0.001)", "M(0)"),
            ("MX(0.001)", "MX(0)"),
            ("MY(0.001)", "MY(0)"),
        ] {
            text = text.replace(noise, zero);
        }
        let circuit = parse_ticit_text(&text).expect("CCZ fixture parses");
        let options = SamplerOptions {
            postselection_mask: vec![1; circuit.detector_count()],
            normalize_syndromes: true,
            ..Default::default()
        };
        let counts = Sampler::new(&circuit, options)
            .expect("CCZ fixture compiles")
            .sample_counts_with_seed(64, 7)
            .expect("CCZ fixture samples")
            .counts;
        assert_eq!(counts.discarded, 0);
    }

    /// `test_detectors`: an inverted measurement flags every shot's observable;
    /// adding an always-fired detector discards every shot instead.
    #[test]
    fn estimate_matches_the_cli_conventions() {
        let parsed = parse_ticit_text("M !0\nOBSERVABLE_INCLUDE(0) rec[-1]\n").expect("parses");
        let summary = estimate_logical_error_rate(&parsed, 5, 1).expect("estimates");
        assert_eq!(summary.shots, 5);
        assert_eq!(summary.discarded, 0);
        assert_eq!(summary.accepted, 5);
        assert_eq!(summary.logical_errors, 5);
        assert!((logical_error_rate(&summary) - 1.0).abs() < 1e-15);
        assert!((discard_rate(&summary) - 0.0).abs() < 1e-15);

        let parsed = parse_ticit_text("M !0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
            .expect("parses");
        let summary = estimate_logical_error_rate(&parsed, 5, 1).expect("estimates");
        assert_eq!(summary.discarded, 5);
        assert_eq!(summary.accepted, 0);
        assert!(logical_error_rate(&summary).is_nan());

        let parsed = parse_ticit_text(
            "M !0\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-1]\n",
        )
        .expect("parses");
        let summary = estimate_logical_error_rate(&parsed, 5, 1).expect("estimates");
        assert_eq!(summary.logical_errors, 5, "classifies by observable 0 only");
    }

    #[test]
    fn batch_sampler_defaults_follow_the_cpp_sizing() {
        let options = SamplerOptions::default();
        let input = sampler_input("H 0\nT 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n", &options);
        let sampler = Sampler::from_input(input, options).expect("sampler builds");
        let info = sampler.info();
        // max_k = 1 for one active qubit: batch = min(2048, 32768/2) = 2048.
        assert_eq!(info.batch_size, 2048);
        assert_eq!(info.sample_chunk_shots, 2048);
        assert_eq!(info.threads, 1);
        assert!(!info.detector_postselection);
    }

    #[test]
    fn logical_counts_flag_a_deterministic_flip() {
        let options = SamplerOptions::default();
        let input = sampler_input("M !0\nOBSERVABLE_INCLUDE(0) rec[-1]\n", &options);
        let mut sampler = Sampler::from_input(input, options).expect("sampler builds");
        let result = sampler.sample(100, false).expect("sampling succeeds");
        assert_eq!(result.counts.shots, 100);
        assert_eq!(result.counts.discarded, 0);
        assert_eq!(result.counts.accepted, 100);
        assert_eq!(result.counts.logical_errors, 100);
    }

    #[test]
    fn noiseless_reference_normalizes_outputs_before_counting() {
        let circuit = Circuit::from_text(
            "X 0\nM 0\nDETECTOR rec[-1]\n\
             OBSERVABLE_INCLUDE(0)\nOBSERVABLE_INCLUDE(1) rec[-1]\n",
        )
        .expect("circuit parses");
        let reference = circuit.reference_sample().expect("reference samples");
        assert_eq!(reference.detectors, [1]);
        assert_eq!(reference.observables, [0, 1]);

        let mut sampler = circuit
            .compile(SamplerOptions {
                observable: 1,
                postselection_mask: vec![1],
                normalize_syndromes: true,
                ..Default::default()
            })
            .expect("circuit compiles");
        let result = sampler
            .sample_with_seed(4, 7, false)
            .expect("sampling succeeds");
        assert_eq!(
            result.counts,
            SampleCounts {
                shots: 4,
                discarded: 0,
                accepted: 4,
                logical_errors: 0,
            }
        );
        assert_eq!(result.detectors, [0; 4]);
        assert_eq!(result.observables, [0; 8]);
        assert_eq!(result.observable_ones, [0, 0]);
    }

    #[test]
    fn packed_rows_match_numpy_little_bit_order() {
        let columns = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1];
        let mut packed = Vec::new();
        append_bit_rows(&mut packed, &columns, &[], columns.len(), 1, 1, true);
        assert_eq!(packed, [0xff, 0x04]);
    }

    #[test]
    fn sample_returns_row_major_clifft_shaped_records() {
        let circuit = Circuit::from_text(
            "M !0\nM 1\nDETECTOR rec[-2] rec[-1]\n\
             OBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(2) rec[-1]\n\
             EXP_VAL Z0\n",
        )
        .expect("circuit parses");
        let mut sampler = circuit
            .compile(SamplerOptions::default())
            .expect("circuit compiles");
        let result = sampler
            .sample_with_seed(3, 7, false)
            .expect("sampling succeeds");

        assert_eq!(result.measurements, vec![1, 0, 1, 0, 1, 0]);
        assert_eq!(result.detectors, vec![1, 1, 1]);
        assert_eq!(result.observables, vec![1, 0, 0, 1, 0, 0, 1, 0, 0]);
        assert_eq!(result.observable_ones, vec![3, 0, 0]);
        assert_eq!(result.exp_vals, vec![1.0; 3]);

        let packed = sampler
            .sample_with_seed(3, 7, true)
            .expect("packed sampling succeeds");
        assert!(packed.bit_packed);
        assert_eq!(packed.measurements, vec![1; 3]);
        assert_eq!(packed.detectors, vec![1; 3]);
        assert_eq!(packed.observables, vec![1; 3]);
        assert_eq!(packed.observable_ones, result.observable_ones);
        assert_eq!(packed.exp_vals, result.exp_vals);

        let counts_only = sampler
            .sample_counts_with_seed(3, 7)
            .expect("count sampling succeeds");
        assert_eq!(counts_only.counts, result.counts);
        assert_eq!(counts_only.observable_ones, result.observable_ones);
        assert_eq!(counts_only.record_rows, 0);
        assert!(counts_only.measurements.is_empty());
    }

    #[test]
    fn postselection_returns_only_survivor_rows() {
        let circuit = Circuit::from_text(
            "EXP_VAL Z1\nM !1\nDETECTOR rec[-1]\n\
             X_ERROR(0.5) 0\nM 0\nDETECTOR rec[-1]\n\
             OBSERVABLE_INCLUDE(0) rec[-2]\n",
        )
        .expect("circuit parses");
        let mut sampler = circuit
            .compile(SamplerOptions {
                postselection_mask: vec![0, 1],
                batch_size: 16,
                ..Default::default()
            })
            .expect("circuit compiles");
        let result = sampler
            .sample_with_seed(128, 9, false)
            .expect("sampling succeeds");
        let rows = result.counts.accepted as usize;

        assert!(rows > 0);
        assert!(rows < 128);
        assert!(result.measurements.chunks_exact(2).all(|row| row == [1, 0]));
        assert!(result.detectors.chunks_exact(2).all(|row| row == [1, 0]));
        assert_eq!(result.observables, vec![1; rows]);
        assert_eq!(result.observable_ones, vec![rows as u64]);
        assert_eq!(result.exp_vals, vec![1.0; rows]);
    }

    #[test]
    fn source_discards_and_the_passed_mask_are_unioned() {
        let circuit = crate::Circuit::from_text("M !0\nDETECTOR rec[-1]\nM 0\nDISCARD rec[-1]\n")
            .expect("circuit parses");
        assert_eq!(circuit.detector_count(), 2);

        let mut source_only = Sampler::new(&circuit, SamplerOptions::default()).expect("prepares");
        let counts = source_only
            .sample_with_seed(8, 1, false)
            .expect("samples")
            .counts;
        assert_eq!((counts.accepted, counts.discarded), (8, 0));

        let mut union = Sampler::new(
            &circuit,
            SamplerOptions {
                postselection_mask: vec![1],
                ..Default::default()
            },
        )
        .expect("prepares with a short mask");
        let counts = union.sample_with_seed(8, 1, false).expect("samples").counts;
        assert_eq!((counts.accepted, counts.discarded), (0, 8));

        let source_fires = crate::Circuit::from_text("M !0\nDISCARD rec[-1]\n").expect("parses");
        let mut sampler = Sampler::new(&source_fires, SamplerOptions::default()).expect("prepares");
        let counts = sampler
            .sample_with_seed(8, 1, false)
            .expect("samples")
            .counts;
        assert_eq!((counts.accepted, counts.discarded), (0, 8));
    }

    #[test]
    fn identical_seeds_reproduce_counts() {
        let text = "X_ERROR(0.125) 0\nM 0\nDETECTOR rec[-1]\nH 1\nT 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
        let options = SamplerOptions {
            postselection_mask: vec![1],
            batch_size: 32,
            ..Default::default()
        };
        let input = sampler_input(text, &options);
        let mut sampler = Sampler::from_input(input, options).expect("sampler builds");
        let first = sampler
            .sample_with_seed(4096, 12, false)
            .expect("sampling succeeds");
        let second = sampler
            .sample_with_seed(4096, 12, false)
            .expect("sampling succeeds");
        assert_eq!(first.counts, second.counts);
        assert_eq!(first.measurements, second.measurements);
        assert_eq!(first.detectors, second.detectors);
        assert_eq!(first.observables, second.observables);
        assert!(first.counts.discarded > 0, "the X error should fire");
        assert_eq!(
            first.counts.accepted + first.counts.discarded,
            first.counts.shots
        );
        let other = sampler
            .sample_with_seed(4096, 13, false)
            .expect("sampling succeeds");
        assert_ne!(first.counts, other.counts);
    }

    #[test]
    fn batch_threads_preserve_chunk_seeded_counts() {
        let text = "X_ERROR(0.125) 0\nM 0\nDETECTOR rec[-1]\nH 1\nT 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
        let single_options = SamplerOptions {
            postselection_mask: vec![1],
            sample_chunk_shots: 96,
            batch_size: 32,
            threads: 1,
            ..Default::default()
        };
        let threaded_options = SamplerOptions {
            threads: 3,
            ..single_options.clone()
        };
        let mut single = Sampler::from_input(sampler_input(text, &single_options), single_options)
            .expect("single-worker sampler builds");
        let mut threaded =
            Sampler::from_input(sampler_input(text, &threaded_options), threaded_options)
                .expect("threaded sampler builds");

        let single_result = single
            .sample_with_seed(385, 12, false)
            .expect("single worker samples");
        let threaded_result = threaded
            .sample_with_seed(385, 12, false)
            .expect("three workers sample");
        assert_eq!(single_result.active_threads, 1);
        assert_eq!(threaded_result.active_threads, 3);
        assert_eq!(threaded_result.counts, single_result.counts);
        assert_eq!(threaded_result.counts.shots, 385);

        let capped = threaded
            .sample_with_seed(95, 13, false)
            .expect("one chunk uses one worker");
        assert_eq!(capped.active_threads, 1);
    }
}
