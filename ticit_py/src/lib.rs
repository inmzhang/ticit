//! Python bindings for ticit, exposed as `ticit._core`.

#[cfg(feature = "gpu")]
use std::hash::{BuildHasher, Hasher, RandomState};
#[cfg(feature = "gpu")]
use std::num::NonZeroUsize;
use std::sync::Mutex;

use numpy::{IntoPyArray, PyArray1, PyArray2, ndarray::Array2};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

mod simulator;

pyo3_stub_gen::create_exception!(
    ticit._core,
    ParseError,
    PyValueError,
    "A circuit failed to parse."
);

fn ticit_error(error: ticit::TicitError) -> PyErr {
    match error {
        ticit::TicitError::Parse { .. } => ParseError::new_err(error.to_string()),
        ticit::TicitError::Io { .. } => PyOSError::new_err(error.to_string()),
        ticit::TicitError::InvalidInput { .. } | ticit::TicitError::Unsupported { .. } => {
            PyValueError::new_err(error.to_string())
        }
        ticit::TicitError::Internal { .. } | ticit::TicitError::WorkerPanic => {
            PyRuntimeError::new_err(error.to_string())
        }
    }
}

#[cfg(feature = "gpu")]
fn random_seed() -> u64 {
    RandomState::new().build_hasher().finish()
}

/// A parsed `.ticit`/Stim-style circuit.
///
/// Call `circuit.compile(...)` when the circuit will be sampled.
///
/// Examples:
///     >>> import ticit
///     >>> circuit = ticit.Circuit("H 0\nM 0\nDETECTOR rec[-1]")
///     >>> (circuit.num_qubits, circuit.num_measurements, circuit.num_detectors)
///     (1, 1, 1)
#[gen_stub_pyclass]
#[pyclass(
    name = "Circuit",
    module = "ticit._core",
    frozen,
    unsendable,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
struct PyCircuit(ticit::Circuit);

#[gen_stub_pymethods]
#[pymethods]
impl PyCircuit {
    /// Parses `stim_text` into a circuit. An empty string creates an empty circuit.
    #[new]
    #[pyo3(signature = (stim_text=""))]
    fn new(stim_text: &str) -> PyResult<Self> {
        ticit::Circuit::from_text(stim_text)
            .map(Self)
            .map_err(ticit_error)
    }

    /// Parses a circuit from source text.
    #[staticmethod]
    fn from_text(stim_text: &str) -> PyResult<Self> {
        Self::new(stim_text)
    }

    /// Parses a circuit from a UTF-8 file.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        ticit::Circuit::from_file(path)
            .map(Self)
            .map_err(ticit_error)
    }

    /// Compiles this circuit into a reusable `Program`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        postselection_mask=None,
        *,
        normalize_syndromes=false,
        expected_detectors=None,
        expected_observables=None,
        backend="cpu",
        observable=0,
        threads=1,
        sample_chunk_shots=0,
        batch_size=0,
        gpu_chunk_shots=1_048_576,
    ))]
    fn compile(
        &self,
        postselection_mask: Option<Vec<u8>>,
        normalize_syndromes: bool,
        expected_detectors: Option<Vec<u8>>,
        expected_observables: Option<Vec<u8>>,
        backend: &str,
        observable: usize,
        threads: usize,
        sample_chunk_shots: usize,
        batch_size: usize,
        gpu_chunk_shots: usize,
    ) -> PyResult<PyProgram> {
        compile_circuit(
            &self.0,
            postselection_mask.unwrap_or_default(),
            normalize_syndromes,
            expected_detectors.unwrap_or_default(),
            expected_observables.unwrap_or_default(),
            backend,
            observable,
            threads,
            sample_chunk_shots,
            batch_size,
            gpu_chunk_shots,
        )
    }

    /// Computes the full noiseless detector and observable sample.
    fn reference_sample(&self) -> PyResult<PyReferenceSample> {
        self.0
            .reference_sample()
            .map(PyReferenceSample::from)
            .map_err(ticit_error)
    }

    /// Number of qubits named by the circuit.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.0.qubit_count()
    }

    /// Number of measurement records produced by the circuit.
    #[getter]
    fn num_measurements(&self) -> usize {
        self.0.measurement_record_count()
    }

    /// Number of detector declarations.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.0.detector_count()
    }

    /// Number of observable indices.
    #[getter]
    fn num_observables(&self) -> usize {
        self.0.observable_count()
    }

    /// Number of expectation values produced by the circuit.
    #[getter]
    fn num_exp_vals(&self) -> usize {
        self.0.expectation_value_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "Circuit(num_qubits={}, num_measurements={}, num_detectors={}, num_observables={})",
            self.num_qubits(),
            self.num_measurements(),
            self.num_detectors(),
            self.num_observables(),
        )
    }
}

/// Full noiseless detector and observable parity vectors.
#[gen_stub_pyclass]
#[pyclass(
    name = "ReferenceSample",
    module = "ticit._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
struct PyReferenceSample {
    /// One bool per detector declaration.
    detectors: Vec<bool>,
    /// One bool per observable index.
    observables: Vec<bool>,
}

impl From<ticit::ReferenceSample> for PyReferenceSample {
    fn from(sample: ticit::ReferenceSample) -> Self {
        Self {
            detectors: sample.detectors.into_iter().map(|bit| bit != 0).collect(),
            observables: sample.observables.into_iter().map(|bit| bit != 0).collect(),
        }
    }
}

enum Backend {
    Cpu(Box<ticit::Sampler>),
    #[cfg(feature = "gpu")]
    Gpu {
        circuit: Box<ticit::Circuit>,
        chunk_shots: NonZeroUsize,
        postselect_detectors: bool,
        expected_detectors: Vec<u8>,
        expected_observables: Vec<u8>,
    },
}

/// A compiled circuit that can be sampled repeatedly.
///
/// Construct programs with `Circuit.compile`; this class has no public constructor.
#[gen_stub_pyclass]
#[pyclass(name = "Program", module = "ticit._core", skip_from_py_object)]
struct PyProgram {
    backend: Mutex<Backend>,
    backend_name: &'static str,
    num_qubits: usize,
    num_measurements: usize,
    num_detectors: usize,
    num_observables: usize,
    num_exp_vals: usize,
    observable: usize,
    has_postselection: bool,
}

impl PyProgram {
    fn run(
        &self,
        py: Python<'_>,
        shots: u64,
        seed: Option<u64>,
        keep_records: bool,
        bit_packed: bool,
    ) -> PyResult<PySampleResult> {
        if shots == 0 {
            return Err(PyValueError::new_err("shots must be positive"));
        }
        let observable = self.observable;
        let result = py.detach(|| -> PyResult<ticit::SampleResult> {
            match &mut *self
                .backend
                .lock()
                .map_err(|_| PyRuntimeError::new_err("the program sampler lock was poisoned"))?
            {
                Backend::Cpu(sampler) => {
                    let compile_s = sampler.preprocessing_timing().compile_s;
                    let mut result = match (seed, keep_records) {
                        (Some(seed), true) => sampler.sample_with_seed(shots, seed, bit_packed),
                        (Some(seed), false) => sampler.sample_counts_with_seed(shots, seed),
                        (None, true) => sampler.sample(shots, bit_packed),
                        (None, false) => sampler.sample_counts(shots),
                    }
                    .map_err(ticit_error)?;
                    result.timing.compile_s = compile_s;
                    Ok(result)
                }
                #[cfg(feature = "gpu")]
                Backend::Gpu {
                    circuit,
                    chunk_shots,
                    postselect_detectors,
                    expected_detectors,
                    expected_observables,
                } => ticit::gpu::sample_circuit_with_reference(
                    circuit,
                    shots,
                    seed.unwrap_or_else(random_seed),
                    *chunk_shots,
                    *postselect_detectors,
                    observable,
                    expected_detectors,
                    expected_observables,
                )
                .map_err(|error| PyRuntimeError::new_err(error.to_string())),
            }
        })?;
        let mut result = result;
        result.bit_packed = bit_packed;
        if keep_records && u64::try_from(result.record_rows) != Ok(result.counts.accepted) {
            return Err(PyRuntimeError::new_err(format!(
                "the {} backend does not support per-shot record output",
                self.backend_name,
            )));
        }
        PySampleResult::new(
            py,
            result,
            observable,
            self.num_measurements,
            self.num_detectors,
            self.num_observables,
            self.num_exp_vals,
        )
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProgram {
    /// Samples this compiled circuit and returns per-shot records and counters.
    ///
    /// With `bit_packed=True`, each record array has `ceil(num_bits / 8)`
    /// columns and uses NumPy's little bit order.
    #[pyo3(signature = (shots, seed=None, *, bit_packed=false))]
    fn sample(
        &self,
        py: Python<'_>,
        shots: u64,
        seed: Option<u64>,
        bit_packed: bool,
    ) -> PyResult<PySampleResult> {
        self.run(py, shots, seed, true, bit_packed)
    }

    /// Selected execution backend: `"cpu"` or `"gpu"`.
    #[getter]
    fn backend(&self) -> &str {
        self.backend_name
    }

    /// Number of qubits in the circuit.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of measurement records in the circuit.
    #[getter]
    fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    /// Number of detectors in the circuit.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Number of observable indices in the circuit.
    #[getter]
    fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Number of expectation values in each result row.
    #[getter]
    fn num_exp_vals(&self) -> usize {
        self.num_exp_vals
    }

    /// Observable index counted as a logical error.
    #[getter]
    fn observable(&self) -> usize {
        self.observable
    }

    /// Whether detector postselection is active.
    #[getter]
    fn has_postselection(&self) -> bool {
        self.has_postselection
    }

    fn __repr__(&self) -> String {
        format!(
            "Program(backend={:?}, num_qubits={}, num_detectors={}, num_observables={})",
            self.backend_name, self.num_qubits, self.num_detectors, self.num_observables,
        )
    }
}

/// Per-shot records, aggregate counters, and timing from one sampling call.
#[gen_stub_pyclass]
#[pyclass(
    name = "SampleResult",
    module = "ticit._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug)]
struct PySampleResult {
    /// Row-major `uint8` measurement array.
    measurements: Py<PyArray2<u8>>,
    /// Row-major `uint8` detector array.
    detectors: Py<PyArray2<u8>>,
    /// Row-major `uint8` observable array.
    observables: Py<PyArray2<u8>>,
    /// `float64` array shaped `(passed_shots, num_exp_vals)`.
    exp_vals: Py<PyArray2<f64>>,
    /// Attempted shots. Alias of `shots`.
    total_shots: u64,
    /// Attempted shots.
    shots: u64,
    /// Shots rejected by detector postselection. Alias of `discarded`.
    discards: u64,
    /// Shots rejected by detector postselection.
    discarded: u64,
    /// Shots retained after postselection. Alias of `accepted`.
    passed_shots: u64,
    /// Shots retained after postselection.
    accepted: u64,
    /// Accepted shots where the selected observable was one.
    logical_errors: u64,
    /// `uint64` count for every observable column.
    observable_ones: Py<PyArray1<u64>>,
    /// Observable index represented by `logical_errors`.
    observable: usize,
    /// Circuit parsing/planning time in seconds.
    compile_s: f64,
    /// Exogenous-noise generation time in seconds.
    presample_s: f64,
    /// Circuit execution time in seconds.
    execute_s: f64,
    /// Total steady-state sampling time in seconds.
    sample_s: f64,
    /// CPU worker count used by the call; one for GPU sampling.
    active_threads: usize,
    /// Whether the three bit arrays pack eight bits into each byte.
    bit_packed: bool,
}

impl PySampleResult {
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        result: ticit::SampleResult,
        observable: usize,
        num_measurements: usize,
        num_detectors: usize,
        num_observables: usize,
        num_exp_vals: usize,
    ) -> PyResult<Self> {
        let counts = result.counts;
        let rows = result.record_rows;
        let bit_packed = result.bit_packed;
        Ok(Self {
            measurements: array2(
                py,
                result.measurements,
                rows,
                output_columns(num_measurements, bit_packed),
            )?,
            detectors: array2(
                py,
                result.detectors,
                rows,
                output_columns(num_detectors, bit_packed),
            )?,
            observables: array2(
                py,
                result.observables,
                rows,
                output_columns(num_observables, bit_packed),
            )?,
            exp_vals: array2(py, result.exp_vals, rows, num_exp_vals)?,
            total_shots: counts.shots,
            shots: counts.shots,
            discards: counts.discarded,
            discarded: counts.discarded,
            passed_shots: counts.accepted,
            accepted: counts.accepted,
            logical_errors: counts.logical_errors,
            observable_ones: result.observable_ones.into_pyarray(py).unbind(),
            observable,
            compile_s: result.timing.compile_s,
            presample_s: result.timing.presample_s,
            execute_s: result.timing.execute_s,
            sample_s: result.timing.sample_s,
            active_threads: result.active_threads,
            bit_packed,
        })
    }
}

fn output_columns(bits: usize, bit_packed: bool) -> usize {
    if bit_packed { bits.div_ceil(8) } else { bits }
}

fn array2<T: numpy::Element>(
    py: Python<'_>,
    values: Vec<T>,
    rows: usize,
    columns: usize,
) -> PyResult<Py<PyArray2<T>>> {
    Array2::from_shape_vec((rows, columns), values)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        .map(|array| array.into_pyarray(py).unbind())
}

#[gen_stub_pymethods]
#[pymethods]
impl PySampleResult {
    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tuple = PyTuple::new(
            py,
            [
                self.measurements.clone_ref(py),
                self.detectors.clone_ref(py),
                self.observables.clone_ref(py),
            ],
        )?;
        Ok(tuple.call_method0("__iter__")?.unbind())
    }

    /// Fraction of attempted shots rejected by postselection, or NaN for zero shots.
    #[getter]
    fn discard_rate(&self) -> f64 {
        self.discarded as f64 / self.shots as f64
    }

    /// Fraction of accepted shots with a logical error, or NaN if none passed.
    #[getter]
    fn logical_error_rate(&self) -> f64 {
        self.logical_errors as f64 / self.accepted as f64
    }

    fn __repr__(&self) -> String {
        format!(
            "SampleResult(total_shots={}, passed_shots={}, discards={}, logical_errors={})",
            self.total_shots, self.passed_shots, self.discards, self.logical_errors,
        )
    }
}

/// Parses a quantum circuit from source text.
///
/// Examples:
///     >>> import ticit
///     >>> ticit.parse("M 0").num_measurements
///     1
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
fn parse(text: &str) -> PyResult<PyCircuit> {
    PyCircuit::new(text)
}

/// Parses a quantum circuit from a UTF-8 file.
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
fn parse_file(path: &str) -> PyResult<PyCircuit> {
    PyCircuit::from_file(path)
}

/// Compiles circuit text into a reusable `Program`.
///
/// The first five parameters mirror Clifft's `compile` call.
/// `normalize_syndromes=True` computes a noiseless reference on the CPU before
/// sampling. `backend="gpu"` requires a package built with Cargo feature `gpu`.
///
/// Args:
///     stim_text: Circuit in ticit's Stim-style text format.
///     postselection_mask: Zero/nonzero flag for each detector.
///     expected_detectors: Explicit detector reference bits.
///     expected_observables: Explicit observable reference bits.
///     normalize_syndromes: Compute and apply a noiseless reference sample.
///     backend: `"cpu"` or `"gpu"`.
///     observable: Observable index counted as a logical error.
///     threads: CPU worker count.
///     sample_chunk_shots: CPU shots per scheduling chunk; zero selects ticit's default.
///     batch_size: CPU execution batch size; zero selects ticit's default.
///     gpu_chunk_shots: Maximum shots in one GPU launch group.
///
/// Examples:
///     >>> import ticit
///     >>> program = ticit.compile("H 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]", threads=1)
///     >>> program.backend
///     'cpu'
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    stim_text,
    postselection_mask=None,
    expected_detectors=None,
    expected_observables=None,
    normalize_syndromes=false,
    *,
    backend="cpu",
    observable=0,
    threads=1,
    sample_chunk_shots=0,
    batch_size=0,
    gpu_chunk_shots=1_048_576,
))]
fn compile(
    stim_text: &str,
    postselection_mask: Option<Vec<u8>>,
    expected_detectors: Option<Vec<u8>>,
    expected_observables: Option<Vec<u8>>,
    normalize_syndromes: bool,
    backend: &str,
    observable: usize,
    threads: usize,
    sample_chunk_shots: usize,
    batch_size: usize,
    gpu_chunk_shots: usize,
) -> PyResult<PyProgram> {
    let circuit = ticit::Circuit::from_text(stim_text).map_err(ticit_error)?;
    compile_circuit(
        &circuit,
        postselection_mask.unwrap_or_default(),
        normalize_syndromes,
        expected_detectors.unwrap_or_default(),
        expected_observables.unwrap_or_default(),
        backend,
        observable,
        threads,
        sample_chunk_shots,
        batch_size,
        gpu_chunk_shots,
    )
}

#[allow(clippy::too_many_arguments)]
fn compile_circuit(
    circuit: &ticit::Circuit,
    mask: Vec<u8>,
    normalize_syndromes: bool,
    expected_detectors: Vec<u8>,
    expected_observables: Vec<u8>,
    backend: &str,
    observable: usize,
    threads: usize,
    sample_chunk_shots: usize,
    batch_size: usize,
    gpu_chunk_shots: usize,
) -> PyResult<PyProgram> {
    #[cfg(not(feature = "gpu"))]
    let _ = gpu_chunk_shots;
    if threads == 0 {
        return Err(PyValueError::new_err("threads must be positive"));
    }
    if !mask.is_empty() && mask.len() != circuit.detector_count() {
        return Err(PyValueError::new_err(format!(
            "postselection_mask has length {}, expected {}",
            mask.len(),
            circuit.detector_count(),
        )));
    }
    if normalize_syndromes && (!expected_detectors.is_empty() || !expected_observables.is_empty()) {
        return Err(PyValueError::new_err(
            "normalize_syndromes cannot be combined with expected_detectors or expected_observables",
        ));
    }
    if !expected_detectors.is_empty() && expected_detectors.len() != circuit.detector_count() {
        return Err(PyValueError::new_err(format!(
            "expected_detectors has length {}, expected {}",
            expected_detectors.len(),
            circuit.detector_count(),
        )));
    }
    if !expected_observables.is_empty() && expected_observables.len() != circuit.observable_count()
    {
        return Err(PyValueError::new_err(format!(
            "expected_observables has length {}, expected {}",
            expected_observables.len(),
            circuit.observable_count(),
        )));
    }
    let num_qubits = circuit.qubit_count();
    let num_measurements = circuit.measurement_record_count();
    let num_detectors = circuit.detector_count();
    let num_observables = circuit.observable_count();
    let num_exp_vals = circuit.expectation_value_count();

    let (backend, backend_name, has_postselection) = match backend.to_ascii_lowercase().as_str() {
        "cpu" => {
            let sampler = circuit
                .compile(ticit::SamplerOptions {
                    observable,
                    postselection_mask: mask,
                    normalize_syndromes,
                    expected_detectors,
                    expected_observables,
                    sample_chunk_shots,
                    batch_size,
                    threads,
                })
                .map_err(ticit_error)?;
            let postselection = sampler.info().detector_postselection;
            (Backend::Cpu(Box::new(sampler)), "cpu", postselection)
        }
        "gpu" => {
            #[cfg(feature = "gpu")]
            {
                let reference = if normalize_syndromes {
                    circuit.reference_sample().map_err(ticit_error)?
                } else {
                    ticit::ReferenceSample {
                        detectors: expected_detectors,
                        observables: expected_observables,
                    }
                };
                let requested_all = !mask.is_empty() && mask.iter().all(|&flag| flag != 0);
                let requested_any = mask.iter().any(|&flag| flag != 0);
                let postselect_detectors = requested_all || circuit.all_detectors_postselected();
                let selective_source =
                    circuit.has_detector_postselection() && !circuit.all_detectors_postselected();
                if (requested_any && !requested_all) || (selective_source && !requested_all) {
                    return Err(PyValueError::new_err(
                        "the GPU backend supports only no detector postselection or all detectors postselected",
                    ));
                }
                let chunk_shots = NonZeroUsize::new(gpu_chunk_shots)
                    .ok_or_else(|| PyValueError::new_err("gpu_chunk_shots must be positive"))?;
                (
                    Backend::Gpu {
                        circuit: Box::new(circuit.clone()),
                        chunk_shots,
                        postselect_detectors,
                        expected_detectors: reference.detectors,
                        expected_observables: reference.observables,
                    },
                    "gpu",
                    postselect_detectors,
                )
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err(PyRuntimeError::new_err(
                    "the GPU backend requires ticit_py built with Cargo feature `gpu`",
                ));
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown backend {other:?}; expected 'cpu' or 'gpu'",
            )));
        }
    };

    Ok(PyProgram {
        backend: Mutex::new(backend),
        backend_name,
        num_qubits,
        num_measurements,
        num_detectors,
        num_observables,
        num_exp_vals,
        observable,
        has_postselection,
    })
}

/// Samples a compiled program and returns per-shot records and counters.
///
/// `seed=None` uses OS-provided entropy. A fixed seed makes a call
/// reproducible. Postselected programs return one row per surviving shot.
///
/// Examples:
///     >>> import ticit
///     >>> result = ticit.sample(ticit.compile("M 0"), shots=20, seed=5)
///     >>> (result.total_shots, result.passed_shots, result.discards)
///     (20, 20, 0)
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
#[pyo3(signature = (program, shots, seed=None, *, bit_packed=false))]
fn sample(
    py: Python<'_>,
    program: PyRef<'_, PyProgram>,
    shots: u64,
    seed: Option<u64>,
    bit_packed: bool,
) -> PyResult<PySampleResult> {
    program.run(py, shots, seed, true, bit_packed)
}

/// Clifft-compatible postselected sampling with optional survivor records.
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
#[pyo3(signature = (program, shots, seed=None, keep_records=false, *, bit_packed=false))]
fn sample_survivors(
    py: Python<'_>,
    program: PyRef<'_, PyProgram>,
    shots: u64,
    seed: Option<u64>,
    keep_records: bool,
    bit_packed: bool,
) -> PyResult<PySampleResult> {
    program.run(py, shots, seed, keep_records, bit_packed)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ParseError", m.py().get_type::<ParseError>())?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyReferenceSample>()?;
    m.add_class::<PyProgram>()?;
    m.add_class::<PySampleResult>()?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(parse_file, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(sample_survivors, m)?)?;
    simulator::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
pyo3_stub_gen::module_variable!("ticit._core", "__version__", String);
