//! Bindings for ticit's Pauli and procedural tableau-simulator APIs.

use num_complex::Complex64;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

pyo3_stub_gen::create_exception!(
    ticit._core,
    SimulatorError,
    PyRuntimeError,
    "A simulator operation failed on the live quantum state."
);

fn sim_error(error: ticit::SimError) -> PyErr {
    match error {
        ticit::SimError::InvalidProbability(_)
        | ticit::SimError::InvalidProbabilityDistribution
        | ticit::SimError::InvalidRotationAngle(_)
        | ticit::SimError::RepeatedQubit(_)
        | ticit::SimError::NonCommutingControlledPaulis
        | ticit::SimError::InvalidControlledPauli
        | ticit::SimError::NonHermitianPauli
        | ticit::SimError::QubitIndexOutOfRange { .. } => PyValueError::new_err(error.to_string()),
        _ => SimulatorError::new_err(error.to_string()),
    }
}

fn check_qubit(q: usize, nqubits: usize) -> PyResult<()> {
    if q < nqubits {
        Ok(())
    } else {
        Err(PyValueError::new_err("qubit index is out of range"))
    }
}

/// A packed Pauli operator with a phase exponent of `i`.
///
/// String position is the qubit index. Use `pauli_string("IXYZ")` to parse a
/// dense literal, or construct identity storage with `PauliString(nqubits)` and
/// set its bits explicitly.
///
/// Examples:
///     >>> import ticit
///     >>> p = ticit.pauli_string("XYZ")
///     >>> str(p)
///     'XYZ'
///     >>> str(-p)
///     '-XYZ'
#[gen_stub_pyclass]
#[pyclass(
    name = "PauliString",
    module = "ticit._core",
    unsendable,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub(crate) struct PyPauliString(pub(crate) ticit::PauliString);

#[gen_stub_pymethods]
#[pymethods]
impl PyPauliString {
    /// Constructs identity on `nqubits` qubits.
    #[new]
    #[pyo3(signature = (nqubits=0))]
    fn new(nqubits: usize) -> Self {
        Self(ticit::PauliString::new(nqubits))
    }

    /// Parses a dense literal such as `"IXYZ"`; `_` is identity.
    #[staticmethod]
    fn from_text(text: &str) -> PyResult<Self> {
        ticit::pauli_string(text)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Number of qubits represented by the operator.
    #[getter]
    fn nqubits(&self) -> usize {
        self.0.nqubits
    }

    /// Copy of the packed LSB-first X words.
    #[getter]
    fn x(&self) -> Vec<u64> {
        self.0.x.clone()
    }

    /// Copy of the packed LSB-first Z words.
    #[getter]
    fn z(&self) -> Vec<u64> {
        self.0.z.clone()
    }

    /// Stored phase exponent of `i`, in `0..=3`.
    #[getter]
    fn phase_exponent(&self) -> i32 {
        self.0.phase_exponent()
    }

    /// Returns whether qubit `q` has an X component.
    fn xbit(&self, q: usize) -> PyResult<bool> {
        check_qubit(q, self.0.nqubits)?;
        Ok(self.0.xbit(q))
    }

    /// Returns whether qubit `q` has a Z component.
    fn zbit(&self, q: usize) -> PyResult<bool> {
        check_qubit(q, self.0.nqubits)?;
        Ok(self.0.zbit(q))
    }

    /// Sets or clears the X component on qubit `q`.
    fn set_xbit(&mut self, q: usize, value: bool) -> PyResult<()> {
        check_qubit(q, self.0.nqubits)?;
        self.0.set_xbit(q, value);
        Ok(())
    }

    /// Sets or clears the Z component on qubit `q`.
    fn set_zbit(&mut self, q: usize, value: bool) -> PyResult<()> {
        check_qubit(q, self.0.nqubits)?;
        self.0.set_zbit(q, value);
        Ok(())
    }

    /// Sets the phase exponent modulo four.
    fn set_phase(&mut self, phase_exponent: i32) {
        self.0.set_phase(phase_exponent);
    }

    /// Adds `delta` to the phase exponent modulo four.
    fn phase_shift(&mut self, delta: i32) {
        self.0.phase_shift(delta);
    }

    /// Whether any qubit carries X, Y, or Z.
    fn has_nonidentity_body(&self) -> bool {
        self.0.has_nonidentity_body()
    }

    /// Structural body equality ignoring phase.
    fn same_body(&self, other: &PyPauliString) -> bool {
        self.0.same_body(&other.0)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("PauliString({:?})", self.0.to_string())
    }

    fn __eq__(&self, other: &PyPauliString) -> bool {
        self.0 == other.0
    }

    fn __mul__(&self, other: &PyPauliString) -> Self {
        Self(&self.0 * &other.0)
    }

    fn __neg__(&self) -> Self {
        Self(ticit::neg(self.0.clone()))
    }
}

/// Constructs identity on `nqubits` qubits.
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
fn pauli_identity(nqubits: usize) -> PyPauliString {
    PyPauliString(ticit::pauli_identity(nqubits))
}

/// Constructs X on qubit `q` and identity elsewhere.
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
fn pauli_x(nqubits: usize, q: usize) -> PyResult<PyPauliString> {
    check_qubit(q, nqubits)?;
    Ok(PyPauliString(ticit::pauli_x(nqubits, q)))
}

/// Constructs Y on qubit `q` and identity elsewhere.
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
fn pauli_y(nqubits: usize, q: usize) -> PyResult<PyPauliString> {
    check_qubit(q, nqubits)?;
    Ok(PyPauliString(ticit::pauli_y(nqubits, q)))
}

/// Constructs Z on qubit `q` and identity elsewhere.
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
fn pauli_z(nqubits: usize, q: usize) -> PyResult<PyPauliString> {
    check_qubit(q, nqubits)?;
    Ok(PyPauliString(ticit::pauli_z(nqubits, q)))
}

/// Parses a dense Pauli literal such as `"IXYZ"`.
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
fn pauli_string(text: &str) -> PyResult<PyPauliString> {
    PyPauliString::from_text(text)
}

/// Returns a copy of `pauli` multiplied by -1.
#[gen_stub_pyfunction(module = "ticit._core")]
#[pyfunction]
fn neg(pauli: &PyPauliString) -> PyPauliString {
    PyPauliString(ticit::neg(pauli.0.clone()))
}

/// Result of a Pauli measurement.
#[gen_stub_pyclass]
#[pyclass(
    name = "MeasureResult",
    module = "ticit._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Clone, Copy, Debug)]
struct PyMeasureResult {
    /// Observed eigenvalue bit: `False` is +1 and `True` is -1.
    outcome: bool,
    /// Probability assigned to this outcome before projection.
    probability: f64,
    /// Whether the state forced the outcome.
    deterministic: bool,
}

impl From<ticit::MeasureResult> for PyMeasureResult {
    fn from(result: ticit::MeasureResult) -> Self {
        Self {
            outcome: result.outcome,
            probability: result.probability,
            deterministic: result.deterministic,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMeasureResult {
    fn __repr__(&self) -> String {
        format!(
            "MeasureResult(outcome={}, probability={:?}, deterministic={})",
            if self.outcome { "True" } else { "False" },
            self.probability,
            if self.deterministic { "True" } else { "False" },
        )
    }
}

/// A procedural Clifford+T state simulator shaped like Stim's TableauSimulator.
///
/// Operations apply immediately. Qubit-writing operations grow the register;
/// measurements return `MeasureResult`; non-collapsing `peek_*` calls reject
/// qubits outside the current register.
///
/// Examples:
///     >>> import ticit
///     >>> sim = ticit.TableauSimulator(2, seed=7)
///     >>> sim.h(0)
///     >>> sim.cx(0, 1)
///     >>> sim.peek_observable_expectation(ticit.pauli_string("XX"))
///     1.0
#[gen_stub_pyclass]
#[pyclass(
    name = "TableauSimulator",
    module = "ticit._core",
    unsendable,
    skip_from_py_object
)]
#[derive(Clone, Debug)]
struct PyTableauSimulator(ticit::TableauSimulator);

#[gen_stub_pymethods]
#[pymethods]
impl PyTableauSimulator {
    /// Starts in `|0...0>`; `seed=None` uses OS entropy.
    #[new]
    #[pyo3(signature = (num_qubits, seed=None))]
    fn new(num_qubits: usize, seed: Option<u64>) -> Self {
        Self(match seed {
            Some(seed) => ticit::TableauSimulator::with_seed(num_qubits, seed),
            None => ticit::TableauSimulator::new(num_qubits),
        })
    }

    /// Current register size.
    #[getter]
    fn num_qubits(&self) -> usize {
        self.0.num_qubits()
    }

    /// Number of live amplitude terms.
    #[getter]
    fn rank(&self) -> usize {
        self.0.rank()
    }

    /// Reseeds outcome sampling without changing the state.
    fn reseed_rng(&mut self, seed: u64) {
        self.0.reseed_rng(seed);
    }

    /// Copies the RNG position from `snapshot` without changing the state.
    fn restore_rng_from(&mut self, snapshot: &PyTableauSimulator) {
        self.0.restore_rng_from(&snapshot.0);
    }

    /// Hadamard on `q`.
    fn h(&mut self, q: usize) {
        self.0.h(q);
    }

    /// Phase gate S on `q`.
    fn s(&mut self, q: usize) {
        self.0.s(q);
    }

    /// S dagger on `q`.
    fn s_dag(&mut self, q: usize) {
        self.0.s_dag(q);
    }

    /// Pauli X on `q`.
    fn x(&mut self, q: usize) {
        self.0.x(q);
    }

    /// Pauli Y on `q`.
    fn y(&mut self, q: usize) {
        self.0.y(q);
    }

    /// Pauli Z on `q`.
    fn z(&mut self, q: usize) {
        self.0.z(q);
    }

    /// Square-root X on `q`.
    fn sqrt_x(&mut self, q: usize) {
        self.0.sqrt_x(q);
    }

    /// Square-root X dagger on `q`.
    fn sqrt_x_dag(&mut self, q: usize) {
        self.0.sqrt_x_dag(q);
    }

    /// Square-root Y on `q`.
    fn sqrt_y(&mut self, q: usize) {
        self.0.sqrt_y(q);
    }

    /// Square-root Y dagger on `q`.
    fn sqrt_y_dag(&mut self, q: usize) {
        self.0.sqrt_y_dag(q);
    }

    /// Order-three Pauli cycle X -> Y -> Z -> X.
    fn c_xyz(&mut self, q: usize) {
        self.0.c_xyz(q);
    }

    /// Inverse Pauli cycle X -> Z -> Y -> X.
    fn c_zyx(&mut self, q: usize) {
        self.0.c_zyx(q);
    }

    /// Hadamard-like exchange of X and Y.
    fn h_xy(&mut self, q: usize) {
        self.0.h_xy(q);
    }

    /// Hadamard-like exchange of Y and Z.
    fn h_yz(&mut self, q: usize) {
        self.0.h_yz(q);
    }

    /// CNOT from `control` onto `target`.
    fn cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.cx(control, target).map_err(sim_error)
    }

    /// CNOT alias matching Stim's spelling.
    fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.cnot(control, target).map_err(sim_error)
    }

    /// Controlled Y from `control` onto `target`.
    fn cy(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.cy(control, target).map_err(sim_error)
    }

    /// Controlled Z on `a` and `b`.
    fn cz(&mut self, a: usize, b: usize) -> PyResult<()> {
        self.0.cz(a, b).map_err(sim_error)
    }

    /// Swaps `a` and `b`.
    fn swap(&mut self, a: usize, b: usize) {
        self.0.swap(a, b);
    }

    /// ISWAP on `a` and `b`.
    fn iswap(&mut self, a: usize, b: usize) -> PyResult<()> {
        self.0.iswap(a, b).map_err(sim_error)
    }

    /// ISWAP dagger on `a` and `b`.
    fn iswap_dag(&mut self, a: usize, b: usize) -> PyResult<()> {
        self.0.iswap_dag(a, b).map_err(sim_error)
    }

    /// X-controlled X.
    fn xcx(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.xcx(control, target).map_err(sim_error)
    }

    /// X-controlled Y.
    fn xcy(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.xcy(control, target).map_err(sim_error)
    }

    /// X-controlled Z.
    fn xcz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.xcz(control, target).map_err(sim_error)
    }

    /// Y-controlled X.
    fn ycx(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.ycx(control, target).map_err(sim_error)
    }

    /// Y-controlled Y.
    fn ycy(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.ycy(control, target).map_err(sim_error)
    }

    /// Y-controlled Z.
    fn ycz(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.ycz(control, target).map_err(sim_error)
    }

    /// Z-controlled X, equivalent to CNOT.
    fn zcx(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.zcx(control, target).map_err(sim_error)
    }

    /// Z-controlled Y.
    fn zcy(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.0.zcy(control, target).map_err(sim_error)
    }

    /// Z-controlled Z, equivalent to CZ.
    fn zcz(&mut self, a: usize, b: usize) -> PyResult<()> {
        self.0.zcz(a, b).map_err(sim_error)
    }

    /// Applies a Pauli operator to the state.
    fn pauli(&mut self, pauli: &PyPauliString) {
        self.0.pauli(&pauli.0);
    }

    /// Applies commuting control and target Pauli axes.
    fn controlled_pauli(
        &mut self,
        control: &PyPauliString,
        target: &PyPauliString,
    ) -> PyResult<()> {
        self.0
            .controlled_pauli(&control.0, &target.0)
            .map_err(sim_error)
    }

    /// Applies T on `q`.
    fn t(&mut self, q: usize) -> PyResult<()> {
        self.0.t(q).map_err(sim_error)
    }

    /// Applies T dagger on `q`.
    fn t_dag(&mut self, q: usize) -> PyResult<()> {
        self.0.t_dag(q).map_err(sim_error)
    }

    /// Applies a T rotation about `axis`; `adjoint=True` selects T dagger.
    fn t_pauli(&mut self, axis: &PyPauliString, adjoint: bool) -> PyResult<()> {
        self.0.t_pauli(&axis.0, adjoint).map_err(sim_error)
    }

    /// Applies CCZ to three distinct qubits.
    fn ccz(&mut self, a: usize, b: usize, c: usize) -> PyResult<()> {
        self.0.ccz(a, b, c).map_err(sim_error)
    }

    /// Measures Z on `q` and collapses the state.
    fn measure(&mut self, q: usize) -> PyResult<PyMeasureResult> {
        self.0.measure(q).map(Into::into).map_err(sim_error)
    }

    /// Measures a Hermitian Pauli observable and collapses the state.
    fn measure_observable(&mut self, observable: &PyPauliString) -> PyResult<PyMeasureResult> {
        self.0
            .measure_observable(&observable.0)
            .map(Into::into)
            .map_err(sim_error)
    }

    /// Forces a Pauli measurement to `desired_value`.
    fn postselect_observable(
        &mut self,
        observable: &PyPauliString,
        desired_value: bool,
    ) -> PyResult<PyMeasureResult> {
        self.0
            .postselect_observable(&observable.0, desired_value)
            .map(Into::into)
            .map_err(sim_error)
    }

    /// Forces a Z measurement on `q`.
    fn postselect_z(&mut self, q: usize, desired_value: bool) -> PyResult<PyMeasureResult> {
        self.0
            .postselect_z(q, desired_value)
            .map(Into::into)
            .map_err(sim_error)
    }

    /// Forces an X measurement on `q`.
    fn postselect_x(&mut self, q: usize, desired_value: bool) -> PyResult<PyMeasureResult> {
        self.0
            .postselect_x(q, desired_value)
            .map(Into::into)
            .map_err(sim_error)
    }

    /// Forces a Y measurement on `q`.
    fn postselect_y(&mut self, q: usize, desired_value: bool) -> PyResult<PyMeasureResult> {
        self.0
            .postselect_y(q, desired_value)
            .map(Into::into)
            .map_err(sim_error)
    }

    /// Returns a Pauli expectation without collapsing the state.
    fn peek_observable_expectation(&self, observable: &PyPauliString) -> PyResult<f64> {
        self.0
            .peek_observable_expectation(&observable.0)
            .map_err(sim_error)
    }

    /// Returns the X expectation on `q` without measuring.
    fn peek_x(&self, q: usize) -> PyResult<f64> {
        self.0.peek_x(q).map_err(sim_error)
    }

    /// Returns the Y expectation on `q` without measuring.
    fn peek_y(&self, q: usize) -> PyResult<f64> {
        self.0.peek_y(q).map_err(sim_error)
    }

    /// Returns the Z expectation on `q` without measuring.
    fn peek_z(&self, q: usize) -> PyResult<f64> {
        self.0.peek_z(q).map_err(sim_error)
    }

    /// Resets `q` to |0>.
    fn reset(&mut self, q: usize) -> PyResult<()> {
        self.0.reset(q).map_err(sim_error)
    }

    /// Resets `q` to |0>.
    fn reset_z(&mut self, q: usize) -> PyResult<()> {
        self.0.reset_z(q).map_err(sim_error)
    }

    /// Resets `q` to |+>.
    fn reset_x(&mut self, q: usize) -> PyResult<()> {
        self.0.reset_x(q).map_err(sim_error)
    }

    /// Resets `q` to |+i>.
    fn reset_y(&mut self, q: usize) -> PyResult<()> {
        self.0.reset_y(q).map_err(sim_error)
    }

    /// Reconstructs the dense state vector; intended for small registers.
    fn state_vector(&self) -> Vec<Complex64> {
        self.0.state_vector()
    }

    fn __repr__(&self) -> String {
        format!(
            "TableauSimulator(num_qubits={}, rank={})",
            self.0.num_qubits(),
            self.0.rank(),
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SimulatorError", m.py().get_type::<SimulatorError>())?;
    m.add_class::<PyPauliString>()?;
    m.add_class::<PyMeasureResult>()?;
    m.add_class::<PyTableauSimulator>()?;
    m.add_function(wrap_pyfunction!(pauli_identity, m)?)?;
    m.add_function(wrap_pyfunction!(pauli_x, m)?)?;
    m.add_function(wrap_pyfunction!(pauli_y, m)?)?;
    m.add_function(wrap_pyfunction!(pauli_z, m)?)?;
    m.add_function(wrap_pyfunction!(pauli_string, m)?)?;
    m.add_function(wrap_pyfunction!(neg, m)?)?;
    Ok(())
}
