# ticit v0.1 Python API Reference

The `ticit` package performs exact batch sampling of noisy, adaptive,
Clifford-dominated quantum circuits:

```python
import ticit

compiled_sampler = ticit.Circuit("""
    H 0
    M 0
    OBSERVABLE_INCLUDE(0) rec[-1]
""").compile()
result = compiled_sampler.sample(shots=10_000, seed=42)
print(result.logical_error_rate)
```

Like Clifft, `SampleResult` contains NumPy arrays for per-shot measurements,
detectors, observables, and expectation values. It also includes aggregate
postselection and logical-error counters.

## Index

- [`ticit.Circuit`](#ticitcircuit)
  - [`ticit.Circuit.__init__`](#ticitcircuit__init__)
  - [`ticit.Circuit.from_text`](#ticitcircuitfrom_text)
  - [`ticit.Circuit.from_file`](#ticitcircuitfrom_file)
  - [`ticit.Circuit.compile`](#ticitcircuitcompile)
  - [`ticit.Circuit.num_qubits`](#ticitcircuitnum_qubits)
  - [`ticit.Circuit.num_measurements`](#ticitcircuitnum_measurements)
  - [`ticit.Circuit.num_detectors`](#ticitcircuitnum_detectors)
  - [`ticit.Circuit.num_observables`](#ticitcircuitnum_observables)
  - [`ticit.Circuit.num_exp_vals`](#ticitcircuitnum_exp_vals)
- [`ticit.Program`](#ticitprogram)
  - [`ticit.Program.sampde`](#ticitprogramsample)
  - [`ticit.Program.backend`](#ticitprogrambackend)
  - [`ticit.Program.num_qubits`](#ticitprogramnum_qubits)
  - [`ticit.Program.num_measurements`](#ticitprogramnum_measurements)
  - [`ticit.Program.num_detectors`](#ticitprogramnum_detectors)
  - [`ticit.Program.num_observables`](#ticitprogramnum_observables)
  - [`ticit.Program.num_exp_vals`](#ticitprogramnum_exp_vals)
  - [`ticit.Program.observable`](#ticitprogramobservable)
  - [`ticit.Program.has_postselection`](#ticitprogramhas_postselection)
- [`ticit.SampleResult`](#ticitsampleresult)
  - [record arrays](#record-arrays)
  - [count fields](#count-fields)
  - [rate properties](#rate-properties)
  - [timing fields](#timing-fields)
- [`ticit.PauliString`](#ticitpaulistring)
- [Pauli constructor functions](#pauli-constructor-functions)
- [`ticit.MeasureResult`](#ticitmeasureresult)
- [`ticit.TableauSimulator`](#ticittableausimulator)
- [`ticit.SimulatorError`](#ticitsimulatorerror)
- [`ticit.ParseError`](#ticitparseerror)
- [`ticit.parse`](#ticitparse)
- [`ticit.parse_file`](#ticitparse_file)
- [`ticit.compile`](#ticitcompile)
- [`ticit.sample`](#ticitsample)
- [`ticit.sample_survivors`](#ticitsample_survivors)
- [GPU backend](#gpu-backend)

## Installation

From the repository, build the mixed Rust/Python package with Maturin:

```sh
cd ticit_py
maturin develop
```

Python 3.10 or newer is required. The wheel uses PyO3's `abi3-py310` stable
ABI. To include CUDA support, build with the `gpu` Cargo feature:

```sh
cd ticit_py
maturin develop --features gpu
```

The checked-in type stub is generated from the PyO3 declarations:

```sh
cargo run -p ticit_py --bin stub_gen
```

## `ticit.Circuit`

```python
class ticit.Circuit(stim_text: str = "")
```

A parsed and lowered circuit. Constructing a `Circuit` validates the complete
input but does not allocate sampling workers.

### `ticit.Circuit.__init__`

```python
def __init__(self, stim_text: str = "") -> None
```

Parses circuit source. An empty string creates an empty circuit.

```python
import ticit

circuit = ticit.Circuit("H 0\nM 0\nDETECTOR rec[-1]")
assert circuit.num_qubits == 1
assert circuit.num_measurements == 1
assert circuit.num_detectors == 1
```

Raises:

- `ticit.ParseError`: the input is malformed or cannot be lowered.
- `ValueError`: the input uses a valid but unsupported operation.

### `ticit.Circuit.from_text`

```python
@staticmethod
def from_text(stim_text: str) -> ticit.Circuit
```

Equivalent to `ticit.Circuit(stim_text)`.

### `ticit.Circuit.from_file`

```python
@staticmethod
def from_file(path: str) -> ticit.Circuit
```

Reads a UTF-8 circuit file and returns its parsed circuit. Raises `OSError` if
the file cannot be read and `ticit.ParseError` if its contents are invalid.

### `ticit.Circuit.num_qubits`

```python
property num_qubits: int
```

Number of qubits named by the circuit.

### `ticit.Circuit.num_measurements`

```python
property num_measurements: int
```

Number of measurement results written to the measurement record.

### `ticit.Circuit.num_detectors`

```python
property num_detectors: int
```

Number of `DETECTOR` and `DISCARD` declarations.

### `ticit.Circuit.num_observables`

```python
property num_observables: int
```

One more than the largest `OBSERVABLE_INCLUDE` index, or zero when the circuit
has no observable declarations.

### `ticit.Circuit.num_exp_vals`

```python
property num_exp_vals: int
```

Number of expectation values produced by `EXP_VAL` instructions.

### `ticit.Circuit.compile`

```python
def circuit.compile(
    postselection_mask: Sequence[int] | None = None,
    *,
    backend: str = "cpu",
    observable: int = 0,
    threads: int = 1,
    sample_chunk_shots: int = 0,
    batch_size: int = 0,
    gpu_chunk_shots: int = 1_048_576,
) -> ticit.Program
```

Compiles the parsed circuit into a reusable sampler. `postselection_mask`
contains one zero/nonzero flag per detector. `backend` selects `"cpu"` or
`"gpu"`; the remaining options configure the selected backend.

## `ticit.Program`

A circuit prepared for repeated calls to
[`Program.sample`](#ticitprogramsample). Programs are created by
[`Circuit.compile`](#ticitcircuitcompile); `ticit.Program()` has no public
constructor.

CPU programs retain their planned program, expression plan, worker states, and
buffers between calls. GPU programs retain the parsed circuit and backend
configuration; current GPU planning and device allocation occur in each sample
call.

### `ticit.Program.backend`

```python
property backend: str
```

Either `"cpu"` or `"gpu"`.

### `ticit.Program.num_qubits`

```python
property num_qubits: int
```

Number of circuit qubits.

### `ticit.Program.num_measurements`

```python
property num_measurements: int
```

Number of circuit measurement records.

### `ticit.Program.num_detectors`

```python
property num_detectors: int
```

Number of circuit detectors.

### `ticit.Program.num_observables`

```python
property num_observables: int
```

Number of circuit observable indices.

### `ticit.Program.num_exp_vals`

```python
property num_exp_vals: int
```

Number of expectation values in each result row.

### `ticit.Program.observable`

```python
property observable: int
```

Observable index whose accepted one outcomes are counted by
`SampleResult.logical_errors`.

### `ticit.Program.has_postselection`

```python
property has_postselection: bool
```

Whether the compiled program rejects shots using at least one detector.

### `ticit.Program.sample`

```python
def program.sample(
    shots: int,
    seed: int | None = None,
    *,
    bit_packed: bool = False,
) -> ticit.SampleResult
```

Samples the compiled circuit. `shots` must be positive. `seed=None` chooses
fresh OS-provided entropy; an integer makes the result reproducible. Calls
release the Python GIL and are serialized around the program's reusable worker
buffers.

`bit_packed=True` returns the three bit arrays with shape
`(rows, ceil(num_bits / 8))`. Bit `k` is stored in byte `k // 8` at
`1 << (k % 8)`, equivalent to `numpy.packbits(..., axis=1, bitorder="little")`.

For postselected programs, the record arrays contain one row per surviving
shot.

## `ticit.SampleResult`

Per-shot records, aggregate counters, and timing from one sampling call.
Instances are immutable and are returned by `Program.sample`, `ticit.sample`, and
`ticit.sample_survivors`.

### Record arrays

```python
property measurements: numpy.ndarray  # uint8, (rows, output measurement bytes)
property detectors: numpy.ndarray     # uint8, (rows, output detector bytes)
property observables: numpy.ndarray   # uint8, (rows, output observable bytes)
property exp_vals: numpy.ndarray      # float64, (rows, program.num_exp_vals)
property bit_packed: bool
```

`Program.sample` returns `passed_shots` rows. The three bit arrays can also be
tuple-unpacked as `measurements, detectors, observables = result`, matching
Clifft. Without packing, each bit occupies one byte. Count-only
`sample_survivors(..., keep_records=False)` returns zero-row arrays with the
requested packed or unpacked column count.

### Count fields

```python
property total_shots: int
property shots: int
property discards: int
property discarded: int
property passed_shots: int
property accepted: int
property logical_errors: int
property observable_ones: numpy.ndarray  # uint64, (program.num_observables,)
property observable: int
```

The aliases follow both Clifft and ticit terminology:

- `total_shots == shots` is the number of attempted shots.
- `discards == discarded` is the number rejected by detector postselection.
- `passed_shots == accepted` is the number retained.
- `shots == discarded + accepted` always holds.
- `logical_errors` counts accepted shots where the selected observable is one.
- `observable_ones[i]` counts accepted rows where observable `i` is one.
- `observable` identifies that selected observable index.

```python
result = ticit.Circuit("M 0").compile().sample(shots=100, seed=1)
assert result.total_shots == 100
assert result.passed_shots == 100
assert result.discards == 0
```

### Rate properties

```python
property discard_rate: float
property logical_error_rate: float
```

`discard_rate` is `discarded / shots`. `logical_error_rate` is
`logical_errors / accepted`. A zero denominator produces `nan`.

### Timing fields

```python
property compile_s: float
property presample_s: float
property execute_s: float
property sample_s: float
property active_threads: int
```

- `compile_s`: CPU circuit-planning time, or GPU planning/setup/JIT warmup time.
- `presample_s`: exogenous-noise generation and expression evaluation.
- `execute_s`: factored circuit execution and result reduction.
- `sample_s`: wall-clock steady-state sampling time.
- `active_threads`: CPU workers that received work; one for GPU sampling.

With multiple CPU workers, `presample_s` and `execute_s` are sums of worker
time and can exceed `sample_s`.

## `ticit.PauliString`

```python
class ticit.PauliString(nqubits: int = 0)
```

A packed Pauli operator. The represented operator is
`i**phase_exponent * product(X**x * Z**z)`. Constructing by qubit count creates
identity; [`ticit.pauli_string`](#pauli-constructor-functions) parses a dense
literal.

```python
p = ticit.pauli_string("IXYZ")
assert p.nqubits == 4
assert str(p) == "IXYZ"
assert str(-p) == "-IXYZ"
```

Read-only properties:

```python
property nqubits: int
property x: list[int]
property z: list[int]
property phase_exponent: int
```

`x` and `z` are copies of the packed LSB-first 64-bit words. Qubit `q` is bit
`q & 63` of word `q >> 6`.

Methods:

```python
@staticmethod
def from_text(text: str) -> ticit.PauliString

def xbit(self, q: int) -> bool
def zbit(self, q: int) -> bool
def set_xbit(self, q: int, value: bool) -> None
def set_zbit(self, q: int, value: bool) -> None
def set_phase(self, phase_exponent: int) -> None
def phase_shift(self, delta: int) -> None
def has_nonidentity_body(self) -> bool
def same_body(self, other: ticit.PauliString) -> bool
```

Out-of-range qubit indices raise `ValueError`. `set_phase` and `phase_shift`
reduce their inputs modulo four. `same_body` ignores phase.

Operators:

- `str(p)` renders the coefficient and dense body.
- `p * q` performs Pauli multiplication; operands must have equal widths.
- `-p` returns a copy multiplied by -1.
- `p == q` compares width, phase, and packed body structurally.

### Pauli constructor functions

```python
def ticit.pauli_identity(nqubits: int) -> ticit.PauliString
def ticit.pauli_x(nqubits: int, q: int) -> ticit.PauliString
def ticit.pauli_y(nqubits: int, q: int) -> ticit.PauliString
def ticit.pauli_z(nqubits: int, q: int) -> ticit.PauliString
def ticit.pauli_string(text: str) -> ticit.PauliString
def ticit.neg(pauli: ticit.PauliString) -> ticit.PauliString
```

`pauli_string` accepts `I`, `X`, `Y`, and `Z` case-insensitively; `_` aliases
identity. String position is the qubit index. The single-axis constructors
raise `ValueError` when `q >= nqubits`.

## `ticit.MeasureResult`

An immutable result returned by tableau-simulator measurements.

```python
property outcome: bool
property probability: float
property deterministic: bool
```

`outcome=False` represents eigenvalue +1 and `outcome=True` represents -1.
`probability` is the pre-projection branch probability. `deterministic` is true
when the state forced the outcome.

## `ticit.TableauSimulator`

```python
class ticit.TableauSimulator(num_qubits: int, seed: int | None = None)
```

A procedural Clifford+T simulator in Stim's `TableauSimulator` style. It starts
in `|0...0>`. Writing a missing qubit grows the register; read-only `peek_*`
operations reject missing qubits. `seed=None` uses OS entropy.

```python
sim = ticit.TableauSimulator(2, seed=7)
sim.h(0)
sim.cx(0, 1)
assert sim.peek_observable_expectation(ticit.pauli_string("XX")) == 1
assert sim.measure(0).outcome == sim.measure(1).outcome
```

State and RNG:

```python
property num_qubits: int
property rank: int
def reseed_rng(self, seed: int) -> None
def restore_rng_from(self, snapshot: ticit.TableauSimulator) -> None
```

Single-qubit Clifford gates:

```python
def h(self, q: int) -> None
def s(self, q: int) -> None
def s_dag(self, q: int) -> None
def x(self, q: int) -> None
def y(self, q: int) -> None
def z(self, q: int) -> None
def sqrt_x(self, q: int) -> None
def sqrt_x_dag(self, q: int) -> None
def sqrt_y(self, q: int) -> None
def sqrt_y_dag(self, q: int) -> None
def c_xyz(self, q: int) -> None
def c_zyx(self, q: int) -> None
def h_xy(self, q: int) -> None
def h_yz(self, q: int) -> None
```

Two-qubit Clifford gates:

```python
def cx(self, control: int, target: int) -> None
def cnot(self, control: int, target: int) -> None
def cy(self, control: int, target: int) -> None
def cz(self, a: int, b: int) -> None
def swap(self, a: int, b: int) -> None
def iswap(self, a: int, b: int) -> None
def iswap_dag(self, a: int, b: int) -> None
def xcx(self, control: int, target: int) -> None
def xcy(self, control: int, target: int) -> None
def xcz(self, control: int, target: int) -> None
def ycx(self, control: int, target: int) -> None
def ycy(self, control: int, target: int) -> None
def ycz(self, control: int, target: int) -> None
def zcx(self, control: int, target: int) -> None
def zcy(self, control: int, target: int) -> None
def zcz(self, a: int, b: int) -> None
```

Repeated operands raise `ValueError`. `cx`, `cnot`, and `zcx` are aliases;
`cz` and `zcz` are aliases; `cy` and `zcy` are aliases.

Pauli and non-Clifford operations:

```python
def pauli(self, pauli: ticit.PauliString) -> None
def controlled_pauli(
    self,
    control: ticit.PauliString,
    target: ticit.PauliString,
) -> None
def t(self, q: int) -> None
def t_dag(self, q: int) -> None
def t_pauli(self, axis: ticit.PauliString, adjoint: bool) -> None
def ccz(self, a: int, b: int, c: int) -> None
```

Controlled Pauli axes must be positive, Hermitian, and commuting. T rotation
axes must be Hermitian. Argument failures raise `ValueError`; exponential rank
growth beyond the engine cap raises `ticit.SimulatorError`.

Measurements and postselection:

```python
def measure(self, q: int) -> ticit.MeasureResult
def measure_observable(self, observable: ticit.PauliString) -> ticit.MeasureResult
def postselect_observable(
    self,
    observable: ticit.PauliString,
    desired_value: bool,
) -> ticit.MeasureResult
def postselect_x(self, q: int, desired_value: bool) -> ticit.MeasureResult
def postselect_y(self, q: int, desired_value: bool) -> ticit.MeasureResult
def postselect_z(self, q: int, desired_value: bool) -> ticit.MeasureResult
```

Forcing an outcome with zero probability raises `ticit.SimulatorError` and
leaves the state unchanged.

Non-collapsing expectations and resets:

```python
def peek_observable_expectation(self, observable: ticit.PauliString) -> float
def peek_x(self, q: int) -> float
def peek_y(self, q: int) -> float
def peek_z(self, q: int) -> float
def reset(self, q: int) -> None
def reset_x(self, q: int) -> None
def reset_y(self, q: int) -> None
def reset_z(self, q: int) -> None
```

`reset` and `reset_z` prepare `|0>`; `reset_x` prepares `|+>`; `reset_y`
prepares `|+i>`.

Dense inspection:

```python
def state_vector(self) -> list[complex]
```

Reconstructs a length-`2**num_qubits` state vector. This is intended only for
tests and small registers because both time and memory are exponential.

## `ticit.SimulatorError`

```python
class ticit.SimulatorError(RuntimeError)
```

Raised for live-state failures such as rank overflow, impossible
postselection, or pruning that would erase the state. Invalid axes, repeated
qubits, and out-of-range read-only access instead raise `ValueError`.

## `ticit.ParseError`

```python
class ticit.ParseError(ValueError)
```

Raised when circuit source is malformed or fails during lowering.

## `ticit.parse`

```python
def ticit.parse(text: str) -> ticit.Circuit
```

Parses source text without preparing a sampler.

```python
circuit = ticit.parse("M 0")
assert circuit.num_measurements == 1
```

## `ticit.parse_file`

```python
def ticit.parse_file(path: str) -> ticit.Circuit
```

Parses a UTF-8 circuit file. This is equivalent to
`ticit.Circuit.from_file(path)`.

## `ticit.compile`

```python
def ticit.compile(
    stim_text: str,
    postselection_mask: Sequence[int] | None = None,
    expected_detectors: Sequence[int] | None = None,
    expected_observables: Sequence[int] | None = None,
    normalize_syndromes: bool = False,
    *,
    backend: str = "cpu",
    observable: int = 0,
    threads: int = 1,
    sample_chunk_shots: int = 0,
    batch_size: int = 0,
    gpu_chunk_shots: int = 1_048_576,
) -> ticit.Program
```

Compatibility wrapper that parses and prepares a circuit. New code should use
[`Circuit.compile`](#ticitcircuitcompile). The positional parameters mirror
Clifft's `compile` function so existing raw-parity call sites need minimal
changes.

Arguments:

- `stim_text`: circuit in ticit's Stim-style text format.
- `postselection_mask`: one zero/nonzero flag per detector. Nonzero flags reject
  a shot when that detector parity is one. Source `DISCARD` declarations are
  unioned with this mask.
- `expected_detectors`: reserved for Clifft call compatibility. Must be `None`
  or empty because ticit deliberately uses raw parity.
- `expected_observables`: same restriction as `expected_detectors`.
- `normalize_syndromes`: reserved for compatibility and must remain `False`.
- `backend`: `"cpu"` or `"gpu"`.
- `observable`: observable index counted as a logical error.
- `threads`: maximum CPU worker count; must be positive.
- `sample_chunk_shots`: CPU shots assigned to one scheduling chunk. Zero uses
  ticit's automatic value.
- `batch_size`: CPU shots executed together in one bit-packed batch. Zero uses
  a value based on peak active width.
- `gpu_chunk_shots`: maximum shots allocated in one GPU launch group; must be
  positive when `backend="gpu"`.

Returns:

- A reusable `ticit.Program`.

Raises:

- `ticit.ParseError`: malformed circuit source.
- `ValueError`: invalid options, mask length, backend name, or unsupported
  reference normalization.
- `RuntimeError`: `backend="gpu"` was requested from a CPU-only build.

```python
program = ticit.compile(
    "H 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]",
    backend="cpu",
    threads=2,
)
assert program.observable == 0
```

## `ticit.sample`

```python
def ticit.sample(
    program: ticit.Program,
    shots: int,
    seed: int | None = None,
    *,
    bit_packed: bool = False,
) -> ticit.SampleResult
```

Compatibility wrapper for [`Program.sample`](#ticitprogramsample). Postselected
programs return survivor rows.

```python
program = ticit.Circuit("M 0").compile()
a = program.sample(shots=64, seed=123)
b = program.sample(shots=64, seed=123)
assert (a.discards, a.logical_errors) == (b.discards, b.logical_errors)
```

## `ticit.sample_survivors`

```python
def ticit.sample_survivors(
    program: ticit.Program,
    shots: int,
    seed: int | None = None,
    keep_records: bool = False,
    *,
    bit_packed: bool = False,
) -> ticit.SampleResult
```

A Clifft-compatible name for postselected sampling. Its counters are identical
to `ticit.sample(program, shots, seed)`. `keep_records=True` returns survivor
rows; `False` returns zero-row arrays and avoids record materialization.
`bit_packed` has the same meaning as on `Program.sample`.

## GPU backend

GPU selection is a compile option, not a separate Python module:

```python
program = ticit.Circuit(
    "H 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"
).compile(backend="gpu", gpu_chunk_shots=1_048_576)
result = ticit.sample_survivors(program, shots=1_000_000, seed=42)
```

Requirements and current limits:

- Build `ticit_py` with Cargo feature `gpu`.
- A working CUDA environment supported by ticit's `cutile` backend is required.
- GPU detector postselection is all-or-none. An empty/zero mask selects none;
  an all-nonzero mask selects every detector. Selective masks raise
  `ValueError` instead of silently changing semantics.
- The GPU currently supports count-only sampling; `Program.sample` raises
  `RuntimeError` because GPU kernels do not yet return per-shot records.
- GPU planning, device allocation, and one-time cuTile JIT warmup currently
  occur during the sampling call and are reported in `compile_s`.
