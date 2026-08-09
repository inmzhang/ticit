import numpy as np
import pytest

import ticit


def test_circuit_compile_and_program_sample_api():
    circuit = ticit.Circuit("H 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]")
    compiled_sampler = circuit.compile(threads=2)
    first = compiled_sampler.sample(shots=257, seed=9)
    second = compiled_sampler.sample(shots=257, seed=9)

    assert compiled_sampler.backend == "cpu"
    assert compiled_sampler.num_qubits == 1
    assert first.total_shots == first.shots == 257
    assert first.passed_shots == first.accepted == 257
    assert first.discards == first.discarded == 0
    assert first.measurements.shape == (257, 1)
    assert first.detectors.shape == (257, 0)
    assert first.observables.shape == (257, 1)
    assert first.measurements.dtype == np.uint8
    assert first.observable_ones.dtype == np.uint64
    assert np.array_equal(first.measurements, second.measurements)
    assert np.array_equal(first.observables, second.observables)
    assert first.observable_ones.tolist() == [first.logical_errors]
    assert first.logical_errors == second.logical_errors
    measurements, detectors, observables = first
    assert measurements is first.measurements
    assert detectors is first.detectors
    assert observables is first.observables


def test_postselection_and_survivor_alias():
    program = ticit.Circuit("X 0\nM 0\nDETECTOR rec[-1]").compile([1])
    result = ticit.sample_survivors(program, shots=32, seed=1, keep_records=True)

    assert program.has_postselection
    assert result.discards == 32
    assert result.passed_shots == 0
    assert result.logical_error_rate != result.logical_error_rate  # NaN
    assert result.measurements.shape == (0, 1)
    assert result.detectors.shape == (0, 1)
    assert result.observables.shape == (0, 0)

    counts_only = ticit.sample_survivors(ticit.Circuit("M 0").compile(), shots=32, seed=1)
    assert counts_only.passed_shots == 32
    assert counts_only.measurements.shape == (0, 1)
    assert counts_only.detectors.shape == (0, 0)
    assert counts_only.observables.shape == (0, 0)


def test_bit_packed_records_match_numpy_little_bit_order():
    program = ticit.Circuit(
        "X 0 1 2 3 4 5 6 7 10\nM 0 1 2 3 4 5 6 7 8 9 10"
    ).compile()
    result = program.sample(shots=2, seed=1, bit_packed=True)

    assert result.bit_packed
    assert result.measurements.dtype == np.uint8
    assert result.measurements.shape == (2, 2)
    assert result.measurements.tolist() == [[0xFF, 0x04], [0xFF, 0x04]]
    assert result.detectors.shape == (2, 0)
    assert result.observables.shape == (2, 0)

    counts_only = ticit.sample_survivors(
        program, shots=2, seed=1, bit_packed=True
    )
    assert counts_only.bit_packed
    assert counts_only.measurements.shape == (0, 2)


def test_parse_metadata_and_argument_errors():
    circuit = ticit.parse("M 0\nDETECTOR rec[-1]")
    assert repr(circuit).startswith("Circuit(")
    assert (circuit.num_qubits, circuit.num_measurements, circuit.num_detectors) == (
        1,
        1,
        1,
    )

    with pytest.raises(ValueError, match="postselection_mask"):
        circuit.compile([1, 1])
    with pytest.raises(ValueError, match="raw parity"):
        ticit.compile("M 0", normalize_syndromes=True)
    with pytest.raises(ValueError, match="shots must be positive"):
        ticit.Circuit("M 0").compile().sample(shots=0)


def test_pauli_and_tableau_public_apis():
    x = ticit.pauli_string("XI")
    z = ticit.pauli_string("ZI")
    assert str(x * z) == "-i*YI"
    assert str(-ticit.pauli_y(2, 0)) == "-YI"
    assert x.same_body(ticit.pauli_x(2, 0))

    sim = ticit.TableauSimulator(2, seed=7)
    sim.h(0)
    sim.cx(0, 1)
    assert sim.peek_observable_expectation(ticit.pauli_string("XX")) == pytest.approx(1)
    first = sim.measure(0)
    second = sim.measure(1)
    assert first.outcome == second.outcome
    assert second.deterministic
