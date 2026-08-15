"""Compare PPVM and ticit on PPVM's 85-qubit MSD example.

Run from the repository root after installing both bindings into one Python
environment. Timings include tableau construction, every gate, and all 85
measurements for each shot; imports and validation are excluded.

Example:
    RUSTFLAGS="-C target-cpu=native" uv run --project ticit_py \
      --with "ppvm @ git+https://github.com/QuEraComputing/ppvm.git#subdirectory=ppvm-python" \
      python ticit_py/benchmarks/ppvm_msd.py --shots 1000 --repeats 7
"""

from __future__ import annotations

import argparse
import gc
import platform
import statistics
import time
from collections.abc import Callable
from importlib.metadata import version

import ppvm
import ticit

QUBITS_PER_BLOCK = 17
N_BLOCKS = 5
N_QUBITS = QUBITS_PER_BLOCK * N_BLOCKS


def encode_scalar(tab, qubits: list[int]) -> None:
    for i in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        tab.sqrt_y(qubits[i])
    for i, j in [(1, 3), (7, 10), (12, 14), (13, 16)]:
        tab.cz(qubits[i], qubits[j])
    for i in [7, 16]:
        tab.sqrt_y_dag(qubits[i])
    for i, j in [(4, 7), (8, 10), (11, 14), (15, 16)]:
        tab.cz(qubits[i], qubits[j])
    for i in [4, 10, 14, 16]:
        tab.sqrt_y_dag(qubits[i])
    for i, j in [(2, 4), (6, 8), (7, 9), (10, 13), (14, 16)]:
        tab.cz(qubits[i], qubits[j])
    for i in [3, 6, 9, 10, 12, 13]:
        tab.sqrt_y(qubits[i])
    for i, j in [(0, 2), (3, 6), (5, 8), (10, 12), (11, 13)]:
        tab.cz(qubits[i], qubits[j])
    for i in [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 14]:
        tab.sqrt_y(qubits[i])
    for i, j in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (12, 15)]:
        tab.cz(qubits[i], qubits[j])
    for i in [0, 2, 5, 6, 8, 10, 12]:
        tab.sqrt_y_dag(qubits[i])


def blocks() -> list[list[int]]:
    return [
        list(range(i * QUBITS_PER_BLOCK, (i + 1) * QUBITS_PER_BLOCK))
        for i in range(N_BLOCKS)
    ]


def finish_msd_scalar(tab, ql: list[list[int]]) -> None:
    for i in [0, 1, 4]:
        for q in ql[i]:
            tab.sqrt_x(q)
    for control, target in zip(ql[0], ql[1]):
        tab.cz(control, target)
    for control, target in zip(ql[2], ql[3]):
        tab.cz(control, target)
    for q in ql[0]:
        tab.sqrt_y(q)
    for q in ql[3]:
        tab.sqrt_y(q)
    for control, target in zip(ql[0], ql[2]):
        tab.cz(control, target)
    for control, target in zip(ql[3], ql[4]):
        tab.cz(control, target)
    for q in ql[0]:
        tab.sqrt_x_dag(q)
    for control, target in zip(ql[0], ql[4]):
        tab.cz(control, target)
    for control, target in zip(ql[1], ql[3]):
        tab.cz(control, target)
    for block in ql:
        for q in block:
            tab.sqrt_x_dag(q)


def build_msd_scalar(tab) -> None:
    ql = blocks()
    for block in ql:
        tab.h(block[7])
        tab.t(block[7])
        encode_scalar(tab, block)
    finish_msd_scalar(tab, ql)


def _at(qubits: list[int], indices: list[int]) -> list[int]:
    return [qubits[i] for i in indices]


def _pairs(qubits: list[int], pairs: list[tuple[int, int]]) -> list[int]:
    return [qubits[i] for pair in pairs for i in pair]


def encode_ppvm_fused(tab: ppvm.GeneralizedTableau, qubits: list[int]) -> None:
    tab.sqrt_y(_at(qubits, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
    tab.cz(_pairs(qubits, [(1, 3), (7, 10), (12, 14), (13, 16)]))
    tab.sqrt_y_dag(_at(qubits, [7, 16]))
    tab.cz(_pairs(qubits, [(4, 7), (8, 10), (11, 14), (15, 16)]))
    tab.sqrt_y_dag(_at(qubits, [4, 10, 14, 16]))
    tab.cz(_pairs(qubits, [(2, 4), (6, 8), (7, 9), (10, 13), (14, 16)]))
    tab.sqrt_y(_at(qubits, [3, 6, 9, 10, 12, 13]))
    tab.cz(_pairs(qubits, [(0, 2), (3, 6), (5, 8), (10, 12), (11, 13)]))
    tab.sqrt_y(_at(qubits, [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 14]))
    tab.cz(_pairs(qubits, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (12, 15)]))
    tab.sqrt_y_dag(_at(qubits, [0, 2, 5, 6, 8, 10, 12]))


def build_msd_ppvm_fused(tab: ppvm.GeneralizedTableau) -> None:
    ql = blocks()
    for block in ql:
        tab.h(block[7])
        tab.t(block[7])
        encode_ppvm_fused(tab, block)
    for i in [0, 1, 4]:
        tab.sqrt_x(ql[i])
    tab.cz_block(ql[0][0], ql[1][0], QUBITS_PER_BLOCK)
    tab.cz_block(ql[2][0], ql[3][0], QUBITS_PER_BLOCK)
    tab.sqrt_y(ql[0])
    tab.sqrt_y(ql[3])
    tab.cz_block(ql[0][0], ql[2][0], QUBITS_PER_BLOCK)
    tab.cz_block(ql[3][0], ql[4][0], QUBITS_PER_BLOCK)
    tab.sqrt_x_dag(ql[0])
    tab.cz_block(ql[0][0], ql[4][0], QUBITS_PER_BLOCK)
    tab.cz_block(ql[1][0], ql[3][0], QUBITS_PER_BLOCK)
    for block in ql:
        tab.sqrt_x_dag(block)


def ppvm_scalar(shots: int) -> int:
    initial = ppvm.GeneralizedTableau(N_QUBITS, seed=0)
    ones = 0
    for seed in range(shots):
        sim = initial.fork(seed=seed)
        build_msd_scalar(sim)
        ones += sum(
            sim.measure(q) == ppvm.MeasurementResult.ONE for q in range(N_QUBITS)
        )
    return ones


def ppvm_fused(shots: int) -> int:
    initial = ppvm.GeneralizedTableau(N_QUBITS, seed=0)
    ones = 0
    for seed in range(shots):
        sim = initial.fork(seed=seed)
        build_msd_ppvm_fused(sim)
        ones += sum(
            result == ppvm.MeasurementResult.ONE
            for result in sim.measure_many(range(N_QUBITS))
        )
    return ones


def ticit_scalar(shots: int) -> int:
    ones = 0
    for seed in range(shots):
        sim = ticit.TableauSimulator(N_QUBITS, seed=seed)
        build_msd_scalar(sim)
        ones += sum(sim.measure(q).outcome for q in range(N_QUBITS))
    return ones


def verify_states() -> tuple[int, int, int, int]:
    ppvm_tab = ppvm.GeneralizedTableau(N_QUBITS, seed=0)
    ppvm_fused_tab = ppvm.GeneralizedTableau(N_QUBITS, seed=0)
    ticit_tab = ticit.TableauSimulator(N_QUBITS, seed=0)
    build_msd_scalar(ppvm_tab)
    build_msd_ppvm_fused(ppvm_fused_tab)
    build_msd_scalar(ticit_tab)

    words = []
    for axis in "XYZ":
        for q in range(N_QUBITS):
            words.append("I" * q + axis + "I" * (N_QUBITS - q - 1))
    quartets = [(0, 1, 2, 3), (0, 2, 4, 5), (0, 2, 6, 7), (0, 2, 8, 9)]
    for block in range(N_BLOCKS):
        for quartet in quartets:
            word = ["I"] * N_QUBITS
            for q in quartet:
                word[block * QUBITS_PER_BLOCK + q] = "Z"
            words.append("".join(word))

    nonzero = 0
    for word in words:
        expected = ppvm_tab.expectation(word)
        fused = ppvm_fused_tab.expectation(word)
        actual = ticit_tab.peek_observable_expectation(ticit.pauli_string(word))
        if max(abs(actual - expected), abs(fused - expected)) > 1e-9:
            raise AssertionError(
                f"{word}: ticit={actual}, ppvm={expected}, ppvm fused={fused}"
            )
        nonzero += abs(expected) > 1e-9
    return ppvm_tab.num_coefficients(), ticit_tab.rank, len(words), nonzero


def measure(cases: list[tuple[str, Callable[[int], int]]], shots: int, repeats: int):
    for _, run in cases:
        run(min(shots, 3))
    elapsed: dict[str, list[float]] = {name: [] for name, _ in cases}
    checksums: dict[str, int] = {}
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(repeats):
            for offset in range(len(cases)):
                name, run = cases[(repeat + offset) % len(cases)]
                start = time.perf_counter_ns()
                checksum = run(shots)
                elapsed[name].append((time.perf_counter_ns() - start) / 1e9)
                if name in checksums and checksums[name] != checksum:
                    raise AssertionError(f"unstable checksum for {name}")
                checksums[name] = checksum
    finally:
        if was_enabled:
            gc.enable()
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.shots < 1 or args.repeats < 1:
        parser.error("--shots and --repeats must be positive")

    ppvm_terms, ticit_rank, expectations, nonzero = verify_states()
    cases = [
        ("ppvm scalar", ppvm_scalar),
        ("ppvm fused", ppvm_fused),
        ("ticit scalar", ticit_scalar),
    ]
    elapsed = measure(cases, args.shots, args.repeats)
    medians = {name: statistics.median(values) for name, values in elapsed.items()}
    baseline = medians["ppvm scalar"]

    print(
        f"Python {platform.python_version()}; ppvm {version('ppvm')}; ticit {version('ticit')}"
    )
    print(
        f"MSD: {N_QUBITS} qubits, 5 T gates, {args.shots} shots x {args.repeats} repeats"
    )
    print(
        f"state check: ppvm terms={ppvm_terms}, ticit rank={ticit_rank}, "
        f"{expectations} expectations matched ({nonzero} nonzero)"
    )
    print("path                 median ms/shot       shots/s   speedup vs ppvm scalar")
    for name, _ in cases:
        seconds = medians[name]
        print(
            f"{name:<20} {seconds / args.shots * 1e3:>14.3f} {args.shots / seconds:>13,.1f} {baseline / seconds:>24.2f}x"
        )


if __name__ == "__main__":
    main()
