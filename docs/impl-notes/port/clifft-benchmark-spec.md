# clifft — Porting/Benchmarking Spec

(Explorer report; verified empirically where noted. Basis for task: benchmark ticit vs symft vs clifft.)

## 1. Language & build

C++20 core + Python 3.12+ nanobind bindings; CMake ≥3.20 via scikit-build-core, orchestrated by `just`. Deps `stim` + `fast_float` via FetchContent (stim is a C++ library dependency — tableau simulator for Clifford-frame absorption). `CLIFFT_CPU_BASELINE` defaults `native` → `-march=native -mtune=native -ffast-math`.

**Prebuilt and working locally**: `.venv` has editable install `clifft 0.0.1.dev95+g6ce41a179.d20260729` (Python 3.13 venv, cp312 abi3 — fine), `_clifft_core.abi3.so` importable, `CPU_BASELINE == "native"`. No build needed. `uv`, `just`, `cmake`, `ninja` on PATH.

Build from scratch if needed: `uv venv && uv pip install -e .` in `CLIFFT_ROOT`.

## 2. CLI — there is none

No standalone binary. Profilers (`profile_svm` etc.) take no argv (hardcoded circuits). No console_scripts. **Use the Python API.**

## 3. Python API

```python
clifft.lower(hir, postselection_mask=[], expected_detectors=[], expected_observables=[]) -> Program
clifft.sample_survivors(program, shots, seed=None, keep_records=False) -> SampleResult
```

`SampleResult` fields: `total_shots`, `passed_shots`, `discards` (= total−passed), `logical_errors`, `observable_ones` (u64 per-observable counts) always; `measurements`/`detectors`/`observables` (uint8 (rows, n)) and **`exp_vals` (float64 (rows, num_exp_vals))** only with `keep_records=True`. `clifft.sample()` raises ValueError on postselected programs — always `sample_survivors`.

`clifft.compile(stim_text, postselection_mask=None, expected_detectors=None, expected_observables=None, normalize_syndromes=False, hir_passes=default, bytecode_passes=default)` wrapper.

## 4. Raw-parity mode (SOFT#8)

- `expected_detectors` baked into bytecode at lowering (parity accumulator seeded with 1; **changes which shots get discarded** in postselect kernels).
- `expected_observables` stored on module, XORed at sample time (4 sites in svm.cc).
- **Empty is the default and means raw parity** (docs/guide/simulation.md:136). `normalize_syndromes=True` mutually exclusive with explicit parities.

Exact raw-parity invocation:
```python
clifft.set_num_threads(1)
circuit = clifft.parse_file(path)
hir = clifft.trace(circuit)
clifft.default_hir_pass_manager().run(hir)
program = clifft.lower(hir, postselection_mask=mask)  # omit both expected_*
clifft.default_bytecode_pass_manager().run(program)
result = clifft.sample_survivors(program, shots, seed=seed, keep_records=False)
```
Matches SOFT's discard rule (raw detector bits) exactly.

Empirical proof (nonzero-reference circuit, 20k shots): raw vs normalized flips logical_errors 19795 ↔ 205 (~96×); raw+postselect discards every shot. That's the SOFT#8 incomparability.

### ⚠️ Two additional incomparabilities

**(a) `logical_errors`: OR-across-observables in clifft, XOR-across in SOFT** (`svm.cc:399-415` vs `stim_sampling.cpp:206-212`). Agree only when `num_observables <= 1`. Raw mode does NOT fix; needs `keep_records=True` + manual XOR (or use `observable_ones[0]` per AGENTS.md).
**(b) Denominator**: SOFT divides by `accepted`; divide clifft's `logical_errors` by `passed_shots`, not `total_shots`.

## 5. Original detector-free ccz-nontels snapshot (historical)

These trace times and counts describe the initially imported fixture snapshot,
before decoder annotations were restored:

| file | qubits | meas | det | obs | exp_vals | T | clifft trace time |
|---|---|---|---|---|---|---|---|
| d05_p0 | 835 | 8220 | 0 | 0 | 28 | 8 | 0.312 s |
| d05_p1e-3 | 835 | 8220 | 0 | 0 | 28 | 8 | 2.78 s |
| d07_p0 | 1475 | 21004 | 0 | 0 | 28 | 8 | 1.69 s |
| d07_p1e-3 | 1475 | 21004 | 0 | 0 | 28 | 8 | 20.8 s |
| d09_p0 | 2307 | 42812 | 0 | 0 | 28 | 8 | 8.37 s |
| d09_p1e-3 | 2307 | 42812 | 0 | 0 | 28 | 8 | 94.8 s |
| d11_p0 | 3331 | 76044 | 0 | 0 | 28 | 8 | 33.4 s |
| d11_p1e-3 | 3331 | 76044 | 0 | 0 | 28 | 8 | **327 s** |

**Zero DETECTOR / OBSERVABLE_INCLUDE / REPEAT in every file.** ⇒ normalization moot; `logical_errors`/`observable_ones` identically 0; **logical error rate not a meaningful metric on this suite — the signal is `exp_vals`** (requires keep_records=True). Compile/trace time is not negligible (327 s for Clifft and 788 s for SymFT on d11_p1e-3) — exclude it from or report it separately from throughput.

The fixtures regenerated on 2026-08-13 use direct logical-T preparation and
inline the generated decoder boundary. Their `(qubits, measurements,
detectors)` counts are d05 `(834, 8,012, 6,992)`, d07
`(1,474, 20,604, 18,786)`, d09 `(2,306, 42,156, 39,312)`, and d11
`(3,330, 75,068, 70,970)`. Every file retains 28 `EXP_VAL` probes and has no
`OBSERVABLE_INCLUDE`; the historical trace and throughput numbers above do not
apply.

Instructions used across the 8 files: QUBIT_COORDS, CX, CZ, CY, H, S, R, RX, M, MX, MY, MPP, E, TICK, T, EXP_VAL (+ DEPOLARIZE1/2, X_ERROR, Z_ERROR in p1e-3). Only clifft-extensions appearing: `T` (64), `EXP_VAL` (224).

## 6. `.clifft` format notes

Stim-superset, extension is filename convention only (no sniffing; ≤1 GB size check). clifft LACKS: `SPP`, `SPP_DAG`, `HERALDED_ERASE`, `HERALDED_PAULI_CHANNEL_1`, `sweep[k]`, Pauli targets on OBSERVABLE_INCLUDE, spaced combiners `X0 * Y1`, general `[tag]`s. clifft-only: `T`/`T_DAG`, `R_X/R_Y/R_Z` (half-turns), `U3`/`U`, `R_XX/R_YY/R_ZZ` (+ unprefixed aliases), `R_PAULI`, `EXP_VAL`, `READOUT_NOISE`, `LEAKAGE`/`LOSS`/`LEVEL_TRANSITION`, `CH`/`CCX`/`CCZ` rewrites, `DEPOLARIZE3`, `PAULI_CHANNEL_3`. 94 named gates.

`EXP_VAL`: one probe per product, non-destructive, Pauli-frame aware, clamped [-1,1], **hard optimizer reordering barrier** (`optimizer/commutation.cc:90-100`). Each benchmark now retains only its 28 logical CCZ probes; terminal stabilizers are closed by the compiled detector boundary instead of duplicated as probes. Unsupported in noncomp sampler.

No .clifft ↔ .stim converter anywhere (no circuit writer at all).

## 7. Throughput measurement

clifft exposes NO timing — time `sample_survivors` externally (wall-clock), as SOFT's benchmark.py does. `clifft.set_num_threads(1)` essential (default 20 here; threading engages only at peak rank ≥ 18 — irrelevant at peak_rank 8). Compile time reported separately.

In-repo benches: `just bench` (pytest-benchmark, wall-clock; d3 surface, deep Clifford, QV20, noncomp) — they use `sample()`, not `sample_survivors()`.
