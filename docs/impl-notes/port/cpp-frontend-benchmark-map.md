# SOFT frontend / tools / benchmark map

(Explorer report. Reference paths are relative to `SOFT_ROOT`.)

## 1. Frontend — `cpp/src/frontend/`

### 1.1 Public API (`stim.hpp`, `stim_prepared_sampler.hpp`)

Two parse layers: **circuit layer** returns flattened IR; **parse layer** additionally lowers to factored state.

| Function | File:line | Returns |
|---|---|---|
| `parse_stim_circuit_lines/_text/_file` | `stim_parser.cpp:1331,1340,1350` | `QuantumCircuit` |
| `parse_stim_lines/_text/_file` | `stim_parser.cpp:1363,1374,1385` | `StimParseResult` |
| `plan_stim_factored_program` | `stim_sampling.cpp:110` | `FactoredInstructionProgram` |
| `make_stim_circuit_sampling_input[_from_file]` | `stim_sampling.cpp:137,155` | `CircuitSamplingInput` |
| `prepare_single_shot_sampler_from_stim_file` / `..._batch_...` | `stim_sampling.cpp:171,179` | `PreparedCircuit{SingleShot,Batch}Sampler` |
| `estimate_stim_logical_error_rate(parsed, shots, seed=1)` | `stim_sampling.cpp:187` | `StimSampleSummary` |
| `discard_rate` / `logical_error_rate` | `stim_sampling.cpp:219,224` | `double` (NaN when denominator 0) |

Structs: `StimParseResult{state, measurement_records, detectors, observables}` (`stim.hpp:14`); `StimSampleSummary{shots, discarded, accepted, logical_errors}` as `int` (`stim.hpp:29`). `StimDetector`/`StimObservableInclude` alias `CircuitDetector`/`CircuitObservableInclude` (`stim.hpp:11-12`).

IR types in `circuit/circuit.hpp`: `CircuitInstructionKind` (71 variants, `:9-86`), `CircuitInstruction{kind, probability, kernel_angle, qubits, measurement_targets, pauli_products, feedback_targets, probabilities, exp_val, line}` (`:103`), `CircuitDetector{records, coords, line, after_instruction, after_pending_operation}` (`:116`), `CircuitObservableInclude{index, records, line}` (`:124`), `QuantumCircuit{nqubits, nrecords, nexpvals, instructions, detectors, observables}` (`:130`).

Sampler-side structs in `sampler/prepared_sampler.hpp`: `CircuitSamplingOptions{observable=0, postselect_detectors=false, sample_chunk_shots=0, batch_size=0, batch_mask_threshold_denominator=2, threads=1}` (`:31`), `CircuitSamplingCounts` (u64 `shots/discarded/accepted/logical_errors`, `:15`), `CircuitSamplingTiming{parse_s, plan_s, presample_s, execute_s, accumulate_s, sample_s}` (`:22`), `CircuitSamplingInfo` (`:40`), `CircuitSamplingRunResult{counts, timing, active_threads}` (`:55`).

### 1.2 Lexical structure

- Line-oriented. `strip_stim_comment` (`:209`) drops `#`-to-EOL **only when not inside `[...]`** (naive flag, not nesting-aware).
- `parse_instruction` (`:274`): op = leading `[A-Za-z][A-Za-z0-9_]*`, uppercased → **case-insensitive op names**.
- Optional tag `OP[tag]` parsed into `inst.tag` (`:286-293`) then **completely ignored**.
- Optional parens `(...)` matched by depth (`find_matching_paren:170`), split on top-level commas (`split_arguments:185`).
- Each arg via `parse_numeric_expression` (`:161`): fast `strtod`, else recursive-descent (`ArgumentExpressionParser:66`) with `+ - * /`, unary sign, parens, constant `PI` (case-insensitive; `:145`). Unknown identifier → `"unknown numeric constant: <NAME>"`.
- Targets = whitespace-split remainder.
- **`REPEAT N {`** one line ending `{`; body ends with a line exactly `}`. No parens, exactly one target (`:334`). Count in `[1, 10^18]` (`kStimMaxRepeatCount:242`); above `INT_MAX` → `"REPEAT count is too large for this flattened circuit frontend"`. Blocks **fully flattened by literal repetition** (`append_nodes:1294`).

### 1.3 Target grammar

| Form | Recognizer | Notes |
|---|---|---|
| `5`, `!5` | `stim_qubit_target:355` | `!` = inverted; rejected for non-measurement ops |
| `rec[-k]` | `is_record_target:389` | must start `rec[`, end `]`, offset `-` + digits. `stim_record_index:397` → **1-based** `idx = nrecords + offset + 1`, in `[1, nrecords]` |
| `sweep[k]` | `is_sweep_target:417` | recognized only to reject: `"sweep-controlled operations are not supported"` (`:996`) |
| `X5`, `!Z3`, `*` | `is_pauli_target_or_combiner:424` | `stim_mpp_pauli:578` splits on `*`, each `!` toggles product `inverted` |
| `X1 * X2` spaced | `stim_mpp_targets:602` | regroups; `"misplaced MPP combiner"` / `"dangling MPP combiner"` |

**Qubit count** (`all_qubits:546`): max qubit index + 1 over all targets, **excluding** `DETECTOR`, `OBSERVABLE_INCLUDE`, `MPAD`, `TICK`, `SHIFT_COORDS` (`:554`). `QUBIT_COORDS` targets *do* count.

### 1.4 Accepted instruction set

**Single-qubit Cliffords** (`single_qubit_clifford_kind:839`, no parens): `H`/`H_XZ`, `H_NXY`, `H_NXZ`, `H_NYZ`, `H_XY`, `H_YZ`, `C_NXYZ`, `C_NZYX`, `C_XNYZ`, `C_XYNZ`, `C_XYZ`, `C_ZNYX`, `C_ZYNX`, `C_ZYX`, `S`/`SQRT_Z`, `S_DAG`/`SQRT_Z_DAG`, `SQRT_X`, `SQRT_X_DAG`, `SQRT_Y`, `SQRT_Y_DAG`, `X`, `Y`, `Z`.

**Two-qubit Cliffords** (`two_qubit_clifford_kind:912`): `CX`/`CNOT`/`ZCX`, `CY`/`ZCY`, `CZ`/`ZCZ`, `SWAP`, `CXSWAP`, `CZSWAP`/`SWAPCZ`, `ISWAP`, `ISWAP_DAG`, `SQRT_XX(_DAG)`, `SQRT_YY(_DAG)`, `SQRT_ZZ(_DAG)`, `SWAPCX`, `XCX`, `XCY`, `XCZ`, `YCX`, `YCY`, `YCZ`.

**Non-Clifford / rotations** (`:1145-1171`):
- `T`, `T_DAG` — dedicated kinds, no parens.
- `R_X`, `R_Y`, `R_Z` — one angle arg, per-qubit (`append_single_axis_rotation:742`).
- `R_XX`, `R_YY`, `R_ZZ` — one angle, **paired** targets, product `P_q1·P_q2` (`append_two_axis_rotation:755`); `q1 == q2` → `"two-qubit Pauli operation requires distinct qubits"`.
- `R_PAULI(θ) X0*Z1 …` — one angle, MPP-style products.
- `U3`/`U` — exactly 3 args `(θ, φ, λ)`; emits **three** `PauliRotation`s in order Z(λ), Y(θ), Z(φ) using `parens[2], parens[0], parens[1]` (`append_u3_rotation:772`).
- `SPP` / `SPP_DAG` — no parens, hardcoded kernel angle `±π/4` (`:1216`).

**Angle convention (critical):** all `R_*`/`U3` args are **half-turns**; `pauli_rotation_kernel_angle_from_half_turns` (`:726`): `angle_kernel = half_turns · π / 2`; kernel applies `exp(-i·angle·P)`. `SPP` bypasses conversion.

**Measurement / reset** (`:1172-1210`): `M`/`MZ`, `MX`, `MY`; `MR`/`MRZ`, `MRX`, `MRY`; `R`/`RZ`, `RX`, `RY` (no parens); `MPP`; `MXX`/`MYY`/`MZZ` (paired → `MPP` kind, `inverted` = XOR of pair's `!` flags, `circuit_pair_measurement_products:713`); `MPAD` (targets are pad *values* `0`/`1`/`!0`, appends records, does **not** grow `nqubits`). All measurement ops accept optional single-probability paren = measurement flip probability (`stim_paren_probability:447`, default 0).

**Noise** (`:1218-1278`): `X_ERROR`, `Y_ERROR`, `Z_ERROR` (required p); `DEPOLARIZE1/2/3`; `PAULI_CHANNEL_1/2/3` with exactly 3/15/63 probs (`:1251`); `HERALDED_ERASE(p)`, `HERALDED_PAULI_CHANNEL_1(4 probs)` — both append **one record per qubit** (`:1269,1277`); `I_ERROR`, `II_ERROR` — validated (disjoint probs sum ≤ 1 + 1e-12, `check_disjoint_probability_list:503`) then **dropped**.

**Correlated errors** (`append_correlated_error:979`): `E`/`CORRELATED_ERROR` starts group, `ELSE_CORRELATED_ERROR` extends (error if no open group). Targets form implicit Pauli product, **no `*` allowed** (`circuit_implicit_pauli_product:688`). Conditional → absolute probs: `p_i · Π_{j<i}(1 − p_j)` (`:987-988`). Group flushes into one `PauliProductChannel` on any other instruction (`:1079`), REPEAT boundary (`:1293`), end of input (`:1311`).

**Feedback** (`append_classical_controlled_pairs:991`): triggered when **any** target is `rec[-k]`; then all targets consumed pairwise. `CX`/`CNOT`/`ZCX`/`CY`/`ZCY` with rec first → `FeedbackX`/`FeedbackY`; `CZ`/`ZCZ` rec on **either** side → `FeedbackZ`; `XCZ`/`YCZ` rec **second** → `FeedbackX`/`FeedbackY`. Pairs without rec fall back to ordinary gate. Other op with rec → `"unsupported classically controlled gate"`.

**Annotations**: `TICK` (emits `Tick`, rejects parens/targets); `QUBIT_COORDS` (1–16 coords when parens present, requires targets); `SHIFT_COORDS` (1–16 coords **required**, no targets, accumulates `builder.coord_shift`); `DETECTOR` (optional 1–16 coords, records all `rec[-k]`, coords stored `coords + coord_shift` via `coords_with_shift:629`, `after_instruction = instructions.size()`); `OBSERVABLE_INCLUDE` (exactly one paren = nonneg integer via `nonnegative_integer_argument:495`; `rec[-k]` targets collected, Pauli/`*` targets **silently ignored**, else fail, `stim_observable_record_indices:435`).

**`EXP_VAL`** (`:1194`): no parens, MPP-style products, ≥1 required, allocates `nexpvals` slots. SymFT/Clifft extension, not Stim.

**No postselection marker syntax.** Postselection is a *sampler option*: `estimate_stim_logical_error_rate` (`stim_sampling.cpp:196-202`) discards a shot if **any** detector bit is 1; `CircuitSamplingOptions::postselect_detectors` selects same all-detectors policy in prepared samplers. No per-detector mask on SOFT side.

### 1.5 Parse-error behavior

Every failure: `detail::fail` → throws `symft::Error : std::runtime_error`. `check_probability` (`internal.hpp:113`): `"probability must be between 0 and 1"`, NaN rejected by negated-comparison form.

**Gap:** `inst.line` tracked and stored but **no error message includes it** (`stim_qubit_target` discards with `(void)line`, `:363`). Rust port can improve at zero compat cost.

Unknown op → `"unsupported Stim operation: " + op`. Unmatched `}` → `"unmatched Stim block terminator"`; missing `}` → `"unterminated Stim block"`; `REPEAT` without `{` → `"REPEAT must start a block"`.

### 1.6 Detector positioning (subtle, load-bearing)

Detectors carry source instruction index, remapped twice: `detectors_with_lowered_positions` (`stim_parser.cpp:1314`, duplicated `stim_sampling.cpp:93`) converts `after_instruction` → `after_pending_operation` using `instruction_pending_operation_counts`; then `plan_stim_factored_program` (`stim_sampling.cpp:110`) remaps through `optimize_pending_operations(...).prefix_remap`, failing with `"detector pending-operation prefix was not preserved by optimization"`. Finally `insert_stim_detector_events` (`:55`) splices `RecordDetector` instructions at resolved checkpoints, 1-based `detector` ids, XOR-of-records `SymbolicBool` outcome (`detector_expression:42`). **Pending optimizer must preserve every prefix a detector observes — hard constraint on the Rust planner.**

## 2. Tools — `cpp/tools/`

### `symft_cli` (36 lines)
`usage: symft_cli <circuit.stim> [shots]`, shots default 1000, 1–2 positional args else exit 2. Runs `estimate_stim_logical_error_rate` (single-shot path, seed 1). Prints `key value` per line: `qubits`, `records`, `max_active_qubits`, `simd_backend`, `shots`, `discarded`, `accepted`, `logical_errors`, `discard_rate`, `logical_error_rate`. Errors → `symft_cli: <what>` stderr, exit 1.

**Always postselects all detectors and XORs across ALL observables** (`stim_sampling.cpp:205-211`) — differs from prepared samplers which filter to single `observable` index via `logical_records_for_observable` (`prepared_sampler.cpp:309`).

### `symft_rate_bench` (444 lines) — the throughput tool the port must match

Flags (`parse_options:214`), `--name value` or `--name=value`:

| Flag | Default |
|---|---|
| `--circuit PATH` / `--file PATH` | `benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim` |
| `--shots N` | `100000000` |
| `--sampler single\|single-shot\|single_shot\|batch\|batched\|both\|all` | `batch` |
| `--sample-chunk-shots N\|auto\|none\|off` | `0` (auto) |
| `--repeats N` | 1 |
| `--observable N` | 0 |
| `--threads N\|auto` | 1 (`auto` = hardware_concurrency) |
| `--active-components auto\|1\|on\|enabled\|0\|off\|disabled\|dense` | `auto` |
| `--postselect-detectors [=bool]` / `--no-postselect-detectors` | false |
| `--batch-size N\|auto` | `0` (auto) |
| `--batch-mask-threshold-denominator N` | 2 |

Positionals in fixed order: path, shots, batch_size, repeats, observable, postselect_detectors.

Output (`print_result:379`), `key value` lines: `sampler` (`single`/`single_postselected`/`batch`/`batch_postselected`), `file`, `shots` (**requested**), `sampled_shots` (actual), `active_components` (`enabled`|`dense_fallback`), `detector_postselection`, `batch_size` (batch only), `sample_chunk_shots`, `repeats`, `threads`, `requested_threads` (if differs), `batch_mask_threshold_denominator` (batch_postselected only), `sample_s_avg`, `sample_shots_per_s`, `presample_s_avg` + `execute_s_avg` (**only when threads == 1**), `discarded`, `accepted`, `logical_errors`, `discard_rate`, `logical_error_rate`. `--sampler both`: single result, blank line, batch result.

**Throughput definition:** `sample_shots_per_s = requested_shots / timing.sample_s` (`:381`) — requested, not sampled; timings averaged over repeats (`average_repeated_timings:336`), counts accumulate. Repeat `r` passed as `stream_id` (`:355`).

### `symft_plan` (96 lines)
Parse + plan only. Prints `qubits`, `records`, `detectors`, `instructions`, `max_active_qubits`, then if component plan exists: `active_components`, `component_count`, `dense_peak_dimension`, `component_peak_live_dimension`, `component_allocated_dimension`, `estimated_dense_vector_work`, `estimated_component_vector_work`, then `pending_operations_before/after`, `fused_rotations`, `cancelled_rotations`, `measurement_left_swaps`, `parse_seconds`, `plan_seconds`, `peak_rss_kib`.

## 3. Benchmark harness — `benchmark/`

### Methodology (`benchmark.py`, `README.md:177-192`)
- **Pinning**: `pin_cpu` (`benchmark.py:83`) hard-pins to one logical CPU via `sched_setaffinity`.
- **Thread limiting**: env at `:29-43` before simulator imports — `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/BLIS_NUM_THREADS=1`, `JAX_ENABLE_X64=true`, `JAX_PLATFORMS=cpu`, etc.
- **Compile timeout**: SIGALRM 300 s → `COMPILE_TIMEOUT`.
- **Warmup**: stim `sample(min(shots,1024))`; tsim full-size JAX warmup; clifft `sample_survivors(program, 1, seed=seed-1)`; symft `sample(shots=1, stream_id=seed-1)`.
- **Measured loop** (`measure:272`): per repeat, fixed-size calls until accumulated `sample_s ≥ seconds` (60). Seed per call = `seed + repeat*1_000_000 + calls`, base offset `case["seed"] + 10_000`.
- **Metric**: `shots_per_second = total_shots / elapsed` per repeat; arithmetic mean across repeats (2). Attempted shots, not surviving.
- **Config** (`config.json`): `run.cpu=0`, `sample_seconds=60`, `repeats=2`, `seed=20260723`. Per-tool per-circuit shot counts differ by up to 6 orders of magnitude.

### How each tool is invoked

| Tool | Construction | Counting |
|---|---|---|
| stim | `compile_detector_sampler()`; `sample(shots, separate_observables=True, bit_packed=True)` | `errors = count_nonzero(obs[:,0] & 1)`; **discarded hardcoded 0** |
| tsim | `compile_detector_sampler(strategy, seed)`; postselection mask = all-ones | `discarded = any(dets, axis=1)`, `errors = count_nonzero(~discarded & obs[:,0])` |
| clifft | `parse_file → trace → hir passes → compute_reference_syndrome → lower(postselection_mask=[1]*ndet, expected_detectors=ref, expected_observables=ref) → bytecode passes` | native `total_shots/.discards/.logical_errors` |
| symft | `Circuit(path).compile_counts_sampler(batch=True, observable=0, postselect_detectors, batch_size=0, threads=1)` | native counts |

### SOFT#8 / SOFT#9 mismatches as visible in code

**(a) Output-convention mismatch** (`benchmark.py:184-192`): clifft compiled with `expected_detectors`/`expected_observables` from `compute_reference_syndrome` → XOR-normalized against noiseless reference (`clifft/__init__.py:351-354`). tsim/symft use **raw parity** (symft discard test = "any detector bit set", `stim_sampling.cpp:196-202`). Differs when noiseless reference detectors nonzero → tools postselect different events → different work per shot. Throughput-comparability bug.

**(b) Logical-error-metric mismatch**: stim counts `obs[:,0]` over all shots; tsim over non-discarded; symft prepared sampler XOR-parity over `OBSERVABLE_INCLUDE` entries with `index == options.observable` (`prepared_sampler.cpp:309-319`), while `symft_cli` XORs across **every** observable (`stim_sampling.cpp:205-211`); clifft against normalized reference. Not comparable across tools; not even self-consistent within SOFT.

**(c) Timing asymmetry**: stim/tsim/clifft timed wall-clock around the call; symft reports internal `timing.sample_s` — excludes Python-boundary overhead, systematically favors symft.

### Circuits (`benchmark/circuit/`, 12 files + manifest.json)
`pure_surface_d7_r7_p1e-3.stim`, `pure_surface_d9_r9_p1e-3.stim` (genuine Stim); `msc_d3_inject_cultivate_p1e-3.stim`, `msc_d5_inject_cultivate_p1e-3.stim`, `msc_proxy_d7_unverified_p1e-3.stim`, `MSC_circuit_d7_p0.0005.stim`, `distillation.stim`, `coherent_surface_d{3,5}_r{1,3,5}_p1e-3_rz0p02.stim` (extended dialect, Stim cannot parse). manifest: `angle_convention: "half_turns (angle * pi = radians)"`, per-circuit metadata + SHA-256. d=7 proxy's detector statistics NOT valid for correctness claims (README:73-87).

### GPU harness (`GPU_benchmark.py`, `GPU_config.json`)
Tsim + SymFT only. Fresh child process per case (GPU memory release), `CUDA_VISIBLE_DEVICES` selection. `symft_gpu` config: per-circuit `mode` (`gpu`|`gpu_presample_expressions`), `threads_per_block` (32/64/128), `shots_per_launch`. Passes `cuda=True, cuda_mode, shots_per_launch, threads_per_block, sample_chunk_shots=shots_per_launch`; asserts `backend == "cuda"`.

## 4. `.clifft` circuits — original ccz-nontels bundle

The initial detector-free snapshot had 8 files
`d{05,07,09,11}_p{0,1e-3}.clifft`, 292 KB – 7.3 MB.

**No `.clifft` format exists — files are plain Stim-dialect text**, same dialect SOFT parses; both parsers extension-agnostic. `d05_p0.clifft`: ~834 `QUBIT_COORDS` lines, circuit body, 100 trailing `EXP_VAL` lines.

Opcode census `d05_p0.clifft` (3113 lines): QUBIT_COORDS 834, TICK 539, CX 459, RX 269, MX 262, M 255, R 254, EXP_VAL 100, CZ 88, MPP 27, T 8, E 8, CY 4, S 2, MY 2, H 2. `d05_p1e-3` adds DEPOLARIZE2 1202, DEPOLARIZE1 543, Z_ERROR 272, X_ERROR 262. All supported by SOFT parser.

The initial imported snapshot had no `DETECTOR`, `OBSERVABLE_INCLUDE`, or
`REPEAT` instructions. The fixtures regenerated on 2026-08-13 inline their
separate decoder companions: detector counts are 6,992 (d05), 18,786 (d07),
39,312 (d09), and 70,970 (d11), with no observables. They use direct logical-T
preparation and have 8,012, 20,604, 42,156, and 75,068 measurements,
respectively. The census and benchmark description in this section are
historical.

`E` lines = 8 T-proxy source markers, each on the T corner before its source
MPP, for example `E(0.125) Z741`. `run_benchmark.py:54` rewrites via regex and
requires exactly 8 matches.

Runner (`run_benchmark.py`): `--engine symft|clifft`, `--path`, `--shots 2000`, `--batches 3`, `--source-probability 0.0`, `--sample-chunk-shots 2048`, `--unpacked`, `--drop-exp`, `--cpu`, `--monitor-interval 0.02`. 200-shot warmup, 3 timed batches, `median_batch_shots_per_second`; peak RSS from the Linux process-status watchdog; blake2b digest over measurements+expectations for cross-engine agreement. JSON on stdout. `run_all.sh`: 16 one-core processes (clifft CPUs 0–7, symft 8–15), `OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1`; symft d=11/p=1e-3 compile documented non-completing (killed 3192.6 s).

## 5. clifft — `CLIFFT_ROOT`

C++20 + CMake (not Rust). `src/clifft/{api,backend,circuit,frontend,noncomp,optimizer,sampling,svm,util}`, nanobind Python binding. **No CLI binary** — Python package is the entry point (`just py-install`, `just bench`).

Running: `parse_file → trace → default_hir_pass_manager().run → lower(hir, postselection_mask, expected_detectors, expected_observables) → default_bytecode_pass_manager().run → sample_survivors(program, shots, seed, keep_records)`. One-shot: `clifft.compile(stim_text, postselection_mask=None, expected_detectors=None, expected_observables=None, normalize_syndromes=False, ...)` (`__init__.py:331`).

Output: `.measurements`, `.detectors`, `.observables`, `.exp_vals`, `.total_shots`, `.passed_shots`, `.discards`, `.logical_errors`, `.observable_ones` (`bindings.cc:978-1212`).

**Raw-parity mode is the DEFAULT**: `expected_detectors`/`expected_observables` default empty, `normalize_syndromes=False`. `normalize_syndromes=True` mutually exclusive with explicit parities. CCZ bundle uses `clifft.compile(text, hir_passes=None, bytecode_passes=None)`, no parities/mask + `clifft.set_num_threads(1)`.

Port-relevant: `clifft.get_num_threads()`/`set_num_threads()` (OpenMP); AGENTS.md invariants — 32-byte instruction ABI, allocation-free hot dispatch, uniform `[0,1)` = `(rng() >> 11) * 0x1.0p-53`.

## 6. Python tests — `python/tests/` (8 files, 544 lines)

- **test_imports.py** — exports: `Circuit`, `CompiledMeasurementSampler`, `CompiledCountsSampler`, `SymFTError`, `read_stim_file`, `sample`, `simd_backend`, `cuda_enabled`, `active_cuda_backend`; NOT `active_simd_backend`/`active_batch_backend`; `SymFTError` subclasses RuntimeError; exact signature defaults; `__version__ == "0.1.0"`.
- **test_circuit_metadata.py** — empty circuit → all-zero metadata, `(shots, 0)` sample; `detectors[0]`: `records=(1,)` (1-based), `coords=(1.5,2.5)`, `line=2`; `OBSERVABLE_INCLUDE(2)` → `num_observables == 3` (**= max index + 1**), `num_observable_includes == 1`; `REPEAT 3 { M 0 }` → `num_measurements == 3` (flattening observable).
- **test_errors.py** — unknown gate → SymFTError; sweep message exact; both text+path → TypeError; negative shots → ValueError; `batch_size=-1` → ValueError; double `__init__` → RuntimeError, original intact.
- **test_stim_features.py** — `M !0`+`CX rec[-1] 1`+`M 1` both records 1; `T 0` then `MX 0` nondeterministic (20 < ones < 180 of 200); `MPAD 1 0` → records `(1, 0)`, no qubits; heralded channel adds record.
- **test_measurement_sampling.py** — samples `np.bool_` `(shots, nrecords)`; `shots=0` preserves width; `bit_packed=True` → `np.uint8` LSB-first across byte boundaries; same seed bit-identical; **single-shot and batch backends produce identical arrays for same seed**; `compile_sampler(batch=True, batch_size=2)` handles `shots=5`.
- **test_detector_sampling.py** — detector samples dtype/shape; bit-packed layout same.
- **test_counts_sampling.py** — `sample_counts` keys exactly `shots, discarded, accepted, logical_errors, discard_rate, logical_error_rate, active_threads, timing`; **single-shot backend discards on any fired detector BY DEFAULT; batch backend does NOT postselect unless `postselect_detectors=True`** — real behavioral divergence, port must decide deliberately; prepared sampler reuse across stream_ids; identical stream_id reproduces counts; **concurrent sample() calls from two threads serialize and return identical results**; invalid `cuda_mode` → ValueError.
- **test_exp_val.py** — `EXP_VAL` non-destructive (records bit-identical with/without probe at same seed); `H 0; CX 0 1; EXP_VAL Z0*Z1 X0*X1 X2` → `(1.0, 1.0, 0.0)`; 13-qubit dense-path → `2^(-13/2)`; expectation columns ordered by EXP_VAL appearance.

## Port-relevant summary

Frontend = ~1400 lines line-oriented parsing; three tricky parts: (1) half-turn → kernel-angle with `SPP` hardcoded exception; (2) CORRELATED_ERROR group accumulation, conditional→absolute conversion, three flush triggers; (3) two-stage detector position remapping (lowering + pending-optimizer prefix preservation). Else mechanical opcode table. Parse errors: untyped strings, no line numbers — free improvement.

Benchmarking: reproduce `symft_rate_bench` output format + attempted-shots/internal-sample_s definition for symft comparison. A fair ticit-vs-symft-vs-clifft comparison must now execute the same detector/observable output contract in every backend and report normalization and postselection explicitly.
