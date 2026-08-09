# C++ Test Suite Catalogue — SOFT/cpp/tests

(Explorer report; basis for porting the C++ test suite to Rust.)

## 0. Framework & organization (no framework at all)

There is **no Catch2/doctest/gtest**. Every test binary is a hand-rolled `main()` that calls plain `void test_*()` functions in an anonymous namespace, and a `require(bool, std::string)` helper that prints `FAILED: <msg>` to `stderr` and calls `std::exit(1)` on the first failure.

Three binaries, registered in `SOFT/cpp/CMakeLists.txt:147-161` via `add_test`:

| CMake target | Source | Notes |
|---|---|---|
| `symft_cpp_tests` | `tests/symft_tests.cpp` (2089 lines) | 28 test functions, dispatch list at `symft_tests.cpp:2058-2089` |
| `symft_frames_tests` | `tests/frames_tests.cpp` (231 lines) | 4 test functions, dispatch at `frames_tests.cpp:223-230` |
| `symft_cuda_tests` | `tests/symft_cuda_tests.cpp` | only built when `SYMFT_CPP_ENABLE_CUDA=ON`; reads `benchmark/circuit/*.stim` fixtures |

Rust port implication: `require(cond, msg)` maps to `assert!(cond, "{}", msg)`; `test_*` functions map to `#[test]` fns; table-driven style maps to slice-of-tuples loops. `main()` ordering carries no semantics.

## 1. `core/pauli.hpp|cpp`

### 1.1 `test_pauli_algebra` — `symft_tests.cpp:548-570`

**Exercises**: `pauli_x/y/z/identity`, `operator*`, `phase_shift`, `set_xbit/set_zbit`, `pauli_anticommutes`, `pauli_string`, `xbit`/`zbit`, `operator==`.

Invariants, each an exact structural equality (`operator==` compares `nqubits`, `phase` byte, and both `x`/`z` word vectors — **not** a semantic Pauli comparison):

| line | assertion | pins down |
|---|---|---|
| 550 | `X·X == I` (1 qubit) | phase carry 0 |
| 551 | `Y·Y == I` | Y's stored `phase=1` must cancel: `1+1 + 2*(z&x popcount=1 → carry 1)` = `4 ≡ 0` |
| 552-554 | `X·Z == pauli_y(1,0)` with `phase_shift(3)` applied | `X·Z` has `phase = (1+3)&3 = 0`, body `x=1,z=1`. Encodes the **`i^phase` with Y carrying an implicit `i`** convention |
| 555 | `pauli_anticommutes(X, Z)` true | |
| 556-562 | 65-qubit `X_0 X_64` vs `Z_0 Z_64` → **commutes** | forces cross-word XOR-then-single-parity implementation (`pauli.cpp:180-184` XORs all words into one accumulator *before* popcount parity — per-word `bool` OR would be wrong) |
| 563-564 | drop `Z_0` → `X_0X_64` vs `Z_64` **anticommutes** | one cross-word site remains |
| 565-569 | `pauli_string("IXYZ")` bit layout | `I`→(0,0), `X`→(1,0), `Y`→(1,1), `Z`→(0,1), index = string position |

**Not asserted anywhere in the suite**: `PauliString::str()`, `pauli_squares_to_identity`, `pauli_body_y_count`, `measurement_phase_sign`, `neg()`, `has_nonidentity_body`, `pauli_identity` of size 0, `'_'` alias in `pauli_string`, `i*`/`-`/`-i*` prefix formatting in `str()` (`pauli.cpp:72-96`), the `fail("unsupported Pauli character")` path. Coverage gaps a Rust port should fill — `measurement_phase_sign` has a `fail()` branch for non-Hermitian Paulis (`pauli.cpp:207`) reached only through `factored_planner.cpp:532,648`.

### 1.2 Indirect `PauliString` fixtures (helpers a port needs)

- `uniform_xmask_pauli(k, pivot, lower_mask)` — `symft_tests.cpp:222-231`: X on `pivot` plus X on every bit set in `lower_mask` below pivot. Pure-X body.
- `real_xmask_pauli` — `:233-238`: same, plus `set_zbit(pivot)` and `set_phase(1)` → Y at pivot with real coefficient.
- `general_xmask_pauli` — `:240-244`: same, plus `set_zbit(pivot-1)` → mixed X/Z body.

These generate the "uniform / real / general" kernel classes used in §5.

## 2. `core/symbolic.hpp|cpp`

**No dedicated `test_symbolic*` function.** `SymbolicBool` asserted only through frames and program-lowering tests. `SymbolicContext::fresh_bernoulli_bool`, `fresh_categorical_bools`, `fresh_categorical_conditions`, `SymbolicCategoricalDistribution`, `SymbolicBool::str()`, `max_condition()`, `operator!` on constants, `SymbolicBoolEvaluationPlan` word/mask packing have **zero direct assertions**. Port should add unit tests.

What *is* pinned:

| location | assertion | pins down |
|---|---|---|
| `symft_tests.cpp:583` | `conjugated.sign == SymbolicBool(false, {1, 66})` | 2-arg ctor **normalizes**: sorts + XOR-cancels duplicate conditions (`symbolic.cpp:16-17` → `internal.hpp:132-138 normalize_conditions`) |
| `:585` | `signed_query.sign == symbolic_bool(1)` | `xor_bool` = symmetric difference of sorted condition lists (`symbolic.cpp:59-70`): `{1,66} ⊕ {66} = {1}` |
| `:760-761` | fused rotation sign preserved | fusion preserves symbolic sign identity |
| `:813, 818` | `!sign` used as fused sign | `operator!` flips only `constant`, keeps `conditions` (`symbolic.cpp:55-57`) |
| `:1208-1213` | `bump_next_condition(dense_sign)` then `fresh_condition()` → 9 | `bump_next_condition` uses `max_condition()` (`symbolic.cpp:105-118`) |
| `:1222` | measurement outcome has `dense_sign.conditions.size() + 1` conditions | 8-term dense sign XOR branch symbol, un-simplified |
| `:1232` | second measurement outcome: `!constant && conditions.empty()` | full cancellation to literal `false` |
| `:1268, 1303` | `SymbolicContext(12)` explicit next_condition ctor | `SymbolicContext(n)` with `n<=0` fails (`symbolic.cpp:99-103`) — untested |
| `:1272` | `reduced == xor_bool(symbolic_bool(9), symbolic_bool(10))` | see §4.4 |
| `:1557, 1586` | row index `(condition - 1) * shot_words` | **condition ids are 1-based**, index presampled storage as `id-1` |

`SymbolicBoolEvaluationPlan` constructed 11 times in tests but its `word_indices`/`word_masks` coalescing (consecutive conditions in same 64-bit word merge into one mask, `symbolic.cpp:85-97`) never asserted directly. **Port should add direct test.**

## 3. `core/frames.hpp|cpp`

### 3.1 `test_active_pauli_frame_index` — `symft_tests.cpp:572-586`

Setup: `k=65`. Loop `term = 0..69`: `add_pauli(pauli_x(65, term % 65), term + 1)` → 70 terms, conditions 1..70. Then `add_pauli(pauli_x(65,0), 100)` **twice** → terms 70, 71 both condition 100.

Assertions:
1. `:582` — `conjugate_by(frame, pauli_z(65,0)).pauli == pauli_z(65,0)`: Pauli-frame conjugation never changes body, only symbolic sign.
2. `:583` — `.sign == SymbolicBool(false, {1, 66})`. Load-bearing: forces exact block layout (`frames.cpp:46-81`): terms packed **64 per block**; each block occupies `k` consecutive `uint64_t` in `x_term_blocks`/`z_term_blocks`, indexed `[block*k + qubit]`, term bit at `1 << (term & 63)`. Terms with X on qubit 0: term 0 (cond 1), 65 (cond 66), 70+71 (cond 100 ×2). Collected `[1, 66, 100, 100]` → sorted → **odd-multiplicity filter** (`frames.cpp:142-151`) drops pair → `{1, 66}`. Pins: 64-term blocking, `base + q` stride `k`, >64-qubit multi-word body scan, duplicate-condition XOR cancellation.
3. `:584-585` — `SymbolicPauliString` overload XORs incoming sign into frame-derived sign (`frames.cpp:157-164`).

Buffer grows lazily: `add_pauli` resizes by `k` words when `(terms.size() & 63) == 0` (`frames.cpp:51-54`).

**Untested**: 1-arg `add_pauli` (auto-fresh condition), dimension-mismatch fail (`frames.cpp:47-49`), both `conjugate_by(ConditionalPauliString, ...)` overloads (`frames.cpp:83-97`), `ConditionalPauliString` `condition <= 0` rejection (`frames.cpp:15-17`). **`DormantState` has zero direct test coverage** — `dormant_bit`, `set_dormant_bit`, `assign_dormant_symbol` need new Rust tests.

### 3.2 `CliffordFrame` + `preimage()` — the whole of `frames_tests.cpp`

**Convention to replicate exactly.** `CliffordFrame::rows` stores `U† X_q U` (rows `0..n-1`) and `U† Z_q U` (rows `n..2n-1`) (`frames.hpp:75`; `xrow(q)=q`, `zrow(q)=n+q`). `preimage(frame, P)` returns **`U† P U`** = Stim's forward tableau of the *inverse* gate. Visible in test data: `ISWAP` expects `X_ → -ZY` while `ISWAP_DAG` expects `X_ → +ZY` (`frames_tests.cpp:98-99`), the opposite of Stim's own `ISWAP` tableau. Cross-checking against Stim requires daggering the gate first.

#### `test_clifford_frame` — `frames_tests.cpp:32-43`
- `:36-37` — after `left_CX(cf, 0, 1)` on `n=2`: `preimage(X_0) == pauli_string("XX")`, `preimage(Z_1) == pauli_string("ZZ")`.
- `:39-42` — `left_H(h,0)` then `left_S(h,0)` on `n=1`: `preimage(X_0) == pauli_y(1,0)` (phase word 1).

#### `test_extended_clifford_frame_preimages` — `frames_tests.cpp:45-130`

Helper `expected_pauli_preimage(string)` (`:19-30`) parses leading `-` → `phase_shift(2)`, then `pauli_string(rest)`; `_` = identity.

**Single-qubit table** (`:54-73`), 18 gates, `n=1`, assert `preimage(X_0)` / `preimage(Z_0)`:

| gate | `U†XU` | `U†ZU` |
|---|---|---|
| `left_H_NXY` | `-Y` | `-Z` |
| `left_H_NXZ` | `-Z` | `-X` |
| `left_H_NYZ` | `-X` | `-Y` |
| `left_H_XY` | `Y` | `-Z` |
| `left_H_YZ` | `-X` | `Y` |
| `left_C_NXYZ` | `-Z` | `Y` |
| `left_C_NZYX` | `Y` | `-X` |
| `left_C_XNYZ` | `Z` | `-Y` |
| `left_C_XYNZ` | `-Z` | `-Y` |
| `left_C_XYZ` | `Z` | `Y` |
| `left_C_ZNYX` | `-Y` | `X` |
| `left_C_ZYNX` | `-Y` | `-X` |
| `left_C_ZYX` | `Y` | `X` |
| `left_SQRT_X` | `X` | `Y` |
| `left_SQRT_X_DAG` | `X` | `-Y` |
| `left_SQRT_Y` | `Z` | `-X` |
| `left_SQRT_Y_DAG` | `-Z` | `X` |
| `left_Y` | `-X` | `-Z` |

**Two-qubit table** (`:94-113`), 18 gates, `n=2`, assert `preimage(X_0)`, `preimage(Z_0)`, `preimage(X_1)`, `preimage(Z_1)`:

| gate | `U†X_0U` | `U†Z_0U` | `U†X_1U` | `U†Z_1U` |
|---|---|---|---|---|
| `left_CY` | `XY` | `Z_` | `ZX` | `ZZ` |
| `left_CXSWAP` | `_X` | `ZZ` | `XX` | `Z_` |
| `left_CZSWAP` | `ZX` | `_Z` | `XZ` | `Z_` |
| `left_ISWAP` | `-ZY` | `_Z` | `-YZ` | `Z_` |
| `left_ISWAP_DAG` | `ZY` | `_Z` | `YZ` | `Z_` |
| `left_SQRT_XX` | `X_` | `YX` | `_X` | `XY` |
| `left_SQRT_XX_DAG` | `X_` | `-YX` | `_X` | `-XY` |
| `left_SQRT_YY` | `ZY` | `-XY` | `YZ` | `-YX` |
| `left_SQRT_YY_DAG` | `-ZY` | `XY` | `-YZ` | `YX` |
| `left_SQRT_ZZ` | `-YZ` | `Z_` | `-ZY` | `_Z` |
| `left_SQRT_ZZ_DAG` | `YZ` | `Z_` | `ZY` | `_Z` |
| `left_SWAPCX` | `XX` | `_Z` | `X_` | `ZZ` |
| `left_XCX` | `X_` | `ZX` | `_X` | `XZ` |
| `left_XCY` | `X_` | `ZY` | `XX` | `XZ` |
| `left_XCZ` | `X_` | `ZZ` | `XX` | `_Z` |
| `left_YCX` | `XX` | `ZX` | `_X` | `YZ` |
| `left_YCY` | `XY` | `ZY` | `YX` | `YZ` |
| `left_YCZ` | `XZ` | `ZZ` | `YX` | `_Z` |

**Gate coverage gap.** 44 `left_*` + 8 `right_*` = 52 functions. Covered: 39. **Never called in any test**: `left_SDG`, `left_X`, `left_Z`, `left_CZ`, `left_SWAP`, all eight `right_*` (no non-`frames.cpp` callers repo-wide). **`coordinates_in_frame()` zero coverage**; only caller `factored_planner.cpp:775`. Sparse/dense branch selector in `preimage` (`support.size()*2 <= pauli.x.size()`, `frames.cpp:348`) only incidentally exercised — small `n` always takes one branch. **Add a large-`n` preimage test for the sparse path.**

#### `test_sqrt_gate_directions` — `frames_tests.cpp:132-168`

1. `:139-155` — four circuits sampled `sample_measurements(program, 8, 31)`, **every** shot's record 0 equals fixed bit:
   - `SQRT_X 0 / MY 0` → all `true`; `SQRT_X_DAG 0 / MY 0` → all `false`
   - `SQRT_Y 0 / H 0 / M 0` → all `false`; `SQRT_Y_DAG 0 / H 0 / M 0` → all `true`
2. `:157-167` — bit-exact record-vector equality: `SQRT_X ≡ H S H` (via `MY`), `SQRT_X_DAG ≡ H S_DAG H`, 8 shots seed 37. Requires identical randomness consumption.

#### `test_extended_clifford_gate_directions` — `frames_tests.cpp:170-219`

12 gates: parse native circuit + reference decomposition; assert both have empty `pending_operations` and `native.state.clifford == reference.state.clifford` (full row-by-row equality incl. phases):

| gate | reference |
|---|---|
| `C_NXYZ` | `H S H S_DAG` |
| `C_NZYX` | `S S H S_DAG` |
| `C_XNYZ` | `S H` |
| `C_XYNZ` | `H S_DAG H S` |
| `C_XYZ` | `S_DAG H` |
| `C_ZNYX` | `H S_DAG` |
| `C_ZYNX` | `S H S_DAG H` |
| `C_ZYX` | `H S` |
| `CXSWAP` | `CX 0 1; SWAP 0 1` |
| `SWAPCX` | `SWAP 0 1; CX 0 1` |
| `ISWAP` | `CX 1 0; CX 0 1; CX 1 0; S 0; H 1; CX 0 1; H 1; S 1` |
| `ISWAP_DAG` | `S_DAG 1; H 1; CX 0 1; H 1; S_DAG 0; CX 1 0; CX 0 1; CX 1 0` |

Only place `CliffordFrame::operator==` asserted; only place `left_CZ`/`left_SWAP` exercised.

## 4. `circuit/circuit.hpp` + `circuit/circuit_lowering.cpp`

Two parse entry points from `frontend/stim.hpp`: `parse_stim_circuit_text` → `QuantumCircuit` (structural), `parse_stim_text` → `StimParseResult` (lowered).

### 4.1 `test_stim_frontend_circuit_lowering` — `symft_tests.cpp:1319-1356`

Circuit: `REPEAT 2 { M !0 }` + `DETECTOR rec[-1] rec[-2]`.

- `nqubits == 1`, `nrecords == 2`, `instructions.size() == 2` — **REPEAT flattened at parse time**
- `instructions[0].kind == MZ`; `measurement_targets[0].inverted == true` (`!0`)
- `detectors[0].records == {2, 1}` — `rec[-1]`→2, `rec[-2]`→1, **1-based absolute record indices, source order preserved**; `after_instruction == 2`
- `instruction_pending_operation_counts.size() == 3` — **prefix table length `ninstructions + 1`** seeded 0 (`circuit_lowering.cpp:708-714`); `[2] == 2`
- `M 0 / H 0 / DETECTOR rec[-1]`: `after_instruction == 2`, `after_pending_operation == 1` — Clifford advances instruction counter, not pending-op counter
- `M !0 / CY rec[-1] 1 / M 1` → both records 1 (`FeedbackY`); `X 0 / CY 0 1 / M 1` → gate path. **Same mnemonic, two lowering paths**

### 4.2 `test_parser_feedback` — `symft_tests.cpp:1309-1317`

`M !0 / CX rec[-1] 1 / M 1` → both records 1. Pins `FeedbackX` + `CircuitFeedbackTarget{record, qubit}`. `FeedbackZ` **not covered**.

### 4.3 `test_extended_stim_frontend` — `symft_tests.cpp:1358-1497`

**(a) Rotation angle conversion, `:1360-1389`.** Stim's `R_*(t)` argument in half-turns; stored `kernel_angle` = `t·π/4`:
- `R_X(0.5) 0` → `kernel_angle == π/4` (±1e-12), body via `same_body` (**phase ignored**)
- `R_Z(pi/pi) 0` → `π/2` — parser evaluates **arithmetic with `pi`**
- `R_XX(0.25)` → `π/8`; `R_PAULI(-0.5) X0*Z1` → `-π/4`; `U3(0.5,0.25,-0.5) 0` → **3 instructions**, angles `-π/4, +π/4, +π/8`

**(b) MPAD / observables, `:1390-1405`.**
- `MXX !0 1 / MPAD 1 0 / OBSERVABLE_INCLUDE(0) rec[-1] X0` → `nrecords == 3`, `instructions.size() == 2`, `observables[0].records == {3}` (Pauli targets in observables dropped)
- `MPAD 1` alone → `nqubits == 0` (**MPAD targets are literal 0/1 values, not qubit ids**, `circuit_lowering.cpp:411-414`), `nrecords == 1`

**(c) Parse-error rejection, `:1406-1456`.**
- `CX sweep[x] 0` throws; `CX sweep[0] 0`, `CZ 0 sweep[0]` → message contains **"sweep-controlled operations are not supported"**
- `REPEAT 0 { M 0 }`, `REPEAT(2) 1 { M 0 }` throw; `OBSERVABLE_INCLUDE(0.6) rec[-1]` throws; `R(0.25) 0` throws
- `I_ERROR(0.8,0.8) 0`, `II_ERROR(0.8,0.8) 0 1` throw (prob sum > 1)
- `TICK 0`, `TICK()`, `SHIFT_COORDS(1) 0`, `SHIFT_COORDS`, `QUBIT_COORDS foo`, `QUBIT_COORDS() 0` throw

**(d) Correlated error chain, `:1457-1467`.** `E(0.25) X0 / ELSE_CORRELATED_ERROR(0.5) Z0 / M 0` → 2 instructions (chain collapses to one `PauliProductChannel`, flushes before measurement); `probabilities == {0.25, 0.375}` (±1e-12). **0.375 = 0.5 × (1 − 0.25)** — ELSE renormalized to absolute at parse time.

**(e) Aliases, `:1468-1481`.** `H_XY / ZCX / SWAPCZ / SQRT_YY_DAG / R_YY(0.25) / R_PAULI(1) X0*Z1 / M 0 1` smoke, 4 shots seed 11. `ZCX`→`CX`, `SWAPCZ`→`CZSWAP`.

**(f) Noise channels, `:1482-1496`.** `SPP X0*Z1 / PAULI_CHANNEL_1 / PAULI_CHANNEL_2 / DEPOLARIZE3(0.1) / HERALDED_ERASE(0.1) / HERALDED_PAULI_CHANNEL_1 / M 0 1`: `nrecords == 4` — **heralded channels each append one record** ahead of `M` records (`circuit_lowering.cpp:400-401`).

**Critical gap**: channel *distributions* never numerically checked. Lowering (`circuit_lowering.cpp:251-340`): `DEPOLARIZE1(p)` = 4-way `{1−p, p/3 ×3}` over 2 bits; `DEPOLARIZE2(p)` = 16-way `{1−p, p/15 ×15}` over 4 bits; `PAULI_CHANNEL_n` = `4^n`-way over `2n` bits, axis code `remainder & 3` read **most-significant-target-first** (`:316`); `DEPOLARIZE3` via `apply_depolarize_n` `p/63`; `HERALDED_*` = 5-way over 3 bits, assignments `{000, 100, 110, 111, 101}` → `{no-herald, herald+I, +X, +Y, +Z}`, probs `{1−Σp, p0..p3}` (`:389-399`). **Transcribe from source; add tests.**

### 4.4 Measurement-relation simplification — `symft_tests.cpp:1242-1274`, `1276-1307`

`FactoredInstructionProgram` constructed directly; constructor rewrites a later `RecordMeasurement`:
- dense case: `IntroduceDormantMeasurementBranch{branch=10, outcome = dense_sign ⊕ s10, record=1, record_condition=9}` then `RecordMeasurement{outcome = dense_sign}` → rewritten to `s9 ⊕ s10`.
- sparse case: `RecordMeasurement{outcome = symbolic_bool(9)}` → **unchanged** (rewrite only fires when it shrinks the expression).

### 4.5 `test_detectors` — `symft_tests.cpp:1910-1922`

`estimate_stim_logical_error_rate(parsed, 5)`:
- `M !0 / OBSERVABLE_INCLUDE(0) rec[-1]` → `shots==5, discarded==0, logical_errors==5`
- `M !0 / DETECTOR rec[-1] / OBSERVABLE_INCLUDE(0) rec[-1]` → `discarded==5, accepted==0`

### 4.6 Detector postselection — `test_batch_postselection`, `symft_tests.cpp:1840-1908`

- `M !0 / DETECTOR rec[-1]`, 8 shots → all discarded, `active_shots==0`
- `M 0 / DETECTOR rec[-1]`, 8 shots → none discarded, `measurement_words[0]==0`
- `X_ERROR(0.125) 0 / M 0 / DETECTOR rec[-1] / H 1 / T 1 / T_DAG 1 / H 1 / M 1`, 64 shots seed 41: default vs `BatchDetectorPostselectionOptions{1}`; `discarded > 0`, `active_shots == accepted`, **identical results between compaction strategies**.

## 5. Cross-validation methods used

**No comparison against Stim, no state-vector brute force.** Five techniques:

1. **Naive kernel reference** — `reference_active_measurement_probability` (`:246-274`), `reference_active_measurement_projection` (`:276-306`). Diagonal branch: `pivot_value = diagonal_phase_bit ⊕ parity(source0 & z_without_pivot) ⊕ branch`; non-diagonal: `inv_sqrt2·α[s0] + (branch ? −1 : +1)·conj(η)·inv_sqrt2·α[s1]`, `η = odd ? action.odd_phase : action.even_phase`, selected by `parity(s0 & action.zmask)`. Tolerance **1e-9** absolute.
2. **Generic-vs-precomputed kernel** — `check_high_pivot_single_rotation_kernel` (`:191-220`): `rotate_pauli` generic as oracle vs AoS precomputed kernel + SoA `FactoredExecutorState`, 1e-9. Batch version (`:466-518`) incl. `mixed_signs` (odd shots get `−theta` via packed sign bits).
3. **SIMD-vs-scalar** — `test_d5_nondiagonal_measurement_simd_kernel` (`:1099-1146`). Fixture `X_0 X_1 Y_9` at `k=10`, exact masks `xmask == 0x203`, `zmask == 0x200`. `simd::scalar_table()` vs `simd::dispatch_table()` for `measure_nondiagonal_probability_soa` / `project_nondiagonal_soa`, both branches, **1e-12**.
4. **Optimized-vs-unoptimized, bit-exact** — `plan_without_pending_optimization` (`:666-672`) drains pending ops one at a time; `test_pending_operation_optimizer_end_to_end` (`:870-1033`): `sample_measurements(reference, N, seed) == sample_measurements(optimized, N, seed)` for 5 circuits (512/512/1024/1024/1024 shots, seeds 811/821/827/829/831).
5. **Dense-vs-component, serial-vs-parallel** — `test_active_component_factorization` (`:1598`) forces `use_active_components` on/off, compares records/words/survivors. `test_prepared_sampler_multithreading` (`:1963`) 1-thread vs 4-thread `CircuitSamplingCounts`. CUDA tests: GPU vs CPU discard rate within 0.02 at 20000/30000 shots.

Statistical windows:
- `test_t_gate_exact_rotation` (`:1499`): `H 0/T 0/H 0/M 0`, 200 shots seed 7 → ones ∈ (10, 50)
- `test_batch_sampler` (`:1801`): `H 0/T 0/M 0`, 200 shots seed 23, batch 32 → ones ∈ (50, 150)
- `test_presampled_exogenous` (`:1560`): `X_ERROR(0.5)`, 11 shots seed 31 → word ≠ 0 and ≠ all-ones; (`:1595`): `X_ERROR(0.3)`, 4096 shots seed 1234 → ones ∈ (1000, 1450)

## 6. Test helper functions a Rust port needs

**`frames_tests.cpp`**: `require` (12-17), `expected_pauli_preimage(string)` (19-30).

**`symft_tests.cpp`**:
| helper | line | purpose |
|---|---|---|
| `require_same_counts` | 34-42 | compares all 4 fields of `CircuitSamplingCounts` |
| `require_throws` / `require_throws_with_message` | 44-65 | `symft::Error`, substring match |
| `approx(Complex, Complex, eps=1e-10)` | 67-69 | `abs(a−b) <= eps` |
| `deterministic_amplitude(basis, shot)` | 71-75 | `re = 0.001·((basis%97)+1+3·shot)`, `im = −0.0015·(((basis+5·shot)%89)+1)` — **replicate bit-for-bit** |
| `deterministic_alpha(k, shot)` | 77-84 | fills `2^k` amplitudes |
| `active_only_program(k)` | 86-92 | empty `FactoredInstructionProgram(k,k,{},k)` |
| `high_pivot_rotation_instruction` | 94-109 | `ApplyPrecomputedActivePauliRotation`, constant sign |
| `component_test_promotion/rotation/measurement` | 111-140 | pending-op builders |
| `component_test_program()` | 142-189 | 10-instruction fixture; global order `[q0,q2]` vs merged component order `[q2,q0]` forces non-highest local pivot |
| `check_*_kernel` family | 191-518 | oracle comparisons, see §5 |
| `execute_expression_postselected_for_test` | 520-541 | plan+evaluate+postselect wrapper |
| `plan_without_pending_optimization` | 666-672 | unoptimized oracle planner |
| `pending_record_order` | 674-688 | record ids in pending order |
| `require_pending_record_conditions_are_causal` | 690-728 | **causality checker**: every condition consumed by an op that is produced by some `record_condition` must have producer strictly earlier. Port verbatim |

**`symft_cuda_tests.cpp`**: `require`, `write_temp_stim`, `fixture_path`, `run_cuda_file`/`run_cpu_batch_file`, `discard_rate`, `require_close`.

## 7. Other planner-adjacent behaviors pinned

- `test_high_pivot_selection` (`:645-664`): `PrecomputedActivePauliRotationKernel(X_1·X_3, 0.125).pair_bit == 3`; diagonal kernel on `Z_0·Z_3` → `is_diagonal && pivot == 3`; non-diagonal on `X_1·Z_2·X_3` → `!is_diagonal && pivot == 3`. `PendingFactoredState(4,0)` + rotation `X_0·X_2` + measurement `Z_2`: after one `process_next_pending_operation`, `pending.k == 1`, measurement Pauli rewritten to `pauli_z(4, 0)` — **dormant promotion picks highest dormant pivot, remaps coordinates**.
- `test_active_rotation` (`:588-643`): `rotate_pauli(ActiveState(1), X, π/4)` → `α₀ = cos(π/4)`, `α₁ = −i·sin(π/4)`; `Y, π/6` → `α₀ = cos(π/6)`, `α₁ = +sin(π/6)`; precomputed `Z, 0.31` → `α₀ *= e^{−0.31i}`, `α₁ *= e^{+0.31i}`. `sign=true` negates angle. 1e-10.
- `test_active_h_rewrite_stays_virtual` (`:1154`), `test_dormant_measurement_tableau_reuse` (`:1169`), `test_dormant_measurement_sign_feeds_promotion` (`:1187`): `program.max_k` = 2, 1, 1; per-shot record relations (`bit0 == bit1` repeated dormant measurement `Z_0X_1`; `bit0 != bit1` after `S` between two `MX`).
- `test_pending_operation_optimizer` (`:730-868`): commuting same-body rotations fuse with **angle addition** (0.1+0.3=0.4); anticommuting interleave blocks fusion; different symbolic signs don't fuse; **opposite** signs (`sign` vs `!sign`) fuse with **angle subtraction** (0.2−0.3 → 0.1, sign `!sign`); exact inverses cancel (`cancelled_rotations == 1`); preserved pending prefix `{1}` blocks cross-detector fusion, `prefix_remap[1]==1, prefix_remap[2]==2`; `PendingClassicalRecord` = movement barrier.
- `test_batch_expectation_sampler` (`:1812-1838`): expectation values `{1.0, 0.0, 1/√2, 1/√2, 0.0}` at 1e-12 for `EXP_VAL Z0 X0 / H / T / EXP_VAL X0 Y0 Z0 / T_DAG / H / M 0`; probe non-destructive (record 0 always `false`).
- Constants: `default_single_shot_sample_chunk_shots() == 2048`, `default_batch_count(10) == 32` (`:1545-1546`).
