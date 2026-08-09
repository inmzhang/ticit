# SOFT C++: `core/` + `circuit/` Consumption Surface Map

(Explorer report. All paths relative to `SOFT_ROOT`.)

## 0. Layering correction (important for a Rust port)

The layering is **not** `core → circuit → factored`. It is:

```
core/common.hpp
  ├─ core/pauli.hpp ──┬─ core/internal.hpp (detail::)
  └─ core/symbolic.hpp┘
       └─ core/frames.hpp ──┐
                            ├─→ factored/factored.hpp ──→ circuit/circuit.hpp ──→ frontend/stim.hpp
sampler/active.hpp ─────────┘                                    │                      │
  (depends only on core/pauli.hpp)                               │                      │
                                                     sampler/prepared_sampler.hpp ──────┘
                                                                 └─→ cuda/*
simd/*  ← NO core dependency at all
```

- `factored/factored.hpp:3-4` includes **`core/frames.hpp` and `sampler/active.hpp`** — `factored` sits *above* part of `sampler`.
- `circuit/circuit.hpp:3` includes **`factored/factored.hpp`** — `circuit` is *downstream* of `factored`. `CircuitLoweringResult` embeds a `FrameFactoredState` by value (`circuit.hpp:140`).
- `simd/simd.hpp` and `simd/batch_interleaved.hpp` include **only `<complex>/<cstddef>/<cstdint>`** and define their own `Complex`. **The entire `simd/` subsystem is core-independent** — flat C-style function-pointer `KernelTable` (`simd.hpp:11-64`). Cleanest port boundary in the repo.

## 1. Include / dependency graph (project-local)

Direct includers of `core/` or `circuit/`:

| File | Includes |
|---|---|
| `factored/factored.hpp` | `core/frames.hpp`, `sampler/active.hpp` |
| `factored/factored_internal.hpp` | `core/internal.hpp`, `factored/factored.hpp` |
| `factored/pending_optimizer.cpp` | `factored/factored.hpp`, `core/internal.hpp` |
| `factored/factored_planner.cpp` | `factored/factored_internal.hpp`, `sampler/component_plan.hpp` |
| `factored/factored_state.cpp` | `factored/factored_internal.hpp` |
| `circuit/circuit_lowering.cpp` | `core/internal.hpp`, `circuit/circuit.hpp` |
| `sampler/active.hpp` | `core/pauli.hpp` |
| `sampler/active_internal.hpp` | `core/internal.hpp`, `sampler/active.hpp` |
| `sampler/random.hpp` | `core/internal.hpp` |
| `sampler/component_plan.cpp` | `core/internal.hpp`, `sampler/active_internal.hpp` |
| `sampler/exogenous_presample.cpp` | `core/internal.hpp` |
| `sampler/presampled_expression.cpp` | `core/internal.hpp` |
| `sampler/prepared_sampler.hpp` | **`circuit/circuit.hpp`** |
| `frontend/stim.hpp` | **`circuit/circuit.hpp`** |
| `frontend/stim_parser.cpp` | `core/internal.hpp`, `frontend/stim.hpp` |
| `frontend/stim_sampling.cpp` | `frontend/stim.hpp`, `core/internal.hpp` |
| `cuda/cuda_program.cpp` | `core/internal.hpp`, `sampler/active_internal.hpp` |
| `python/_native.cpp` | `core/common.hpp` |

Transitive-only: `sampler/{exogenous,presampled_expression,single_shot,batch_sampler,component_plan}.hpp` via `factored/factored.hpp`; `sampler/contiguous_active.hpp` via `sampler/active.hpp`; `sampler/{batch_internal,active_kernels}.hpp` via `active_internal.hpp`.

Zero core/circuit dependency: all 7 `simd/*` files, `cuda/cuda_runtime.cu`.

## 2. Per-symbol consumption + layout contracts

### 2.1 `core/common.hpp`

| Symbol | Consumers | How |
|---|---|---|
| `Error` | cuda (7 sites), `_native.cpp:108` (`catch` → `symft.SymFTError`) | error contract |
| `Complex` (`std::complex<double>`) | `sampler/active.hpp:14,28`, active/batch kernels | `ActiveState::alpha` value type. **`simd/` uses own alias** |
| `checked_nqubits` | factored_state, planner, active_state | validation |
| `packed_bit` | planner, stim_sampling, samplers, cuda, `_native.cpp:347` | read-only bit extraction from `Vec<u64>` |
| `packed_bits` | `circuit_lowering.cpp:254-396` only | builds `SymbolicCategoricalDistribution::assignments` rows |
| `set_packed_bit`, `bit_word_count`, `nwords_for` | **core-only** | internal |

### 2.2 `core/pauli.hpp`

| Symbol | Consumers | How |
|---|---|---|
| `PauliString` | everywhere | value type |
| **`::x` / `::z` raw word vectors** | **`factored_planner.cpp:76, 240-242, 283-285, 334-336`; `active_internal.hpp:51,55`** | **LAYOUT CONTRACT §2.7** |
| `xbit`/`zbit`, `set_xbit`/`set_zbit` | factored_internal, planner | accessors |
| `phase_exponent` | `active_state.cpp:45,131` | feeds `phase_factor()` → `even_phase` |
| `set_phase` | planner `:539,549`, pending_optimizer `:24` | re-canonicalized to `pauli_body_y_count(p)` |
| `has_nonidentity_body` | factored_state `:296`, planner | guard |
| `same_body` | pending_optimizer `:60,157`, planner | rotation-fusion identity |
| `operator*` | planner `:548,833,837,840`, stim_parser `:593` | MPP products / tableau rows |
| `pauli_identity` | stim_parser `:579,689` | MPP accumulator seed |
| `pauli_x/y/z` | circuit_lowering, planner, stim_parser | constructors |
| `pauli_anticommutes` | planner `:78,547,564`, pending_optimizer `:29` | sign push-through & fusion |
| `pauli_squares_to_identity` | planner `:536,623,859`, pending_optimizer `:19`, active_state `:40` | Hermiticity guard |
| `pauli_body_y_count` | planner `:539,549`, pending_optimizer `:24` | phase re-canonicalization |
| `measurement_phase_sign` | planner `:532,648`, pending_optimizer `:23` | extracts symbolic sign from `i^phase` |
| `neg`, `operator<<`, `str()` | **unused downstream** | |

### 2.3 `core/symbolic.hpp`

| Symbol | Consumers | How |
|---|---|---|
| `SymbolicBool` | factored, circuit, frontend | value type |
| **`::conditions`** direct r/w | planner (many sites: 88-186, 382-477), pending_optimizer `:60`, presampled_expression `:39-52` | **LAYOUT CONTRACT §2.7** |
| `::constant` direct r/w | factored_state `:306`, planner, pending_optimizer `:64` | XOR-toggled, bypassing ctors |
| `xor_bool` | planner (8), factored_state `:334`, pending_optimizer `:23`, circuit_lowering `:27,416,457`, stim_sampling `:50` | |
| `max_condition()` | factored_internal (26 overloads :47-125), planner, factored_state `:399` | drives `bump_next_condition` |
| `operator!` | unused downstream | |
| **`SymbolicBoolEvaluationPlan`** | built in planner (9 sites); stored in 6 `FactoredInstruction` variants (`factored.hpp:50-90`) + `PresampledExpression::residual_plan`; consumed in single_shot_sampler `:70-133`, batch_symbols `:127-330`, batch_internal `:186-249`, cuda_program `:67-144` | **LAYOUT CONTRACT §2.7** |
| `SymbolicCategoricalDistribution` | planner `:1076-1158` (rarity classification), exogenous_presample, batch_symbols, single_shot_sampler, cuda | fields `nbits`, `conditions`, `assignments`, `probabilities` read directly |
| `SymbolicContext` | shared_ptr in `FrameFactoredState`/`PendingFactoredState`/`ActivePauliFrame`/`DormantState`; copied by value into `FactoredInstructionProgram::context` | see §4 |

### 2.4 `core/frames.hpp`

| Symbol | Consumers | Notes |
|---|---|---|
| `ConditionalPauliString` | `factored.hpp:182`, factored_state `:290-302` | |
| `SymbolicPauliString` | factored, planner, pending_optimizer | `.pauli`/`.sign` raw fields, mutated in place |
| `ActivePauliFrame` | field of `FrameFactoredState`; `add_pauli(pre, condition)` at factored_state `:297` | **`x_term_blocks`/`z_term_blocks` never touched downstream — private cache behind `conjugate_by`** |
| `conjugate_by(ActivePauliFrame, PauliString)` | factored_state `:285` (single call site) | sole gateway reading term blocks |
| `DormantState` | factored.hpp `:130,210` | only `.bits` read (factored_state `:394`). **`dormant_bit`/`set_dormant_bit`/`assign_dormant_symbol`: zero downstream callers — dead API** |
| `CliffordFrame` | `FrameFactoredState::clifford`, `PendingFactoredState::pending_frame` | |
| **`::rows`** | **planner `:801-803`** read+write | **LAYOUT CONTRACT §2.7 item 5** |
| `xrow`/`zrow`/`copy_pauli_to_row` | planner (16 call sites) | only sanctioned row-mutation path |
| `support_words`, `support_for_row`, `ensure_coordinate_columns`, etc. | **zero downstream** | internal memoization behind `preimage`/`coordinates_in_frame` |
| `preimage` | factored_state `:284,295`, planner `:325,802`, frames_tests (19×) | read-only |
| `coordinates_in_frame` | planner `:775` (single call) | |
| `left_*`/`right_*` (CliffordFrame) | factored_state `:102-276` 1:1 forwarding shims for all 46 `left_*`; **`right_*`: no downstream callers outside tests** | |

### 2.5 `core/internal.hpp` (`detail::`)

| Helper | Downstream |
|---|---|
| `fail` | 16 files |
| `kPi` | circuit_lowering, stim_parser, layout benches |
| `kLowProbabilitySampleThreshold` (0.02) | planner `:1092,1212`, exogenous_presample |
| `symbol_bit_mask`/`symbol_word_index`/`symbol_word_count` | all samplers, planner, cuda |
| `is_odd_popcount` | active_state, active_internal, component_plan, single_shot_sampler |
| `popcount64` | active_internal `:90`, prepared_sampler, batch_runtime, benches |
| `trailing_zeros64` | planner `:244-305`, batch_symbols, batch_internal, batch_runtime |
| `highest_set_bit64` | active_state `:65,75,76` only |
| `check_probability` | random.hpp, circuit_lowering, stim_parser, exogenous_presample, batch |
| `optional_equal` | factored_state `:14-66` only |
| `kWordBits`, `bit_mask`, `word_index`, `check_qubit`, `check_same_nqubits`, `toggle_condition`, `normalize_conditions` | **core-only** |

### 2.6 `circuit/circuit.hpp`

| Symbol | Consumers |
|---|---|
| `CircuitInstructionKind` (86 enumerators) | produced by `stim_parser.cpp:650-703`; consumed by giant `switch` in `circuit_lowering.cpp:105-701` |
| `CircuitInstruction`, `CircuitMeasurementTarget`, `CircuitPauliProduct`, `CircuitFeedbackTarget` | built in stim_parser, read in circuit_lowering |
| `CircuitDetector` (= `StimDetector`) | stim_sampling `:42-106,119-129`; `after_pending_operation` rewritten twice: `:102` (from `instruction_pending_operation_counts`) then `:127` (through `PendingOptimizationStats::prefix_remap`) |
| `CircuitObservableInclude` | prepared_sampler `:309-320` (`logical_records_for_observable`), `_native.cpp:490-492` |
| `QuantumCircuit` | stim.hpp, stim_sampling, `_native.cpp` (owns `unique_ptr<QuantumCircuit>`) |
| `lower_circuit_to_factored` | **single call site: `stim_sampling.cpp:140`** |

### 2.7 Raw-bit-layout contracts a Rust port must preserve

1. **`PauliString::x`/`z` as `Vec<u64>`, LSB-first, qubit `q` → word `q>>6`, bit `q&63`.**
   - `active_internal.hpp:50-56`: `active_mask_x(pauli) = pauli.x[0]` — word 0 reinterpreted wholesale as active-basis mask. Requires `k < 63` (enforced `active_state.cpp:37`) and bit `i` of `x[0]` = qubit `i`. Every SIMD/CUDA `xmask`/`zmask` derives from here.
   - `factored_planner.cpp:76`: hand-inlined `zbit()` in hot anticommutation path.
   - `factored_planner.cpp:240-257, 283-300, 334-346`: pops bits with `trailing_zeros64`/`x &= x-1`, `q = word*64 + ctz`; builds `pending_x_operation_blocks`/`pending_z_operation_blocks` bitset transpose (`factored.hpp:216-217`).
   - **`x.size() == z.size() == nwords_for(nqubits)` assumed** (`:336` indexes `z[word]` with loop bound from `x.size()`).

2. **`SymbolicBool::conditions` is strictly-ascending, duplicate-free `Vec<i32>` of positive ids.** Ctor enforces via `normalize_conditions`; downstream writes the field directly and must restore invariant:
   - planner `:127-138` merge-join two-pointer walk — wrong if unsorted.
   - planner `:182, 444` direct moves; `:401-415 normalize_xor_conditions` private re-implementation (sort + odd-multiplicity keep) after substitution expansion.
   - planner `:381-382` `std::binary_search` requires sortedness.
   - pending_optimizer `:60`: vector equality as *semantic* equality — only valid under canonical form.
   - planner `:88-96, 462-468`: ascending order ⇒ `symbol_word_index` non-decreasing; adjacent-dedup counts distinct words.

3. **`SymbolicBoolEvaluationPlan::word_indices` ascending & deduped; `word_masks[i]` = OR of all bits in that word.** Built `symbolic.cpp:90-94`. Semantics: **XOR-parity of masked bits, then XOR `constant`** (`single_shot_sampler.cpp:90`). `word_indices.back()` used as max word for bounds checks (`single_shot_sampler.cpp:71`, `batch_symbols.cpp:186,246`). cuda flattens into `CudaWordMask{word, mask}`; `batch_internal.hpp:206` fast path depends on `conditions` retained alongside word plan.

4. **Condition ids 1-based, dense** (`symbol_bit_mask(c) = 1 << ((c-1)&63)`, `symbol_word_index(c) = (c-1)>>6`). `batch_symbols.cpp:144,202,286` index `value_words[condition - 1]` directly. `nsymbols = context.next_condition - 1` (planner `:1203`) is sole authority for the dense range.

5. **`CliffordFrame::rows` flat `Vec<PauliString>` length `2*nqubits`, indexed `xrow(q)`/`zrow(q)`.** planner `:800-803` reads+writes `rows` directly, **bypassing `copy_pauli_to_row` and skipping `invalidate_support_cache()`** — safe only because `composed` is freshly constructed. Latent hazard for a port with different lazy-cache init.

6. **`SymbolicCategoricalDistribution::assignments[row]` packed-bit `Vec<u64>` addressed by `packed_bit(row, bit)`**, `conditions[bit_idx]` ↔ bit `bit_idx`: exogenous_presample `:196-200`, cuda_program `:89-100` (caps `nbits ≤ 64`), planner `:1080-1082`.

7. **`ActivePauliFrame::x_term_blocks`/`z_term_blocks` and `CliffordFrame` caches are NOT part of the downstream contract** — restructure freely as long as `conjugate_by`, `preimage`, `coordinates_in_frame` keep semantics.

## 3. `factored/factored.hpp` API surface used by circuit lowering

`CircuitLoweringAccumulator` (`circuit_lowering.cpp:13-16`) = `{FrameFactoredState state; std::vector<SymbolicBool> records;}`.

### `FrameFactoredState` — `factored.hpp:125-136`

```cpp
struct FrameFactoredState {
    int n = 0;                                       // total qubits
    int k = 0;                                       // active qubit count
    CliffordFrame clifford;                          // sized n
    ActivePauliFrame active_frame;                   // ActivePauliFrame(n, context)  ← sized n, not k
    DormantState dormant;                            // DormantState(n - k, context)
    std::shared_ptr<SymbolicContext> context;        // shared with active_frame & dormant
    std::vector<PendingOperation> pending_operations;// append-only queue
};
```
Ctors (`factored_state.cpp:84-100`): validate `checked_nqubits`, fail if `k > n`, fresh `SymbolicContext` when null. Lowering always starts at `FrameFactoredState(circuit.nqubits, 0)` (`circuit_lowering.cpp:707`).

### Mutating entry points

| Signature | Semantics | Lowering call sites |
|---|---|---|
| `apply_pauli(state, ConditionalPauliString)` (`factored_state.cpp:290-299`) | bump `next_condition`, `preimage(clifford, pauli)`, if non-identity append to `active_frame` via `add_pauli(pre, condition)`; **no pending op queued** | — |
| `apply_pauli(state, PauliString, int condition)` (`:301-303`) | thin forward | — |
| `apply_pauli(state, PauliString, SymbolicBool)` (`:305-312`) | expands XOR expr into one `apply_pauli` per condition id; `true` constant → fresh probability-1.0 Bernoulli condition | `263-264, 285-288, 328-329, 370, 402-403, 434, 444, 457, 623, 669` |
| `apply_pauli_rotation(state, PauliString, double kernel_angle) → PendingPauliRotation` (`:314-318`) | `prepare_pending_pauli` (= `conjugate_by(active_frame, preimage(clifford, p))`), push `PendingPauliRotation{angle, symbolic_pauli}`. Angle = internal φ of `exp(-iφP)` | `533, 539` |
| `apply_pauli_measurement(state, PauliString)` (`:320-324`) | queue measurement, no record | — (unused by lowering) |
| `apply_pauli_measurement(state, PauliString, sign, record?, record_condition?)` (`:326-340`) | prepare, XOR caller's `sign` into conjugation-derived sign, queue; `record` = 1-based Stim record index | `443, 456, 468, 602` |
| `apply_pauli_expectation(state, PauliString, exp_val)` (`:342-357`) | rejects negative; queue `PendingPauliMeasurement` with `exp_val`, no record — **non-destructive probe**; sets `has_expectation` → forces source-order planning | `612` |
| `apply_classical_record(state, outcome, record?, record_condition?)` (`:359-368`) | purely classical record event — heralds, MPAD | `401, 419` |
| `pending_operations` field | lowering snapshots queue length per instruction → `instruction_pending_operation_counts` → `CircuitDetector::after_pending_operation` (stim_sampling `:102`) | `713` |

### `left_*(FrameFactoredState&, …)` — `factored_state.cpp:102-276`

All **46** are pure forwarding shims to the `CliffordFrame` overloads. 24 single-qubit (`H, H_NXY, H_NXZ, H_NYZ, H_XY, H_YZ, C_NXYZ, C_NZYX, C_XNYZ, C_XYNZ, C_XYZ, C_ZNYX, C_ZYNX, C_ZYX, S, SDG, SQRT_X, SQRT_X_DAG, SQRT_Y, SQRT_Y_DAG, X, Y, Z`) + 22 two-qubit (`CX, CY, CZ, SWAP, CXSWAP, CZSWAP, ISWAP, ISWAP_DAG, SQRT_XX, SQRT_XX_DAG, SQRT_YY, SQRT_YY_DAG, SQRT_ZZ, SQRT_ZZ_DAG, SWAPCX, XCX, XCY, XCZ, YCX, YCY, YCZ`). **They mutate only `state.clifford` — Clifford gates absorbed into frame for free.**

### Post-lowering pipeline

`PendingFactoredState(const FrameFactoredState&)` (`factored_state.cpp:389-408`) adopts `n, k, dormant.bits, context, pending_operations`, scans for `exp_val` (→ `has_expectation`) and max `record` (→ `next_record`). Then `optimize_pending_operations(state, preserved_prefixes)` → `plan_factored_updates(state)` → `FactoredInstructionProgram`. Entry: `stim_sampling.cpp:110-135`.

## 4. `SymbolicContext` field consumers

### `bernoulli_probabilities` (`std::map<int,double>`)
- Written by `fresh_bernoulli_condition`/`fresh_bernoulli_bool`, driven from circuit_lowering `:27, 416, 623, 672`, factored_state `:307`.
- **Read at exactly one place**: planner `:1211-1218` (`FactoredInstructionProgram` ctor). Partitions by `kLowProbabilitySampleThreshold` (0.02) into `sampled_bernoulli_conditions` + `sampled_bernoulli_probabilities` (parallel vectors) and `sampled_low_probability_bernoulli_groups` (`BernoulliSampleGroup{probability, conditions}`, grouped by exact double equality `:1110-1118`).
- **`std::map` ordering (ascending condition id) determines fill order and hence RNG draw order** — reproducibility contract for a Rust port (use `BTreeMap`).

### `categorical_distributions` (`std::vector<SymbolicCategoricalDistribution>`)
- Written by `fresh_categorical_conditions`/`fresh_categorical_bools`, driven exclusively from circuit_lowering `:259, 284, 326, 368, 396`.
- **Read at exactly one place**: planner `:1207-1210` → `build_categorical_sample_plan` (`:1146-1158`), classifies via `rare_categorical_sample_info` (`:1076-1108`, finds all-false row, checks `1 - P(all-false) < 0.02`) into `sampled_categorical_distributions` (dense) or `sampled_rare_categorical_groups` (geometric-skip, deduplicated by structural equality `:1124-1134`).

### `condition_to_categorical` (`std::unordered_map<int,size_t>`)
- Written/read only inside `core/symbolic.cpp:123,165` (guard against a condition joining two categorical groups). **Zero external consumers — a Rust port can make it a private construction-time check.**

### `next_condition`
- Read once: planner `:1203` → `nsymbols = max(0, next_condition - 1)`. Mutated via `bump_next_condition` (planner 7 sites, factored_state 3) and `fresh_condition()` (circuit_lowering `:19,442`, planner `:726,865`).

## 5. Build system

- Top-level CMake: `enable_testing()` + `add_subdirectory(cpp)`. C++20. Single static lib `symft_cpp`, includes rooted at `cpp/src/`.
- Flags: `SYMFT_CPP_ENABLE_AVX2` (default ON, per-file `-mavx2 -mfma`), `SYMFT_CPP_ENABLE_AVX512` (ON, `-mavx512f -mavx512dq -mfma`), `SYMFT_CPP_NATIVE` (ON, `-march=native`), `SYMFT_CPP_ENABLE_CUDA` (OFF; `SYMFT_CPP_CUDA_REAL_DOUBLE` OFF flips `CudaReal` float→double).
- Executables: `symft_cli`, `symft_bench`, `symft_plan`, `symft_batch_bench`, `symft_active_layout_bench`, `symft_batch_active_layout_bench`, `symft_rate_bench`, CUDA-gated `symft_cuda_rate_bench`.
- Python binding: hand-written CPython C-API (`_native.cpp`), **not pybind11**; compiles same C++ source list as CMake (duplicated); exposes `symft.Circuit`, `CompiledMeasurementSampler`, `CompiledCountsSampler`, `SymFTError`, `simd_backend()`, `cuda_enabled()`. Only core touchpoints: `symft::Error`, `symft::packed_bit`.
