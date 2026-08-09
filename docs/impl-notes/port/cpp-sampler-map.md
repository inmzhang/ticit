# SOFT C++ → Rust port map: `factored/`, `sampler/`, `simd/`

(Explorer report. Paths relative to `SOFT_ROOT/cpp/src/`. Namespace `symft`, internals in `symft::detail`.)

## 0. Type foundations

| Type | Definition | Shape |
|---|---|---|
| `PauliString` | `core/pauli.hpp:10-29` | `nqubits`, `Vec<u64> x`, `Vec<u64> z`, `u8 phase` (i^phase) |
| `SymbolicBool` | `core/symbolic.hpp:12-22` | `bool constant` XOR a **sorted, deduped** `Vec<i32> conditions` (1-based ids) |
| `SymbolicBoolEvaluationPlan` | `core/symbolic.hpp:33-41` | `constant`, `conditions`, word-grouped `word_indices`/`word_masks` for packed XOR-parity |
| `SymbolicContext` | `core/symbolic.hpp:50-72` | `next_condition`, `map<int,double> bernoulli_probabilities`, `categorical_distributions` |
| `CliffordFrame` | `core/frames.hpp:66-92` | rows = `U† X_q U` / `U† Z_q U`; `xrow(q)`/`zrow(q)` |

Condition-id bit packing (`core/internal.hpp:57-76`): condition `c` → word `(c-1)>>6`, bit `(c-1)&63`. Reused for symbols, records, detectors. `packed_bit` is 0-based.

Sign conventions (`core/pauli.cpp:187-208`):
- `pauli_squares_to_identity(P)` ⇔ `(phase_exponent − y_count) & 1 == 0`
- `measurement_phase_sign(P)` = `(phase_exponent − y_count) & 3 == 2`; **throws** for ±i coefficients.

## 1. The planner — `factored/factored_planner.cpp` (1254 lines)

### 1.1 Input

`PendingFactoredState` (`factored.hpp:205-233`) from lowered `FrameFactoredState` (`factored_state.cpp:389-408`). Lowering already conjugated every Pauli through frames (`factored_state.cpp:280-287`: `prepare_pending_pauli` = `conjugate_by(active_frame, preimage(clifford, P))`); planner sees flat `PendingOperation` list:

```
PendingPauliRotation    { kernel_angle: f64, pauli: SymbolicPauliString }
PendingPauliMeasurement { pauli, record: Option<i32>, record_condition: Option<i32>, exp_val: Option<i32> }
PendingClassicalRecord  { outcome: SymbolicBool, record, record_condition }
```

Qubits `[0, k)` **active** (dense amplitude coordinates), `[k, n)` **dormant** (|0⟩ stabilizer, tableau only).

### 1.2 Core loop

`process_next_pending_operation` (`:938-977`):
1. Checkpoint instruction index into `pending_prefix_instruction_indices` (frontend re-anchors DETECTORs via this — `stim_sampling.cpp:23-39`).
2. Pop front op (or advance `pending_operation_cursor` in expectation mode).
3. `reduce_pending_operation_signs` — substitutions + XOR relations (§1.5).
4. Expectation mode only: transform op by accumulated `pending_frame`.
5. Dispatch by variant; append to `state.instructions`.
6. Push another checkpoint.

`process_pending_operations_in_place` (`:981-989`) runs `optimize_pending_operations` first if planning hasn't started.

### 1.3 Instruction opcodes (6, `factored.hpp:94-100`)

| Opcode | Emitted by | Runtime effect | Δk |
|---|---|---|---|
| `ApplyPrecomputedActivePauliRotation` | `:600-627` | `exp(−i·φ·P)` on dense vector; `sign` flips φ | — |
| `PromoteDormantRotation` | `:629-645` | doubles vector: `α' = [c·α, ∓i·s·α]` | +1 |
| `RecordMeasurement` | `:651-671`, `:924-936` | deterministic outcome bit / expectation ±1 | — |
| `RecordDetector` | **frontend-injected** (`stim_sampling.cpp:54-88`), not planner | XOR of records, or symbolic outcome | — |
| `MeasurePrecomputedActivePauli` | `:853-890` (sampled), `:751-772` (exp_val) | Born-rule sample + project | −1 (0 exp_val) |
| `IntroduceDormantMeasurementBranch` | `:701-749` | fair coin, no quantum work | — |

Each carries a `SymbolicBoolEvaluationPlan` (`sign_plan`/`outcome_plan`).

### 1.4 Basis changes resolved entirely at plan time

**Runtime α vector never touched by basis change.** Three planner-only `CliffordFrame`s applied to *remaining pending operations*, so kernels always act on coordinate `k−1` or `k` (comment `:554-555`).

**(a) Dormant rotation promotion** — `dormant_rotation_promotion_tableau_frame` (`:556-598`). Picks highest dormant qubit with X bit (`highest_dormant_x_qubit`, `:522-529`, scans n−1 down to k). Frame: `X_{old_k} := P` (sign-stripped via `positive_hermitian_body`, `:535-541`), `Z_{old_k} := Z_{picked}`, other rows ×`Z_{picked}` if anticommute with P (`:543-552`), surviving dormant rows repacked. `k += 1`.

**(b) Dormant measurement replacement** — `:673-699`. `Z_{picked} := P`, `X_{picked} := Z_{picked}^{old}`. Measured Pauli becomes exactly `Z_{picked}` = uniform coin. k unchanged.

**(c) Active measurement coordinate frame** — `:813-851`. Measured active Pauli → `Z_{k−1}`, anticommuting partner → `X_{k−1}`, remaining k−1 coordinates relabeled around dropped pivot. `k -= 1`.

After each frame, `push_symbolic_pauli_through_pending_from` (`:315-358`): fresh condition `b`, Pauli `X_{pivot}` with sign `b` pushed through later pending ops, XOR-ing `b` into signs of anticommuting ops. Two impls:
- **Bitset path** (`:272-313`): transposed bitmap `pending_{x,z}_operation_blocks[block*n + q]` (built `:218-260`), 64 ops per word-XOR. Used when `has_expectation`.
- **Linear path** (`:332-357`): `single_x_qubit` fast test → one Z-bit lookup.

`transform_pending_operations_by_frame` (`:798-811`): normally rewrites pending Paulis in place; expectation mode composes into `state.pending_frame` and defers.

### 1.5 Symbolic-expression minimization

Cost model (`:85-105`): primary = distinct 64-bit **words** touched (`symbolic_word_cost`), tiebreak = term count. (Runtime eval = one XOR per touched word.)

Both mechanisms keyed on `record_condition ⊕ outcome = 0`:
- **Substitutions** (`pending_substitutions: map<i32, SymbolicBool>`): when outcome cheaper than the condition and non-self-referential, record `pivot → outcome` (`:380-388`). `substitute_pending_symbols` (`:417-445`) expands to fixpoint with `normalize_xor_conditions` (`:401-415`).
- **Relations** (`pending_relations` + index + words): `reduce_pending_symbolic_bool` (`:447-506`) applies a relation only if it provably lowers cost (`2·overlap > |relation|` or shared-words > new-words). `reduce_by_relation_once` (`:107-148`) single merge-scan, strictly-improving only.

**Final global pass** in program ctor: `reduce_program_symbolic_expressions` (`:1023-1040`) with `SymbolicRelationReducer` (`:150-212`, single-condition relations as `fixed_conditions` map, fixpoint), then `refresh_instruction_plans` (`:1042-1068`) recompiles every plan.

### 1.6 `FactoredInstructionProgram` construction (`:1162-1222`)

Computes `nsymbols/nrecords/ndetectors/nexpvals` by scan, then exogenous sampling plan:
- `build_categorical_sample_plan` (`:1146-1158`): categorical is *rare* if all-false row exists and `1 − P(false) < 0.02` (`kLowProbabilitySampleThreshold`). Rare grouped by identical parameters into `RareCategoricalSampleGroup` (`factored.hpp:115-123`, conditional-on-event row probabilities); rest scalar.
- Bernoullis split at 0.02 into `sampled_bernoulli_{conditions,probabilities}` (per-shot) vs `sampled_low_probability_bernoulli_groups` (geometric-gap, grouped by exact double equality `:1110-1118`).
- `active_component_plan = build_active_component_plan(*this)`; `use_active_components = plan->selected`.

## 2. Factored state — `factored/factored_state.cpp` (411 lines)

~45 `left_*` forwards to CliffordFrame (`:102-276`). Notable:
- `apply_pauli` (`:290-312`): SymbolicBool-conditioned Pauli decomposed into one `add_pauli` per condition id; `constant` true → `fresh_bernoulli_condition(1.0)` (`:307`).
- `PendingFactoredState(const FrameFactoredState&)` (`:389-408`): scans `exp_val` → `has_expectation`, bumps `next_record`.
- `operator==` for ops/instructions (`:9-82`); `ApplyPrecomputedActivePauliRotation` compares only `pauli, kernel_angle, sign` (kernels derived).

### Active-state representation (`sampler/active.hpp`, `active_state.cpp`)

`ActiveState` (`active.hpp:12-22`): `Vec<Complex<f64>>` length 2^k — **reference impl only**; real samplers use split-complex SoA `Vec<f64> re` + `Vec<f64> im`.

`ActivePauliAction` (`active.hpp:24-34`, ctor `active_state.cpp:36-48`): `xmask`, `zmask` (single u64 — **hard limit `k < 63`**, `:38`), `even_phase = i^phase_exponent`, `odd_phase = −even_phase`, `xz_overlap_odd`.

`PrecomputedActivePauliRotationKernel` (`active.hpp:38-53`, ctor `active_state.cpp:50-67`) — **O(1) memory, ≤128 bytes** (static_assert `active.hpp:72`). Coefficients derived per basis index:
- `is_diagonal = (xmask == 0)`; `uniform_imag_pairs = (zmask == 0)`; `real_pair_flip = can_rotate_real_pair_flip` (`active_internal.hpp:70-75`: `zmask≠0 && xz_overlap_odd && even_phase` purely imaginary)
- `pair_bit = highest_set_bit64(xmask)`, `pair_count = 2^k / 2`
- `minus_even_coefficient = −i·sin(φ)·even_phase`; per-basis coefficient `compact_rotation_coefficient` (`active_internal.hpp:77-83`): sign-flip on `sign XOR parity(basis & zmask)`.

`PrecomputedActivePauliMeasurementKernel` (`active.hpp:57-70`, ctor `active_state.cpp:69-119`): pivot = highest set bit of xmask, else zmask. Diagonal: validates `even_phase` real ±1, stores `diagonal_phase_bit`/`z_without_pivot`; nondiagonal: `nondiagonal_coefficient1_even = conj(even_phase)/√2`. Source-index arithmetic via `insert_zero_bit` (`active_internal.hpp:43-48, 85-114`).

## 3. Pending optimizer — `factored/pending_optimizer.cpp` (253 lines)

Runs **before** planning (hard error otherwise `:188-190`). Per segment:

**(a) `fuse_commuting_rotations` (`:74-126`)** — O(n²) with early break. For rotation `i`, walk forward while `rotation_can_cross` (`:32-51`: commutes with rotations/measurements; classical records always cross; `exp_val` measurement = hard barrier). Later rotation with same canonical body + same condition set → fuse: `angle = later + (signs agree ? + : −)·earlier` (`:53-72`). Fused angle exactly `0.0` → both deleted. Canonicalization (`:18-26`): strip ± into `sign` via `measurement_phase_sign`, normalize body phase to y_count.

**(b) `move_measurements_earlier` (`:128-170`)** — move measurement left over commuting rotations. Unconditional sliding only if fusion already removed something this segment (`allow_all_commuting` `:172-181`); else stops at first rotation with same Pauli body. Measurements never cross measurements or classical records.

**Segmentation + prefix remap (`:213-249`)**: `preserved_prefixes` (DETECTOR anchors) split into independently-optimized segments; `stats.prefix_remap` original → new prefix (`stim_sampling.cpp:117-129` consumes). Non-preserved entries = −1.

**Expectation escape hatch (`:203-212`)**: any `exp_val` → optimizer no-op with identity remap.

## 4. Single-shot sampler — `sampler/single_shot_sampler.cpp` (1551 lines)

### 4.1 Runtime state

`FactoredExecutorState` (`single_shot.hpp:28-50`): split-complex `active_re/im` + scratch, bitsets `value_words`, `assigned_words`, `measurement_words`, `detector_words`, `Vec<f64> exp_values`, `u64 rng_state`. Two-array symbol model: `assigned_words` = has value, `value_words` = value; conflicting double-assign = error (`:50-64`).

### 4.2 Symbolic evaluation

Four variants (`:70-133`); **executor calls `eval_symbolic_bool_unchecked` (`:126-133`)** — skips assignment check (planner guarantees assignment-before-use), short-circuits to `plan.constant` when `word_indices` empty. Semantics: XOR `value_words[w] & mask` per plan word, `is_odd_popcount` of accumulated parity, XOR constant.

### 4.3 Per-instruction execution (`:918-979`)

- **Rotation**: eval sign → `rotate_contiguous_active`.
- **Promote**: `promote_first_dormant_rotation(runtime, sign ? −φ : φ)` (`:436-464`) — grow to 2·dim, `promote_contiguous_active`, `k += 1, ndormant −= 1`.
- **RecordMeasurement**: eval outcome; `exp_val` → write ∓1.0, else record bit + assign `record_condition`.
- **RecordDetector**: `detector_outcome_from_runtime` (`:149-159`) prefers measurement-record XOR when `records` non-empty, else symbolic plan.
- **MeasurePrecomputedActivePauli** (`:942-968`): `prob_true` (diagonal or nondiagonal), clamp [0,1]; `exp_val` → write `(±1)·(1−2p)` **without collapsing**; else `branch = sample_bernoulli(rng, prob_true)`, **assign branch symbol BEFORE evaluating outcome plan** (outcome references it), project with `invnorm = 1/√p`, `k −= 1, ndormant += 1`, write record.
- **IntroduceDormantMeasurementBranch** (`:970-979`): `sample_bernoulli(rng, 0.5)`, assign, eval outcome, write record.

### 4.4 Dense kernels — `sampler/contiguous_active.cpp` (169 lines)

Shared by single-shot AND prepared batch sampler.

| Function | Algorithm |
|---|---|
| `rotate_contiguous_active:10-77` | diagonal → per-basis multiply by `c + coeff(basis)`. Uniform-imag → inline kernel if `pair_count < 16384` (`kSimdPairRotationThreshold`) else `simd::dispatch_table()`. Else scalar general-pair loop |
| `promote_contiguous_active:79-94` | `re[dim+i]=−q·im[i]; im[dim+i]=q·re[i]`; low half ×c |
| `diagonal_probability_contiguous:96-109` | sum abs² over `compact_diagonal_measurement_source`, clamped |
| `nondiagonal_probability_contiguous:111-127` | always `simd::dispatch_table().measure_nondiagonal_probability_soa` |
| `project_diagonal_contiguous:129-142` | gather-compact into low half × invnorm |
| `project_nondiagonal_contiguous:144-167` | SIMD into scratch, `copy_n` back |

### 4.5 Exogenous symbol sampling (`:253-391`)

`sample_exogenous_symbols` fixed order — **RNG-consumption-critical**: scalar categoricals → rare categorical groups → per-shot bernoullis → low-probability bernoulli groups. Rare/low-prob groups: geometric-gap skipping (`:316-373`), O(#events).

### 4.6 Active components (`:564-916`)

`use_active_components` → `Vec<SingleShotActiveComponent>` (`single_shot.hpp:19-26`), each independent 2^k_i vector. `ScopedSingleShotComponent` (`:720-754`) RAII-swaps a component to *be* the runtime's active vector so same kernels apply. `merge_single_shot_components` (`:650-718`): tensor product written backwards (descending source_basis) for in-place expansion: `target[(s << target.k) + t] = target[t] * source[s]`. Component promotion (`:773-799`) writes `[cos φ, ∓i sin φ]` directly.

### 4.7 Postselection (`:1341-1440`)

`execute_postselected_in_place`: `RecordDetector` returns `!outcome`, does **not** write detector record; any fired detector aborts shot immediately.

### 4.8 Chunked driver (`:1460-1507`)

```
prepare packed exogenous + expression plan (once)
runtime seeded (seed ^ 0x5eed1234)
exogenous_rng_state = seed
per chunk of 2048:
    resample_prepared_exogenous_packed_in_place(..., exogenous_rng_state)
    exogenous_rng_state = samples.next_rng_state   // chained, not re-derived
    evaluate_presampled_expression_block(...)
    per shot: reset_executor(); execute_in_place(..., shot)
```

`reset_executor` (`:1201-1245`) has `clear_detector_records` flag — postselecting path passes false.

## 5. Batch sampler

### 5.1 Two layouts, one struct

`BatchFactoredExecutorState` (`batch_sampler.hpp:55-95`), selected by `dense_shot_major_active` (`batch_internal.hpp:96-101`):
- **shot-major** (true): `active_re[shot * active_stride + basis]`, `active_stride = 2^max_k`. Contiguous per shot → reuses single-shot kernels.
- **basis-major/interleaved** (default false): `active_re[basis * active_pitch + shot]`, `active_pitch = padded_batch_active_pitch(batches)` (multiple of 4, `kBatchActiveLaneAlignment`).

**Prepared batch sampler forces shot-major** (`prepared_sampler.cpp:543`) + `store_detector_records = false` (`:542`). `batch_interleaved` family NOT on production path. **Port shot-major first.**

Bit-plane storage column-major over shots: `value_words[(condition−1) * batch_words + word]` (`batch_internal.hpp:119-129`).

### 5.2 Batch symbols — `batch_symbols.cpp` (685 lines)

`eval_symbolic_bool_batch` (`:127-233`): fill with constant, XOR one 64-shot word row per condition. Specialized on `nwords ∈ {1,2,n}`, `batch_words == 1`. `kBatchScalarSymbolicEvalThreshold = 32` selects per-condition vs bulk assignment checks. `xor_symbolic_bool_batch_into` (`:260-332`) accumulates into existing buffer (residual half of presampled expression). `write_direct_branch_measurement_record` (`batch_internal.hpp:200-214`): outcome plan exactly the branch condition (± negation) → write `eval_scratch` directly.

### 5.3 Batch active kernels — `batch_active.cpp` (651 lines)

`fill_shot_coefficient_scalars` (`:6-27`): per-shot ±1/±q scalar array from sign bit vector, O(active_pitch). `batch_sign_mode` (`:29-43`): AllMinus/AllPlus/Mixed. `rotate_pauli_batch` (`:268-322`) dispatch: shot-major → per-shot `rotate_contiguous_active`; pitch 1 → single contiguous; diagonal → per-basis loop; uniform_imag → `rotate_uniform_imag_pairs_batch` (`:178-227`; `kXmaskRotationPairThreshold = 64`); else `rotate_compact_pairs_batch` (`:229-266`). Measurement (`:381-600`): `sample_batch_measurement_branches_from_true` (`:381-416`) one Bernoulli per shot **in shot order within each 64-bit word**, `invnorms[shot] = 1/√p`.

### 5.4 Batch runtime — `batch_runtime.cpp` (2453 lines)

**Shot-major rotation runs (`:1493-1588`)** — key optimization. Run of up to `kShotMajorRotationRunLimit = 32` consecutive rotations: evaluate all sign bit-vectors up front into `rotation_run_sign_words`, then loop shot-outer/rotation-inner — each shot's vector touched once across ≤32 rotations. Driven from `execute_batch_in_place:2305-2328`.

**Detector postselection (`:1957-2083`)** — lazy dead-shot scheme:
1. `mark_dead_from_*` (`:1296-1379`) OR fired detectors into `scratch.dead_bits`, track `dead_count`.
2. `should_compact_dead_before_instruction` (`:789-812`): pure-over-dead instructions (rotations, records `:733-759`) may run on dead lanes; measurements/branches force compaction. Shot-major dense: pure instructions never trigger compaction (`:803-805`). Threshold `dead_count * denominator >= active_shots`; denominator = `kExpensivePureCompactionDenominator = 64` for expensive-pure.
3. Compaction (`:845-1253`): `compress_bits` = **BMI2 `_pext_u64`** with portable fallback (`:814-843`, `__builtin_cpu_supports("bmi2")`). Only live columns compacted, via last-use tables (`condition_last_uses:621-646`, `measurement_record_last_uses:648-677` — logical_records used at very end, `block_expression_last_use_by_index`).
4. `compact_active_columns` (`:1102-1183`) skips identity prefix (`first_moved`).

**Expression workspace (`:1929-2005`)** lazy: pre-compaction reads immutable `PresampledExpressionBlock` via bit-shifted slice (`expression_slice_word:1405-1423`); first compaction materializes mutable copy.

`default_batch_count(max_k)` (`:2088-2094`) = `min(2048, max(1, 32768 / 2^max_k))` (`kDefaultBatchShots`, `kDefaultBatchActiveAmplitudes`).

### 5.5 Component plan — `component_plan.cpp` (471 lines)

Plan-time tensor-factor connectivity simulation. `PlanningComponent { active, coordinates }`, `active_order`, `coordinate_components`. Per quantum instruction: `touched_components` (`:91-119`), merge target = largest (`select_merge_target:121-140`), record merge sources, `remap_action` (`:58-89`) into component-local bits.

Cost: dense = 2^k per rotation, 2·2^k per promote/measure; component = Σ 2^merged_k + 2^component_k + 32 (`kComponentDispatchWork`).

Selection gates (`should_select_component_plan:216-236`), all must hold: `max_k >= 8` && ≥4 quantum instructions; `dense_work >= 1.8 × component_work` (`kRequiredWorkRatio`); `dense − component >= 8192` (`kMinimumSavedWork`); `component_allocated_dimension <= 1.25 × dense_peak_dimension` (`kMaximumAllocationRatio`). Early bail `:248`: `max_k < 8 || nexpvals != 0`. exp_val steps → `ActiveComponentStepKind::None` (`:372-377`).

### 5.6 Exogenous presampling — `exogenous_presample.cpp` (541 lines)

`PresampledExogenous` shot-major; `PackedPresampledExogenous` **symbol-major** (`value_words[(condition−1) * shot_words + shot_word]`) — prepared samplers use packed (condition = contiguous bit-plane).

`generate_packed_biased_bits` (`:88-158`):
- `p >= 1` → fill live mask; `p == 0.5` → one `next_random_u64` per word
- `p > 0.5` → sample 1−p, invert
- `p < 0.02` → geometric-gap sparse OR (`or_low_probability_bits_packed:63-86`)
- else → **bit-sliced fair-coin decomposition**: truncate p to 8 binary digits; for `bit = 6..0`: `shoot` = fair word; `result |= shoot & alive; alive &= ~shoot`. Residual mass added by geometric pass at `p_leftover / (1 − p_truncated)`.

`exogenous_assigned_words` (`:170-185`): which conditions are exogenous — split key for expression plan.

### 5.7 Presampled expression plan — `presampled_expression.cpp` (287 lines)

`split_presampled_expression` (`:33-54`): partition plan conditions into exogenous (bulk) + residual (runtime-dependent). Exogenous partials **interned** by `(constant, exogenous_conditions)` into `block_expressions` (`:94-107`). `prepare_block_expression_parent_deltas` (`:127-157`): greedy parent minimizing `|symmetric_difference| + (constant differs)`; `evaluate_presampled_expression_block` (`:221-285`) = `copy(parent) XOR delta_conditions`. Runtime eval = one bit lookup + optional residual XOR (`SingleShotExpressionEvaluator::eval`, `single_shot_sampler.cpp:173-193`; `BatchExpressionEvaluator::eval`, `batch_runtime.cpp:1433-1490`).

## 6. SIMD — `simd/`

### 6.1 Dispatch

`simd.hpp:11-64`: 9-entry function-pointer `KernelTable`. `simd_dispatch.cpp:21-40`: function-local static, `__builtin_cpu_supports`; AVX512 = `avx512f && avx512dq`; AVX2 = `avx2 && fma`; else scalar. `dispatch_name()` → `"avx512"|"avx2"|"scalar"`.

Build: library gets `-march=native` (SYMFT_CPP_NATIVE); `simd_avx2.cpp`/`simd_avx512.cpp` separate TUs with explicit flags. Two SIMD paths: `-march=native` inline kernels in `active_kernels.hpp` (pair_count < 16384) and runtime-dispatched table (above).

### 6.2 Kernel table (all split-complex SoA)

| Kernel | Semantics |
|---|---|
| `mul_assign` | interleaved Complex* — legacy |
| `norm_sum` | interleaved, gather by index list |
| `mul_assign_soa` | `α *= c + coeff[i]` |
| `norm_sum_soa` | gather by index list |
| `measure_nondiagonal_probability_soa` | `source0 = insert_zero_bit(idx, pivot)`, `source1 = source0 ^ xmask`, sign from `parity(source0 & zmask) XOR branch` |
| `project_nondiagonal_soa` | same, writes out × invnorm |
| `rotate_uniform_imag_pairs_soa` | shared q all pairs |
| `rotate_real_pair_flip_soa` | per-pair ±q from `phase_signs` |
| `rotate_general_pairs_soa` | per-pair complex left/right coeff arrays |

### 6.3 Specialization pattern (AVX2 example `simd_avx2.cpp:460-588`)

1. `selector == 1 && xmask == 1` → in-register lane swap (`_mm256_permute_pd 0b0101`).
2. `pair_bit == 1 && xmask ∈ {2,3}` → `_mm256_permute4x64_pd` 0x4e/0x1b.
3. `pair_bit >= 2 && dim >= 4` → sub-lane xmask part as **compile-time template param LaneMask (0–3)**, constant-folded permute.
4. Fallback: `lower_mask == 0` → contiguous partner blocks, two loads. `xor_contiguous_run(lower_mask, selector) >= 4` → contiguous sub-segments. Else gather/scatter (scatter emulated on AVX2 via aligned stack buffer).
5. Scalar tail.

Nondiagonal coefficient sign computed vectorially: `nondiagonal_coefficient_sign_mask4` (`:254-276`) 6-step XOR-fold parity of `source0 & zmask` → sign bit XOR into coefficient. Branchless.

**AVX512 diffs**: 8-wide, sub-lane mask 3 bits (`permute_lanes_xor3:241-268`, 8-case runtime switch `_mm512_permutexvar_pd`); chunked partner path needs `pair_bit >= 3`, `dim >= 8`; real `_mm512_i64scatter_pd`; re/im deinterleave via strided gathers (`complex_indices8`); falls back to scalar_table for `pivot < 2` (AVX2 same).

### 6.4 `batch_interleaved` (1051 lines)

Separate 17-entry table for basis-major `active[basis * leading_shots + shot]`; `_const`/per-shot/`_xmask` flavors. Table populated **entirely with scalar functions**; AVX2 = inline pitch2 special cases only. No runtime dispatch. **Not on production path — deprioritize.**

## 7. Public API — `sampler/prepared_sampler.{hpp,cpp}`

### 7.1 Structs (`prepared_sampler.hpp:15-59`)

```rust
struct CircuitSamplingOptions {
    observable: i32,                       // 0
    postselect_detectors: bool,            // false
    sample_chunk_shots: i32,               // 0 = auto
    batch_size: i32,                       // 0 = auto
    batch_mask_threshold_denominator: i32, // 2
    threads: i32,                          // 1
}
struct CircuitSamplingCounts { shots, discarded, accepted, logical_errors: u64 }
struct CircuitSamplingTiming { parse_s, plan_s, presample_s, execute_s, accumulate_s, sample_s: f64 }
struct CircuitSamplingRunResult { counts, timing, active_threads: i32 }
```

Auto-sizing (`:215-229`): `sample_chunk_shots = max(2048, min_auto)`; min_auto = 1 single-shot, `batch_size` for batch (`:519-521`); `batch_size = default_batch_count(program.max_k)`. `accumulate_s` never written.

### 7.2 `sample()` semantics

`sample(shots)` / `sample(shots, stream_id)`; no-arg stream = `next_stream_id_++` from 0. Worker storage reused — **calls on one instance must not overlap** (`:87`).

Threading (`:259-292`): `std::jthread` per worker, **static strided chunk assignment** (`chunk_index = worker_id; chunk_index += active_threads`). `active_threads = min(requested, nchunks)`. Per-worker exception_ptr, rethrown after join. Worker contexts pre-allocated (`:374-383`, `:537-557`) — no shared mutable state, no locks. Results merge additively (`:294-305`).

Counting: `accumulate_single_counts` (`:109-122`) — discard if any detector bit set, else XOR logical observable's record parities. Batch bitwise with popcounts (`accumulate_block_counts:124-164`) using `detector_any_words` (OR of detector outcomes; why `store_detector_records = false` is safe).

## 8. RNG — exact reproduction requirements

### 8.1 Generator: SplitMix64 (`sampler/random.hpp:13-19`)

```rust
fn next_random_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
fn rand_float(state: &mut u64) -> f64 { ((next_random_u64(state) >> 11) as f64) * 2f64.powi(-53) }
```

### 8.2 Consumption rules (must match exactly)

- `sample_bernoulli` (`:25-34`): `p <= 0` → false, `p >= 1` → true, **no RNG consumed**. Only strict interior draws. Wrong = desync everything after.
- `sample_categorical_row` (`:36-46`): one draw; cumulative walk `r <= cumulative`; falls through to len−1.
- `sample_geometric_gap` (`:48-58`): one draw; `u = max(rand_float, f64::MIN_POSITIVE)`; `floor(ln(u) / ln_1p(-p))`; i32::MAX if non-finite/too large. **`(-p).ln_1p()`, not `(1.0-p).ln()`.**
- `fill_batch_random_half_bits` (`batch_internal.hpp:162-177`): one raw u64 per batch word for fair coins — **not** per-shot Bernoulli.

### 8.3 Seed derivation (`prepared_sampler.cpp:26-30`)

```rust
fn block_seed(base: u64, stream_id: u64, block_index: u64) -> u64 {
    base ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(block_index + 1))
         ^ (0xbf58_476d_1ce4_e5b9u64.wrapping_mul(stream_id + 1))
}
```

| Sampler | Exogenous seed | Branch rng_state |
|---|---|---|
| Single-shot (`:421-434`) | `block_seed(0x7eed0000, stream_id, chunk_index)` | `block_seed(0x5eed1234, stream_id, chunk_index)`, once per chunk, advances across shots |
| Batch (`:629-653`) | `block_seed(0x7eed0000, stream_id, chunk_index)` | `block_seed(0x5eed1234, stream_id, block_index)`, `block_index = chunk_index * blocks_per_chunk + local` |

Consequences:
- Results **independent of thread count** — chunk seeds depend only on (stream_id, chunk_index). Rules out work-stealing that reassigns chunk indices.
- Single-shot and batch produce **different** streams for same (seed, stream_id); not expected to agree shot-for-shot.
- Standalone helpers different scheme: `sample_measurements(program, shots, seed, chunk)` (`single_shot_sampler.cpp:1460-1500`) seeds executor `seed ^ 0x5eed1234` (`kSingleShotBranchSeedXor`) once, **chains** exogenous via `packed_samples.next_rng_state` from `seed`. `sample_measurements_batch` (`batch_runtime.cpp:2371-2405`) carries one rng_state through.

## 9. Port sequencing notes

1. Instruction stream small — 6 opcodes, ~300 lines dispatch. Hard parts: planner tableau-frame construction (§1.4), symbolic cost model (§1.5) — off-by-one in condition indexing silently produces wrong-but-plausible statistics.
2. C++ `std::variant`/`std::visit`/`if constexpr (requires {...})` structural dispatch → Rust explicit `match` (clearer with 6 variants).
3. **Prioritize shot-major.** `batch_interleaved` (1268 lines) + basis-major branches deferrable/droppable.
4. SIMD: KernelTable → `&'static KernelTable` of fns via `is_x86_feature_detected!`. LaneMask templating → const generics. `_pext_u64` behind bmi2 check.
5. **Differential-testing hook**: `sample_measurements(program, shots, seed)` (`single_shot.hpp:70-71`) → packed record bits; seed-deterministic, single-threaded, chunk-independent as long as `sample_chunk_shots` matches.
6. **`k < 63` hard invariant** (`active_internal.hpp:36-41`) — active masks single u64.
