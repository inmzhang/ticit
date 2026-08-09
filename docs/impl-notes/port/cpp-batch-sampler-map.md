# Batch Sampler — Rust Port Map (`SOFT/cpp/src/sampler/batch_*`)

(Explorer report. Paths relative to `SOFT_ROOT/cpp/src/`.)

## 1. Public API — `sampler/batch_sampler.hpp`

### 1.1 `BatchActiveComponent` (`batch_sampler.hpp:16-24`)
`k` (live coordinate count), `active`, `stride` (= `2^component_max_k[c]`, shot-major row stride), `re/im` (size `stride * active_pitch`), `scratch_re/im`.

### 1.2 `BatchDetectorPostselectionScratch` (`:26-38`)
Shot-bit-packed u64 vectors (`batch_words` words): `dead_bits`, `keep_bits`, `scratch` (**dead field — never used**), `compact_scratch`, `expression_words` (`[expr * batch_words + word]`), `live_sources` (`[dst] = src`), `condition_last_use_by_index` (size nsymbols+1), `record_last_use_by_index` (size nrecords+1), cache keys `postselection_metadata_program` / `_retained_record_uses` (pointer identity), `dead_count`.

### 1.3-1.4 Result/options
`BatchDetectorPostselectionResult{discarded, accepted}` (accepted = `active_shots` after final compaction). `BatchDetectorPostselectionOptions{mask_dead_shots_min_fraction_denominator = 2, retained_record_uses = nullptr}` — compaction fires when `dead_count * denom >= active_shots`; retained records pinned live to program end.

### 1.5 `BatchFactoredExecutorState` (`:55-95`)

| field | semantics |
|---|---|
| `n, k, ndormant` | `k + ndormant == n` |
| `batches` | **shot capacity**, set once in ctor |
| `active_shots` | **live shots** ≤ batches; shrinks under postselection |
| `active_pitch` | `padded_batch_active_pitch(batches)` — constant for run |
| `active_stride` | shot-major row stride = 2^max_k (dense) or component stride |
| `batch_words` | `ceil(batches/64)` — **capacity** words ≠ live words |
| `store_detector_records` | default **true**; false → only `detector_any_words` maintained |
| `dense_shot_major_active` | default **false** |
| `active_re/im, scratch_re/im` | `2^max_k * active_pitch` doubles both layouts |
| `value_words` | `[(cond-1)*batch_words + word]` |
| `assigned_words` | **symbol-indexed** (`ceil(nsymbols/64)` words, bit `(cond-1)&63`), not shot-indexed |
| `measurement_words` | `[(record-1)*batch_words + word]` |
| `detector_words` / `detector_any_words` | `[(det-1)*batch_words + word]` / OR of all |
| `exp_values` | f64 `[exp_val * batches + shot]` — **stride = batches** |
| `eval_scratch` | `batch_words` words; universal single-expression temp (heavily aliased, §10) |
| `rotation_run_sign_words` | `[run_offset * ceil(active_shots/64) + word]` |
| `shot_coefficient_scalars` / `branch_prob_true` / `branch_invnorms` | f64[active_pitch]; invnorms tail set 1.0 |
| `rng_state` | **one shared splitmix64 state for whole batch**, default 1 |

Ctor (`batch_runtime.cpp:2122-2129`): `batches = arg > 0 ? arg : default_batch_count(max_k)`; `rng_state = seed`; `reset_batch_executor`.

### 1.6 Free functions (defs in `batch_runtime.cpp`)
`default_batch_count` (`:2088`); `reset_batch_executor(rt, prog, shots, clear_symbol_values=true)` (`:2131-2207`, detector_words cleared only if store_detector_records; value_words clear gated); `execute_batch_in_place` ×4 overloads (`:2209-2329`; the ExpressionPlan/Block overload is the **PreparedCircuitBatchSampler fast path** with shot-major rotation-run fusion); `prepare_batch_detector_postselection_scratch` ×2 (`:2331-2351`, last-use tables cached on `(&program, retained_record_uses)`); `execute_batch_postselected_in_place` (`:2353-2369` → `:1957-2083`); `active_batch_backend()` = `simd::dispatch_name()`; `sample_measurements_batch(prog, shots, batches=0, seed=1)` (`:2371-2405`; row = record bitmask `row[(rec-1)>>6] |= 1<<((rec-1)&63)`); `sample_measurements_and_expectations_batch` (`:2407-2451`); `execute_batch_instruction_in_place` **fails if components enabled** (`:2113-2120`).

## 2. Layouts and sizing

### 2.1 Index formulas (`batch_active_offset`, `batch_internal.hpp:96-101` = single source of truth)
- shot-major: `active_re[shot * active_stride + basis]`, `active_stride = 2^max_k`.
- basis-major: `active_re[basis * active_pitch + shot]`.
- Allocation identical: `2^max_k * active_pitch` (`ensure_dense_batch_active_storage`, `batch_runtime.cpp:226-239`).
- Components: `stride = 2^plan.component_max_k[c]`; when enabled dense vectors **freed** (swap with empty `:157-160`).

Bit/word buffers: `word = shot>>6; mask = 1<<(shot&63)`. External tables: `PresampledExogenous.value_words[shot * nwords + symbol_word]` (shot-major); `PackedPresampledExogenous.value_words[(cond-1) * shot_words + shot_word]` (symbol-major); `PresampledExpressionBlock.expression_words[expr * shot_words + shot_word]`.

**Live vs capacity words trap**: `batch_words = ceil(batches/64)` allocation stride; `runtime_batch_word_count(rt) = ceil(active_shots/64)` loop bound (`batch_internal.hpp:80-82`); `batch_live_word_mask(rt, word) = low_bits_mask(active_shots - word*64)` (`:75-78`).

### 2.2 `padded_batch_active_pitch` (`batch_internal.hpp:57-63`)
`cap <= 2 → max(cap,1)`; else round `max(cap,4)` up to multiple of 4 (`kBatchActiveLaneAlignment = 4`). Called once per reset with `batches` (not active_shots). Pitch 1 degenerates to single-shot contiguous kernels; pitch 2 blocks `rotate_uniform_imag_pairs` kernel choice (AVX2 pitch-2 xmask kernel used instead).

### 2.3 `default_batch_count` (`batch_runtime.cpp:2088-2094`)
`clamp(32768 >> max_k, 1, 2048)` — target ~32768 doubles working set. `kDefaultBatchShots = 2048`, `kDefaultBatchActiveAmplitudes = 32768`.

### 2.4 Frontend sizing (`prepared_sampler.cpp`)
`batch_size_or_default` (`:222-229`); `sample_chunk_or_default = requested > 0 ? requested : max(2048, min_auto)` (`:215-220`), batch min_auto = batch_size (`:519-521`). Chunks = presample unit; blocks (batch_size) = execution unit; `blocks_per_chunk = ceil(chunk/batch_size)` (`:608-610`). **Prepared batch sampler sets `store_detector_records = false` and `dense_shot_major_active = true` AFTER construction (`:542-543`)** — ctor's reset ran with old flags.

Thresholds: `kXmaskRotationPairThreshold = 64`, `kBatchScalarSymbolicEvalThreshold = 32`, `kShotMajorRotationRunLimit = 32`, `kExpensivePureCompactionDenominator = 64`, `kSimdPairRotationThreshold = 16384`.

## 3. `batch_active.cpp` kernel inventory

Universal branchless trick: **materialize per-shot sign/branch as f64 ±direction vector** (`fill_shot_coefficient_scalars:6-27`, writes only `[0, active_shots)`, tail stale), multiply into coefficients. `batch_sign_mode` (`:29-43`): AllMinus/AllPlus/Mixed (active_shots == 0 ⇒ AllMinus).

Key functions: `finish_active_measurement_branch` (`:50-57`, `--k; ++ndormant; assign branch symbol`); `copy_projected_active_prefix_from_scratch` (`:59-67`, copies only active_shots lanes — padding stale); `measure_nondiagonal_true_prob_batch` (`:69-94`); `compute_active_measurement_true_prob_batch` (`:96-142`, dispatcher); `project_nondiagonal_batch` (`:144-176`, per-lane coefficient select by `directions[lane] < 0.0`); `rotate_uniform_imag_pairs_batch` (`:178-227`, uniform sign → `_const` kernel; else pairs kernel if `pair_count < 64 && pitch != 2` else xmask kernel); `rotate_compact_pairs_batch` (`:229-266`); `rotate_pauli_batch` (`:268-322`, entry validates `kernel.action.nqubits == k`); `promote_first_dormant_rotation_batch` (`:324-379`, then `++k; --ndormant`); `sample_batch_measurement_branches_from_true` (`:381-416`, §7); `measure_shot_major_active_branch_batch` (`:418-450`, packs branch bits into `eval_scratch`); `measure_diagonal_active_pauli_branch_batch` (`:452-526`, per-lane source-index recomputation `compact_diagonal_measurement_source(kernel, idx, branch_directions[lane] < 0.0)`); `measure_nondiagonal_active_pauli_branch_batch` (`:528-583`); `measure_precomputed_active_pauli_batch` (`:602-620`, `write_direct_branch_measurement_record` shortcut); `measure_precomputed_active_pauli_expectation_batch` (`:622-649`, `exp_values[exp*batches+shot] = sign * (1 - 2*p_true)`).

**Contract**: after branch measurement, `eval_scratch` holds branch bits; `write_direct_branch_measurement_record` (`batch_internal.hpp:200-214`) relies on it (outcome plan == [branch_condition] ⇒ skip symbolic eval, invert if constant).

## 4. Dispatch axes

Two-level dispatch inside each kernel (no top-level switch):
- Level 1 `dense_shot_major_active`: true ⇒ loop shots × shared `*_contiguous` kernels (vectorize within shot); false ⇒ basis-major (vectorize across shots). Set only at `prepared_sampler.cpp:543` — **production always shot-major**.
- Level 2 `active_pitch == 1`: degenerates to contiguous path even in basis-major.
- Components (orthogonal): `configure/reset_batch_components` (`batch_runtime.cpp:119-224`); fails if `instruction_steps.size() != instructions.size()`. `ScopedBatchComponent` (`:390-429`) swaps component vectors into runtime, sets k/stride, runs same kernel, swaps back (Drop-style restore set: k, ndormant, active_stride from saved; component.k/stride from runtime). `merge_batch_components` (`:267-388`) in-place tensor expansion, `source_basis` **descending** — reversing corrupts. `execute_batch_component_measurement` (`:493-521`): component k decremented inside scope, global `--k; ++ndormant` applied again after (`:513-514`) — double bookkeeping essential. Component selection thresholds (`component_plan.cpp:216-235`): `max_k >= 8`, ≥4 quantum instr, `dense >= 1.8 × component`, saved ≥ 8192, alloc ≤ 1.25 × dense peak. Shot-major rotation-run fusion disabled when components on (`:2026-2027`).

## 5. `batch_symbols.cpp` — batched symbolic eval

`eval_symbolic_bool_batch(out, plan, rt)` (`:127-233`): fill constant (live-masked, tail zeroed), then XOR one condition column per condition. Assignment-check strategy (`:136`): per-condition checks if `word_indices.empty() || conditions.size() <= 32`, else bulk `word_masks & ~assigned_words` upfront. Three unrolled shapes: nwords 1 / 2 / general; `nwords==1 && batch_words==1` indexes `value_words[cond-1]` directly. `xor_symbolic_bool_batch_into` (`:260-332`) accumulates into existing out (residual half of presampled expressions); constant folded as `out ^= live_word_mask`.

**No CSE here** — CSE lives in `presampled_expression.cpp`: `split_presampled_expression` (exogenous vs residual conditions), `intern_exogenous_partial` (`:94-107`), `prepare_block_expression_parent_deltas` (`:127-157`, greedy min |symmetric_difference| + constant-differs), evaluation = copy parent + XOR delta (`:235-284`). `block_expression_last_use_by_index` (`:179-183`) for compaction.

Other: `assign_batch_symbol` (`:106-125`, idempotent-with-check, fails "assigned inconsistent concrete batch values"; memcpy fast path when `nwords == batch_words && active_shots % 64 == 0`); `ensure_batch_measurement/detector_storage` (`:334-366`, stride-multiplier growth ⇒ **must recopy every column**); `write_batch_measurement_record` (`:368-392`, writes column + assigns record_condition); `write_batch_detector_record` (`:394-425`, `!store_detector_records` ⇒ only ORs `detector_any_words`).

## 6. Exogenous presample — two implementations, DIFFERENT RNG consumption

### 6.A In-runtime — `sample_exogenous_symbols_batch` (`batch_symbols.cpp:667-683`)
Group order (RNG contract): 1. scalar categoricals; 2. rare categorical groups; 3. bernoulli conditions by index; 4. low-probability bernoulli groups.

1. `sample_categorical_distribution_batch` (`:521-554`): all-assigned ⇒ return 0 draws; some ⇒ fail. Else per shot one `sample_categorical_row` draw (always consumed).
2. `sample_rare_categorical_group_batch` (`:575-612`): any preassigned ⇒ per-set dense fallback. Else geometric skip over `total_draws = active_shots * nsets`; index `d ⇒ (shot = d/nsets, set = d%nsets)` (**set-minor**); per iteration one gap draw (consumed even on terminating iteration); realized event: second draw `sample_categorical_row(event_probabilities)`; `draw += gap; …; ++draw`.
3. `sample_bernoulli_condition_batch` (`:614-634`): assigned/p≤0/p≥1 ⇒ 0 draws; else one draw per shot. **No bit-slicing.**
4. `sample_low_probability_bernoulli_group_batch` (`:636-665`): geometric skip over `shots * nconditions`, `d ⇒ (shot = d/ncond, cond = d%ncond)`; one gap draw per iteration.

### 6.B Offline packed — `resample_prepared_exogenous_packed_in_place` (`exogenous_presample.cpp:473-524`)
Used by PreparedCircuitBatchSampler. Same 4 phases. `value_words` zeroed then XORed.
3'. `generate_packed_biased_bits` (`:88-158`): p≤0 ⇒ 0 draws; p≥1 ⇒ fill, 0 draws; `invert = p > 0.5` (sample 1−p); p == 0.5 exactly ⇒ one draw per shot word; p < 0.02 ⇒ geometric OR (`or_low_probability_bits_packed:63-86`); else **bit-sliced fair-coin**: `raw = floor(p*256)`, `top_bits = raw < 128 ? raw : 127`, per shot word draw `alive` then for bit 6..0 draw `shoot` (**exactly 8 draws per shot word**), `result |= shoot & alive` if bit set, `alive &= ~shoot` always; residual `p_leftover/(1−p_truncated)` geometric pass; invert vs live mask at end.

Shot-major variant `resample_prepared_exogenous_in_place` (`:435-471`): plain geometric (p<0.02) or per-shot bernoulli — **no bit-slicing**. **Three inequivalent Bernoulli generators — port all three separately.**

`samples.next_rng_state` written at end for chaining.

### 6.C `sample_geometric_gap` (`random.hpp:48-58`)
Requires 0<p<1. One draw: `u = max(rand_float, f64::MIN_POSITIVE)`; `gap = floor(ln(u)/ln1p(-p))`; non-finite or ≥ i32::MAX ⇒ i32::MAX. Termination `gap >= total - draw` **consumes the draw**.

### 6.D Loading presamples into runtime
- Shot-major variant ORs (`:427-455`) — requires pre-cleared value_words; O(shots × set bits) via trailing_zeros.
- Packed variant **overwrites** columns (`:496-519`) via `packed_condition_slice_word` (`:457-474`) — the unaligned bit-slice: `out = words[src] >> bit_offset | words[src+1] << (64-offset)`, masked by live. `expression_slice_word` (`batch_runtime.cpp:1405-1423`) identical formula.

## 7. Measurement branch sampling

**RNG stream**: single shared `rng_state` splitmix64; no per-shot streams. Prepared sampler reseeds **per block**: `rng_state = block_seed(0x5eed1234, stream_id, block_index)` (`prepared_sampler.cpp:653`), `block_index = chunk_index * blocks_per_chunk + local` (`:646-647`); exogenous base `0x7eed0000` keyed by chunk (`:633`). `sample_measurements_batch` does NOT reseed per block (continuous).

**Order**: basis-major `sample_batch_measurement_branches_from_true` (`batch_active.cpp:381-416`): all prob_true first, then strictly ascending shot order; per shot `pt = clamp(...)`, `branch = sample_bernoulli(rng, pt)`, `probability = branch ? pt : 1−pt` (fail if ≤ 0), `invnorms[shot] = 1/sqrt`. Branch-bit words `[nwords, batch_words)` zeroed; invnorm tail 1.0. Shot-major path interleaves per shot but identical draw sequence.

**`sample_bernoulli` consumes NO randomness at p ≤ 0 or p ≥ 1** — deterministic measurements free; always-drawing port desynchronizes.

`IntroduceDormantMeasurementBranch` → `fill_batch_random_half_bits` (`batch_internal.hpp:162-177`): exactly `ceil(active_shots/64)` raw draws, one per live word regardless of partial last word.

Postselection: Measure/IntroduceDormant classified impure (`:749-755`) ⇒ **compaction forced before them whenever dead_count ≠ 0** (`:800-802`) ⇒ measurement draws always over compacted live set; dead shots never consume randomness. Load-bearing for reproducibility.

## 8. Postselection

Driver `execute_batch_postselected_with_expressions` (`batch_runtime.cpp:1957-2083`).

### 8.1 Mark, lazily compact
`mark_dead_from_detector_records` (`:1339-1379`, XOR record columns, single-record fast path) / `_detector_bits` / `_constant_detector` OR into `dead_bits`; `dead_count += popcount(fired & ~dead_bits)` (each shot counted once). Detectors handled before dispatch (`:2040-2052`); never write detector columns in postselected mode. Early exit: `dead_count >= active_shots ⇒ active_shots = 0; break`.

### 8.2 Trigger — `should_compact_dead_before_instruction` (`:789-812`)
`active_shots == 0` or `dead_count == 0` ⇒ false; impure instruction ⇒ true; shot-major dense && !components ⇒ false (dead shots skipped by branch: `postselected_shot_is_dead`, `:1536-1538`, in `execute_shot_major_rotation_run:1571-1573`, `rotate_shot_major_postselected:1590-1614`, `promote_..._postselected:1616-1649`); else `denom = max(1, options.denominator)`, raised to 64 for expensive-pure; `dead_count * denom >= active_shots`.

Purity (`:733-759`): pure = rotation, promotion, RecordMeasurement, RecordDetector; impure = MeasurePrecomputedActivePauli, IntroduceDormantMeasurementBranch. Expensive (`:761-787`): rotation, promotion, measure, dormant-branch.

### 8.3 Compaction — `compact_surviving_shots` (`:1185-1253`)
1. survivor_count; == old ⇒ return; == 0 ⇒ active_shots = 0.
2. `collect_live_sources` (`:1072-1100`, ≤64 fast path).
3. `compact_active_columns` (`:1102-1183`): `first_moved` skip; shot-major `copy_n(re + src*stride, dim, re + dst*stride)`, dim = 2^k; basis-major element-wise per row; components per-component.
4. Bit columns via **PEXT**: `compress_bits` (`:835-843`, `_pext_u64` behind cached `__builtin_cpu_supports("bmi2")`, portable fallback `:821-833` walks keep_mask low-to-high — bit-identical); `append_compressed_bits` (`:845-860`) splices across word boundaries (writes out[word+1] — spare word required). Live-column filters: expressions `last_use > idx`; assigned conditions `last_use > idx`; measurements `last_use >= idx` if include_current_use else >; detectors only when `compact_detector_records` (postselected path never passes it).
5. `active_shots = survivor_count`; `dead_bits` cleared, `dead_count = 0`.

**NOT compacted**: `exp_values`, `branch_prob_true`, `branch_invnorms`, `shot_coefficient_scalars`, `detector_any_words`, `detector_words`. **Expectation values invalid under postselection.**

### 8.4 Expression workspace materialization
Evaluator initially reads immutable `PresampledExpressionBlock` with `first_sample_shot` offset (`:1981-1987`). First compaction: `materialize_expression_workspace` (`:1989-2005`) copies block expressions into `scratch.expression_words` (`initialize_expression_workspace:1929-1955`), re-points evaluator (`expression_stride_words = batch_words`, `first_sample_shot = 0`); workspace then compacted alongside. Hard requirement.

### 8.5 Accounting
Final unconditional compaction (`:2073-2081`) with `instruction_index = instructions.size()`, `include_current_record_use = true`, records pinned via `retained_record_uses` (`measurement_record_last_uses:648-677` marks them at final index). Returns `{discarded, active_shots}`. All-dead: `{discarded, 0}`. **No resampling loop** — `sample(shots)` counts attempted; `accepted <= shots`; callers scale up.

## 9. Detector/observable accumulation

Runtime layouts §1.5. With `store_detector_records == false` only `detector_any_words` written.

Frontend (`prepared_sampler.cpp`): non-postselected `accumulate_block_counts` (`:124-164`): `discard_bits = detector_any & live`; `logical_bits[w] ^= measurement column XORs` per observable record-group (`xor_records_into:70-82`); `logical_errors += popcount(logical & ~discard & live)`. Postselected `accumulate_logical_counts_for_survivors` (`:166-188`): `accepted += active_shots`; XOR over compacted columns masked by live. Merge `merge_worker_result` (`:294-305`); workers stripe chunk indices (`:619-621`).

## 10. Hazards for a faithful Rust port (numbered)

**Aliasing**
1. `eval_scratch` aliased ≥4 ways: (a) BatchExpressionEvaluator out; (b) branch-bit output of branch measurements; (c) implicit input to `write_direct_branch_measurement_record`; (d) read back after RecordDetector (`:1871-1873`). Rust: index-based or split-borrow design; preserve ordering contract.
2. `BatchExpressionEvaluator::eval` returns reference into `eval_scratch` while runtime passed mutably next line (`:1656-1657`) — copy bits or detach buffer.
3. `ScopedBatchComponent` exact save/restore set (see §4).

**In-place reuse**
4. `merge_batch_components` descending source_basis — reversing corrupts.
5. `project_nondiagonal_batch` copies only active_shots lanes back — padded lanes stale.
6. `project_nondiagonal_contiguous` requires distinct scratch.
7. Shot-major compaction `copy_n` disjoint only because `dim ≤ stride`.

**Stale tails**
8. `fill_shot_coefficient_scalars` writes only `[0, active_shots)`.
9. `branch_invnorms` tail = 1.0 is a deliberate invariant.
10. `initialize_dense_batch_active` (`:241-265`) shot-major zeroes only first 2^initial_k per row — safe only because promote fully writes upper half. Basis-major leaves rows ≥ dim stale.
11. `reset_batch_executor` fills after resize — except `value_words` gated on `clear_symbol_values` (false when postselecting, `prepared_sampler.cpp:652`).
12. `shot_coefficient_scalars`/`branch_prob_true`/`branch_invnorms` only grow, never cleared.

**RNG order**
13. `sample_bernoulli` zero draws at p ≤ 0 / p ≥ 1.
14. `sample_geometric_gap` always one draw incl. terminating iteration.
15. `sample_categorical_row` uses `r <= cumulative` (inclusive), falls through to last.
16. Three inequivalent Bernoulli batch generators (§6).
17. `rand_float = (next_u64 >> 11) * 2^-53` exact.

**Integer/index**
18. `active_length(k)` fails `k < 0 || k >= 62`.
19. Conditions/records/detectors **1-based**; exp_val **0-based**.
20. `total_draws` in i64, narrowed back to int; geometric saturates i32::MAX.
21. `or_low_probability_bits_packed` narrows gap to int, guards word overflow.
22. `append_compressed_bits` writes out[word+1] — spare word needed.
23. Measurement/detector storage growth changes stride multiplier — recopy all columns.

**Layout/dispatch**
24. `batch_words` (capacity) vs `runtime_batch_word_count` (live) — most error-prone distinction.
25. `active_pitch` from `batches`, never shrinks.
26. `exp_values` stride = batches.
27. `dense_shot_major_active`/`store_detector_records` mutated after construction (`prepared_sampler.cpp:542-543`).
28. `compress_bits` PEXT/portable agree bit-for-bit.
29. `BatchDetectorPostselectionScratch::scratch` dead field — drop.
30. `detector_words` allocated by ctor under old flags, never resized/cleared/compacted in postselected path.
