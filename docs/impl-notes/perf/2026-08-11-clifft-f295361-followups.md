# Clifft performance follow-ups after `f295361`

## Scope

Audited Clifft `f295361..550cfbb` against ticit `a410f4d` plus the retained
working-tree change. Measurements used the native release build on the local
Intel Core i5-14600KF, one thread pinned to CPU 10. Hardware counters were not
available (`perf_event_paranoid=2`), so profiles use `cpu-clock:u`.

## Commit audit

| Clifft commit | Technique | Ticit action |
|---|---|---|
| `b80225c` / #282 | Specialized scalar symbolic Pauli rotations | Already present in `active.rs`: pair enumeration, precomputed phases, and explicit butterflies. |
| `ddaef3d` / #283 | Fuse constant symbolic rotations | Already covered by the retained register-run and fixed five-step diagonal fusion. The earlier general X-basis fusion regressed `msc_d5`, so no general fusion framework was added. |
| `5c058fd` / #284 | AVX-512 fused symbolic rotations | Already covered by ticit's AVX-512 register-resident rotation runs. |
| `48b84ff` | Forced-ISA tests | Useful validation practice, but not a runtime optimization. |
| `cbb6a1e` | Executor refactor | No transferable speedup by itself. |
| `b2a501d` / #288 | AVX-512 direct symbolic rotations | Ticit already dispatches high-pivot AVX2/AVX-512 direct kernels and keeps small pivots scalar. |
| `e2be936` / #290 | Lane-paired low-pivot rotations | Ticit already has in-lane rotation kernels. The analogous low-pivot measurement experiment below regressed and was removed. |
| `4750623` / #289 | Forward coordinate and symbolic-Pauli frames | Ticit already has a lazy cumulative frame and indexed sign propagation for expectation probes. Generalizing it did not improve ordinary planning. The retained lesson was narrower: construct the three planner tableaus directly in their final row order. |
| `0e93617` / #291 | Preserve uniform fixed-k probabilities | No CPU batch fixed-k contract exists in ticit, and Clifft reports no ordinary-sampling win. Deferred until that API exists. |
| `550cfbb` / #292 | AVX-512 lane-paired active measurements | The AVX2 width-three analogue regressed end to end. Revisit only as an AVX-512 width-at-least-four kernel on `riling` if a profile still shows the gap. |

## Baseline profile

CCZ `d09_p0`, one attempted shot, spent 10.33 seconds of sampled CPU time in
compilation. The largest self-costs were:

| Symbol / work | Self CPU |
|---|---:|
| `CliffordFrame::ensure_support_words` row-vector construction | 12.74% |
| `preimage` | 10.37% |
| `Vec<PauliString>::extend_with` from identity tableau construction | 7.70% |
| `cfree` | 5.16% |
| `coordinates_in_frame` | 4.72% |

The sampling profile on CCZ `d05_p0` had no 20% hotspot. Nondiagonal
probability plus projection accounted for about 12.9%, while symbolic sign
evaluation and XOR work were spread across several functions.

## Retained: build planner tableaus from final rows

The old planner allocated a full identity `CliffordFrame`, then cloned over all
`2n` rows for dormant promotion, dormant measurement, and active measurement.
The retained implementation builds final X and Z row vectors once and moves
them into the frame. This also removes the now-unused copy-and-invalidate helper.

Pinned compile comparisons:

| Circuit | Before (s) | Direct rows (s) | Change |
|---|---:|---:|---:|
| CCZ `d05_p0` | 0.739 | 0.698 | **-5.5%** |
| CCZ `d07_p0` | 2.775 | 2.553 | **-8.0%** |
| CCZ `d09_p0` | 9.976 | 9.687 | **-2.9%** |
| CCZ `d09_p1e-3` (`cpu-clock:u`) | 38.079 | 37.392 | **-1.8%** |

On the post-change `d09_p0` profile, sampled CPU time fell to 9.90 seconds,
`Vec<PauliString>::extend_with` fell from 7.70% to 4.11%, and `cfree` fell from
5.16% to 3.59%. Peak RSS was neutral within noise (317,988 versus 319,872 KiB).
Sampling was also neutral: a three-repeat `d05_p0` check measured 290,051
before and 293,612 shots/s after.

The final public-Python validation compiled and sampled all 11 SOFT circuits
and all eight CCZ circuits. With empty CCZ reference vectors and one sampler
call, normalized preparation times were:

| CCZ circuit | Preparation (s) |
|---|---:|
| `d05_p0` | 0.692 |
| `d05_p1e-3` | 2.48 |
| `d07_p0` | 2.55 |
| `d07_p1e-3` | 10.4 |
| `d09_p0` | 9.69 |
| `d09_p1e-3` | 36.1 |
| `d11_p0` | 34.2 |
| `d11_p1e-3` | 111 |

## Rejected experiments

- Low-pivot AVX2 measurement probability and in-place projection: exhaustive
  small-Pauli differential tests passed, but pinned ABBA throughput fell from
  294,909 to 291,369 shots/s (-1.2%). Removed.
- Lazy cumulative-frame planning for every circuit: CCZ `d07_p0` compile time
  was 2.756 -> 2.766 seconds and `d09_p0` was 9.964 -> 10.064 seconds. Removed.
- Reusing inner support-cache allocations: about 0.4% on `d09_p0`, below run
  noise. Removed.
- Single-symbol `SymbolicBool` XOR fast path: only 0.7% on noisy d07 and 1.7%
  on noisy d09 because ticit's value-returning API still allocates and copies.
  Removed rather than complicating the core operation.

## Next profile-guided targets

1. The remaining compile profile points at packed/contiguous `CliffordFrame`
   row and support storage. This is the closest analogue to Clifft's compact
   planner frame, but it is a real data-layout project and needs isolated
   benchmarks before changing the shared frame.
2. If AVX-512 profiling on `riling` still attributes meaningful time to
   width-at-least-four low-pivot measurements, test Clifft's lane-paired kernel
   with the same minimum-width guard. Do not enable the local AVX2 width-three
   variant.
3. Fixed-k probability propagation remains out of scope until the CPU batch API
   exposes fixed-k sampling.
