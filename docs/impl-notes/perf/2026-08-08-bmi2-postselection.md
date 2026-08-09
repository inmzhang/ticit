# BMI2 postselection compaction — 2026-08-08

Local P-core 0 profiling of 30 million postselected `msc_d3` shots put 15.37%
of ticit's cycles in the portable bit-column compression loop. SymFT uses
runtime-dispatched BMI2 `_pext_u64` for the identical operation.

## Retained boundary

ticit now dispatches one private `PEXT` wrapper after
`is_x86_feature_detected!("bmi2")`. The target-feature precondition is the
only unsafe invariant: the intrinsic accepts and returns plain `u64` values
and does not access memory. Crate-wide `unsafe_code` remains denied, with a
function-local allowance at the checked call and intrinsic only;
`unsafe_op_in_unsafe_fn` is also denied. An independent scalar reference checks
4,096 pseudorandom input/mask pairs, the portable implementation, the
dispatched implementation, and edge masks.

The native binary contains `pext` instructions. In a second 30-million-shot
profile, bit-column compaction fell from 15.37% to 1.04% of cycles:

| `msc_d3`, postselected | Before | BMI2 | Change |
|---|---:|---:|---:|
| Sampling throughput | 3.256 Mshot/s | 3.790 Mshot/s | **+16.4%** |
| Sample time | 9.214 s | 7.915 s | -14.1% |
| Execute time | 8.374 s | 7.074 s | -15.5% |

Longer ABBA runs ranged from +17% to +21%. `msc_d5` improved about 3% amid
host variance. Non-postselected d05/d07 stayed within -0.7% to +1.2%,
consistent with the new function being unreachable on that path. Under the
same profiler, SymFT measured 4.311 Mshot/s, so the local postselected d3 gap
is now about 12%.

Both profiled implementations produced the same aggregate result over all 30
million shots: 9,400,620 discarded, 20,599,380 accepted, and 24 logical
errors. The full Rust gate is 252/252 with clean formatting and Clippy.

## Rejected follow-up

A checked-once unsafe boundary around diagonal measurement removed loop bounds
checks and improved `msc_d3` only 0.9%, but regressed CCZ d07 by about 9% due
the resulting unrolled code shape. It was removed completely; unsafe code that
did not buy a broad win was not retained.
