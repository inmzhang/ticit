# Native build and instruction layout — 2026-08-08

This pass followed the [Rust Performance Book](https://nnethercote.github.io/perf-book/title-page.html):
measure release-build knobs independently, then inspect oversized hot types.
All throughput measurements used P-core 0, one sampler thread, native SIMD,
detector postselection disabled, and matched ABBA runs.

## Release-build configuration

`RUSTFLAGS=-Ctarget-cpu=native` is a broad win over the portable release
binary: about +9.1% on `msc_d3`, +10.8% on CCZ `d05_p0`, and +5.5% on
`d07_p0` (the latter also used 2.74% fewer core cycles). Competitive local and
server benchmarks should therefore use native builds. It is not a repository
default because that would make ordinary release artifacts host-specific.

The remaining knobs were built in separate target directories on top of the
native baseline:

| Release option | `msc_d3` | CCZ `d05_p0` | Decision |
|---|---:|---:|---|
| `codegen-units=1` | +3.6% | -2.4% | reject |
| `lto=fat` | +0.9% | -0.6% | reject |
| `panic=abort` | +2.2% | -2.7% | reject |
| all three | +3.3% | -1.6% | reject |
| LLVM `fp-contract=fast` | -2.4% | not run | reject |
| PGO, broad training corpus | about +8% | about +36% | reject: not distributable via `cargo install` |

The combined profile was effectively flat on `d07_p0` (+0.2%). No Cargo
profile setting was retained: each narrow `msc_d3` gain traded away d05
throughput. Sampling profiles also showed no allocator symbols among the hot
functions, so changing the global allocator was not justified.

PGO also improved d07 by about 24%, but it is not an accepted build baseline:
crates.io packages installed through `cargo install` cannot include the trained
profile. The generated local profiles and binaries were deleted; retained work
must improve the distributable build.

## Planned-instruction layout

`FactoredInstruction` was 368 bytes. Planned rotation instructions retained a
`PauliString`, an `ActivePauliAction`, and an angle already encoded in their
precomputed kernel; planned measurements likewise retained a redundant
`PauliString`. Removing those values cuts the enum to 256 bytes (-30.4%) and
avoids the two Pauli bitset allocations for every planned rotation and active
measurement. A unit test pins the new size. No boxing, dependency, or unsafe
code was added.

| Circuit | Before | Compact layout | Change |
|---|---:|---:|---:|
| SOFT `msc_d3` | 3.109 Mshot/s | 3.107 Mshot/s | -0.1% |
| CCZ `d05_p0` | 286,009 shots/s | 286,630 shots/s | +0.2% |
| CCZ `d07_p0` | 118,188 shots/s | 123,668 shots/s | **+4.6%** |

A five-repeat `perf stat` confirmation on d07 measured 119,606 versus 124,410
shots/s (+4.0%), 3.2% fewer core cycles, 5.8% fewer cache references, and
29.2% fewer cache misses. Instructions increased 0.6%, while branches were
unchanged. The layout change is retained because it materially improves the
instruction-dense case without a representative regression.

Correctness: 251/251 under `cargo nextest run`, clean formatting and Clippy,
and the current scalar d05 batch output remains byte-identical to the saved C++
oracle. CCZ d09/d11 were skipped only for this fast performance loop.

## Rejected AVX2 bounds-check removal

Direct `_mm256_loadu_pd`/`_mm256_storeu_pd` calls in the uniform-imaginary
rotation kernel raised postselected `msc_d3` throughput from 3.83 to
3.97--3.98 Mshot/s (+3.7--3.9%), but reduced CCZ `d05_p0` from 289--290k to
278--280k shots/s (about -3.7%). Restricting unchecked accesses to the
profiled high-pivot branch removed the msc gain and still trended down on d05.
The entire experiment was removed: the existing safe SIMD slice API is the
broader winner, and the proposed unsafe boundary bought nothing consistently.
Replacing the low-pivot index loops with safe `chunks_exact_mut(4)` iteration
was also rejected after it reduced msc throughput to 3.58--3.68 Mshot/s.
Contracting only the profiled diagonal squared-norm expression emitted the
same scalar FMA as SymFT but was neutral to slower (3.78--3.81 Mshot/s versus
3.81 Mshot/s), so that source change was removed too.
Pre-staging direct rotation-kernel references removed repeated enum checks in
the shot-major runner, but msc/d05 stayed within noise and d07 fell about 1%
in ABBA runs. It too was removed; reduced branch counts did not reduce cycles.

## Rejected AVX2 lane-dispatch hoist

`perf annotate` attributed 5.2% of uniform-rotation samples to the indirect
jump used to select a four-lane XOR permutation. Expanding four loop variants
and selecting once removed that inner-loop jump. Five fixed-workload runs cut
`msc_d3` core instructions by 1.36% and branches by 5.81%, but core cycles by
only 0.30% (16.729 B to 16.679 B). On CCZ `d05_p0`, instructions and branches
were unchanged while core cycles rose 1.64% (25.397 B to 25.812 B). The larger
hot function traded control flow for instruction-cache/layout cost, so the
source experiment was removed.

A compact alternative selected one runtime `vpermps` index vector before the
loop. It cut `msc_d3` branches by 4.64%, but raised core cycles by 0.98% and
instructions by 0.17%; it was removed without widening the benchmark.

## Rejected small diagonal source table

The scalar diagonal probability and projection loops recompute source indices,
so an experiment packed the common small-state mappings into the measurement
kernel. A first eight-byte table grew `FactoredInstruction`; a second encoding
replaced an existing field and preserved the 256-byte instruction size. The
zero-growth form cut `msc_d3` core cycles by 8.05% and instructions by 6.50%,
while `d05_p0` stayed effectively flat. Every compact/helper variant regressed
`d07_p0` by roughly 0.8--2.2%; the final compact form also raised cache misses
about 20%. The table and its extra code were removed because the narrow win did
not survive the representative circuit set. No unsafe code was needed or kept.
