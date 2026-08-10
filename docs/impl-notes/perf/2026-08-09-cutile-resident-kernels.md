# cuTile resident-kernel checks — 2026-08-09

> Historical tuning record. Current corrected rates and retained launch
> settings are in
> [`2026-08-10-gpu-correctness-and-comparison.md`](2026-08-10-gpu-correctness-and-comparison.md).

RTX 4090 through Slurm on `gpucluster`, ticit commits shown below, SymFT commit
`bd77739`. Rates use the CLI's steady-state `sample_s`, which excludes cuTile
JIT warmup.

This is an optimization diary, not the final cross-GPU comparison. Its
intermediate "current" rates use smaller or matched launch sizes and are
superseded by the full-batch results in
[`2026-08-09-gpu-comparison.md`](2026-08-09-gpu-comparison.md).

| Circuit | Commit / variant | Shots | `sample_s` | shots/s | Result |
|---|---|---:|---:|---:|---|
| `msc_d3` | `60970e2`, dim-16 | 1 M | 0.003661 | 273 M | baseline |
| `msc_d3` | `af669ba`, bounded/uniform parity | 1 M | 0.003308 | 302 M | retained |
| `msc_d3` | `4b0df13`, fixed state slots | 1 M | 0.003242 | 308 M | retained |
| `msc_d3` | `d7a8891`, original detector placement | 1 M | 0.002985 | 335 M | retained |
| `msc_d5` | `60970e2`, dim-1024 | 100 k | 0.060382 | 1.66 M | baseline |
| `msc_d5` | `af669ba`, bounded/uniform parity | 100 k | 0.056569 | 1.77 M | retained |
| `msc_d5` | `4b0df13`, fixed state slots | 100 k | 0.054089 | 1.85 M | retained |
| `msc_d5` | `45da53b`, diagonal X-rotation runs | 100 k | 0.037436 | 2.67 M | retained |
| `msc_d5` | `f2e1e22`, span state-neutral ops | 100 k | 0.035650 | 2.81 M | retained |
| `msc_d5` | `8822174`, defer detector evaluation | 100 k | 0.034225 | 2.92 M | superseded |
| `msc_d5` | `d8e5f72`, stop after first detector | 100 k | 0.032497 | 3.08 M | superseded |
| `msc_d5` | `b463445`, likely detectors first | 100 k | 0.031599 | 3.16 M | superseded |
| `msc_d5` | `d7a8891`, stop rejected shots in place | 100 k | 0.016145 | 6.19 M | retained |
| `msc_d5` | `5137fe7`, 64-shot detector finalizer | 100 k | 0.035008 | 2.86 M | reverted |
| `msc_d5` | `bcf052d`, direct promotion expansion | 100 k | 0.035296 | 2.83 M | reverted |
| `msc_d5` | `4b3783f`, accumulated X-run phases | 100 k | 0.037670 | 2.65 M | reverted |
| `msc_d5` | `aca6f03`, 32-bit basis/parity | 100 k | 0.039215 | 2.55 M | reverted |
| `msc_d5` | dim-1024, 2 shots/tile | 100 k | 140.087694 | 714 | reverted |

Basis/Pauli parity only needs the low four or ten bits in these kernels, and a
rotation with `zmask == 0` has no per-basis parity at all. Removing the wider
folds and the uniform-imaginary arithmetic specialization reduced timed kernel
execution from 1.899 to 1.345 ms for `msc_d3` and 58.004 to 53.110 ms for
`msc_d5`.

Commit `4b0df13` keeps active qubits in fixed physical bit slots. Measurement
zeros the measured slot and a later promotion reuses it, deleting whole-state
pivot compaction and the dynamic active-dimension masks. Exact msc counters
were unchanged. Timed kernels fell again to 1.231 ms for `msc_d3` and 52.249 ms
for `msc_d5`.

Commit `45da53b` surrounds profitable consecutive X-only rotation runs with
Hadamard transforms, making every rotation in the run diagonal and avoiding
state-vector permutations. The `msc_d5` timed kernel fell from 52.249 to
35.555 ms; exact 100,000-shot counters remained 85,296 discarded and 14,704
accepted. Accumulating all T-phase exponents and applying them only at the end
of a run (`4b3783f`) was slightly slower at 35.971 ms kernel time and was
reverted by `35dea00`.

Commit `f2e1e22` keeps the state in the X basis across dormant-branch and
detector instructions, which do not modify amplitudes. It diagonalizes 73 of
75 rotations in five runs. Two 100,000-shot kernel timings were 33.710 and
33.702 ms, with exact counters unchanged.

Commit `8822174` stable-partitions all 107 detector expressions into a tail
loop, so the 1,024-amplitude state is carried through 153 instructions instead
of all 260. Two kernel timings were 32.005 and 32.006 ms; exact counters and
the roughly 10.8 s JIT warmup were unchanged.

Because a medium block contains one shot, commit `d8e5f72` stops the detector
tail as soon as that shot is rejected. The circuit rejects 85.3% of shots; two
kernel timings fell to 30.493 and 30.498 ms with exact counters unchanged.

Commit `b463445` orders the side-effect-free detector tail by constant term and
affine support size, putting likely rejections first without runtime sampling.
Two kernel timings fell again to 29.693 and 29.700 ms; exact counters and JIT
time were unchanged.

Commit `d7a8891` instead leaves detectors at their original circuit positions
and stops the entire one-shot state loop on rejection. This is valid for the
aggregate API because a rejected shot cannot contribute a logical error, so
its later state is unobservable. Three 100,000-shot runs produced 14.271,
14.274, and 14.285 ms kernel times and 16.038, 16.145, and 16.281 ms total
sampling times, with exact counters unchanged. The median is 6.19 Mshot/s,
1.26x SymFT's measured 4.91 Mshot/s. cuTile JIT warmup increased from roughly
10.8 to 17.1 seconds.

Moving the detector tail to a separate 64-shot kernel (`5137fe7`) reduced the
count-buffer copy but required four branch-word writes per shot. The combined
kernel time regressed to 33.275 ms, so `af8dbca` restored the in-kernel tail.

The retained `msc_d3` path is about 4.6x SymFT's measured 73.2 Mshot/s. The
current `msc_d5` path reaches 6.19 Mshot/s, about 26% above SymFT's measured
4.91 Mshot/s.

Nsight Systems reports a `(128, 1, 1)` block for `sample1024`; `cuobjdump`
reports 131 registers/thread and 4 KiB shared memory. Nsight Compute hardware
counters are disabled on the cluster (`ERR_NVGPUCTRPERM`). Narrowing the
ten-bit basis and parity tiles from `u64` to `u32` reduced allocation to exactly
128 registers/thread, but two kernel timings regressed consistently from
35.555 ms to 37.342 and 37.343 ms. Commit `6ce0b3c` reverted the experiment;
raising theoretical occupancy is not enough to offset its extra conversions
and instruction mix.

Directly constructing the low/high state halves during promotion (`bcf052d`)
reduced the 100,000-shot kernel to 33.089 ms, but expanded cuTile JIT warmup
from about 10.8 to 40.8 s for only a 1.8% kernel win. Exact counters were
unchanged. Commit `b4c2a22` reverted the poor compile-time tradeoff.

The two-shot experiment (`5767651`, reverted by `8eaa1e9`) made the leading
state dimension generic to amortize scalar metadata loads. It instead expanded
the generated program to about 10.6 MB of PTX. `ptxas` reached roughly 13 GB
RSS, JIT warmup took 701.15 s, and the timed kernel took 140.08 s. Its
1,000-shot host/device exogenous check completed successfully, and the 100,000
shot counters (`85,296` discarded, `14,704` accepted) matched the one-shot
run, so this was a performance failure rather than a correctness failure.

Do not widen the resident dim-1024 tile across shots. Optimize the one-shot
instruction path or use a different state decomposition instead.

## Final resident results

Later commits added a 4,096-amplitude resident kernel for `k=11..12`, device
count reduction, occupancy tuning, nibble parity, and reuse of the true
measurement projection. On `coherent_surface_d5_r1` (`k=12`), two retained
100,000-shot runs took 73.382 and 73.348 ms total (72.722 and 72.663 ms in the
sampling kernel), with the same 7,094 discarded and 92,906 accepted shots.
The median rate is 1.363 Mshot/s, 3.4% above SymFT's matched 1.318 Mshot/s.

The retained resident rates on the RTX 4090 are therefore:

| Circuit | `max_k` | ticit shots/s | SymFT shots/s | ticit / SymFT |
|---|---:|---:|---:|---:|
| `msc_d3` | 4 | 412 M | 73.2 M | 5.6x |
| `msc_d5` | 10 | about 6.5 M | 4.91 M | 1.33x |
| `coherent_surface_d5_r1` | 12 | 1.363 M | 1.318 M | 1.034x |

The last `k=12` improvement stores the partner contribution computed for the
measurement probability and reuses it for projection. This removed a second
state permutation and parity pass, reducing kernel time from 80.11 to 72.66
ms. A `zmask == 0` measurement shortcut immediately before it reduced kernel
time from 84.92 to 80.11 ms. Both retained exact counters.

## Wide global-state path (`k=22`)

`coherent_surface_d5_r5` uses 4,194,304 amplitudes per shot and cannot remain
resident. ticit uses one CTA per shot and global primary/scratch buffers. The
reference SymFT kernel uses the same one-CTA ownership model.

The reference source exposed a missing specialization: all 430 rotations in
this circuit have `zmask == 0`, so their pair coefficient is purely imaginary.
Commit `4c20101` uses the four-term uniform-pair update instead of the general
complex formula. At 1,000 shots in 100-shot chunks, total time fell from
11.021 to 10.913 s and kernel time from 6.440 to 6.328 s, with the same 997
discarded shots.

cuTile maps a 256-lane wide tile to 128 CUDA threads (56 registers/thread).
Changing only the tile work quantum produced:

| Tile lanes | Commit / outcome | JIT warmup | Kernel time | Total sample time |
|---:|---|---:|---:|---:|
| 128 | `730f248`, reverted | 59.39 s | 10.313 s | 14.928 s |
| 256 | `4c20101`, baseline | 64.94 s | 6.328 s | 10.913 s |
| 512 | `b7bc62f`, retained step | 68.64 s | 4.802 s | 9.387 s |
| 1,024 | `1640c5c`, retained | 74.04 s | 4.287 s | 8.869 s |
| 2,048 | `8a264ba`, reverted | 244.89 s | 3.977 s | 8.557 s |

The 2,048-lane tile bought only 3.5% total throughput while tripling JIT time,
so commit `c363fd4` restored 1,024 lanes. Nsight Compute remains unavailable
(`ERR_NVGPUCTRPERM`); Nsight Systems confirms 128-thread blocks for the wide
step kernel.

Chunk width matters more than another local arithmetic tweak. A 100-shot
chunk underfills the 4090's 128 SMs, while the old 170-shot memory-budget cap
requires two waves. With 1,024 attempted shots in eight 128-shot chunks, ticit
took 4.260 s (240.4 shot/s), discarded 1,018, and accepted 6. SymFT with the
same 1,024 shots, 128-shot launch size, 128 threads/block, GPU exogenous
sampling, postselection, and observable 0 took 9.197 s (111.3 shot/s),
discarded 1,019, and accepted 5. ticit is 2.16x faster on this matched `k=22`
case. Commit `e810935` makes 128 the default wide chunk cap.

CUDA Graph replay, fused detector checks, and a parity-only wide
specialization were neutral (within 0.2%), so they were reverted. The retained
wide wins are the reference's uniform-imaginary pair formula, a 1,024-lane
cuTile work quantum, and one-wave 128-shot chunks.

## Sparse wide path (`k=19`)

`MSC_circuit_d7_p0.0005` has 11,937 low-probability exogenous sets in three
sparse groups. A 1,024-shot host/device differential check originally found a
rare disagreement caused by Rust and CUDA `log` rounding to opposite sides of
an integer geometric-gap boundary. Commits `2be3ca9` through `2ff8b6d` derive
the gap from the shared 24 random bits and integer CDF thresholds instead. The
same 1,024-shot check then passed all eight 128-shot chunks exactly; ticit
reported 1,022 discarded, 2 accepted, and 2 logical errors.

With 8,192 attempted shots in 128-shot chunks, ticit took 0.74944 s (10.93
kshot/s) and SymFT took 1.51852 s (5.395 kshot/s). Both used GPU exogenous
sampling, detector postselection, observable 0, and 128-thread blocks on the
RTX 4090. ticit is 2.03x faster on this matched case. Both runs discarded 8,151
shots and accepted 41; their logical-error counts differ because their branch
RNG streams are not shared. ticit's 77.87 s one-time cuTile JIT warmup is
reported separately and excluded from `sample_s`.

## Generic wide bit-mask shift

Commit `bf46366` replaces the wide kernel's scalar multiply loop for `1 << bit`
with one cuTile shift. It applies to every `max_k > 12` plan and does not inspect
the circuit, instruction mix, or rotation masks.

| Circuit | Shots / chunk | 4090 baseline / current | 4090 throughput | H200 baseline / current | H200 throughput |
|---|---:|---:|---:|---:|---:|
| `coherent_surface_d5_r5` (`k=22`) | 1,024 / 256 | 3.78618 / 3.73457 s | +1.38% | 1.90324 / 1.77325 s | +7.33% |
| `MSC_circuit_d7` (`k=19`) | 8,192 / 2,048 | 0.228677 / 0.227106 s | +0.69% | 0.111587 / 0.105783 s | +5.49% |

Each paired run used the same seed and produced identical aggregate counters.
A plan-specific uniform-rotation kernel and a shared resident/wide dynamic
shift both had architecture-dependent tradeoffs and were not retained.
