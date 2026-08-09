# GPU throughput comparison — 2026-08-09

ticit GPU commit `c6d0db1` was compared with SymFT commit `bd77739` on one RTX
4090 and one H200 allocated through Slurm. The two wide-state rows were
remeasured in paired runs after the generic bit-mask improvement at `bf46366`.
On the 11 SOFT circuits, ticit wins 9/11 on the 4090 with a 1.665x geometric-mean
speedup and 8/11 on the H200 with a 1.192x speedup. It wins all eight CCZ
non-tels cases, with geometric means of 207.7x and 180.5x respectively.

## Method

- Rates are attempted shots divided by each CLI's `sample_s`. This is the
  public aggregate batch API, not ticit's internal single-shot validation hook.
- `sample_s` excludes parsing, planning, RNG setup, and ticit's one-time cuTile
  JIT warmup. JIT latency is listed separately below.
- Both tools use raw detector parity for postselection and observable 0 for
  logical-error counting. This avoids the convention and metric mismatches in
  SOFT issues #8 and #9.
- SymFT's `gpu` and `gpu_presample_expressions` modes, 32--1,024 threads per
  block, and progressively larger launches were swept where applicable. The
  tables use the strongest memory-feasible measured setting, not an arbitrary
  shared microbatch.
- ticit uses its public automatic chunk policy: a 16 GiB workspace ceiling,
  two-thirds of currently free VRAM, tensor-dimension guards, and power-of-two
  chunks. Wide `k=19` and `k=22` rows used 2,048 and 256 shots per ticit chunk.
- The two dim-16 rows use 10 million attempted shots to stabilize their very
  short kernels. Wide rows use 8,192 (`k=19`) and 1,024 (`k=22`) shots. Every
  CCZ row uses one 65,536-shot batch.
- RNG streams are independent. Aggregate counts agree statistically, while
  ticit's GPU path is checked exactly against its CPU reference on shared input
  words and seeds.
- The `bf46366` wide-row retest used the same seed and chunking for the
  candidate and preserved `c92d484` binary; their aggregate counters matched
  exactly.

## SOFT benchmark circuits

| Circuit | 4090 ticit | 4090 SymFT | Ratio | H200 ticit | H200 SymFT | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `msc_d3` | 434,129,753 | 72,441,500 | 5.993x | 374,906,545 | 109,523,000 | 3.423x |
| `coherent_d3_r1` | 603,430,090 | 132,196,000 | 4.565x | 759,354,430 | 190,117,000 | 3.994x |
| `coherent_d3_r3` | 10,112,752 | 20,973,100 | 0.482x | 9,802,852 | 21,277,800 | 0.461x |
| `distillation` | 16,733,813 | 13,913,000 | 1.203x | 15,886,143 | 15,713,600 | 1.011x |
| `msc_d5` | 6,889,508 | 4,240,030 | 1.625x | 6,658,351 | 6,423,480 | 1.037x |
| `msc_proxy_d7` | 6,149,098 | 3,153,800 | 1.950x | 5,783,066 | 4,365,040 | 1.325x |
| `pure_d7` | 12,429,261 | 7,009,280 | 1.773x | 10,969,191 | 8,040,580 | 1.364x |
| `pure_d9` | 8,910,993 | 4,393,970 | 2.028x | 7,827,662 | 5,211,010 | 1.502x |
| `coherent_d5_r1` | 1,362,645 | 1,494,890 | 0.912x | 790,391 | 2,193,520 | 0.360x |
| `coherent_d5_r5` (`k=22`) | 274.195 | 191.725 | 1.430x | 577.471 | 676.336 | 0.854x |
| `MSC_d7` sparse (`k=19`) | 36,071.3 | 31,176.7 | 1.157x | 77,441.6 | 61,924.8 | 1.251x |
| **Geometric mean** |  |  | **1.665x** |  |  | **1.192x** |

The 4090 losses are `coherent_d3_r3` and `coherent_d5_r1`. The H200 also
loses `coherent_d5_r5`. The largest remaining deficit is the H200
`coherent_d5_r1` resident `k=12` kernel at 0.360x SymFT.

The stabilized dim-16 timing inputs were:

| GPU / circuit | ticit `sample_s` | SymFT `sample_s` |
|---|---:|---:|
| 4090 `msc_d3` | 0.023034588 | 0.1380420 |
| 4090 `coherent_d3_r1` | 0.016571928 | 0.0756453 |
| H200 `msc_d3` | 0.026673314 | 0.0913054 |
| H200 `coherent_d3_r1` | 0.013169081 | 0.0525992 |

For the wide rows, SymFT's strongest feasible 4090 launches were 256 shots
for `k=22` and 2,048 for `k=19`; 512 and 4,096 shots respectively ran out of
memory. On H200 its strongest measured launches were 1,024 and 8,192. ticit's
power-of-two chunking avoids both tensor-dimension overflow and a costly
extra cuTile specialization for a non-power-of-two remainder.

## CCZ non-tels circuits

| Circuit | 4090 ticit | 4090 SymFT | Ratio | H200 ticit | H200 SymFT | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `d05_p0` | 4,515,761.5 | 183,867 | 24.56x | 4,394,548.0 | 290,345 | 15.14x |
| `d05_p1e-3` | 4,260,327.8 | 17,757.1 | 239.92x | 4,164,019.5 | 36,105.9 | 115.33x |
| `d07_p0` | 3,757,110.9 | 47,622 | 78.89x | 3,628,096.6 | 79,815.3 | 45.46x |
| `d07_p1e-3` | 3,342,902.0 | 2,010.33 | 1,662.86x | 3,133,146.6 | 5,379.47 | 582.43x |
| `d09_p0` | 2,697,589.1 | 12,603.3 | 214.04x | 2,932,302.5 | 25,466.1 | 115.15x |
| `d09_p1e-3` | 2,085,604.3 | 10,186.6 | 204.74x | 2,250,250.2 | 630.75 | 3,567.58x |
| `d11_p0` | 2,073,344.5 | 5,264.81 | 393.81x | 2,457,349.4 | 8,113.28 | 302.88x |
| `d11_p1e-3` | 1,341,969.9 | 5,161.59 | 259.99x | 1,475,122.6 | 7,545.54 | 195.50x |
| **Geometric mean** |  |  | **207.7x** |  |  | **180.5x** |

These large ratios are sampling-only and need compile-time context. For the
largest noisy case, `d11_p1e-3`, parse plus planning took 159.83 s for ticit and
160.73 s for SymFT on the 4090 allocation, and 162.98 s versus 166.80 s on the
H200 allocation. ticit then paid about 24.1 s (4090) or 22.7 s (H200) for cuTile
JIT. Sampling throughput dominates only once those fixed costs are amortized.

## JIT and interpretation

cuTile JIT warmup ranged from roughly 9 seconds for the compact dim-16 kernel
to 73--79 seconds for the wide kernels. Resident dim-1024 and dim-4096 kernels
were roughly 21 and 45 seconds. SymFT's CUDA kernels are compiled ahead of
time, so first-process wall time favors SymFT on small jobs even where ticit has
higher steady-state throughput.

The result is therefore: ticit currently leads most steady-state GPU batch
throughput, dramatically so on noisy CCZ sampling, but it does not dominate
every circuit and is not yet the lower-latency choice for one-off runs. The
GPU backend remains experimental.
