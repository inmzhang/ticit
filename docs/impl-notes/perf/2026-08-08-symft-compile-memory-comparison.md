# Preprocessing time and peak memory: ticit vs native SymFT — 2026-08-08

Requested comparison alongside the throughput work: circuit compilation
(parse + plan + prepare) wall time and peak RSS for ticit at 728f02's code
against the native SymFT reference build
(`SOFT/build-strict`, `-O3` with `-march=native` kernels,
`symft_rate_bench`).

Method: each case is one pinned process (`taskset -c 0`) run under a
Python wrapper reporting wall seconds and `RUSAGE_CHILDREN` peak RSS
(GNU time is not installed on this host). Shot counts are kept small so
the run is compile-dominated; compile time is wall minus the tool's own
`sample_s_avg`. Two full passes agreed within ~3%; the table shows the
second pass.

| Circuit | ticit compile | SymFT compile | ticit peak RSS | SymFT peak RSS |
|---|---:|---:|---:|---:|
| SOFT `msc_d3` | ~0.005 s | ~0.01 s | 17.8 MB | 18.0 MB |
| SOFT `pure_surface_d9` | 0.05 s | 0.07 s | 29.3 MB | 29.0 MB |
| CCZ `d05_p0` | 0.75 s | 0.69 s | 39.0 MB | 41.6 MB |
| CCZ `d05_p1e-3` | **2.6 s** | **8.1 s** | 348 MB | 303 MB |
| CCZ `d07_p0` | 2.82 s | 2.72 s | 113 MB | 135 MB |
| CCZ `d07_p1e-3` | **11.1 s** | **55.9 s** | 1.17 GB | 1.00 GB |

Reading:

- On the planning-heavy noisy CCZ circuits ticit compiles 3.1x (d05) and
  5.0x (d07) faster than SymFT. This is this session's hash-map
  interning + candidate-bounded parent search in
  `prepare_presampled_expression_plan_from_words`; before that fix ticit
  was at 17.1 s / 131.7 s on these circuits, i.e. slower than SymFT.
- The noiseless (p0) circuits are planner-light in both tools and land
  within ~10% of each other, ticit slightly behind on d05_p0/d07_p0.
- Peak RSS is planning-dominated and comparable: ticit is ~15% higher on
  the noisy circuits and ~15% lower on the p0 circuits. Small circuits
  are identical at ~18-29 MB. (The noisy-side gap has not been
  attributed; ticit's owned `Vec<i32>` interner keys are a plausible but
  unverified suspect.)
- Sampling-phase memory is flat in total shots for both tools (chunked
  streaming), so these peaks are also the full-run peaks; the msc_d3 and
  d9 rows confirm the sampling-side footprint parity.

TODO: if the noisy-circuit RSS gap matters for larger circuits, profile
which plan structure holds the extra ~45 MB (d05) / ~170 MB (d07).

## Addendum: msc_d5 and CCZ d09/d11, plus Clifft

Extended per user request; full three-way tables live in `docs/benchmark.md`
(measured at ce3eb52, single process per tool with in-process repeats —
SymFT's d11_p1e-3 compile alone runs ~15 minutes). Highlights:

- Noisy d09/d11: ticit samples 1.9x faster than SymFT and compiles
  6.7x / 7.7x faster (38.5 s vs 257 s; 115 s vs 885 s), but the RSS
  premium grows with circuit size: +29% at d09_p1e-3 (3.30 vs 2.56 GB),
  +38% at d11_p1e-3 (7.90 vs 5.74 GB). The TODO above gets more
  important at these sizes.
- msc_d5, d09_p0, d11_p0: ticit and SymFT are at parity (0.96-1.02x) —
  large-dimension active kernels dominate and neither tool has
  specialized them. These are the natural next optimization targets.
- Clifft (dev95, Python API): far behind on CCZ throughput (hundreds of
  shots/s) but compiles between the two on noisy circuits and matches
  on memory once the interpreter baseline is discounted.
