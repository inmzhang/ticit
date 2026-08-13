# CCZ logical-`EXP_VAL` GPU benchmark

Measured 2026-08-11 on RTX 4090 D cards on `gpu07` with ticit
`dd631a198eced9b577a6470fe5258a6c1c1fdcfe` and SymFT
`925078bb5190137eba6996a6485293b5b7dcd55f`. Every fixture has exactly 28
logical `EXP_VAL` probes; the redundant terminal-stabilizer probes are absent.

Each final row uses 65,536 shots without detector postselection. Ticit's rate
is the median of three independent processes; SymFT's rate is the mean reported
by one process with three repeats. Sampling excludes parsing, planning, RNG
setup, and ticit's cuTile warm-up. All eight final jobs completed successfully
and reported 65,536 accepted shots per repeat with no discard or logical-error
counter activity.

## Final sampling rates

| Circuit | ticit (shots/s) | SymFT (shots/s) | Ratio |
|---|---:|---:|---:|
| `d05_p0` | 5.59 M | 187 k | 30.0x |
| `d05_p1e-3` | 5.51 M | 26.9 k | 205x |
| `d07_p0` | 5.65 M | 48.4 k | 117x |
| `d07_p1e-3` | 5.42 M | 2.07 k | 2,618x |
| `d09_p0` | 5.62 M | 12.1 k | 465x |
| `d09_p1e-3` | 5.48 M | 10.3 k | 532x |
| `d11_p0` | 5.55 M | 5.22 k | 1,063x |
| `d11_p1e-3` | 5.38 M | 5.24 k | 1,026x |
| **Geometric mean** |  |  | **387x** |

## Selected launch settings

| Circuit | ticit chunk shots | SymFT mode | Threads/block | Shots/launch |
|---|---:|---|---:|---:|
| `d05_p0` | 32,768 | GPU exogenous | 32 | 65,536 |
| `d05_p1e-3` | 65,536 | presample expressions | 128 | 65,536 |
| `d07_p0` | 32,768 | GPU exogenous | 32 | 65,536 |
| `d07_p1e-3` | 65,536 | GPU exogenous | 32 | 65,536 |
| `d09_p0` | 65,536 | GPU exogenous | 32 | 65,536 |
| `d09_p1e-3` | 32,768 | GPU exogenous | 32 | 65,536 |
| `d11_p0` | 65,536 | GPU exogenous | 32 | 65,536 |
| `d11_p1e-3` | 32,768 | GPU exogenous | 32 | 65,536 |

Ticit swept chunk sizes 8,192, 32,768, and 65,536. SymFT swept GPU
exogenous versus presampled expressions, 32/128/256 threads per block, and
8,192/32,768/65,536 shots per launch. The large noisy presample variants were
not safe or competitive: d=7 and d=9 exceeded device shared memory, while the
d=11 candidate produced no result after four minutes and was stopped before
the remaining safe configurations were resumed.

## Fixed cost

Ticit fixed cost is the median of parse + plan + RNG setup + cuTile warm-up
across its three processes. SymFT fixed cost is parse + plan from the final
three-repeat process.

| Circuit | ticit (s) | SymFT (s) |
|---|---:|---:|
| `d05_p0` | 24.2 | 0.166 |
| `d05_p1e-3` | 30.2 | 0.986 |
| `d07_p0` | 28.1 | 0.518 |
| `d07_p1e-3` | 44.6 | 4.05 |
| `d09_p0` | 38.6 | 1.60 |
| `d09_p1e-3` | 82.0 | 13.3 |
| `d11_p0` | 63.9 | 4.51 |
| `d11_p1e-3` | 172.9 | 36.6 |

The ticit binary was built with `cargo build --release --features gpu` against
the retained CUDA 13.3 redistributable. Slurm tuning jobs were 25043, 25048,
and 25060; final jobs were 25061, 25064, and 25069 under `qos_4090`. Raw logs
remain at
`/dssg/home/zhangyiming/workspace/gpu-comparison-logical-expval-2026-08-11/logs`.
