# Refactor performance check — 2026-08-09

Commit `0626249`, Intel Core i5-14600KF P-core 10, one worker, release with
`-Ctarget-cpu=native -Cdebuginfo=1`. Each rate is five in-process repeats.

| Circuit | Shots/repeat | shots/s | README baseline |
|---|---:|---:|---:|
| `msc_d3` | 20 M | 5.194 M | 5.19 M |
| `msc_d3`, postselected | 20 M | 6.013 M | 5.93 M |
| `msc_d5` | 200 k | 68.9 k | 66.8 k |
| CCZ `d05_p0` | 1 M | 286 k | 287 k |
| CCZ `d05_p1e-3` | 500 k | 122 k | 121 k |

No broad regression; the README table remains representative. A cycle profile
of `msc_d5` attributed 66.3% self time to active rotation, then 8.6% to
diagonal probability, 6.9% to promotion, and 6.3% to diagonal projection.
Those hot x86 kernels keep their measured permutation/parity intrinsic island.
