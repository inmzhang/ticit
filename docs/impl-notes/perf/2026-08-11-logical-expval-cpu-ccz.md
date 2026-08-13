# CCZ logical-`EXP_VAL` CPU benchmark

Measured overnight on 2026-08-11/12 on an Intel Core i5-14600KF, pinned to
CPU 10 with one sampler thread. The fixtures and Ticit source are commit
`dd631a198eced9b577a6470fe5258a6c1c1fdcfe`; the freshly built Ticit 0.2.2
extension used `-Ctarget-cpu=native` and has SHA-256
`681531e3af0d18e911606aa505ece7630eccead8200e3dd963d483152a7fa164`.
SymFT is `686051afe06e28c433fffec1c61686458728ca2e`; Clifft is
`0.7.1.dev34+gb2a501ddb`.

Each fixture contains exactly 28 logical `EXP_VAL` probes. Throughput is the
arithmetic mean of three 10-second repeats after preparation. Preparation is
reported separately and includes the sampler's parse/plan/reference setup.

| Circuit | Group | Tool | Status | CPU | Input shots | Call shots | Compile (s) | Runs (shots/s) | Average (shots/s) | Discard rate | Logical error rate |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ccz_d05_p0 | ccz | clifft | OK | 10 | 1000 | 1000 | 0.311616 | 4160.53, 4175.34, 4167.97 | 4167.95 | 0 | 0 |
| ccz_d05_p0 | ccz | symft | OK | 10 | 100000 | 100000 | 0.0684841 | 305701, 306119, 305489 | 305769 | 0 | 0 |
| ccz_d05_p0 | ccz | ticit | OK | 10 | 100000 | 100000 | 0.668368 | 299647, 299949, 300216 | 299938 | 0 | 0 |
| ccz_d05_p1e-3 | ccz | clifft | OK | 10 | 1000 | 1000 | 2.77672 | 3432.77, 3407.87, 3424.58 | 3421.74 | 0 | 0 |
| ccz_d05_p1e-3 | ccz | symft | OK | 10 | 50000 | 50000 | 6.03937 | 59437.1, 58888.6, 59094.2 | 59140 | 0 | 0 |
| ccz_d05_p1e-3 | ccz | ticit | OK | 10 | 50000 | 50000 | 2.34652 | 114679, 114612, 114138 | 114476 | 0 | 0 |
| ccz_d07_p0 | ccz | clifft | OK | 10 | 500 | 500 | 1.69191 | 1815.26, 1807.21, 1813.1 | 1811.86 | 0 | 0 |
| ccz_d07_p0 | ccz | symft | OK | 10 | 50000 | 50000 | 0.263087 | 133199, 133723, 133240 | 133388 | 0 | 0 |
| ccz_d07_p0 | ccz | ticit | OK | 10 | 50000 | 50000 | 2.53891 | 128186, 129066, 128415 | 128556 | 0 | 0 |
| ccz_d07_p1e-3 | ccz | clifft | OK | 10 | 500 | 500 | 20.7606 | 1286.01, 1280.52, 1279.93 | 1282.15 | 0 | 0 |
| ccz_d07_p1e-3 | ccz | symft | OK | 10 | 10000 | 10000 | 44.1054 | 19519.9, 19484.1, 19449.6 | 19484.5 | 0 | 0 |
| ccz_d07_p1e-3 | ccz | ticit | OK | 10 | 10000 | 10000 | 10.2219 | 30860, 30653, 29962.4 | 30491.8 | 0 | 0 |
| ccz_d09_p0 | ccz | clifft | OK | 10 | 200 | 200 | 8.3713 | 735.277, 736.916, 740.055 | 737.416 | 0 | 0 |
| ccz_d09_p0 | ccz | symft | OK | 10 | 20000 | 20000 | 0.822511 | 60184.2, 59943.3, 60251.5 | 60126.3 | 0 | 0 |
| ccz_d09_p0 | ccz | ticit | OK | 10 | 20000 | 20000 | 9.15942 | 39123.2, 40524.8, 40520.9 | 40056.3 | 0 | 0 |
| ccz_d09_p1e-3 | ccz | clifft | OK | 10 | 200 | 200 | 94.7543 | 403.456, 401.998, 411.981 | 405.811 | 0 | 0 |
| ccz_d09_p1e-3 | ccz | symft | OK | 10 | 5000 | 5000 | 222.759 | 9142.59, 9187.48, 9148.4 | 9159.49 | 0 | 0 |
| ccz_d09_p1e-3 | ccz | ticit | OK | 10 | 5000 | 5000 | 35.5535 | 12533.8, 12424.2, 12432.4 | 12463.5 | 0 | 0 |
| ccz_d11_p0 | ccz | clifft | OK | 10 | 100 | 100 | 33.4263 | 345.804, 345.332, 346.181 | 345.772 | 0 | 0 |
| ccz_d11_p0 | ccz | symft | OK | 10 | 10000 | 10000 | 2.58136 | 34026.2, 33921.5, 33998.1 | 33981.9 | 0 | 0 |
| ccz_d11_p0 | ccz | ticit | OK | 10 | 10000 | 10000 | 32.8286 | 22623.7, 22699.6, 22622.7 | 22648.6 | 0 | 0 |
| ccz_d11_p1e-3 | ccz | clifft | OK | 10 | 100 | 100 | 327.033 | 188.943, 190.292, 189.975 | 189.737 | 0 | 0 |
| ccz_d11_p1e-3 | ccz | symft | OK | 10 | 2000 | 2000 | 788.353 | 4882.41, 4900.24, 4896.23 | 4892.96 | 0 | 0 |
| ccz_d11_p1e-3 | ccz | ticit | OK | 10 | 2000 | 2000 | 108.32 | 6727.02, 6781.86, 6711.08 | 6739.99 | 0 | 0 |

| Sample seconds per repeat | Repeats |
|---:|---:|
| 10 | 3 |

## Peak memory

Max RSS was measured from one isolated process per row during preparation plus
one configured sample call. The tools for each circuit were pinned to CPUs
10-12 during this memory-only pass; the d=11 noisy trio used no swap.

| Circuit | ticit | SymFT | Clifft |
|---|---:|---:|---:|
| `d05_p0` | 70.9 MiB | 73.8 MiB | 63.6 MiB |
| `d05_p1e-3` | 265 MiB | 290 MiB | 306 MiB |
| `d07_p0` | 145 MiB | 159 MiB | 113 MiB |
| `d07_p1e-3` | 523 MiB | 787 MiB | 1.14 GiB |
| `d09_p0` | 336 MiB | 337 MiB | 225 MiB |
| `d09_p1e-3` | 1.07 GiB | 1.67 GiB | 3.41 GiB |
| `d11_p0` | 743 MiB | 746 MiB | 455 MiB |
| `d11_p1e-3` | 2.42 GiB | 3.21 GiB | 8.54 GiB |

Raw kernel `ru_maxrss` values (KiB), ordered as Clifft / SymFT / ticit:

| Circuit | Clifft | SymFT | ticit |
|---|---:|---:|---:|
| `d05_p0` | 65,100 | 75,576 | 72,636 |
| `d05_p1e-3` | 313,424 | 296,636 | 271,316 |
| `d07_p0` | 115,508 | 163,244 | 148,576 |
| `d07_p1e-3` | 1,190,816 | 805,564 | 536,000 |
| `d09_p0` | 230,436 | 345,152 | 344,040 |
| `d09_p1e-3` | 3,571,968 | 1,748,472 | 1,125,892 |
| `d11_p0` | 466,360 | 764,208 | 761,184 |
| `d11_p1e-3` | 8,957,200 | 3,365,832 | 2,536,508 |
