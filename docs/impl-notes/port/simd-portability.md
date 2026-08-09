# SIMD portability

## ticit

`fearless_simd::Level` selects one backend at runtime:

- x86-64: AVX-512, then AVX2/FMA, then scalar;
- AArch64: baseline NEON, then scalar when `TICIT_SIMD=scalar`;
- other targets: scalar.

The AArch64 kernels use Fearless SIMD's safe `f64x2` API for diagonal and
uniform-pair rotations, promotion, diagonal probability, and nondiagonal
probability/projection. The dim-16 register-run specialization remains AVX2;
ARM executes that run as its NEON-vectorized component operations instead.

The remaining x86 intrinsics implement whole-vector XOR permutations, packed
parity masks, and horizontal reductions that Fearless SIMD 0.6 does not expose
directly. Fearless SIMD still owns target dispatch and checked loads/stores.
`perf` puts these kernels on the hot path, so they should only be replaced as
whole primitives after representative benchmarks show parity.

## Reference implementations

SymFT builds dedicated AVX2 and AVX-512 translation units and runtime-dispatches
between them and its scalar table. It has no NEON table; `-march=native` may let
the compiler auto-vectorize scalar loops on ARM.

Clifft likewise runtime-dispatches dedicated AVX2/AVX-512 SVM translation units
only on x86. Its ARM build uses the scalar source, with either a portable
baseline or `-mcpu=native`, and relies on compiler auto-vectorization.

## Verification

- `cargo clippy --workspace --all-targets --all-features --target
  aarch64-unknown-linux-gnu -- -D warnings` checks the Linux ARM build locally.
- The equivalent `aarch64-apple-darwin` check covers Apple's target ABI.
- A release AArch64 assembly build contains NEON `v*.2d` operations and fused
  multiply-add instructions, proving the path did not scalarize.
- CI's `macos-15` ARM64 runner executes the full test suite natively, including
  the scalar-oracle comparisons in `contiguous`.

Performance claims still require an ARM benchmark host; cross-compilation and
CI establish correctness and code generation, not throughput.
