# ticit

Ticit is a pure-Rust implementation of [SymFT](https://arxiv.org/abs/2607.28600) with CPU and an experimental CUDA backend utilizing [cutile-rs](https://github.com/NVlabs/cutile-rs), plus a Tableau-style [`TableauSimulator`](https://docs.rs/ticit/latest/ticit/tableau_simulator/struct.TableauSimulator.html) for Clifford and Pauli-rotation circuits.

## Installation

### Build from source

```sh
# rust
git clone https://github.com/inmzhang/ticit.git
cd ticit
cargo build --release

# python
uv sync --project ticit_py
```

### CLI binary

```sh
cargo install ticit
```

### Rust crate

```sh
cargo add ticit
```

### Python package

```sh
uv add ticit
```

## Rust API

See the [Rust API documentation](https://docs.rs/ticit).

```rust
use ticit::{Circuit, SamplerOptions};

let circuit = Circuit::from_text("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]")?;
let mut compiled_sampler = circuit.compile(SamplerOptions::default())?;
let result = compiled_sampler.sample(1_000, false)?;
```

Use `sample_with_seed(shots, seed, bit_packed)` when reproducible records are
required. Setting `bit_packed` packs eight record bits into each output byte.

## Python API

See the [Python API reference](docs/python_api_reference.md).

```python
import ticit

compiled_sampler = ticit.Circuit("M 0").compile(backend="cpu")
result = compiled_sampler.sample(shots=1_000, seed=42)
```

## Benchmarks

See [ticit vs SymFT vs Clifft](docs/benchmark.md) for CPU and GPU results,
methodology, and caveats.

## AI acknowledgement

Extensive use of generative AI is an explicit initial goal of this project:
to port SymFT's C++ implementation to Rust and experiment with Rust portable
SIMD and [cuTile-rs](https://github.com/NVlabs/cutile-rs). AI tools assist with
implementation, review, analysis, benchmarking, and documentation; human
contributors make and verify the substantive design, validation, and release
decisions.

## License

[Apache-2.0](LICENSE)
