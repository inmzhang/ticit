# ticit Python bindings

The `ticit` package exposes ticit's exact batch sampler through
`Circuit.compile` then `Program.sample`. See
[`docs/python_api_reference.md`](../docs/python_api_reference.md).

```python
import ticit

compiled_sampler = ticit.Circuit("M 0").compile()
result = compiled_sampler.sample(shots=1_000, seed=42, bit_packed=True)
```

```sh
maturin develop -m ticit_py/Cargo.toml
cargo run -p ticit_py --bin stub_gen
```

Build with `--features gpu` to enable `backend="gpu"`.
