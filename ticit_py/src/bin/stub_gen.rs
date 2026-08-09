//! Generates the checked-in Python stub from pyo3-stub-gen metadata.

fn main() -> pyo3_stub_gen::Result<()> {
    ticit_py::stub_info()?.generate()
}
