use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use ticit::{Circuit, SamplerOptions};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
}

#[derive(Parser)]
#[command(version, about = "Sample a .ticit circuit with the CPU or GPU backend")]
struct Cli {
    /// Circuit to sample.
    circuit: PathBuf,

    /// Number of attempted shots.
    #[arg(short = 'n', long, default_value_t = 1000, value_parser = clap::value_parser!(u64).range(1..))]
    shots: u64,

    /// Seed for deterministic sampling.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Sampling backend.
    #[arg(long, value_enum, default_value = "cpu")]
    backend: Backend,

    /// Number of sampler threads.
    #[arg(short = 'j', long, default_value = "1")]
    threads: NonZeroUsize,

    /// Shots presampled and uploaded per GPU launch group.
    #[arg(long, default_value = "1048576")]
    chunk_shots: NonZeroUsize,

    /// Postselect every detector, in addition to source `DISCARD`s.
    #[arg(long)]
    postselect_detectors: bool,

    /// Write per-shot detector and expectation rows using this path prefix.
    #[arg(long)]
    records_out: Option<PathBuf>,

    /// Condition all ordinary Bernoulli sources on exactly this many faults.
    #[arg(long, requires = "records_out", value_delimiter = ',')]
    exact_k: Vec<usize>,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    match args.backend {
        Backend::Cpu => run_cpu(&args),
        Backend::Gpu => run_gpu(&args),
    }
}

fn run_cpu(args: &Cli) -> Result<()> {
    let circuit = Circuit::from_file(&args.circuit)
        .with_context(|| format!("failed to parse {}", args.circuit.display()))?;
    let options = SamplerOptions {
        postselection_mask: if args.postselect_detectors {
            vec![1; circuit.detector_count()]
        } else {
            Vec::new()
        },
        threads: args.threads.get(),
        ..Default::default()
    };
    let mut sampler = circuit
        .compile(options)
        .context("failed to compile circuit")?;
    let info = *sampler.info();
    if !args.exact_k.is_empty() {
        anyhow::bail!("--exact-k currently requires the GPU backend");
    }
    let result = if args.records_out.is_some() {
        sampler.sample_with_seed(args.shots, args.seed, false)
    } else {
        sampler.sample_counts_with_seed(args.shots, args.seed)
    }
    .context("sampling failed")?;
    if let Some(path) = &args.records_out {
        write_records(path, &circuit, &result)?;
    }
    let counts = result.counts;

    println!("qubits {}", info.qubits);
    println!("records {}", info.measurement_records);
    println!("max_active_qubits {}", info.max_active_qubits);
    println!("simd_backend {}", info.cpu_backend);
    println!("shots {}", counts.shots);
    println!("discarded {}", counts.discarded);
    println!("accepted {}", counts.accepted);
    println!("logical_errors {}", counts.logical_errors);
    println!("discard_rate {}", rate(counts.discard_rate()));
    println!("logical_error_rate {}", rate(counts.logical_error_rate()));
    Ok(())
}

#[cfg(feature = "gpu")]
fn run_gpu(args: &Cli) -> Result<()> {
    if let Some(path) = &args.records_out {
        let circuit = Circuit::from_file(&args.circuit)
            .with_context(|| format!("failed to parse {}", args.circuit.display()))?;
        let exact_ks: Vec<Option<usize>> = if args.exact_k.is_empty() {
            vec![None]
        } else {
            args.exact_k.iter().copied().map(Some).collect()
        };
        for exact_k in exact_ks {
            let result = ticit::gpu::sample_circuit_records(
                &circuit,
                args.shots,
                args.seed,
                args.chunk_shots,
                0,
                exact_k,
            )?;
            let output =
                exact_k.map_or_else(|| path.clone(), |k| suffixed_path(path, &format!("_k{k}")));
            write_records(&output, &circuit, &result)?;
            println!("exact_k {}", exact_k.map_or(-1, |k| k as i64));
            println!("shots {}", result.counts.shots);
            println!("discarded {}", result.counts.discarded);
            println!("accepted {}", result.counts.accepted);
            println!("logical_errors {}", result.counts.logical_errors);
            println!("compile_s {}", result.timing.compile_s);
            println!("sample_s {}", result.timing.sample_s);
        }
        return Ok(());
    }
    ticit::gpu::run(&ticit::gpu::GpuOptions {
        circuit: args.circuit.clone(),
        shots: args.shots,
        seed: args.seed,
        chunk_shots: args.chunk_shots,
        postselect_detectors: args.postselect_detectors,
    })
}

fn suffixed_path(path: &std::path::Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    name.into()
}

fn write_records(
    path: &std::path::Path,
    circuit: &Circuit,
    result: &ticit::SampleResult,
) -> Result<()> {
    std::fs::write(suffixed_path(path, ".detectors.u8"), &result.detectors)?;
    let mut output = BufWriter::new(File::create(suffixed_path(path, ".exp_vals.f64"))?);
    for values in result.exp_vals.chunks(8192) {
        let mut bytes = Vec::with_capacity(values.len() * size_of::<f64>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        output.write_all(&bytes)?;
    }
    output.flush()?;
    std::fs::write(
        suffixed_path(path, ".meta"),
        format!(
            "rows {}\ndetectors {}\nexpectations {}\nlayout row-major\nf64 little-endian\n",
            result.record_rows,
            circuit.detector_count(),
            circuit.expectation_value_count(),
        ),
    )?;
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_gpu(_args: &Cli) -> Result<()> {
    anyhow::bail!("the GPU backend requires a build with `--features gpu`");
}

fn rate(value: f64) -> String {
    if value.is_nan() {
        "nan".into()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_is_choosable() {
        let cpu = Cli::try_parse_from(["ticit", "circuit.ticit"]).expect("CPU CLI parses");
        assert_eq!(cpu.backend, Backend::Cpu);
        let gpu = Cli::try_parse_from(["ticit", "circuit.ticit", "--backend", "gpu"])
            .expect("GPU CLI parses");
        assert_eq!(gpu.backend, Backend::Gpu);
        assert!(Cli::try_parse_from(["ticit", "circuit.ticit", "--exact-k", "2"]).is_err());
        let exact = Cli::try_parse_from([
            "ticit",
            "circuit.ticit",
            "--records-out",
            "rows",
            "--exact-k",
            "0,1,2",
        ])
        .expect("exact-k list parses");
        assert_eq!(exact.exact_k, [0, 1, 2]);
    }
}
