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
    let counts = sampler
        .sample_counts_with_seed(args.shots, args.seed)
        .context("sampling failed")?
        .counts;

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
    ticit::gpu::run(&ticit::gpu::GpuOptions {
        circuit: args.circuit.clone(),
        shots: args.shots,
        seed: args.seed,
        chunk_shots: args.chunk_shots,
        postselect_detectors: args.postselect_detectors,
    })
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
    }
}
