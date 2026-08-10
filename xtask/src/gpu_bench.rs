use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::{Args as ClapArgs, ValueEnum};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum SymftExogenousMode {
    Gpu,
    #[default]
    GpuPresampleExpressions,
}

impl SymftExogenousMode {
    fn flag(self) -> &'static str {
        match self {
            Self::Gpu => "--gpu-exogenous",
            Self::GpuPresampleExpressions => "--gpu-presample-expressions",
        }
    }
}

#[derive(ClapArgs, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Args {
    /// Stim circuit to benchmark with both GPU samplers.
    #[arg(long = "circuit", alias = "file", value_name = "PATH")]
    circuit: PathBuf,

    /// Attempted shots for each sampler.
    #[arg(long, default_value_t = 1024, value_parser = clap::value_parser!(u64).range(1..))]
    shots: u64,

    /// Shots per GPU launch for both samplers.
    #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u64).range(1..=i32::MAX as u64))]
    shots_per_launch: u64,

    /// Compiled ticit GPU executable.
    #[arg(long, default_value = "target/release/ticit")]
    ticit_binary: PathBuf,

    /// Compiled SymFT CUDA benchmark executable.
    #[arg(long)]
    symft_binary: PathBuf,

    /// SymFT CUDA exogenous-sampling mode.
    #[arg(long, default_value = "gpu-presample-expressions")]
    symft_mode: SymftExogenousMode,

    /// SymFT CUDA threads per block.
    #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u64).range(1..=1024))]
    symft_threads_per_block: u64,

    /// Postselect detectors; ticit first normalizes them with a CPU reference.
    #[arg(long)]
    postselect_detectors: bool,
}

pub(crate) fn run(options: &Args) -> Result<()> {
    run_one("ticit", &options.ticit_binary, ticit_arguments(options))?;
    run_one("SymFT", &options.symft_binary, symft_arguments(options))
}

fn run_one(name: &str, binary: &PathBuf, arguments: Vec<OsString>) -> Result<()> {
    eprintln!("==> {name}");
    let status = Command::new(binary)
        .args(arguments)
        .status()
        .with_context(|| format!("failed to run {name} GPU benchmark"))?;
    if !status.success() {
        bail!("{name} GPU benchmark exited with {status}");
    }
    Ok(())
}

fn ticit_arguments(options: &Args) -> Vec<OsString> {
    let mut arguments = vec![
        options.circuit.as_os_str().into(),
        "--backend".into(),
        "gpu".into(),
        "--shots".into(),
        options.shots.to_string().into(),
        "--chunk-shots".into(),
        options.shots_per_launch.to_string().into(),
        "--normalize-syndromes".into(),
    ];
    if options.postselect_detectors {
        arguments.push("--postselect-detectors".into());
    }
    arguments
}

fn symft_arguments(options: &Args) -> Vec<OsString> {
    vec![
        "--circuit".into(),
        options.circuit.as_os_str().into(),
        "--shots".into(),
        options.shots.to_string().into(),
        "--shots-per-launch".into(),
        options.shots_per_launch.to_string().into(),
        "--threads-per-block".into(),
        options.symft_threads_per_block.to_string().into(),
        options.symft_mode.flag().into(),
        if options.postselect_detectors {
            "--postselect-detectors".into()
        } else {
            "--no-postselect-detectors".into()
        },
        "--observable".into(),
        "0".into(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> Args {
        Args {
            circuit: "case.stim".into(),
            shots: 4096,
            shots_per_launch: 128,
            ticit_binary: "ticit".into(),
            symft_binary: "symft-gpu".into(),
            symft_mode: SymftExogenousMode::Gpu,
            symft_threads_per_block: 64,
            postselect_detectors: true,
        }
    }

    #[test]
    fn builds_matched_gpu_commands() {
        let options = options();
        assert_eq!(
            ticit_arguments(&options),
            [
                "case.stim",
                "--backend",
                "gpu",
                "--shots",
                "4096",
                "--chunk-shots",
                "128",
                "--normalize-syndromes",
                "--postselect-detectors",
            ]
        );
        assert_eq!(
            symft_arguments(&options),
            [
                "--circuit",
                "case.stim",
                "--shots",
                "4096",
                "--shots-per-launch",
                "128",
                "--threads-per-block",
                "64",
                "--gpu-exogenous",
                "--postselect-detectors",
                "--observable",
                "0",
            ]
        );
    }
}
