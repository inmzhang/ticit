use anyhow::Result;
use clap::{Parser, Subcommand};

mod bench;
mod gpu_bench;

#[derive(Parser)]
#[command(
    bin_name = "cargo xtask",
    version,
    about = "Developer workflows for ticit"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Benchmark the batch CPU sampler.
    Bench(bench::Args),

    /// Benchmark ticit and SymFT GPU samplers with matched launch settings.
    #[command(name = "bench-gpu", alias = "gpu-bench")]
    BenchGpu(gpu_bench::Args),
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Bench(args) => bench::run(&args),
        Command::BenchGpu(args) => gpu_bench::run(&args),
    }
}
