use anyhow::Result;
use clap::{Parser, Subcommand};

mod bench;

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
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Bench(args) => bench::run(&args),
    }
}
