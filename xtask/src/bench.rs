use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args as ClapArgs;
use ticit::{Circuit, SampleCounts, SamplerOptions, SamplingTiming};

#[derive(ClapArgs, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Args {
    /// Circuit to benchmark.
    #[arg(long = "circuit", alias = "file", value_name = "PATH")]
    path: PathBuf,

    /// Attempted shots per repeat.
    #[arg(long, default_value_t = 100_000_000, value_parser = clap::value_parser!(u64).range(1..))]
    shots: u64,

    /// Batch width, or `auto`.
    #[arg(long, default_value = "auto", value_parser = parse_batch_size)]
    batch_size: usize,

    /// Sampling chunk size, or `auto`.
    #[arg(long, default_value = "auto", value_parser = parse_sample_chunk_shots)]
    sample_chunk_shots: usize,

    /// Number of timed repeats.
    #[arg(long, default_value_t = 1, value_parser = parse_positive)]
    repeats: usize,

    /// Observable used for logical-error counting.
    #[arg(long, default_value_t = 0, value_parser = parse_observable)]
    observable: usize,

    /// Sampler threads, or `auto`.
    #[arg(long, default_value = "1", value_parser = parse_threads)]
    threads: usize,

    /// Flat detector postselection flags, separated by commas.
    #[arg(long, value_delimiter = ',')]
    postselection_mask: Vec<u8>,
}

fn parse_u64(raw: &str, name: &str) -> std::result::Result<u64, String> {
    if raw.starts_with('-') {
        return Err(format!("{name} must be nonnegative"));
    }
    raw.parse().map_err(|_| format!("invalid {name}: {raw}"))
}

fn parse_positive(raw: &str) -> std::result::Result<usize, String> {
    let value = parse_u64(raw, "value")?;
    if value == 0 || value > i32::MAX as u64 {
        Err("value must be in 1:Int32Max".into())
    } else {
        Ok(value as usize)
    }
}

fn parse_batch_size(raw: &str) -> std::result::Result<usize, String> {
    if raw == "auto" {
        Ok(0)
    } else {
        parse_positive(raw)
    }
}

fn parse_sample_chunk_shots(raw: &str) -> std::result::Result<usize, String> {
    if matches!(raw, "auto" | "none" | "off") {
        Ok(0)
    } else {
        parse_positive(raw)
    }
}

fn parse_threads(raw: &str) -> std::result::Result<usize, String> {
    if raw == "auto" {
        Ok(std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1))
    } else {
        parse_positive(raw)
    }
}

fn parse_observable(raw: &str) -> std::result::Result<usize, String> {
    let value = parse_u64(raw, "observable")?;
    usize::try_from(value).map_err(|_| "observable index is too large".into())
}

pub(crate) fn run(options: &Args) -> Result<()> {
    let parse_start = Instant::now();
    let circuit = Circuit::from_file(&options.path)
        .with_context(|| format!("failed to parse {}", options.path.display()))?;
    let parse_s = parse_start.elapsed().as_secs_f64();
    let sampler_options = SamplerOptions {
        observable: options.observable,
        postselection_mask: options.postselection_mask.clone(),
        normalize_syndromes: true,
        sample_chunk_shots: options.sample_chunk_shots,
        batch_size: options.batch_size,
        threads: options.threads,
        ..Default::default()
    };
    let mut sampler = circuit
        .compile(sampler_options)
        .context("failed to compile sampler")?;
    let info = *sampler.info();
    let compile_s = sampler.preprocessing_timing().compile_s;
    let mut counts = SampleCounts::default();
    let mut timing = SamplingTiming::default();
    let mut active_threads = info.threads;
    for repeat in 0..options.repeats {
        let result = sampler
            .sample_with_seed(options.shots, repeat as u64, false)
            .context("sampling failed")?;
        counts.shots += result.counts.shots;
        counts.discarded += result.counts.discarded;
        counts.accepted += result.counts.accepted;
        counts.logical_errors += result.counts.logical_errors;
        timing.presample_s += result.timing.presample_s;
        timing.execute_s += result.timing.execute_s;
        timing.sample_s += result.timing.sample_s;
        active_threads = result.active_threads;
    }
    let repeats = options.repeats as f64;
    timing.presample_s /= repeats;
    timing.execute_s /= repeats;
    timing.sample_s /= repeats;
    let shots_per_s = if timing.sample_s > 0.0 {
        options.shots as f64 / timing.sample_s
    } else {
        0.0
    };
    println!(
        "sampler {}",
        if info.detector_postselection {
            "batch_postselected"
        } else {
            "batch"
        }
    );
    println!("file {}", options.path.display());
    println!("shots {}", options.shots);
    println!("sampled_shots {}", counts.shots);
    println!(
        "active_components {}",
        if info.active_components {
            "enabled"
        } else {
            "dense_fallback"
        }
    );
    println!(
        "detector_postselection {}",
        if info.detector_postselection {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("batch_size {}", info.batch_size);
    println!("sample_chunk_shots {}", info.sample_chunk_shots);
    println!("repeats {}", options.repeats);
    println!("keep_records true");
    println!("threads {active_threads}");
    if active_threads != options.threads {
        println!("requested_threads {}", options.threads);
    }
    println!("parse_s {parse_s}");
    println!("compile_s {compile_s}");
    println!("sample_s_avg {}", timing.sample_s);
    println!("sample_shots_per_s {shots_per_s}");
    if active_threads == 1 {
        println!("presample_s_avg {}", timing.presample_s);
        println!("execute_s_avg {}", timing.execute_s);
    }
    println!("discarded {}", counts.discarded);
    println!("accepted {}", counts.accepted);
    println!("logical_errors {}", counts.logical_errors);
    println!("discard_rate {}", rate(counts.discard_rate()));
    println!("logical_error_rate {}", rate(counts.logical_error_rate()));
    Ok(())
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
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: Args,
    }

    #[test]
    fn clap_accepts_standard_benchmark_flags() {
        let options = TestCli::try_parse_from([
            "bench",
            "--file=c.stim",
            "--shots=7",
            "--batch-size=64",
            "--sample-chunk-shots=auto",
            "--threads=auto",
            "--postselection-mask=1",
        ])
        .expect("flags parse")
        .args;
        assert_eq!(options.path, PathBuf::from("c.stim"));
        assert_eq!(options.shots, 7);
        assert_eq!(options.batch_size, 64);
        assert_eq!(options.sample_chunk_shots, 0);
        assert_eq!(options.postselection_mask, [1]);
        assert!(options.threads >= 1);
    }
}
