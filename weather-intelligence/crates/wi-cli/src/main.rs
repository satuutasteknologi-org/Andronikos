//! # Weather Intelligence CLI
//!
//! Command-line interface for the Weather Intelligence Engine.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod commands;

use commands::{
    ExportCommand, InfoCommand, PredictCommand, ServeCommand, SyncCommand, TrainCommand,
    VerifyCommand,
};

#[derive(Parser, Debug)]
#[command(name = "wi")]
#[command(about = "Weather Intelligence Engine - Unified Weather Forecasting System", long_about = None)]
#[command(author = "Weather Intelligence Team")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    #[arg(short, long, global = true, default_value = "info")]
    log_level: String,

    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Synchronize weather data from all configured sources
    Sync {
        #[arg(short, long, default_value = "24")]
        hours: u64,

        #[arg(short, long)]
        full: bool,

        #[arg(long)]
        sources: Vec<String>,

        #[arg(long)]
        region: Option<String>, // "indonesia" or "global" or "lat,lon,lat,lon"
    },

    /// Train the weather prediction model
    Train {
        #[arg(short, long)]
        config: Option<PathBuf>,

        #[arg(short, long)]
        epochs: Option<usize>,

        #[arg(short, long)]
        batch_size: Option<usize>,

        #[arg(short, long)]
        learning_rate: Option<f32>,

        #[arg(long)]
        resume: Option<String>,

        #[arg(short, long)]
        checkpoint_dir: Option<PathBuf>,

        #[arg(short, long, default_value = "data/train.parquet")]
        data: PathBuf,

        #[arg(long)]
        model_type: Option<ModelType>,
    },

    /// Run inference with a trained model
    Predict {
        #[arg(short, long)]
        model: Option<PathBuf>,

        #[arg(short, long)]
        latitude: f64,

        #[arg(short, long)]
        longitude: f64,

        #[arg(
            short = 's',
            long,
            default_value = "3600",
            help = "Lead time in seconds (30, 60, 300, 3600, etc.)"
        )]
        lead_seconds: u32,

        #[arg(
            short = 'H',
            long,
            alias = "lead-hours",
            help = "Lead time in hours (shortcut for lead-seconds/3600)"
        )]
        lead_hours: Option<f32>,

        #[arg(short, long)]
        output: Option<PathBuf>,

        #[arg(long)]
        format: Option<OutputFormat>,

        #[arg(long)]
        variables: Vec<String>,
    },

    /// Start the API server
    Serve {
        #[arg(short, long, default_value = "0.0.0.0")]
        host: String,

        #[arg(short, long, default_value = "8080")]
        port: u16,

        #[arg(short, long)]
        model: Option<PathBuf>,

        #[arg(long)]
        workers: Option<usize>,

        #[arg(long)]
        cors: bool,

        #[arg(long)]
        enable_grpc: bool,
    },

    /// Evaluate model performance against observations
    Verify {
        #[arg(short, long)]
        model: PathBuf,

        #[arg(short, long)]
        data: PathBuf,

        #[arg(short, long)]
        metrics: Vec<String>,

        #[arg(long)]
        baseline: Vec<String>,

        #[arg(long)]
        output: Option<PathBuf>,

        #[arg(long)]
        temporal_split: bool,

        #[arg(long)]
        spatial_split: bool,
    },

    /// Export model to different formats
    Export {
        #[arg(short, long)]
        model: PathBuf,

        #[arg(short, long)]
        format: ExportFormat,

        #[arg(short, long)]
        output: PathBuf,
    },

    /// Show model information
    Info {
        #[arg(short, long)]
        model: Option<PathBuf>,
    },

    /// Run data ingestion pipeline
    Ingest {
        #[arg(short, long, default_value = "24")]
        hours: u64,

        #[arg(long)]
        sources: Vec<String>,

        #[arg(long)]
        output_format: Option<StorageFormat>,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    GraphCast,
    Downscaler,
}

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Json,
    Csv,
    Parquet,
}

#[derive(Debug, Clone, ValueEnum)]
enum ExportFormat {
    Onnx,
    TorchScript,
    TensorRT,
    BurnNative,
}

#[derive(Debug, Clone, ValueEnum)]
enum StorageFormat {
    Atd,
    Parquet,
    Postgres,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = color_eyre::install();

    let cli = Cli::parse();

    init_logging(&cli.log_level, cli.verbose);

    match cli.command {
        Commands::Sync {
            hours,
            full,
            sources,
            region,
        } => {
            let cmd = SyncCommand {
                hours,
                full,
                sources,
                region,
            };
            cmd.execute().await?;
        }
        Commands::Train {
            config,
            epochs,
            batch_size,
            learning_rate,
            resume,
            checkpoint_dir,
            data,
            model_type,
        } => {
            let cmd = TrainCommand {
                _config: config,
                epochs,
                batch_size,
                learning_rate,
                resume,
                checkpoint_dir,
                data,
                model_type,
            };
            cmd.execute().await?;
        }
        Commands::Predict {
            model,
            latitude,
            longitude,
            lead_seconds,
            lead_hours,
            output,
            format,
            variables,
        } => {
            let effective_lead_seconds = if let Some(hours) = lead_hours {
                (hours * 3600.0) as u32
            } else {
                lead_seconds
            };
            let cmd = PredictCommand {
                model,
                latitude,
                longitude,
                lead_seconds: effective_lead_seconds,
                output,
                format,
                variables,
            };
            cmd.execute().await?;
        }
        Commands::Serve {
            host,
            port,
            model,
            workers,
            cors,
            enable_grpc,
        } => {
            let cmd = ServeCommand {
                host,
                port,
                model,
                workers,
                cors,
                enable_grpc,
            };
            cmd.execute().await?;
        }
        Commands::Verify {
            model,
            data,
            metrics,
            baseline,
            output,
            temporal_split,
            spatial_split,
        } => {
            let cmd = VerifyCommand {
                model,
                data,
                metrics,
                baseline,
                output,
                temporal_split,
                spatial_split,
            };
            cmd.execute().await?;
        }
        Commands::Export {
            model,
            format,
            output,
        } => {
            let cmd = ExportCommand {
                model,
                format,
                output,
            };
            cmd.execute().await?;
        }
        Commands::Info { model } => {
            let cmd = InfoCommand { model };
            cmd.execute().await?;
        }
        Commands::Ingest {
            hours,
            sources,
            output_format,
        } => {
            commands::ingest_command(hours, sources, output_format).await?;
        }
    }

    Ok(())
}

fn init_logging(level: &str, verbose: bool) {
    use tracing_subscriber::{fmt, EnvFilter};

    let env_filter = if verbose {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::new(level)
    };

    fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();
}
