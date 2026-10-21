//! # AetherForge CLI
//! 
//! Command-line interface for AetherForge Weather Intelligence Engine.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod commands;

use commands::{SyncCommand, TrainCommand, PredictCommand, ServeCommand};

#[derive(Parser, Debug)]
#[command(name = "aetherforge")]
#[command(about = "AetherForge - Weather Intelligence Engine", long_about = None)]
#[command(author = "AetherForge Team")]
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
        
        #[arg(short, long, default_value = "dataset.csv")]
        data: PathBuf,
    },
    
    /// Run inference with a trained model
    Predict {
        #[arg(short, long)]
        model: Option<PathBuf>,
        
        #[arg(short, long)]
        latitude: f64,
        
        #[arg(short, long)]
        longitude: f64,
        
        #[arg(short = 's', long, default_value = "3600", help = "Lead time in seconds (30, 60, 300, 3600, etc.)")]
        lead_seconds: u32,
        
        #[arg(short = 'H', long, alias = "lead-hours", help = "Lead time in hours (shortcut for lead-seconds/3600)")]
        lead_hours: Option<f32>,
        
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        #[arg(long)]
        format: Option<OutputFormat>,
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
    },
    
    /// Evaluate model performance
    Eval {
        #[arg(short, long)]
        model: PathBuf,
        
        #[arg(short, long)]
        data: PathBuf,
        
        #[arg(short, long)]
        metrics: Vec<String>,
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = color_eyre::install();
    
    let cli = Cli::parse();
    
    init_logging(&cli.log_level, cli.verbose);
    
    match cli.command {
        Commands::Sync { hours, full, sources } => {
            let cmd = SyncCommand { hours, full, sources };
            cmd.execute().await?;
        },
        Commands::Train { config, epochs, batch_size, learning_rate, resume, checkpoint_dir, data } => {
            let cmd = TrainCommand {
                config,
                epochs,
                batch_size,
                learning_rate,
                resume,
                checkpoint_dir,
                data,
            };
            cmd.execute().await?;
        },
        Commands::Predict { model, latitude, longitude, lead_seconds, lead_hours, output, format } => {
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
            };
            cmd.execute().await?;
        },
        Commands::Serve { host, port, model, workers, cors } => {
            let cmd = ServeCommand { host, port, model, workers, cors };
            cmd.execute().await?;
        },
        Commands::Eval { model, data, metrics } => {
            commands::eval_command(model, data, metrics).await?;
        },
        Commands::Export { model, format, output } => {
            commands::export_command(model, format, output).await?;
        },
        Commands::Info { model } => {
            commands::info_command(model).await?;
        },
    }
    
    Ok(())
}

fn init_logging(level: &str, verbose: bool) {
    use tracing_subscriber::{fmt, EnvFilter};
    
    let env_filter = if verbose {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("debug"))
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