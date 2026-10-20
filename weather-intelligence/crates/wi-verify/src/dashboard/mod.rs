//! # Verification Dashboard
//!
//! Export metrics for visualization and monitoring.

use std::collections::HashMap;
use std::path::PathBuf;

use tracing::info;
use wi_core::{Result, WIError, WeatherVariable};

#[derive(Debug, Clone)]
pub struct DashboardConfig {
    pub output_path: PathBuf,
    pub generate_html: bool,
    pub generate_json: bool,
    pub generate_prometheus: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("/var/weather-intelligence/verification"),
            generate_html: true,
            generate_json: true,
            generate_prometheus: true,
        }
    }
}

/// Verification dashboard
pub struct VerificationDashboard {
    config: DashboardConfig,
}

impl VerificationDashboard {
    pub fn new(config: DashboardConfig) -> Self {
        std::fs::create_dir_all(&config.output_path).ok();
        Self { config }
    }

    pub fn export_metrics(
        &self,
        metrics: &HashMap<WeatherVariable, crate::metrics::VerificationMetrics>,
    ) -> Result<()> {
        if self.config.generate_json {
            self.export_json(metrics)?;
        }
        if self.config.generate_html {
            self.export_html(metrics)?;
        }
        if self.config.generate_prometheus {
            self.export_prometheus(metrics)?;
        }
        Ok(())
    }

    fn export_json(
        &self,
        metrics: &HashMap<WeatherVariable, crate::metrics::VerificationMetrics>,
    ) -> Result<()> {
        let path = self.config.output_path.join("metrics.json");
        let json = serde_json::to_string_pretty(metrics).map_err(|e| {
            WIError::serialization_with_source("Failed to serialize metrics", e.into())
        })?;
        std::fs::write(&path, json)
            .map_err(|e| WIError::storage_with_source("Failed to write metrics", e.into()))?;
        info!("Exported metrics JSON to {:?}", path);
        Ok(())
    }

    fn export_html(
        &self,
        metrics: &HashMap<WeatherVariable, crate::metrics::VerificationMetrics>,
    ) -> Result<()> {
        let path = self.config.output_path.join("dashboard.html");

        let html = self.generate_html(metrics);
        std::fs::write(&path, html)
            .map_err(|e| WIError::storage_with_source("Failed to write HTML", e.into()))?;
        info!("Exported dashboard HTML to {:?}", path);
        Ok(())
    }

    fn generate_html(
        &self,
        metrics: &HashMap<WeatherVariable, crate::metrics::VerificationMetrics>,
    ) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Weather Verification Dashboard</title>
    <style>
        body {{ font-family: monospace; margin: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00ff88; }}
        .metric-card {{ background: #16213e; padding: 15px; margin: 10px; border-radius: 8px; }}
        .metric-value {{ font-size: 24px; color: #00ff88; }}
        .metric-label {{ color: #888; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #00ff88; }}
    </style>
</head>
<body>
    <h1>🌤️ Weather Verification Dashboard</h1>
    <p>Generated: {}</p>
    {}
</body>
</html>
"#,
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            metrics
                .iter()
                .map(|(var, m)| {
                    format!(
                        r#"
<div class="metric-card">
    <div class="metric-label">{:?} - Deterministic</div>
    <div class="metric-value">MAE: {:.2} | RMSE: {:.2} | Bias: {:.2} | Corr: {:.3}</div>
    <div class="metric-label">Probabilistic</div>
    <div class="metric-value">Brier: {:.4} | CRPS: {:.4}</div>
</div>
"#,
                        var,
                        m.deterministic.mae,
                        m.deterministic.rmse,
                        m.deterministic.bias,
                        m.deterministic.correlation,
                        m.probabilistic.brier_score,
                        m.ensemble.as_ref().map(|e| e.crps).unwrap_or(0.0)
                    )
                })
                .collect::<Vec<_>>()
                .join("")
        )
    }

    fn export_prometheus(
        &self,
        metrics: &HashMap<WeatherVariable, crate::metrics::VerificationMetrics>,
    ) -> Result<()> {
        let path = self.config.output_path.join("metrics.prom");
        let mut lines = Vec::new();

        for (var, m) in metrics {
            let var_name = format!("{:?}", var).to_lowercase();
            lines.push(format!(
                "weather_mae{{variable=\"{}\"}} {}",
                var_name, m.deterministic.mae
            ));
            lines.push(format!(
                "weather_rmse{{variable=\"{}\"}} {}",
                var_name, m.deterministic.rmse
            ));
            lines.push(format!(
                "weather_bias{{variable=\"{}\"}} {}",
                var_name, m.deterministic.bias
            ));
            lines.push(format!(
                "weather_correlation{{variable=\"{}\"}} {}",
                var_name, m.deterministic.correlation
            ));
            lines.push(format!(
                "weather_brier_score{{variable=\"{}\"}} {}",
                var_name, m.probabilistic.brier_score
            ));
            if let Some(ens) = &m.ensemble {
                lines.push(format!(
                    "weather_crps{{variable=\"{}\"}} {}",
                    var_name, ens.crps
                ));
            }
        }

        std::fs::write(&path, lines.join("\n")).map_err(|e| {
            WIError::storage_with_source("Failed to write Prometheus metrics", e.into())
        })?;
        info!("Exported Prometheus metrics to {:?}", path);
        Ok(())
    }
}
