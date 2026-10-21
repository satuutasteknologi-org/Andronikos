//! RiskPanel — Enterprise hazard risk display panel overlay for map-hero layout

use leptos::*;

#[component]
pub fn RiskPanel(
    /// Optional landslide risk level
    #[prop(optional)]
    landslide_level: Option<String>,
    /// Optional landslide probability
    #[prop(optional)]
    landslide_prob: Option<f32>,
    /// Optional flood risk level
    #[prop(optional)]
    flood_level: Option<String>,
    /// Optional flood probability
    #[prop(optional)]
    flood_prob: Option<f32>,
    /// Optional smoke risk level
    #[prop(optional)]
    smoke_level: Option<String>,
    /// Optional smoke probability
    #[prop(optional)]
    smoke_prob: Option<f32>,
) -> impl IntoView {
    let landslide_level_val = landslide_level.unwrap_or_else(|| "hijau".to_string());
    let flood_level_val = flood_level.unwrap_or_else(|| "kuning".to_string());
    let smoke_level_val = smoke_level.unwrap_or_else(|| "hijau".to_string());
    let landslide_prob_val = landslide_prob.unwrap_or(0.12);
    let flood_prob_val = flood_prob.unwrap_or(0.34);
    let smoke_prob_val = smoke_prob.unwrap_or(0.08);

    fn risk_level_class(level: &str) -> &'static str {
        match level {
            "merah" => "merah",
            "kuning" => "kuning",
            "hijau" => "hijau",
            _ => "hijau",
        }
    }

    view! {
        <div class="risk-grid-overlay">
            <RiskCardOverlay
                title="Longsor"
                level=landslide_level_val.clone()
                prob=landslide_prob_val
                level_class=risk_level_class(&landslide_level_val)
            />
            <RiskCardOverlay
                title="Banjir"
                level=flood_level_val.clone()
                prob=flood_prob_val
                level_class=risk_level_class(&flood_level_val)
            />
            <RiskCardOverlay
                title="Asap"
                level=smoke_level_val.clone()
                prob=smoke_prob_val
                level_class=risk_level_class(&smoke_level_val)
            />
        </div>
    }
}

#[component]
fn RiskCardOverlay(
    title: &'static str,
    level: String,
    prob: f32,
    level_class: &'static str,
) -> impl IntoView {
    let bar_color_class = match prob {
        v if v > 0.6 => "merah",
        v if v > 0.3 => "kuning",
        _ => "hijau",
    };

    view! {
        <div class="risk-card-overlay">
            <div class="risk-header-overlay">
                <div class="risk-title-overlay">{title}</div>
                <div class=format!("risk-level-overlay {}", level_class)>
                    {level.to_uppercase()}
                </div>
            </div>
            <div class="risk-prob-bar-overlay">
                <div
                    class=format!("risk-prob-fill-overlay {}", bar_color_class)
                    style:width=format!("{}%", prob * 100.0)
                ></div>
            </div>
            <div class="risk-prob-text-overlay">
                {format!("P = {:.0} %", prob * 100.0)}
            </div>
        </div>
    }
}
