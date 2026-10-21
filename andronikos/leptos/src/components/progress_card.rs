//! ProgressCard — Enterprise card with progress bar overlay for map-hero layout

use leptos::*;

#[component]
pub fn ProgressCard(
    /// Card title
    #[prop(into)]
    title: String,
    /// Progress percentage (0-100)
    #[prop(into)]
    progress: f32,
    /// Optional subtitle
    #[prop(optional)]
    subtitle: Option<String>,
    /// Optional status text
    #[prop(optional)]
    status: Option<String>,
) -> impl IntoView {
    let progress_clamped = progress.clamp(0.0, 100.0);

    view! {
        <div class="panel-enterprise">
            <div class="panel-header">
                <div class="card-title">{title}</div>
                {move || subtitle.as_ref().map(|s| {
                    view! { <div class="card-subtitle">{s}</div> }
                })}
            </div>
            <div class="progress-track">
                <div class="progress-fill" style:width=format!("{}%", progress_clamped)></div>
            </div>
            <div class="mt-2 flex justify-between text-xs text-text-muted">
                <span>{format!("{:.1}%", progress_clamped)}</span>
                <span>{status.unwrap_or_default()}</span>
            </div>
        </div>
    }
}
