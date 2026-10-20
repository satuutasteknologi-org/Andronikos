//! StatCard — Enterprise single statistic display overlay for map-hero layout

use leptos::*;

#[component]
pub fn StatCard(
    /// The label/name of the stat
    #[prop(into)]
    label: String,
    /// The main value to display
    #[prop(into)]
    value: String,
    /// Optional change indicator (+/- percentage)
    #[prop(optional)]
    change: Option<String>,
    /// Whether the change is positive (defaults to true)
    #[prop(optional)]
    is_positive: Option<bool>,
    /// Optional subtitle
    #[prop(optional)]
    subtitle: Option<String>,
) -> impl IntoView {
    let is_positive = is_positive.unwrap_or(true);
    let trend_class = if is_positive {
        "stat-trend-overlay positive"
    } else {
        "stat-trend-overlay negative"
    };
    let arrow = if is_positive { "▲" } else { "▼" };

    view! {
        <div class="stat-card-overlay">
            <div class="stat-label-overlay">{label}</div>
            <div class="stat-value-overlay">{value}</div>
            {move || change.as_ref().map(|c| {
                view! {
                    <div class={trend_class}>
                        <span class="stat-trend-arrow">{arrow}</span>
                        {c.clone()}
                    </div>
                }
            })}
            {move || subtitle.as_ref().map(|s| {
                view! {
                    <div class="text-xs text-text-muted">{s}</div>
                }
            })}
        </div>
    }
}
