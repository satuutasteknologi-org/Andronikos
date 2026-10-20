//! StatCardWidget — Enterprise stat card overlay for map-hero layout

use leptos::*;

#[component]
pub fn StatCardWidget(
    /// Card title
    #[prop(into)]
    title: String,
    /// Main value to display
    #[prop(into)]
    value: String,
    /// Optional trend indicator
    #[prop(optional)]
    trend: Option<String>,
    /// Optional trend direction (true = positive, defaults to true)
    #[prop(optional)]
    is_positive: Option<bool>,
    /// Optional subtitle (reserved for future use)
    #[prop(optional)]
    _subtitle: Option<String>,
    /// Optional icon (emoji or text)
    #[prop(optional)]
    icon: Option<String>,
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
            <div class="stat-header">
                <div class="stat-label-overlay">{title}</div>
                {icon.as_ref().map(|icon| {
                    view! {
                        <div class="stat-icon-overlay">{icon}</div>
                    }
                })}
            </div>
            <div class="stat-value-overlay">{value}</div>
            {trend.as_ref().map(move |t| {
                view! {
                    <div class={trend_class}>
                        <span class="stat-trend-arrow">{arrow}</span>
                        {t}
                    </div>
                }
            })}
        </div>
    }
}
