//! SourceBadge — Enterprise source attribution badge overlay for map-hero layout

use leptos::*;

#[component]
pub fn SourceBadge(
    /// Source ID
    #[prop(into)]
    source_id: String,
    /// Reliability score (0.0 to 1.0)
    #[prop(optional)]
    reliability: Option<f32>,
    /// Whether the source is commercial-allowed
    #[prop(optional)]
    commercial_allowed: Option<bool>,
    /// Whether attribution is required
    #[prop(optional)]
    _attribution_required: Option<bool>,
) -> impl IntoView {
    let reliability_val = reliability.unwrap_or(0.0);
    let reliability_class = if reliability_val >= 0.9 {
        "high"
    } else if reliability_val >= 0.7 {
        "medium"
    } else {
        "low"
    };
    let tag_class = if commercial_allowed.unwrap_or(false) {
        "commercial-ok"
    } else {
        "no-commercial"
    };
    let tag_text = if commercial_allowed.unwrap_or(false) {
        "Commercial OK"
    } else {
        "No Commercial"
    };

    view! {
        <div class="source-badge-overlay">
            <div class="source-id-overlay">{source_id}</div>
            <div class=format!("source-reliability-overlay {}", reliability_class)>
                {format!("{:.0}%", reliability_val * 100.0)}
            </div>
            <span class=format!("source-tag-overlay {}", tag_class)>{tag_text}</span>
        </div>
    }
}
