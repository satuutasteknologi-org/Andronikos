//! StatusBadge — Enterprise status indicator overlay for map-hero layout

use leptos::*;

#[component]
pub fn StatusBadge(
    /// Status text
    #[prop(into)]
    label: String,
    /// Status type: "active", "warning", "error", "idle", "info"
    #[prop(optional)]
    status: Option<String>,
) -> impl IntoView {
    let (bg_class, text_class, dot_class) = match status.as_deref() {
        Some("active") | Some("ok") | Some("healthy") => {
            ("badge-success", "text-green-400", "status-dot active")
        }
        Some("warning") | Some("caution") => {
            ("badge-warning", "text-yellow-400", "status-dot warn")
        }
        Some("error") | Some("critical") | Some("danger") => {
            ("badge-error", "text-red-400", "status-dot error")
        }
        Some("idle") | Some("inactive") => ("badge-info", "text-blue-400", "status-dot active"),
        _ => ("badge-info", "text-cyan-400", "status-dot active"),
    };

    let badge_class = format!("badge {}", bg_class);

    view! {
        <span class=badge_class>
            <span class=dot_class></span>
            <span class=text_class>{label}</span>
        </span>
    }
}
