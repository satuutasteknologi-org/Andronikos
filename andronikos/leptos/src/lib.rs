//! # Andronikos Frontend — Leptos + MapLibre
//! Weather Intelligence UI untuk Indonesia.
//! Map-as-Pure-Hero: full-screen MapLibre GL JS rendered from static index.html.
//! Leptos mounts to #leptos-root for UI overlays.
//! Compliance: UU 31/2009 Pasal 36 ayat 1 — informasi risiko, analisis pendukung, prakiraan berbasis probabilitas.

use leptos::*;

mod components;
mod styles;

pub use components::breadcrumb::Breadcrumb;
pub use components::forecast_panel::ForecastPanel;
pub use components::header::Header;
pub use components::map::Map;
pub use components::panel::Panel;
pub use components::progress_card::ProgressCard;
pub use components::risk_panel::RiskPanel;
pub use components::sidebar::Sidebar;
pub use components::source_badge::SourceBadge;
pub use components::source_chip::SourceChip;
pub use components::stat_card::StatCard;
pub use components::stat_card_widget::StatCardWidget;
pub use components::status_badge::StatusBadge;

#[component]
pub fn App() -> impl IntoView {
    view! {
        <>
            {/* Leptos UI overlays rendered into #leptos-root (static HTML) */}
            {/* Map and all MapLibre UI are in static index.html body */}
        </>
    }
}
