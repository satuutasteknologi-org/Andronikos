//! Sidebar — Kept for API compatibility. No longer rendered (replaced by overlay-panel).

use leptos::*;

#[component]
pub fn Sidebar(
    /// Whether the sidebar is open (for mobile)
    #[prop(optional)]
    _is_open: Option<bool>,
) -> impl IntoView {
    view! { <div></div> }
}
