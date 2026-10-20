//! Header — Kept for API compatibility. Navigation is now in index.html overlay.

use leptos::*;

#[component]
pub fn Header(
    /// Page title
    #[prop(optional)]
    _title: Option<String>,
) -> impl IntoView {
    view! { <div></div> }
}
