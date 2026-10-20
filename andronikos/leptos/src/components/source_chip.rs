//! SourceChip — Enterprise source badge chip overlay for map-hero layout
use leptos::*;

#[component]
pub fn SourceChip(#[prop(into)] name: String) -> impl IntoView {
    view! {
        <span class="px-2 py-1 bg-surface border border-ln text-xs rounded text-text-muted">
            {name}
        </span>
    }
}
