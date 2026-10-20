//! Breadcrumb — Enterprise navigation breadcrumb trail

use leptos::*;

#[component]
pub fn Breadcrumb(
    /// List of (label, href) tuples
    items: Vec<(String, Option<String>)>,
) -> impl IntoView {
    view! {
        <nav class="flex items-center gap-2 text-xs text-text-muted">
            {move || {
                items.iter().enumerate().map(|(i, (label, href))| {
                    let is_last = i == items.len() - 1;
                    let label_clone = label.clone();
                    let href_clone = href.clone();
                    view! {
                        <span>
                            {move || {
                                if let Some(href) = &href_clone {
                                    view! {
                                        <a href=href.clone() class="text-cyan-400 hover:text-cyan-300">{label_clone.clone()}</a>
                                    }.into_any()
                                } else {
                                    view! { <span class="text-text-dim">{label_clone.clone()}</span> }.into_any()
                                }
                            }}
                            {move || if !is_last {
                                view! { <span class="mx-1">"\u{203a}"</span> }
                            } else {
                                view! { <span></span> }
                            }}
                        </span>
                    }
                }).collect::<Vec<_>>()
            }}
        </nav>
    }
}
