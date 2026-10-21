//! Panel — Enterprise generic container card overlay for map-hero layout

use leptos::*;

#[component]
pub fn Panel(
    /// Optional title for the panel header
    #[prop(optional)]
    title: Option<String>,
    /// Optional subtitle for the panel header
    #[prop(optional)]
    subtitle: Option<String>,
    /// Children content
    children: Children,
) -> impl IntoView {
    let title_clone = title.clone();
    let subtitle_clone = subtitle.clone();
    let show_header = title_clone.is_some() || subtitle_clone.is_some();

    view! {
        <div class="panel-enterprise">
            {if show_header {
                view! {
                    <div class="panel-header">
                        <div class="panel-title">{title_clone.clone().unwrap_or_default()}</div>
                        {subtitle_clone.as_ref().map(|s| {
                            view! { <div class="card-subtitle">{s.clone()}</div> }
                        })}
                    </div>
                }
            } else {
                view! { <div></div> }
            }}
            <div>
                {children()}
            </div>
        </div>
    }
}
