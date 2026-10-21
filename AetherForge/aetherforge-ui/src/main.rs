use leptos::*;
use crate::app::App;

mod app;

fn main() {
    // Initialize standard tracing for web
    tracing_wasm::set_as_global_default();
    
    // Mount the Leptos app to the body
    mount_to_body(|| {
        view! {
            <App/>
        }
    });
}
