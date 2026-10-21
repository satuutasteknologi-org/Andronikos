use leptos::*;
use wasm_bindgen::JsCast;
use web_sys::{window, HtmlElement};

fn main() {
    _ = console_log::init_with_level(log::Level::Debug);
    console_error_panic_hook::set_once();

    let document = window().unwrap().document().unwrap();
    let mount_point = document
        .get_element_by_id("leptos-root")
        .expect("#leptos-root element to exist in index.html")
        .unchecked_into::<HtmlElement>();

    leptos::mount_to(mount_point, move || {
        view! { <andronikos_ui::App /> }
    });
}
