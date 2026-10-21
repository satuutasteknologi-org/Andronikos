use leptos::*;
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = window)]
    fn initialize_deckgl(container_id: &str);

    #[wasm_bindgen(js_namespace = window)]
    fn update_deckgl_hexagons(data_json: &str);
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeatherNode {
    pub h3: String,
    pub color_r: u8,
    pub color_g: u8,
    pub color_b: u8,
    pub elevation: f64,
    pub lat: f64,
    pub lon: f64,
    pub mslp: f64,
    pub t2m: f64,
    pub u10: f64,
    pub v10: f64,
    pub tp: f64,
    pub z500: f64,
    pub q850: f64,
}

#[component]
pub fn App() -> impl IntoView {
    // UI State Signals
    let (is_running, set_is_running) = create_signal(false);
    let (active_nodes, set_active_nodes) = create_signal(0);
    let (inference_latency, set_inference_latency) = create_signal(0.0);
    let (_epoch, _set_epoch) = create_signal(1);
    let (coriolis_factor, _set_coriolis_factor) = create_signal(0.0000729);
    let (spectral_loss, set_spectral_loss) = create_signal(0.0142);
    let (wind_shear, set_wind_shear) = create_signal(12.4);
    let (gpu_vram, set_gpu_vram) = create_signal(42.1);

    // Target Telemetry Signal
    let (selected_target, set_selected_target) = create_signal::<Option<WeatherNode>>(None);

    // Initialize Deck.gl on mount with DOM-Ready Safety Delay
    create_effect(move |_| {
        set_timeout(move || {
            initialize_deckgl("deck-canvas-container");
            
            // 4. ENGINE STATE PERSISTENCE (F5 Resilience)
            let window = web_sys::window().unwrap();
            let storage = window.session_storage().unwrap().unwrap();
            if let Ok(Some(state)) = storage.get_item("aether_engine_running") {
                if state == "true" {
                    set_is_running.set(true);
                }
            }
        }, std::time::Duration::from_millis(50));
    });

    let (tick, set_tick) = create_signal(0);

    // Dummy Real-time Simulation Effect (Foolproof Tick Pattern)
    create_effect(move |_| {
        if is_running.get() {
            // Bergantung pada tick agar terus me-retrigger dirinya sendiri
            let _ = tick.get();

            set_active_nodes.update(|n| *n = 41920 + (js_sys::Math::random() * 500.0) as i32);
            set_inference_latency.update(|l| *l = 12.4 + (js_sys::Math::random() * 2.0));
            set_spectral_loss.update(|l| *l = 0.0142 + (js_sys::Math::random() * 0.001));
            set_wind_shear.update(|w| *w = 12.0 + (js_sys::Math::random() * 5.0));
            set_gpu_vram.update(|v| *v = 42.1 + (js_sys::Math::random() * 0.5));
            
            let mut data = Vec::new();
            
            // Membuat Grid Statis (Tidak berpindah tempat) untuk merepresentasikan
            // stasiun observasi/titik grid model di seluruh Indonesia
            for x in 0..25 { // Lon steps (95 to 140)
                for y in 0..20 { // Lat steps (-10 to 5)
                    let lat = -10.0 + (y as f64 * 0.75);
                    let lon = 95.0 + (x as f64 * 1.8);
                    
                    let intensity = js_sys::Math::random();
                    let t2m = 22.0 + (intensity * 12.0); // Suhu dinamis
                    let hex_id = format!("GRID-{}-{}", x, y);
                    
                    data.push(WeatherNode {
                        h3: hex_id,
                        color_r: (intensity * 255.0) as u8,
                        color_g: (50.0 + (1.0 - intensity) * 100.0) as u8,
                        color_b: (255.0 - intensity * 150.0) as u8,
                        elevation: 5000.0 + (intensity * 25000.0), // Elevasi dinamis (merepresentasikan badai)
                        lat,
                        lon,
                        mslp: 1005.0 + (js_sys::Math::random() * 15.0),
                        t2m,
                        u10: (js_sys::Math::random() * 20.0) - 10.0,
                        v10: (js_sys::Math::random() * 20.0) - 10.0,
                        tp: if js_sys::Math::random() > 0.8 { js_sys::Math::random() * 45.0 } else { 0.0 },
                        z500: 5500.0 + (js_sys::Math::random() * 300.0),
                        q850: 10.0 + (js_sys::Math::random() * 8.0)
                    });
                }
            }
            
            if let Ok(json) = serde_json::to_string(&data) {
                update_deckgl_hexagons(&json);
            }

            // Jadwalkan iterasi (frame) selanjutnya 1 detik kemudian
            set_timeout(move || {
                set_tick.update(|t| *t += 1);
            }, std::time::Duration::from_millis(1000));
        }
    });

    let toggle_engine = move |_| {
        set_is_running.update(|r| *r = !*r);
        // Simpan status engine agar tidak mati saat di F5
        let window = web_sys::window().unwrap();
        let storage = window.session_storage().unwrap().unwrap();
        let _ = storage.set_item("aether_engine_running", &is_running.get().to_string());
    };

    let trigger_hex_update = move |_| {
        let window = web_sys::window().unwrap();
        if let Ok(val) = js_sys::Reflect::get(&window, &JsValue::from_str("AETHER_SELECTED_HEX")) {
            if val.is_object() {
                if let Ok(json_str) = js_sys::JSON::stringify(&val) {
                    if let Some(json_string) = json_str.as_string() {
                        if let Ok(node) = serde_json::from_str::<WeatherNode>(&json_string) {
                            set_selected_target.set(Some(node));
                            return;
                        }
                    }
                }
            }
        }
        set_selected_target.set(None);
    };

    view! {
        <div class="ui-layer">
            // LEFT PANEL: Core Control & Engine Status
            <div class="left-panel-container">
                <div class="panel controls-panel">
                    <div class="branding">
                        <h1>"AetherForge"</h1>
                        <p>"Omniverse Weather Engine"</p>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">"H3 Mesh Nodes"</div>
                            <div class="metric-value">{active_nodes}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">"GNN Latency"</div>
                            <div class="metric-value">{move || format!("{:.1}ms", inference_latency.get())}</div>
                        </div>
                    </div>
                    
                    <button class="btn-primary" on:click=toggle_engine>
                        {move || if is_running.get() { "PAUSE INFERENCE" } else { "ENGAGE AETHERFORGE" }}
                    </button>
                </div>
                
                <div class="panel">
                    <div class="live-indicator" style="margin-bottom: 16px;">
                        <div class="live-dot" style=move || if is_running.get() { "background-color: #00ff88; box-shadow: 0 0 8px #00ff88;" } else { "background-color: #ff3366; box-shadow: 0 0 8px #ff3366; animation: none;" }></div>
                        {move || if is_running.get() { "DATALINK: SECURE (10Gbps)" } else { "DATALINK: OFFLINE" }}
                    </div>
                    <div class="stat-row">
                        <span class="stat-title">"SwiGLU Kernel Fusion"</span>
                        <span class="stat-val" style="color: #00ff88;">"ACTIVE"</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-title">"Linear Attention O(N)"</span>
                        <span class="stat-val">"PERFORMER"</span>
                    </div>
                    <div class="stat-row" style="border: none;">
                        <span class="stat-title">"VRAM Allocation"</span>
                        <span class="stat-val">{move || format!("{:.1} GB", gpu_vram.get())}</span>
                    </div>
                    <div class="stat-graph">
                        <div class="graph-fill" style=move || format!("width: {}%;", gpu_vram.get() / 80.0 * 100.0)></div>
                    </div>
                </div>
            </div>
            
            // RIGHT PANEL: Advanced Physics Diagnostics
            <div class="right-panel-container">
                <div class="panel advanced-metrics-panel">
                    <div class="branding" style="margin-bottom: 16px;">
                        <h1 style="font-size: 20px; color: #fff; background: none; -webkit-text-fill-color: #fff;">"Atmospheric Diagnostics"</h1>
                        <p>"Real-Time Physics Evaluation"</p>
                    </div>

                    <div class="stat-row">
                        <span class="stat-title">"Spectral Variance Loss"</span>
                        <span class="stat-val">{move || format!("{:.5}", spectral_loss.get())}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-title">"Coriolis Force Modulator"</span>
                        <span class="stat-val">{move || format!("{:.7} rad/s", coriolis_factor.get())}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-title">"Neural ODE Advection"</span>
                        <span class="stat-val">"30s STEP"</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-title">"Vertical Wind Shear (dZ)"</span>
                        <span class="stat-val">{move || format!("{:.1} m/s", wind_shear.get())}</span>
                    </div>
                    <div class="stat-row" style="border: none;">
                        <span class="stat-title">"Adaptive Mesh Refinement"</span>
                        <span class="stat-val" style="color: #00ff88;">"OPTIMIZED"</span>
                    </div>
                </div>
            </div>

            // HIDDEN BUTTON TO BRIDGE JS DOM EVENT TO WASM
            <button id="trigger-hex-update" style="display:none;" on:click=trigger_hex_update></button>

            // TARGET TELEMETRY HUD (Appears when Hexagon is Clicked)
            {move || {
                selected_target.get().map(|target| {
                    // Kalkulasi angin
                    let wind_speed = (target.u10.powi(2) + target.v10.powi(2)).sqrt();
                    let wind_dir = target.u10.atan2(target.v10).to_degrees();
                    let wind_dir_norm = if wind_dir < 0.0 { wind_dir + 360.0 } else { wind_dir };
                    
                    let temp_style = if target.t2m > 30.0 { "color: #ff4444" } else { "color: #00ff88" };
                    let precip_style = if target.tp > 0.0 { "color: #00d2ff" } else { "" };
                    let precip_text = if target.tp > 0.0 { format!("{:.1} mm/hr", target.tp) } else { "0.0 mm/hr (CLEAR)".to_string() };

                    view! {
                        <div class="target-telemetry-hud">
                            <div class="hud-header">
                                <h2>"TARGET ACQUIRED"</h2>
                                <p class="hud-h3">"H3 INDEX: " {target.h3}</p>
                            </div>
                            <div class="hud-body">
                                <div class="hud-col">
                                    <div class="hud-stat">
                                        <span class="hud-label">"COORDINATES"</span>
                                        <span class="hud-val">{format!("{:.4}°, {:.4}°", target.lat, target.lon)}</span>
                                    </div>
                                    <div class="hud-stat">
                                        <span class="hud-label">"MSLP (PRESSURE)"</span>
                                        <span class="hud-val">{format!("{:.1} hPa", target.mslp)}</span>
                                    </div>
                                    <div class="hud-stat">
                                        <span class="hud-label">"Z500 GEOPOTENTIAL"</span>
                                        <span class="hud-val">{format!("{:.0} m", target.z500)}</span>
                                    </div>
                                </div>
                                <div class="hud-col">
                                    <div class="hud-stat">
                                        <span class="hud-label">"SURFACE TEMP"</span>
                                        <span class="hud-val" style=temp_style>
                                            {format!("{:.1} °C", target.t2m)}
                                        </span>
                                    </div>
                                    <div class="hud-stat">
                                        <span class="hud-label">"10M WIND VECTOR"</span>
                                        <span class="hud-val">{format!("{:.1} m/s @ {:.0}°", wind_speed, wind_dir_norm)}</span>
                                    </div>
                                    <div class="hud-stat">
                                        <span class="hud-label">"PRECIPITATION"</span>
                                        <span class="hud-val" style=precip_style>
                                            {precip_text}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    }
                })
            }}

        </div>
    }
}
