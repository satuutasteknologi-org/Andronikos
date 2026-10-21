//! ForecastPanel — Enterprise weather forecast panel overlay for map-hero layout

use leptos::*;

const DEGREE_C: &str = "\u{00b0}C";

#[component]
pub fn ForecastPanel(
    #[prop(optional)] temperature: Option<f32>,
    #[prop(optional)] humidity: Option<f32>,
    #[prop(optional)] precipitation: Option<f32>,
    #[prop(optional)] wind_speed: Option<f32>,
    #[prop(optional)] wind_direction: Option<f32>,
    #[prop(optional)] pressure: Option<f32>,
    #[prop(optional)] prob_rain: Option<f32>,
    #[prop(optional)] lead_hours: Option<u32>,
    #[prop(optional)] model_version: Option<String>,
    #[prop(optional)] sources: Option<Vec<String>>,
) -> impl IntoView {
    let temp = temperature.unwrap_or(28.4);
    let humidity_val = humidity.unwrap_or(78.0);
    let precip_val = precipitation.unwrap_or(12.4);
    let wind_speed_val = wind_speed.unwrap_or(14.2);
    let _wind_dir_val = wind_direction.unwrap_or(245.0);
    let pressure_val = pressure.unwrap_or(1012.0);
    let prob_rain_val = prob_rain.unwrap_or(0.68);
    let lead_hours_val = lead_hours.unwrap_or(3);
    let model_version_val = model_version.unwrap_or_else(|| "GraphCast-1.0.0".to_string());
    let sources_val = sources.unwrap_or_else(|| {
        vec![
            "met.gfs".to_string(),
            "met.gefs".to_string(),
            "obs.bmkg_stn".to_string(),
        ]
    });

    let precip_class = if prob_rain_val > 0.5 {
        "text-red-400 font-bold"
    } else if prob_rain_val > 0.2 {
        "text-yellow-400 font-bold"
    } else {
        "text-green-400 font-bold"
    };

    // Pre-build source chips as static HTML to avoid closure inside view!
    let source_chips: Vec<_> = sources_val
        .iter()
        .map(|s| {
            view! {
                <span class="px-2 py-1 bg-surface border border-ln text-xs rounded text-text-muted">
                    {s.clone()}
                </span>
            }
        })
        .collect();

    view! {
        <div class="forecast-grid-overlay">
            <div class="forecast-item-overlay">
                <div class="forecast-info-overlay">
                    <div class="forecast-label-overlay">Suhu</div>
                    <div class="forecast-value-overlay text-green-400">{format!("{:.1}", temp)}</div>
                    <div class="forecast-unit-overlay">{DEGREE_C}</div>
                </div>
                <div class="forecast-icon-overlay">"🌡"</div>
            </div>
            <div class="forecast-item-overlay">
                <div class="forecast-info-overlay">
                    <div class="forecast-label-overlay">Kelembapan</div>
                    <div class="forecast-value-overlay text-cyan-400">{format!("{:.0}", humidity_val)}</div>
                    <div class="forecast-unit-overlay">%</div>
                </div>
                <div class="forecast-icon-overlay">"💧"</div>
            </div>
            <div class="forecast-item-overlay">
                <div class="forecast-info-overlay">
                    <div class="forecast-label-overlay">Curah Hujan</div>
                    <div class={precip_class}>{format!("{:.1}", precip_val)}</div>
                    <div class="forecast-unit-overlay">mm</div>
                </div>
                <div class="forecast-icon-overlay">"🌧"</div>
            </div>
            <div class="forecast-item-overlay">
                <div class="forecast-info-overlay">
                    <div class="forecast-label-overlay">Angin</div>
                    <div class="forecast-value-overlay text-text">{format!("{:.1}", wind_speed_val)}</div>
                    <div class="forecast-unit-overlay">m/s</div>
                </div>
                <div class="forecast-icon-overlay">"💨"</div>
            </div>
            <div class="forecast-item-overlay">
                <div class="forecast-info-overlay">
                    <div class="forecast-label-overlay">Tekanan Udara</div>
                    <div class="forecast-value-overlay text-text">{format!("{:.0}", pressure_val)}</div>
                    <div class="forecast-unit-overlay">hPa</div>
                </div>
                <div class="forecast-icon-overlay">"📊"</div>
            </div>
            <div class="forecast-item-overlay">
                <div class="forecast-info-overlay">
                    <div class="forecast-label-overlay">Probabilitas Hujan</div>
                    <div class={precip_class}>{format!("{:.0}", prob_rain_val * 100.0)}</div>
                    <div class="forecast-unit-overlay">%</div>
                </div>
                <div class="forecast-icon-overlay">"☔"</div>
            </div>
        </div>

        <div class="mt-4 border-t border-ln pt-4">
            <div class="flex flex-wrap gap-2">
                {source_chips}
            </div>
            <div class="mt-2 text-[10px] text-text-muted">
                {format!("Lead Time: {} jam / Model: {}", lead_hours_val, model_version_val)}
            </div>
        </div>
    }
}
