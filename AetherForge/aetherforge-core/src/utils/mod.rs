//! # Utilities Module
//!
//! Common utility functions for AetherForge.

mod geo_utils {
    use crate::GeoCoord;

    pub fn interpolate_coord(c1: &GeoCoord, c2: &GeoCoord, t: f64) -> GeoCoord {
        GeoCoord::new(
            c1.lat + (c2.lat - c1.lat) * t,
            c1.lon + (c2.lon - c1.lon) * t,
        )
    }

    pub fn bearing(c1: &GeoCoord, c2: &GeoCoord) -> f64 {
        let dlon = (c2.lon - c1.lon).to_radians();
        let lat1 = c1.lat.to_radians();
        let lat2 = c2.lat.to_radians();

        let y = dlon.sin() * lat2.cos();
        let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();

        y.atan2(x).to_degrees()
    }

    pub fn destination(c: &GeoCoord, bearing: f64, dist_km: f64) -> GeoCoord {
        let earth_radius_km = 6371.0;
        let angular_dist = dist_km / earth_radius_km;
        let bearing_rad = bearing.to_radians();

        let lat1 = c.lat.to_radians();
        let lon1 = c.lon.to_radians();

        let lat2 = (lat1.sin() * angular_dist.cos()
            + lat1.cos() * angular_dist.sin() * bearing_rad.cos())
        .asin();

        let lon2 = lon1
            + (bearing_rad.sin() * lat1.cos() * angular_dist.cos()).atan2(
                1.0 - (lat1.sin() * lat2.sin() + lat1.cos() * lat2.cos() * bearing_rad.cos())
                    .powf(2.0),
            );

        GeoCoord::new(lat2.to_degrees(), lon2.to_degrees())
    }

    pub fn midpoint(c1: &GeoCoord, c2: &GeoCoord) -> GeoCoord {
        interpolate_coord(c1, c2, 0.5)
    }

    pub fn distance_km(c1: &GeoCoord, c2: &GeoCoord) -> f64 {
        let earth_radius_km = 6371.0;
        let dlat = (c2.lat - c1.lat).to_radians();
        let dlon = (c2.lon - c1.lon).to_radians();

        let a = (dlat / 2.0).sin().powi(2)
            + c1.lat.to_radians().cos() * c2.lat.to_radians().cos() * (dlon / 2.0).sin().powi(2);

        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt().max(1e-10));

        earth_radius_km * c
    }
}

pub use geo_utils::{bearing, destination, distance_km, interpolate_coord, midpoint};

pub mod time {
    use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Timelike, Utc};

    pub fn hours_ahead(hours: i64) -> DateTime<Utc> {
        Utc::now() + Duration::hours(hours)
    }

    pub fn hours_ago(hours: i64) -> DateTime<Utc> {
        Utc::now() - Duration::hours(hours)
    }

    pub fn days_ago(days: i64) -> DateTime<Utc> {
        Utc::now() - Duration::days(days)
    }

    pub fn start_of_day(dt: DateTime<Utc>) -> DateTime<Utc> {
        let naive = dt.naive_utc();
        let start = NaiveDate::from_ymd_opt(naive.year(), naive.month(), naive.day())
            .unwrap_or(naive.date());
        start
            .and_hms_opt(0, 0, 0)
            .map(|nd| Utc.from_utc_datetime(&nd))
            .unwrap_or(dt)
    }

    pub fn end_of_day(dt: DateTime<Utc>) -> DateTime<Utc> {
        let naive = dt.naive_utc();
        let start = NaiveDate::from_ymd_opt(naive.year(), naive.month(), naive.day())
            .unwrap_or(naive.date());
        start
            .and_hms_opt(23, 59, 59)
            .map(|nd| Utc.from_utc_datetime(&nd))
            .unwrap_or(dt)
    }

    pub fn start_of_month(dt: DateTime<Utc>) -> DateTime<Utc> {
        let naive = dt.naive_utc();
        let first_day =
            NaiveDate::from_ymd_opt(naive.year(), naive.month(), 1).unwrap_or(naive.date());
        first_day
            .and_hms_opt(0, 0, 0)
            .map(|nd| Utc.from_utc_datetime(&nd))
            .unwrap_or(dt)
    }

    pub fn is_same_hour(a: DateTime<Utc>, b: DateTime<Utc>) -> bool {
        a.year() == b.year() && a.month() == b.month() && a.day() == b.day() && a.hour() == b.hour()
    }

    pub fn hourly_range(start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<DateTime<Utc>> {
        let mut times = Vec::new();
        let mut current = start_of_day(start);

        while current <= end {
            if current >= start {
                times.push(current);
            }
            current = current + Duration::hours(1);
        }

        times
    }

    pub fn daily_range(start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<DateTime<Utc>> {
        let mut times = Vec::new();
        let mut current = start_of_day(start);

        while current <= end {
            if current >= start {
                times.push(current);
            }
            current = current + Duration::days(1);
        }

        times
    }

    pub fn format_rfc3339(dt: DateTime<Utc>) -> String {
        dt.to_rfc3339()
    }

    pub fn parse_rfc3339(s: &str) -> Option<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    }
}

pub mod str {
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len.saturating_sub(3)])
        }
    }

    pub fn to_snake_case(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap_or(c));
        }
        result
    }

    pub fn to_kebab_case(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                result.push('-');
            }
            result.push(c.to_lowercase().next().unwrap_or(c));
        }
        result
    }
}

pub mod validation {
    pub fn is_valid_latitude(lat: f64) -> bool {
        (-90.0..=90.0).contains(&lat)
    }

    pub fn is_valid_longitude(lon: f64) -> bool {
        (-180.0..=180.0).contains(&lon)
    }

    pub fn is_valid_coordinate(lat: f64, lon: f64) -> bool {
        is_valid_latitude(lat) && is_valid_longitude(lon)
    }

    pub fn is_valid_probability(p: f64) -> bool {
        (0.0..=1.0).contains(&p)
    }

    pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }
}
