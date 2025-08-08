use super::{DistanceMetric, Point};
use std::ops::{Add, Mul, Sub};
pub enum Geo {}
impl<T, X> DistanceMetric<T, X> for Geo
where
    T: Copy + Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Into<f64>,
    f64: Into<X>,
{
    fn compute(p1: &Point<T>, p2: &Point<T>) -> X {
        let lat1 = p1.0.into();
        let lon1 = p1.1.into();
        let lat2 = p2.0.into();
        let lon2 = p2.1.into();

        let distance = calculate_geo_distance(lat1, lon1, lat2, lon2) as f64;
        distance.into()
    }
}

pub fn calculate_geo_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> i32 {
    const RRR: f64 = 6378.388;
    const PI: f64 = std::f64::consts::PI;

    let deg = lat1.floor();
    let min = lat1 - deg;
    let lat1_rad = PI * (deg + 5.0 * min / 3.0) / 180.0;

    let deg = lon1.floor();
    let min = lon1 - deg;
    let lon1_rad = PI * (deg + 5.0 * min / 3.0) / 180.0;

    let deg = lat2.floor();
    let min = lat2 - deg;
    let lat2_rad = PI * (deg + 5.0 * min / 3.0) / 180.0;

    let deg = lon2.floor();
    let min = lon2 - deg;
    let lon2_rad = PI * (deg + 5.0 * min / 3.0) / 180.0;

    let q1 = (lon1_rad - lon2_rad).cos();
    let q2 = (lat1_rad - lat2_rad).cos();
    let q3 = (lat1_rad + lat2_rad).cos();

    let arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3) + 1.0;
    let distance = RRR * arg.acos();
    distance as i32
}
