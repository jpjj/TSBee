//! Defined as here: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
use core::f64;

use super::{DistanceMetric, Point};
pub enum Geo {}
impl DistanceMetric<f64, i64> for Geo {
    fn compute(p1: &Point<f64>, p2: &Point<f64>) -> i64 {
        let lat1 = p1.0;
        let lon1 = p1.1;
        let lat2 = p2.0;
        let lon2 = p2.1;

        calculate_geo_distance(lat1, lon1, lat2, lon2)
    }
}

pub fn calculate_geo_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> i64 {
    const RRR: f64 = 6378.388;
    const PI: f64 = f64::consts::PI;

    let deg = lat1.trunc();
    let min = lat1 - deg;
    let lat1_rad = PI * (deg + 5.0 * min / 3.0) / 180.0;

    let deg = lon1.trunc();
    let min = lon1 - deg;
    let lon1_rad = PI * (deg + 5.0 * min / 3.0) / 180.0;

    let deg = lat2.trunc();
    let min = lat2 - deg;
    let lat2_rad = PI * (deg + 5.0 * min / 3.0) / 180.0;

    let deg = lon2.trunc();
    let min = lon2 - deg;
    let lon2_rad = PI * (deg + 5.0 * min / 3.0) / 180.0;

    let q1 = (lon1_rad - lon2_rad).cos();
    let q2 = (lat1_rad - lat2_rad).cos();
    let q3 = (lat1_rad + lat2_rad).cos();

    let arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3);
    let distance = RRR * arg.acos() + 1.0;
    distance as i64
}
