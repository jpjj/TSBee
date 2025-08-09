//! Defined as here: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf

use super::{DistanceMetric, Point};

pub enum ATT {}
impl DistanceMetric<f64, i64> for ATT {
    fn compute(p1: &Point<f64>, p2: &Point<f64>) -> i64 {
        let dx = p1.0 - p2.0;
        let dy = p1.1 - p2.1;

        let squared_distance = (dx * dx + dy * dy) / 10.0;
        let r = squared_distance.sqrt();

        let t = r.round();
        let distance = if t < r { t + 1.0 } else { t };
        distance as i64
    }
}
