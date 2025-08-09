//! Defined as here: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf

use super::{DistanceMetric, Point};
pub enum Ceil2d {}
impl DistanceMetric<f64, i64> for Ceil2d {
    fn compute(p1: &Point<f64>, p2: &Point<f64>) -> i64 {
        let dx = p1.0 - p2.0;
        let dy = p1.1 - p2.1;

        let squared_distance = dx * dx + dy * dy;
        let distance = squared_distance.sqrt();

        distance.ceil() as i64
    }
}
