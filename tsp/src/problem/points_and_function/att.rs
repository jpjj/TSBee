pub enum ATT {}
use std::ops::{Add, Mul, Sub};

use super::{DistanceMetric, Point};

impl<T, X> DistanceMetric<T, X> for ATT
where
    T: Copy + Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Into<f64>,
    f64: Into<X>,
{
    fn compute(p1: &Point<T>, p2: &Point<T>) -> X {
        let dx = p1.0 - p2.0;
        let dy = p1.1 - p2.1;

        let squared_distance = dx * dx + dy * dy;
        let r = squared_distance.into().sqrt();

        let t = r.round();
        let distance = if t < r { t + 1.0 } else { t };

        distance.into()
    }
}
