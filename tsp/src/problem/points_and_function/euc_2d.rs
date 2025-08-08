use super::{DistanceMetric, Point};
use std::ops::{Add, Mul, Sub};
pub enum Euc2d {}
impl<T, X> DistanceMetric<T, X> for Euc2d
where
    T: Copy + Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Into<f64>,
    f64: Into<X>,
{
    fn compute(p1: &Point<T>, p2: &Point<T>) -> X {
        let dx = p1.0 - p2.0;
        let dy = p1.1 - p2.1;

        let squared_distance = dx * dx + dy * dy;
        let distance = squared_distance.into().sqrt();

        distance.into()
    }
}
