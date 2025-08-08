use super::{DistanceMetric, Point};
use std::cmp::max;
use std::ops::{Add, Mul, Sub};
pub enum Max2d {}
impl<T, X> DistanceMetric<T, X> for Max2d
where
    T: Copy + Ord + Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Into<f64>,
    f64: Into<X>,
{
    fn compute(p1: &Point<T>, p2: &Point<T>) -> X {
        let dx = max(p1.0 - p2.0, p2.0 - p1.0);
        let dy = max(p1.1 - p2.1, p2.1 - p1.1);

        let distance = max(dx, dy).into();

        distance.into()
    }
}
