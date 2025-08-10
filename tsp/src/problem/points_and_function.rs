pub mod att;
pub mod ceil_2d;
pub mod euc_2d;
pub mod geo;
pub mod man_2d;
pub mod max_2d;
use std::marker::PhantomData;

use crate::{city::City, problem::Problem};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point<T>(pub T, pub T);

pub trait DistanceMetric<T, X> {
    fn compute(p1: &Point<T>, p2: &Point<T>) -> X;
}

pub struct PointsAndFunction<T, X, D> {
    n: usize,
    points: Vec<Point<T>>,
    _phantom_data: PhantomData<(X, D)>,
}

impl<T, X, D> PointsAndFunction<T, X, D> {
    pub fn new(points: Vec<Point<T>>) -> Self {
        let n = points.len();
        Self {
            n,
            points,
            _phantom_data: PhantomData,
        }
    }
    pub fn size(&self) -> usize {
        self.n
    }

    pub fn points(&self) -> &[Point<T>] {
        &self.points
    }
}

impl<T, X, D> Problem for PointsAndFunction<T, X, D>
where
    T: Copy,
    X: Copy,
    D: DistanceMetric<T, X>,
{
    type Distance = X;

    fn distance(&self, c1: City, c2: City) -> Self::Distance {
        let p1 = &self.points[c1.0];
        let p2 = &self.points[c2.0];

        D::compute(p1, p2)
    }

    fn size(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::points_and_function::euc_2d::Euc2d;

    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let points = vec![Point(0.0, 0.0), Point(3.0, 4.0), Point(6.0, 8.0)];

        let euclidean: PointsAndFunction<f64, i64, Euc2d> = PointsAndFunction::new(points);

        assert_eq!(euclidean.distance(City(0), City(1)), 5);
        assert_eq!(euclidean.distance(City(0), City(2)), 10);
        assert_eq!(euclidean.distance(City(1), City(2)), 5);
    }

    #[test]
    fn test_integer_coords_float_distance() {
        let points = vec![Point(0.0, 0.0), Point(3.0, 4.0)];

        let euclidean: PointsAndFunction<f64, i64, Euc2d> = PointsAndFunction::new(points);

        assert_eq!(euclidean.distance(City(0), City(1)), 5);
    }
}
