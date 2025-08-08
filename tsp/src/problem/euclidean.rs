use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

use crate::{city::City, problem::Problem};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point<T>(pub T, pub T);

pub struct Euclidian<T, X> {
    n: usize,
    points: Vec<Point<T>>,
    _phantom_data: PhantomData<X>,
}

impl<T, X> Euclidian<T, X> {
    pub fn new(points: Vec<Point<T>>) -> Self {
        let n = points.len();
        Self {
            n,
            points,
            _phantom_data: PhantomData,
        }
    }
}

impl<T, X> Problem for Euclidian<T, X>
where
    T: Copy + Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Into<f64>,
    X: Copy,
    f64: Into<X>,
{
    type Distance = X;

    fn distance(&self, c1: City, c2: City) -> Self::Distance {
        let p1 = &self.points[c1.0];
        let p2 = &self.points[c2.0];

        let dx = p1.0 - p2.0;
        let dy = p1.1 - p2.1;

        let squared_distance = dx * dx + dy * dy;
        let distance = squared_distance.into().sqrt();

        distance.into()
    }

    fn size(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let points = vec![Point(0.0, 0.0), Point(3.0, 4.0), Point(6.0, 8.0)];

        let euclidean: Euclidian<f64, f64> = Euclidian::new(points);

        assert_eq!(euclidean.distance(City(0), City(1)), 5.0);
        assert_eq!(euclidean.distance(City(0), City(2)), 10.0);
        assert_eq!(euclidean.distance(City(1), City(2)), 5.0);
    }

    #[test]
    fn test_integer_coords_float_distance() {
        let points = vec![Point(0i32, 0i32), Point(3i32, 4i32)];

        let euclidean: Euclidian<i32, f64> = Euclidian::new(points);

        assert_eq!(euclidean.distance(City(0), City(1)), 5.0f64);
    }
}
