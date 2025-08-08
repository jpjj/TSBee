pub mod distance_matrix;
pub mod euclidean;
use crate::city::City;

pub trait Problem {
    type Distance;

    fn distance(&self, c1: City, c2: City) -> Self::Distance;

    fn size(&self) -> usize;
}
