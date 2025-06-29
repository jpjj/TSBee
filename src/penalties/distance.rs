pub mod distance_matrix;
pub mod utils;
use crate::{domain::route::Route, solution::Solution};
pub use distance_matrix::DistanceMatrix;

pub struct DistancePenalizer {
    pub(crate) distance_matrix: DistanceMatrix,
}

impl DistancePenalizer {
    pub fn new(distance_matrix: DistanceMatrix) -> DistancePenalizer {
        DistancePenalizer { distance_matrix }
    }

    pub fn penalize(&self, route: &Route) -> Solution {
        let mut distance = 0;
        for i in 0..route.len() - 1 {
            distance += self.distance_matrix.distance(route[i], route[i + 1]);
        }
        distance += self
            .distance_matrix
            .distance(route[route.len() - 1], route[0]);
        Solution {
            route: route.clone(),
            distance,
        }
    }
}
