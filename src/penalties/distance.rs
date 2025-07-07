pub mod distance_matrix;
pub mod utils;
use crate::{domain::route::Route, solution::Solution};
pub use distance_matrix::DistanceMatrix;

/// Evaluates the quality of TSP routes by calculating their total distance.
///
/// The DistancePenalizer wraps a distance matrix and provides methods to
/// calculate the total distance of a route. It supports penalized distances
/// used in the Held-Karp lower bound calculation.
pub struct DistancePenalizer {
    /// The distance matrix containing distances between all city pairs
    pub(crate) distance_matrix: DistanceMatrix,
}

impl DistancePenalizer {
    /// Creates a new DistancePenalizer with the given distance matrix.
    ///
    /// # Arguments
    ///
    /// * `distance_matrix` - Matrix containing distances between all city pairs
    pub fn new(distance_matrix: DistanceMatrix) -> DistancePenalizer {
        DistancePenalizer { distance_matrix }
    }

    /// Calculates the total distance of a route and creates a Solution.
    ///
    /// This method computes the sum of distances between consecutive cities
    /// in the route, including the return from the last city to the first.
    ///
    /// # Arguments
    ///
    /// * `route` - The route to evaluate
    ///
    /// # Returns
    ///
    /// A `Solution` containing the route and its total distance
    ///
    /// # Example
    ///
    /// For route [0, 2, 1], calculates:
    /// distance(0,2) + distance(2,1) + distance(1,0)
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
