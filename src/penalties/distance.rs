pub mod distance_matrix;
mod utils;
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
        return Solution {
            route: route.clone(),
            distance,
        };
    }

    // pub fn penalize_2opt(&self, route: &Route, i: usize, j: usize) -> i64 {
    //     // we assume that i < j
    //     // and the the slice route[i..j] is reversed, j not belonging to the slice
    //     // imagine the route [..a,b..c,d..] becoming [..a,c..b,d..]
    //     let n = route.len();
    //     let mut distance_delta = 0;

    //     let a = route[(i - 1) % n];
    //     let b = route[i];
    //     let c = route[j - 1];
    //     let d = route[j % n];

    //     // some important notes:
    //     // we assume a symmetric distance matrix
    //     // if j-i == n - 1 or j-i == 1, or j-i == 0,we don't have to do anything.
    //     if j - i == n - 1 || j - i < 2 {
    //         return 0;
    //     }
    //     distance_delta -= self.distance_matrix.distance(a, b);
    //     distance_delta -= self.distance_matrix.distance(c, d);
    //     distance_delta += self.distance_matrix.distance(a, c);
    //     distance_delta += self.distance_matrix.distance(b, d);
    //     distance_delta
    // }
}
