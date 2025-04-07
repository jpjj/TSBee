// this will be what we get from the outside world and what will be inserted into the solver
use crate::penalties::distance::DistanceMatrix;
use chrono;

pub struct Input {
    pub distance_matrix: DistanceMatrix,
    pub time_limit: Option<chrono::Duration>,
}

impl Input {
    pub fn new(distance_matrix: DistanceMatrix, time_limit: Option<chrono::Duration>) -> Input {
        Input {
            distance_matrix,
            time_limit,
        }
    }
}
