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

pub(crate) fn get_input_from_raw(distance_matrix: Vec<Vec<u64>>, time_limit: Option<f64>) -> Input {
    let real_distance_matrix = DistanceMatrix::new(distance_matrix);
    let time_limit = match time_limit {
        Some(limit) => Some(chrono::Duration::milliseconds((limit * 1_000_000.0) as i64)),
        None => None,
    };
    Input::new(real_distance_matrix, time_limit)
}
