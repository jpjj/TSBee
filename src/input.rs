// this will be what we get from the outside world and what will be inserted into the solver
use crate::penalties::distance::DistanceMatrix;

pub struct Input {
    pub distance_matrix: DistanceMatrix,
    pub time_limit: Option<chrono::Duration>,
    pub use_heap_mst: bool,
}

impl Input {
    pub fn new(distance_matrix: DistanceMatrix, time_limit: Option<chrono::Duration>) -> Input {
        Input {
            distance_matrix,
            time_limit,
            use_heap_mst: false, // Default to using heap MST
        }
    }

    #[allow(dead_code)]
    pub fn with_heap_mst(
        distance_matrix: DistanceMatrix,
        time_limit: Option<chrono::Duration>,
        use_heap_mst: bool,
    ) -> Input {
        Input {
            distance_matrix,
            time_limit,
            use_heap_mst,
        }
    }
}
