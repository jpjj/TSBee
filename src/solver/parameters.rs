/// parameters that influence the solver's behavior
pub(super) struct Parameters {
    //  maximum number of iterations
    pub(super) max_iterations: Option<u64>,
    //  maximum time limit for the solver
    pub(super) max_time: Option<chrono::Duration>,
    //  maximum number of iterations without improvement
    pub(super) max_no_improvement: Option<u64>,
    //  maximum number of neighbors to consider in local moves to make graph more sparse
    pub(super) max_neighbors: Option<usize>,
}

impl Parameters {
    pub(super) fn new(
        max_iterations: Option<u64>,
        max_time: Option<chrono::Duration>,
        max_no_improvement: Option<u64>,
        max_neighbors: Option<usize>,
    ) -> Self {
        Self {
            max_iterations,
            max_time,
            max_no_improvement,
            max_neighbors,
        }
    }
}
