use crate::solution::Solution;

use super::stats::Stats;

/// struct to be returned from Solver.solve() containing the best solution found and the stats of the solver run.
pub struct SolutionReport {
    pub best_solution: Solution,
    pub stats: Stats,
}

impl SolutionReport {
    pub(super) fn new(best_solution: Solution, stats: Stats) -> Self {
        Self {
            best_solution,
            stats,
        }
    }
}
