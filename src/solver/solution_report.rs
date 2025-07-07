use crate::solution::Solution;

use super::stats::Stats;

/// struct to be returned from Solver.solve() containing the best solution found and the stats of the solver run.
#[derive(Clone)]
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

    // used for tanslating solutions from a transformed symmetrized atsp instance back to a solution form for the initial problem
    pub fn desymmetrize(self, big_m: i64) -> Self {
        let mut new_solution = self.best_solution.clone();
        new_solution.distance %= big_m;
        if new_solution.distance < 0 {
            new_solution.distance += big_m;
        }
        // find direction
        // are the pairs always I/O or O/I?
        let n = new_solution.route.len() / 2;
        let first_3_ids = new_solution
            .route
            .sequence
            .iter()
            .take(3)
            .map(|x| x.id())
            .collect::<Vec<usize>>();
        // There must be a difference of n or -n between 1 and 0 or 2 and 1.
        // if -n, we know that we have I/O pairs, O/I instead.
        let (id0, id1, id2) = (first_3_ids[0], first_3_ids[1], first_3_ids[2]);
        let mut forward = false;
        if (id1.saturating_sub(id0) == n) | (id2.saturating_sub(id1) == n) {
            forward = true;
        }
        new_solution.route.sequence.retain(|&x| x.id() < n);
        if forward {
            new_solution.route.sequence.reverse();
        }
        let mut new_report = self.clone();
        new_report.best_solution = new_solution;
        new_report
    }
}
