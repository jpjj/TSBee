use crate::solution::Solution;

/// struct managing solutions, like current and best
pub(super) struct SolutionManager {
    pub(super) current_solution: Solution,
    pub(super) best_solution: Solution,
}

impl SolutionManager {
    pub(super) fn new(current_solution: Solution) -> Self {
        let best_solution = current_solution.clone();
        SolutionManager {
            current_solution,
            best_solution,
        }
    }

    pub(super) fn update_best(&mut self) -> bool {
        if self.current_solution < self.best_solution {
            self.best_solution = self.current_solution.clone();
            return true;
        }
        false
    }
}
