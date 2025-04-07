use crate::domain::route::Route;
use crate::local_move::LocalMove;
use crate::penalties::candidates::candidate_set::get_nn_candidates;
use crate::penalties::candidates::Candidates;
use crate::solution::Solution;
use crate::{input::Input, penalties::distance::DistancePenalizer};

use rand::rng;
use rand::seq::SliceRandom;

/// A struct to hold the statistics of the solver
#[derive(Clone)]
pub(super) struct Stats {
    pub(super) start_time: chrono::DateTime<chrono::Utc>,
    pub(super) time_taken: chrono::Duration,
    pub(super) iterations: u64,
    pub(super) iterations_since_last_improvement: u64,
}

impl Stats {
    fn new() -> Self {
        let start_time = chrono::Utc::now();
        let time_taken = chrono::Duration::zero();
        let iterations = 0;
        let iterations_since_last_improvement = 0;
        Self {
            start_time,
            time_taken,
            iterations,
            iterations_since_last_improvement,
        }
    }

    fn reset(&mut self) {
        self.start_time = chrono::Utc::now();
        self.time_taken = chrono::Duration::zero();
        self.iterations = 0;
        self.iterations_since_last_improvement = 0;
    }
}

/// struct managing solutions, like current and best
struct SolutionManager {
    current_solution: Solution,
    best_solution: Solution,
}

impl SolutionManager {
    fn new(current_solution: Solution) -> Self {
        let best_solution = current_solution.clone();
        SolutionManager {
            current_solution,
            best_solution,
        }
    }

    fn update_best(&mut self) -> bool {
        if self.current_solution < self.best_solution {
            self.best_solution = self.current_solution.clone();
            return true;
        }
        false
    }
}

/// parameters that influence the solver's behavior
struct Parameters {
    //  maximum number of iterations
    max_iterations: Option<u64>,
    //  maximum time limit for the solver
    max_time: Option<chrono::Duration>,
    //  maximum number of iterations without improvement
    max_no_improvement: Option<u64>,
    //  maximum number of neighbors to consider in local moves to make graph more sparse
    max_neighbors: Option<usize>,
}

impl Parameters {
    fn new(
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

/// struct to be returned from Solver.solve() containing the best solution found and the stats of the solver run.
pub(super) struct SolutionReport {
    pub(super) best_solution: Solution,
    pub(super) stats: Stats,
}

impl SolutionReport {
    fn new(best_solution: Solution, stats: Stats) -> Self {
        Self {
            best_solution,
            stats,
        }
    }
}

pub struct Solver {
    n: usize,
    penalizer: DistancePenalizer,
    solution_manager: SolutionManager,
    stats: Stats,
    parameters: Parameters,
    candidates: Candidates,
}

impl Solver {
    pub fn new(input: Input) -> Solver {
        let n = input.distance_matrix.len();
        let penalizer = DistancePenalizer::new(input.distance_matrix);
        let time_limit = input.time_limit;
        let route = Route::from_iter(0..n);
        let current_solution = penalizer.penalize(&route);

        let solution_manager = SolutionManager::new(current_solution);
        let stats = Stats::new();
        let parameters = Parameters::new(None, time_limit, None, Some(20));
        let max_neighbors = match parameters.max_neighbors {
            Some(limit) => limit,
            _ => n,
        };
        let candidates = get_nn_candidates(&penalizer.distance_matrix, max_neighbors);
        Solver {
            n,
            penalizer,
            solution_manager,
            stats,
            parameters,
            candidates,
        }
    }

    fn update_best_solution(&mut self) {
        if self.solution_manager.update_best() {
            self.stats.iterations_since_last_improvement = 0;
        }
    }

    fn generate_initial_solution(&mut self) {
        match self.stats.iterations {
            _ => self.generate_random_solution(),
        }
    }

    fn generate_random_solution(&mut self) {
        let mut sequence = (0..self.n).collect::<Vec<usize>>();
        sequence.shuffle(&mut rng());
        let route = Route::from_iter(sequence);
        self.solution_manager.current_solution = self.penalizer.penalize(&route)
    }

    fn get_solution_report(&self) -> SolutionReport {
        let best_solution = self.solution_manager.best_solution.clone();
        let stats = self.stats.clone();
        SolutionReport::new(best_solution, stats)
    }

    fn continuation_criterion(&self) -> bool {
        // returns true if the continuation criterion is met
        // no time limit means we always continue in that regard
        let mut result = true;
        result &= match self.parameters.max_time {
            Some(limit) => chrono::Utc::now() - self.stats.start_time <= limit,
            None => true,
        };
        // no max_no_improvement means we continue as long as iterations since lastimprovment is 0.
        result &= match self.parameters.max_no_improvement {
            Some(limit) => limit < self.stats.iterations_since_last_improvement,
            None => self.stats.iterations_since_last_improvement == 0,
        };
        return result;
    }

    fn one_time(&self) -> bool {
        self.parameters.max_time.is_none()
            && self.parameters.max_iterations.is_none()
            && self.parameters.max_no_improvement.is_none()
    }

    /// function for solving the tsp
    pub fn solve(&mut self) -> SolutionReport {
        self.stats.reset();
        while self.continuation_criterion() {
            self.generate_initial_solution();
            while self.continuation_criterion() {
                self.run_heuristics();
            }
            self.stats.iterations += 1;
            self.stats.iterations_since_last_improvement += 1;
            // sets iterations since  last improvement to 0 if best solution gets updated.
            self.update_best_solution();
            if self.one_time() {
                break;
            }
        }
        self.stats.time_taken = chrono::Utc::now() - self.stats.start_time;
        self.get_solution_report()
    }

    fn run_heuristics(&mut self) {
        let mut local_move = LocalMove::new(
            &self.penalizer.distance_matrix,
            &self.candidates,
            &self.solution_manager.current_solution,
        );
        let new_solution = local_move.execute_2opt();
        self.solution_manager.current_solution = new_solution;
    }
}
