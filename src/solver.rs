mod cache;
mod parameters;
mod solution_manager;
pub mod solution_report;
mod stats;

use crate::domain::city::City;
use crate::domain::route::Route;
use crate::input::Input;
use crate::local_move::LocalSearch;
use crate::penalties::candidates::alpha_nearness::{
    get_alpha_candidates, get_alpha_candidates_v2, get_nn_candidates,
};
use crate::penalties::candidates::held_karp::BoundCalculator;
use crate::penalties::candidates::Candidates;
use crate::penalties::distance::DistancePenalizer;

use cache::SolverCache;
use chrono::TimeDelta;
use parameters::Parameters;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use solution_manager::SolutionManager;
use solution_report::SolutionReport;
use stats::Stats;

pub struct Solver {
    n: usize,
    pub penalizer: DistancePenalizer,
    solution_manager: SolutionManager,
    stats: Stats,
    parameters: Parameters,
    candidates: Candidates,
    cache: SolverCache,
    rng: StdRng,
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
        let parameters = Parameters::new(None, time_limit, None, Some(10));
        let max_neighbors = match parameters.max_neighbors {
            Some(limit) => limit,
            _ => n,
        };
        let candidates = if n <= 3 {
            // For very small problems, use simple nearest neighbor candidates
            get_nn_candidates(&penalizer.distance_matrix, n.saturating_sub(1))
        } else {
            get_alpha_candidates_v2(&penalizer.distance_matrix, max_neighbors, true)
        };
        let cache = SolverCache::new(n);
        let rng = StdRng::seed_from_u64(42);
        Solver {
            n,
            penalizer,
            solution_manager,
            stats,
            parameters,
            candidates,
            cache,
            rng,
        }
    }

    fn update_best_solution(&mut self) {
        if self.solution_manager.update_best() {
            self.stats.iterations_since_last_improvement = 0;
        }
    }

    fn generate_initial_solution(&mut self) {
        self.generate_nearest_neighbor()
    }

    fn generate_nearest_neighbor(&mut self) {
        let mut sequence: Vec<City> = Vec::with_capacity(self.n);
        let start = self.rng.random_range(0..self.n);
        let mut cities_visited = vec![false; self.n];
        sequence.push(City(start));
        while sequence.len() < self.n {
            let mut city_found = false;
            let city = *sequence.last().unwrap();
            cities_visited[city.id()] = true;
            let neighbors = self.candidates.get_neighbors_out(&city);
            for neighbor in neighbors {
                if !cities_visited[neighbor.id()] {
                    sequence.push(*neighbor);
                    city_found = true;
                    break;
                }
            }
            if !city_found {
                // now we have to find the minimum not found, yet
                let mut closest_city = None;
                let mut min_distance = None;
                for (i, _) in cities_visited.iter().enumerate() {
                    if cities_visited[i] {
                        continue;
                    }
                    let new_distance = self.penalizer.distance_matrix.distance(city, City(i));
                    match min_distance {
                        Some(actual_distance) => {
                            if new_distance < actual_distance {
                                min_distance = Some(new_distance);
                                closest_city = Some(City(i));
                            }
                        }
                        None => {
                            min_distance = Some(new_distance);
                            closest_city = Some(City(i));
                        }
                    }
                }
                sequence.push(closest_city.unwrap());
            }
        }
        let route = Route::new(sequence);
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
        // no max_no_improvement means we continue as long as iterations since last improvment is 0.
        result &= match self.parameters.max_no_improvement {
            Some(limit) => self.stats.iterations_since_last_improvement < limit,
            None => true,
        };
        result
    }

    fn one_time(&self) -> bool {
        self.parameters.max_time.is_none()
            && self.parameters.max_iterations.is_none()
            && self.parameters.max_no_improvement.is_none()
    }

    fn double_bridge_kick(&mut self) {
        self.update_best_solution();
        if self.stats.iterations_since_last_improvement == 0 {
            let mut local_move = LocalSearch::new(
                &self.penalizer.distance_matrix,
                &self.candidates,
                &self.solution_manager.best_solution,
                &mut self.cache.dont_look_bits,
            );
            let new_solution = local_move.execute_double_bridge();
            if new_solution < self.solution_manager.current_solution {
                self.solution_manager.current_solution = new_solution;
                self.update_best_solution();

                // if self.solution_manager.update_best() {
                //     // println!("better double bridge move found");
                // };
                return;
            }
        }
        let mut new_solution = self.solution_manager.best_solution.clone();
        let mut random_numbers = (0..4)
            .map(|_| self.rng.random_range(0..self.n))
            .collect::<Vec<usize>>();
        random_numbers.sort();
        let a = 0;
        let b = random_numbers[1] - random_numbers[0];
        let c = random_numbers[2] - random_numbers[0];
        let d = random_numbers[3] - random_numbers[0];
        new_solution.route.sequence.rotate_left(random_numbers[0]);
        new_solution.route.sequence[a..b].reverse();
        new_solution.route.sequence[b..c].reverse();
        new_solution.route.sequence[c..d].reverse();
        new_solution.route.sequence[d..].reverse();
        self.solution_manager.current_solution = self.penalizer.penalize(&new_solution.route);
        self.cache.dont_look_bits[new_solution.route.sequence[a].id()] = true;
        self.cache.dont_look_bits[new_solution.route.sequence[(self.n + a - 1) % self.n].id()] =
            true;
        self.cache.dont_look_bits[new_solution.route.sequence[(self.n + b - 1) % self.n].id()] =
            true;
        self.cache.dont_look_bits[new_solution.route.sequence[(self.n + c - 1) % self.n].id()] =
            true;
        self.cache.dont_look_bits[new_solution.route.sequence[(self.n + d - 1) % self.n].id()] =
            true;

        self.cache.dont_look_bits[new_solution.route.sequence[b].id()] = true;
        self.cache.dont_look_bits[new_solution.route.sequence[c].id()] = true;
        self.cache.dont_look_bits[new_solution.route.sequence[d].id()] = true;
    }

    /// Solves the Traveling Salesman Problem using an iterated local search algorithm.
    ///
    /// This is the main entry point for the TSP solver. It implements an advanced
    /// iterated local search (ILS) algorithm with the following components:
    ///
    /// # Algorithm Overview
    ///
    /// 1. **Initialization**: Generate initial solution using nearest neighbor heuristic
    /// 2. **Local Search**: Apply 3-opt moves until no improvement found
    /// 3. **Diversification**: Use double-bridge kicks to escape local optima
    /// 4. **Lower Bound**: Calculate Held-Karp bound on first iteration
    /// 5. **Iteration**: Repeat steps 2-3 until termination criteria met
    ///
    /// # Key Features
    ///
    /// - **3-opt Local Search**: Considers all ways to remove and reconnect 3 edges
    /// - **Don't Look Bits (DLB)**: Avoids redundant searches for efficiency
    /// - **Alpha-nearness Candidates**: Reduces search space intelligently
    /// - **Double-bridge Perturbation**: 4-opt moves to escape local optima
    /// - **Held-Karp Lower Bound**: Provides optimality gap information
    /// - **Adaptive Penalties**: Adjusts distances based on Lagrangian relaxation
    ///
    /// # Arguments
    ///
    /// * `dlb` - Whether to use Don't Look Bits optimization (recommended: true)
    ///
    /// # Returns
    ///
    /// A `SolutionReport` containing:
    /// - Best tour found
    /// - Total distance
    /// - Execution statistics
    /// - Lower bound information (if calculated)
    ///
    /// # Termination Criteria
    ///
    /// The algorithm stops when any of these conditions are met:
    /// - Time limit exceeded (if specified)
    /// - Maximum iterations reached (if specified)
    /// - No improvement for N iterations (if specified)
    /// - Local optimum reached with no parameters set
    pub fn solve(&mut self, dlb: bool) -> SolutionReport {
        self.stats.reset();

        // Handle trivial cases
        if self.n <= 3 {
            // For 1, 2, or 3 cities, any tour is optimal
            self.stats.time_taken = chrono::Utc::now() - self.stats.start_time;
            return self.get_solution_report();
        }

        self.generate_initial_solution();

        // run while global criterion is met (time, max iterations, ...)
        while self.continuation_criterion() {
            // run until global AND single iteration criterion are met
            while self.continuation_criterion() {
                if !self.run_local_search(dlb) {
                    break;
                }
            }
            self.stats.iterations += 1;
            self.stats.iterations_since_last_improvement += 1;
            // sets iterations since  last improvement to 0 if best solution gets updated.
            self.update_best_solution();
            if self.one_time() {
                break;
            }
            if self.stats.iterations == 1 {
                let upper_bound = self.solution_manager.best_solution.distance;
                let max_iterations = 10000;
                // geben wir uns die hälfte der Zeit, die noch übrig ist.
                let max_time = match self.parameters.max_time {
                    Some(time) => (time - (chrono::Utc::now() - self.stats.start_time)) / 2,
                    None => TimeDelta::seconds(1),
                };
                let mut held_karp_calculator = BoundCalculator::new(
                    self.penalizer.distance_matrix.clone(),
                    upper_bound,
                    max_iterations,
                    max_time,
                );
                let held_carp_result = held_karp_calculator.run();
                // if held_carp_result.optimal {
                //     println!("optimal")
                // }
                self.penalizer
                    .distance_matrix
                    .update_pi(held_carp_result.pi.clone());
                self.solution_manager.current_solution = self
                    .penalizer
                    .penalize(&self.solution_manager.best_solution.route);
                self.solution_manager.best_solution = self
                    .penalizer
                    .penalize(&self.solution_manager.current_solution.route);
                self.candidates = get_alpha_candidates(&self.penalizer.distance_matrix, 5);
                self.stats.held_karp_result = Some(held_carp_result);
            } else {
                // diversification
                self.double_bridge_kick();
            }
        }
        self.stats.time_taken = chrono::Utc::now() - self.stats.start_time;
        self.get_solution_report()
    }

    /// Executes one iteration of 3-opt local search.
    ///
    /// This method attempts to improve the current solution by applying 3-opt moves.
    /// It uses the candidate list to efficiently search for improving moves and
    /// updates the current solution if an improvement is found.
    ///
    /// # Arguments
    ///
    /// * `dlb` - Whether to use Don't Look Bits optimization
    ///
    /// # Returns
    ///
    /// `true` if an improving move was found and applied, `false` otherwise
    ///
    /// # Side Effects
    ///
    /// - Updates `self.solution_manager.current_solution` if improvement found
    /// - Modifies Don't Look Bits in the cache
    fn run_local_search(&mut self, dlb: bool) -> bool {
        let mut local_move = LocalSearch::new(
            &self.penalizer.distance_matrix,
            &self.candidates,
            &self.solution_manager.current_solution,
            &mut self.cache.dont_look_bits,
        );
        let new_solution = local_move.execute_3opt(dlb);
        if new_solution < self.solution_manager.current_solution {
            self.solution_manager.current_solution = new_solution;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {

    use chrono::TimeDelta;

    use super::Solver;
    use crate::{input::Input, penalties::distance::DistanceMatrix};

    #[test]
    fn solves_berlin52() {
        let city_coordinates = vec![
            (565, 575),
            (25, 185),
            (345, 750),
            (945, 685),
            (845, 655),
            (880, 660),
            (25, 230),
            (525, 1000),
            (580, 1175),
            (650, 1130),
            (1605, 620),
            (1220, 580),
            (1465, 200),
            (1530, 5),
            (845, 680),
            (725, 370),
            (145, 665),
            (415, 635),
            (510, 875),
            (560, 365),
            (300, 465),
            (520, 585),
            (480, 415),
            (835, 625),
            (975, 580),
            (1215, 245),
            (1320, 315),
            (1250, 400),
            (660, 180),
            (410, 250),
            (420, 555),
            (575, 665),
            (1150, 1160),
            (700, 580),
            (685, 595),
            (685, 610),
            (770, 610),
            (795, 645),
            (720, 635),
            (760, 650),
            (475, 960),
            (95, 260),
            (875, 920),
            (700, 500),
            (555, 815),
            (830, 485),
            (1170, 65),
            (830, 610),
            (605, 625),
            (595, 360),
            (1340, 725),
            (1740, 245),
        ];
        let distance_matrix = DistanceMatrix::new_euclidian(city_coordinates);

        let input = Input::new(distance_matrix, TimeDelta::new(1, 0));
        let mut solver = Solver::new(input);

        let solution_report = solver.solve(true);
        assert!(solution_report.best_solution.distance <= 7865100);
    }
}
