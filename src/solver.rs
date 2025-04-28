mod cache;
mod parameters;
mod solution_manager;
pub mod solution_report;
mod stats;

use crate::domain::city::City;
use crate::domain::route::Route;
use crate::input::Input;
use crate::local_move::LocalSearch;
use crate::penalties::candidates::alpha_nearness::{get_alpha_candidates, get_alpha_candidates_v2};
use crate::penalties::candidates::candidate_set::get_nn_candidates;
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
    penalizer: DistancePenalizer,
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
        let candidates = get_alpha_candidates_v2(&penalizer.distance_matrix, max_neighbors);
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
            let city = sequence.last().unwrap().clone();
            cities_visited[city.id()] = true;
            let neighbors = self.candidates.get_neighbors_out(&city);
            for neighbor in neighbors {
                if !cities_visited[neighbor.id()] {
                    sequence.push(neighbor.clone());
                    city_found = true;
                    break;
                }
            }
            if !city_found {
                // now we have to find the minimum not found, yet
                let mut closest_city = None;
                let mut min_distance = None;
                for i in 0..self.n {
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
            Some(limit) => limit < self.stats.iterations_since_last_improvement,
            None => true,
        };
        return result;
    }

    fn one_time(&self) -> bool {
        self.parameters.max_time.is_none()
            && self.parameters.max_iterations.is_none()
            && self.parameters.max_no_improvement.is_none()
    }

    fn double_bridge_kick(&mut self) {
        self.update_best_solution();

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

    /// apply double bridge move n times and choose the best one.
    fn double_bridge_kick_v2(&mut self) {
        self.update_best_solution();

        let mut new_solution = self.solution_manager.best_solution.clone();
        let mut best_delta: Option<i64> = None;
        let mut best_random_nums = None;
        for _ in 0..1000 {
            let mut random_numbers = (0..4)
                .map(|_| self.rng.random_range(0..self.n))
                .collect::<Vec<usize>>();
            random_numbers.sort();
            let curr_dist: i64 = random_numbers
                .iter()
                .map(|&i| {
                    self.penalizer
                        .distance_matrix
                        .distance(new_solution.route[i], new_solution.route[(i + 1) % self.n])
                })
                .sum();

            let new_dist: i64 = (0..4)
                .map(|i| {
                    self.penalizer.distance_matrix.distance(
                        new_solution.route[random_numbers[i]],
                        new_solution.route[(random_numbers[(i + 2) % 4] + 1) % self.n],
                    )
                })
                .sum();
            match best_delta {
                None => {
                    best_delta = Some(new_dist - curr_dist);
                    best_random_nums = Some(random_numbers.clone());
                }
                Some(delta) => {
                    if delta > new_dist - curr_dist {
                        best_delta = Some(new_dist - curr_dist);
                        best_random_nums = Some(random_numbers.clone());
                    }
                }
            }
        }
        let random_numbers: Vec<usize> = best_random_nums.unwrap();
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
    }

    /// function for solving the tsp
    pub fn solve(&mut self, dlb: bool) -> SolutionReport {
        self.stats.reset();
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
                if held_carp_result.optimal {
                    println!("optimal")
                }
                self.penalizer
                    .distance_matrix
                    .update_pi(held_carp_result.pi.clone());
                self.candidates = get_alpha_candidates_v2(&self.penalizer.distance_matrix, 5);
                self.stats.held_karp_result = Some(held_carp_result);
            } else {
                // diversification
                self.double_bridge_kick_v2();
            }
        }
        self.stats.time_taken = chrono::Utc::now() - self.stats.start_time;
        self.get_solution_report()
    }

    /// executes local search and returns true if better solution has been found
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
        return false;
    }
}

#[cfg(test)]
mod tests {

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

        let input = Input::new(distance_matrix, None);
        let mut solver = Solver::new(input);

        let solution_report = solver.solve(true);
        assert!(solution_report.best_solution.distance <= 786510);
    }
}
