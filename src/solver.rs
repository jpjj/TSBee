mod cache;
mod parameters;
mod solution_manager;
pub mod solution_report;
mod stats;

use crate::domain::city::City;
use crate::domain::route::Route;
use crate::input::Input;
use crate::local_move::LocalSearch;
use crate::penalties::candidates::candidate_set::get_nn_candidates;
use crate::penalties::candidates::Candidates;
use crate::penalties::distance::DistancePenalizer;

use cache::SolverCache;
use parameters::Parameters;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
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
        let parameters = Parameters::new(None, time_limit, None, Some(20));
        let max_neighbors = match parameters.max_neighbors {
            Some(limit) => limit,
            _ => n,
        };
        let candidates = get_nn_candidates(&penalizer.distance_matrix, max_neighbors);
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
        match self.stats.iterations {
            0 => self.generate_nearest_neighbor(),
            _ => self.generate_random_solution(),
        }
    }

    fn generate_random_solution(&mut self) {
        let mut sequence = (0..self.n).collect::<Vec<usize>>();
        sequence.shuffle(&mut self.rng);
        let route = Route::from_iter(sequence);
        self.solution_manager.current_solution = self.penalizer.penalize(&route)
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
    pub fn solve(&mut self, dlb: bool) -> SolutionReport {
        self.stats.reset();

        // run while global criterion is met (time, max iterations, ...)
        while self.continuation_criterion() {
            self.generate_initial_solution();
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

    use crate::preprocess;

    use super::Solver;

    #[test]
    fn solves_att48() {
        let city_coordinates = vec![
            (6734, 1453),
            (2233, 10),
            (5530, 1424),
            (401, 841),
            (3082, 1644),
            (7608, 4458),
            (7573, 3716),
            (7265, 1268),
            (6898, 1885),
            (1112, 2049),
            (5468, 2606),
            (5989, 2873),
            (4706, 2674),
            (4612, 2035),
            (6347, 2683),
            (6107, 669),
            (7611, 5184),
            (7462, 3590),
            (7732, 4723),
            (5900, 3561),
            (4483, 3369),
            (6101, 1110),
            (5199, 2182),
            (1633, 2809),
            (4307, 2322),
            (675, 1006),
            (7555, 4819),
            (7541, 3981),
            (3177, 756),
            (7352, 4506),
            (7545, 2801),
            (3245, 3305),
            (6426, 3173),
            (4608, 1198),
            (23, 2216),
            (7248, 3779),
            (7762, 4595),
            (7392, 2244),
            (3484, 2829),
            (6271, 2135),
            (4985, 140),
            (1916, 1569),
            (7280, 4899),
            (7509, 3239),
            (10, 2676),
            (6807, 2993),
            (5185, 3258),
            (3023, 1942),
        ];
        let distance_matrix = city_coordinates
            .iter()
            .map(|(x, y)| {
                city_coordinates
                    .iter()
                    .map(|(a, b)| (((x - a) * (x - a) + (y - b) * (y - b)) as u64).isqrt())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let raw_input = preprocess::RawInput::new(distance_matrix, None);
        let input = raw_input.into();
        let mut solver = Solver::new(input);

        let solution_report = solver.solve(true);
        assert_eq!(solution_report.best_solution.distance, 36012);
    }
}
