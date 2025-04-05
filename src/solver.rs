use chrono::TimeDelta;

use crate::candidates::candidate_set::get_nn_candidates;
use crate::solution::Solution;
use crate::{input::Input, penalties::distance::DistancePenalizer, route::Route};

use rand::rng;
use rand::seq::SliceRandom;

pub struct Solver {
    n: usize,
    penalizer: DistancePenalizer,
    current_solution: Solution,
    best_solution: Solution,
    time_limit: Option<TimeDelta>,
    start: chrono::DateTime<chrono::Utc>,
    iterations: u64,
    time_taken: chrono::Duration,
}

impl Solver {
    pub fn new(input: Input) -> Solver {
        let n = input.distance_matrix.len();
        let distance_matrix = input.distance_matrix;
        let candidate_set = get_nn_candidates(&distance_matrix, 10);
        let time_limit = input.time_limit;
        let penalizer = DistancePenalizer::new(distance_matrix);
        let route = Route::new((0..n).collect());
        let current_solution = penalizer.penalize(&route);
        let best_solution = current_solution.clone();
        let start = chrono::Utc::now();
        Solver {
            n,
            penalizer,
            current_solution,
            best_solution,
            time_limit,
            start,
            iterations: 0,
            time_taken: chrono::Duration::zero(),
        }
    }

    pub fn best_distance(&self) -> u64 {
        self.best_solution.distance
    }
    pub fn iterations(&self) -> u64 {
        self.iterations
    }
    pub fn time_taken(&self) -> chrono::Duration {
        self.time_taken
    }
    pub fn best_route(&self) -> Vec<usize> {
        self.best_solution.route.sequence.clone()
    }

    fn generate_initial_solution(&self) -> Solution {
        match self.iterations {
            _ => self.generate_random_solution(),
        }
    }

    fn generate_random_solution(&self) -> Solution {
        let mut sequence = (0..=self.n - 1).collect::<Vec<usize>>();
        sequence.shuffle(&mut rng());
        let route = Route::new(sequence);
        self.penalizer.penalize(&route)
    }

    // fn run_2opt(
    //     &mut self,
    // ) -> bool {
    //     // it is important to understand what is happening herre:
    //     // we assume that i < j
    //     // and the the slice route[i..j] is reversed, j not belonging to the slice
    //     // imagine the route [..a,b..c,d..] becoming [..a,c..b,d..]

    //     // when a,b are defined, we only take c's such that c:
    //     // distance(a, b) > distance(a, c)
    //     // this is the positive gain criterion.
    //     let n = route.len();
    //     let mut distance_delta = 0;

    //     let a = route[(i - 1) % n];
    //     let b = route[i];
    //     let c = route[j - 1];
    //     let d = route[j % n];

    //     // some important notes:
    //     // we assume a symmetric distance matrix
    //     // if j-i == n - 1 or j-i == 1, or j-i == 0,we don't have to do anything.
    //     if j - i == n - 1 || j - i < 2 {
    //         return 0;
    //     }
    //     distance_delta -= self.distance_matrix.distance(a, b);
    //     distance_delta -= self.distance_matrix.distance(c, d);
    //     distance_delta += self.distance_matrix.distance(a, c);
    //     distance_delta += self.distance_matrix.distance(b, d);
    //     distance_delta

    //     let mut improved = false;
    //     for i in 0..self.n {
    //         for j in i + 1 + min_margin..self.n {
    //             let mut new_route = self.current_solution.route.clone();
    //             local_move(&mut new_route, i, j);
    //             let new_solution = self.penalizer.penalize(new_route, false);
    //             if self
    //                 .penalizer
    //                 .is_better(&new_solution, &self.current_solution)
    //             {
    //                 self.current_solution = new_solution;
    //                 improved = true;
    //             }
    //         }
    //     }
    //     improved
    // }
    // fn run_heuristics(&mut self) -> bool {
    //     let mut improved = false;
    //     improved |= self.run_2opt();
    // }

    fn termination_criterion(&self) -> bool {
        // returns true if the termination criterion is met
        // no time limit means we always continue
        match self.time_limit {
            Some(limit) => chrono::Utc::now() - self.start <= limit,
            None => true,
        }
    }

    fn one_time(&self) -> bool {
        self.time_limit.is_none()
    }

    pub fn solve(&mut self) {
        let mut improved = true;
        self.start = chrono::Utc::now();
        while self.termination_criterion() {
            self.iterations += 1;
            improved = true;
            while improved & self.termination_criterion() {
                //improved = self.run_heuristics()
            }

            // if self
            //     .penalizer
            //     .is_better(&self.current_solution, &self.best_solution)
            // {
            //     self.best_solution = self.current_solution.clone();
            // }
            self.current_solution = self.generate_initial_solution();

            if self.one_time() {
                break;
            }
        }
        self.time_taken = chrono::Utc::now() - self.start;
    }
}
