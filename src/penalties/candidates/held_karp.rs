use crate::{domain::city::City, penalties::distance::DistanceMatrix};

use super::{
    min_one_tree::MinOneTree,
    utils::{get_k_argmins_ordered, get_min_spanning_tree},
};

pub struct BoundCalculator {
    distance_matrix: DistanceMatrix,
    upper_bound: i64,
    max_iterations: u64,
    max_time: chrono::Duration,
}

#[derive(Clone)]
pub struct HeldKarpResult {
    pub pi: Vec<i64>,
    pub min_one_tree: MinOneTree,
    pub optimal: bool,
}

impl HeldKarpResult {
    fn new(pi: Vec<i64>, min_one_tree: MinOneTree, optimal: bool) -> Self {
        Self {
            pi,
            min_one_tree,
            optimal,
        }
    }
}

impl BoundCalculator {
    pub fn new(
        distance_matrix: DistanceMatrix,
        upper_bound: i64,
        max_iterations: u64,
        max_time: chrono::Duration,
    ) -> Self {
        Self {
            distance_matrix,
            upper_bound,
            max_iterations,
            max_time,
        }
    }

    /// Creates a lower bound for the optimal length of a tsp given its distance matrix.
    pub fn run(&mut self) -> HeldKarpResult {
        // iteratively:
        // 1. get min 1-tree
        // 2. get its lower bound being the sum of all edge weights - 2* sum(pi)
        // 3. get the degrees of each node minus 2 as a vector g
        // 4. calculate stepsize (polyak) t =
        // beta * (upper_bound - lower_bound) / L2_norm(g)²
        // beta is a parameter between 0 and 2. start with beta_0 = 1 or 2
        // 5. update pi = pi + t * g

        // if L(k) fails to improve for some iterations, halve beta.
        let n = self.distance_matrix.len();
        let mut lower_bound_best = i64::MIN;
        let mut iterations = 0;
        let mut iterations_since_last_improvement_or_beta_change = 0;
        let mut beta = 2.0;
        let mut pi = vec![0; n];
        let mut best_pi = pi.clone();
        let mut best_min_1_tree = get_min_1_tree(&self.distance_matrix);
        let mut min_1_tree: MinOneTree;
        let start = chrono::Utc::now();
        let mut temp_dm = self.distance_matrix.clone();

        while iterations < self.max_iterations && chrono::Utc::now() - start < self.max_time {
            if iterations == 0 || iterations_since_last_improvement_or_beta_change > 10 {
                beta /= 2.0;
                iterations_since_last_improvement_or_beta_change = 0;
                pi = best_pi.clone();
                min_1_tree = best_min_1_tree.clone();
            } else {
                temp_dm.update_pi(pi.clone());
                // 1
                min_1_tree = get_min_1_tree(&temp_dm);
                // 2
            }
            let lower_bound = min_1_tree.score;
            if lower_bound > lower_bound_best {
                lower_bound_best = lower_bound;
                best_pi = pi.clone();
                best_min_1_tree = min_1_tree.clone();
                iterations_since_last_improvement_or_beta_change = 0;
            }
            let degrees = min_1_tree.get_degrees();
            // 3
            let g: Vec<i32> = degrees.into_iter().map(|x| x as i32 - 2).collect();
            if g.iter().min() == Some(&0) {
                // all degrees are two, so we found a min 1-tree that is a hamiltonian cycle, optimal solution found.
                return HeldKarpResult::new(pi, min_1_tree, true);
            }
            // 4
            let t = beta * (self.upper_bound - lower_bound) as f64
                / g.iter().map(|x| x * x).sum::<i32>() as f64;

            // 5
            let pi_new = pi
                .iter()
                .enumerate()
                .map(|(i, pi)| (*pi as f64 + t * g[i] as f64) as i64)
                .collect();
            if pi == pi_new {
                eprintln!("no change");
                return HeldKarpResult::new(best_pi, best_min_1_tree, false);
            }
            pi = pi_new;
            iterations += 1;
            iterations_since_last_improvement_or_beta_change += 1;
        }
        HeldKarpResult::new(best_pi, best_min_1_tree, false)
    }
}

fn get_min_1_tree(distance_matrix: &DistanceMatrix) -> MinOneTree {
    let n = distance_matrix.len();
    let spanning_tree = get_min_spanning_tree(distance_matrix, n - 1);
    let two_nearest_neighbors = get_k_argmins_ordered(distance_matrix.row(n - 1), 2, Some(n - 1));
    let score = spanning_tree.score
        + distance_matrix.distance(City(n - 1), City(two_nearest_neighbors[0]))
        + distance_matrix.distance(City(n - 1), City(two_nearest_neighbors[1]));
    let mut edges = spanning_tree.edges.clone();
    edges.push((City(n - 1), City(two_nearest_neighbors[0])));
    edges.push((City(n - 1), City(two_nearest_neighbors[1])));
    MinOneTree::new(score, edges)
}
