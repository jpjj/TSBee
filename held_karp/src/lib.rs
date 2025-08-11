// use std::vec;

// use graph::Graph;
// use min1tree::{Min1Tree, get_min_1_tree};
// use tsp::edge::Edge;

// pub fn run(graph: &Graph, bound: Option<i64>) -> (Vec<i64>, i64) {
//     // iteratively:
//     // 1. get min 1-tree
//     // 2. get its lower bound being the sum of all edge weights - 2* sum(pi)
//     // 3. get the degrees of each node minus 2 as a vector g
//     // 4. calculate stepsize (polyak) t =
//     // beta * (upper_bound - lower_bound) / L2_norm(g)²
//     // beta is a parameter between 0 and 2. start with beta_0 = 1 or 2
//     // 5. update pi = pi + t * g

//     // if L(k) fails to improve for some iterations, halve beta.
//     let max_iterations = 1000;
//     let max_time = chrono::TimeDelta::milliseconds(200);
//     let max_iterations_no_imrovement = 100;
//     let max_iterations_no_imrovement_beta_change = 10;
//     let mut beta = 2.0;

//     let n = graph.n();
//     let mut lower_bound_best = i64::MIN;
//     let mut iterations = 0;
//     let mut iterations_since_last_improvement_or_beta_change = 0;
//     let mut pi = vec![0; n];
//     let mut best_pi = pi.clone();

//     let mut edges: Vec<Edge> = graph.edges().collect();
//     let mut best_min_1_tree = get_min_1_tree(graph, edges);
//     let mut min_1_tree: Min1Tree;
//     let start = chrono::Utc::now();

//     while iterations < max_iterations && chrono::Utc::now() - start < max_time {
//         if iterations == 0
//             || iterations_since_last_improvement_or_beta_change
//                 > max_iterations_no_imrovement_beta_change
//         {
//             beta /= 2.0;
//             iterations_since_last_improvement_or_beta_change = 0;
//             pi = best_pi.clone();
//             min_1_tree = best_min_1_tree.clone();
//         } else {
//             temp_dm.update_pi(pi.clone());
//             // 1
//             min_1_tree = self.get_min_1_tree(&temp_dm);
//             // 2
//         }
//         let lower_bound = min_1_tree.score;
//         if lower_bound > lower_bound_best {
//             lower_bound_best = lower_bound;
//             best_pi = pi.clone();
//             best_min_1_tree = min_1_tree.clone();
//             iterations_since_last_improvement_or_beta_change = 0;
//         }
//         let degrees = min_1_tree.get_degrees();
//         // 3
//         let g: Vec<i64> = degrees.into_iter().map(|x| x as i64 - 2).collect();
//         if g.iter().min() == Some(&0) {
//             // all degrees are two, so we found a min 1-tree that is a hamiltonian cycle, optimal solution found.
//             return HeldKarpResult::new(pi, min_1_tree, true);
//         }
//         // 4
//         let t = beta * (self.upper_bound - lower_bound) as f64
//             / g.iter().map(|x| x * x).sum::<i64>() as f64;

//         // 5
//         let pi_new: Vec<i64> = pi
//             .iter()
//             .enumerate()
//             .map(|(i, pi)| (*pi as f64 + t * g[i] as f64) as i64)
//             .collect();
//         if pi == pi_new {
//             return HeldKarpResult::new(best_pi, best_min_1_tree, false);
//         }
//         pi = pi_new;
//         iterations += 1;
//         iterations_since_last_improvement_or_beta_change += 1;
//     }
//     HeldKarpResult::new(best_pi, best_min_1_tree, false)
// }
