/// This held-karp approach is inspired by this paper: https://users.cs.cf.ac.uk/C.L.Mumford/papers/HeldKarp.pdf
use std::vec;

use graph::Graph;
use min1tree::get_min_1_tree;
use tsp::edge::Edge;

fn calc_stepsize(m: i32, big_m: i32, t_1: f64) -> f64 {
    let part1_num = ((m - 1) * (2 * big_m - 5)) as f64;
    let part1_den = (2 * (big_m - 1)) as f64;
    let part2 = (m - 2) as f64;
    let part3_num = ((m - 1) * (m - 2)) as f64;
    let part3_den = (2 * (big_m - 1) * (big_m - 2)) as f64;
    t_1 * (part1_num / part1_den - part2 + part3_num / part3_den)
}

fn calc_new_pi(pi: Vec<f64>, t_i: f64, degrees: &[i32], degrees_prev: &[i32]) -> Vec<f64> {
    pi.iter()
        .enumerate()
        .map(|(idx, val)| {
            val + (t_i * (0.6 * (degrees[idx] - 2) as f64 + 0.4 * (degrees_prev[idx] - 2) as f64))
        })
        .collect()
}

pub fn run(graph: &mut Graph) -> (Vec<f64>, f64) {
    // calculate the first min1tree for pi = 0
    // According to Volgenant and Jonker:
    // Calculate t1, the basic stepsize, based on the weight of the min1tree
    //
    // iteratively:
    // Get the degrees of the min1tree d_i
    // calculate stepsize  t_i according to the Volgenant and Jonker Strategy.
    // Get pi_i given t_i, d_i and d_(i-1)
    // Calculate the new min1tree for pi_i

    let n = graph.n();
    let big_m = n as i32;
    // pi is already initialized to vec![0; n] by default in Graph

    // no edges to City n - 1, that is the definition of our 1-tree here.
    let mut edges: Vec<Edge> = graph.edges().filter(|e| e.v.0 < n - 1).collect();

    let mut min_1_tree = get_min_1_tree(graph, Some(&mut edges));
    let mut degrees = vec![2; n];
    let mut degrees_prev;
    let mut t_1 = min_1_tree.total_weight as f64 / (2.0 * n as f64);

    for m in 1..=big_m {
        degrees_prev = degrees.clone();
        // Compute degrees from the min1tree edges
        degrees = min_1_tree.degrees();

        let t_i = calc_stepsize(m, big_m, t_1);
        graph.pi = calc_new_pi(graph.pi.clone(), t_i, &degrees, &degrees_prev);
        min_1_tree = get_min_1_tree(graph, Some(&mut edges));
        t_1 = min_1_tree.total_weight as f64 / (2.0 * n as f64);
    }
    (graph.pi.clone(), min_1_tree.total_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_held_karp_berlin52() {
        // Load berlin52 instance
        let benchmarks_path = PathBuf::from("../benchmarks");
        let problem_path = benchmarks_path.join("data/problems/berlin52.tsp");

        // Parse the TSP file
        let problem = benchmarks::read_problem_file::read_problem_file(&problem_path)
            .expect("Failed to read berlin52.tsp");

        // Create a graph from the problem
        let mut graph = Graph::new_matrix(&problem);

        // Run the Held-Karp algorithm
        let (_pi, held_karp_bound) = run(&mut graph);

        // The optimal solution for berlin52 is 7542
        // The Held-Karp bound should be a lower bound, so it should be <= 7542
        assert!(held_karp_bound <= 7542.0001);

        // The bound should also be reasonably close to the optimal
        // Typically within 1-2% for good implementations
        let ratio = held_karp_bound / 7542.0;
        assert!(ratio > 0.999);
    }
}
