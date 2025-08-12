/// This held-karp approach is inspired by this paper: https://users.cs.cf.ac.uk/C.L.Mumford/papers/HeldKarp.pdf
use std::vec;

use graph::Graph;
use min1tree::get_min_1_tree;
use tsp::edge::Edge;

fn calc_stepsize(m: usize, big_m: usize, t_1: f64) -> f64 {
    let part1_num = ((m - 1) * (2 * big_m - 5)) as f64;
    let part1_den = (2 * (big_m - 1)) as f64;
    let part2 = (m - 2) as f64;
    let part3_num = ((m - 1) * (m - 2)) as f64;
    let part3_den = (2 * (big_m - 1) * (big_m - 2)) as f64;
    t_1 * (part1_num / part1_den - part2 + part3_num / part3_den)
}

fn calc_new_pi(pi: Vec<i64>, t_i: f64, degrees: &[usize], degrees_prev: &[usize]) -> Vec<i64> {
    pi.iter()
        .enumerate()
        .map(|(idx, val)| {
            val + (t_i * (0.6 * (degrees[idx] - 2) as f64 + 0.4 * (degrees_prev[idx] - 2) as f64))
                as i64
        })
        .collect()
}

pub fn run(graph: &mut Graph) -> (Vec<i64>, i64) {
    // calculate the first min1tree for pi = 0
    // According to Volgenant and Jonker:
    // Calculate t1, the basic stepsize, based on the weight of the min1tree
    //
    // iteratively:
    // Get the degrees of the min1tree d_i
    // calculate stepsize  t_i according to the Volgenant and Jonker Strategy.
    // Get pi_i given t_i, d_i and d_(i-1)
    // Calculate the new min1tree for pi_i
    let big_m = 1000;

    let n = graph.n();
    // pi is already initialized to vec![0; n] by default in Graph

    // no edges to City n - 1, that is the definition of our 1-tree here.
    let mut edges: Vec<Edge> = graph.edges().filter(|e| e.v.0 < n - 1).collect();

    let mut min_1_tree = get_min_1_tree(graph, Some(&mut edges));
    let mut degrees = vec![2; n];
    let mut degrees_prev;
    let t_1 = min_1_tree.total_weight as f64 / (2.0 * n as f64);

    for m in 1..=big_m {
        degrees_prev = degrees.clone();
        // Compute degrees from the min1tree edges
        degrees = vec![0; n];
        for edge in &min_1_tree.mst_edges {
            degrees[edge.u.0] += 1;
            degrees[edge.v.0] += 1;
        }
        // Add the two edges from the last city
        degrees[n - 1] = 2;
        degrees[min_1_tree.smallest_edge_last_city.u.0] += 1;
        degrees[min_1_tree.second_smallest_edge_last_city.u.0] += 1;

        let t_i = calc_stepsize(m, big_m, t_1);
        graph.pi = calc_new_pi(graph.pi.clone(), t_i, &degrees, &degrees_prev);
        min_1_tree = get_min_1_tree(graph, Some(&mut edges));
    }
    (vec![], 0)
}
