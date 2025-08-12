use graph::{AdjacencyList, Graph};
use tsp::city::City;

#[derive(Clone, Copy, Debug)]
pub enum CandidateMethod {
    NearestNeighbor,
    AlphaNearness,
    HeldKarp,
    Delaunay,
}

pub fn get_candidates_graph<'a>(
    graph: &'a Graph<'a>,
    method: CandidateMethod,
    k: usize,
) -> Result<Graph<'a>, Box<dyn std::error::Error>> {
    match method {
        CandidateMethod::NearestNeighbor => get_nearest_neighbor_candidates(graph, k),
        CandidateMethod::AlphaNearness => get_alpha_nearness_candidates(graph, k),
        CandidateMethod::HeldKarp => get_held_karp_candidates(graph, k),
        CandidateMethod::Delaunay => get_delaunay_candidates(graph),
    }
}

fn get_nearest_neighbor_candidates<'a>(
    graph: &'a Graph<'a>,
    k: usize,
) -> Result<Graph<'a>, Box<dyn std::error::Error>> {
    let n = graph.n();
    let mut adjacency_list = vec![Vec::new(); n];

    for city in graph.cities() {
        let mut neighbors_with_weights: Vec<(City, f64)> = graph
            .cities()
            .filter(|&other| other != city)
            .map(|other| (other, graph.weight(city, other)))
            .collect();

        neighbors_with_weights.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let k_th_weight = if neighbors_with_weights.len() > k {
            neighbors_with_weights[k - 1].1
        } else {
            f64::INFINITY
        };

        let candidates: Vec<City> = neighbors_with_weights
            .into_iter()
            .filter(|(_, weight)| *weight <= k_th_weight)
            .map(|(neighbor, _)| neighbor)
            .collect();

        adjacency_list[city.0] = candidates;
    }

    Ok(Graph::new_list(graph.problem(), adjacency_list))
}

fn build_candidate_graph_from_alpha_values<'a>(
    graph: &'a Graph,
    alpha_values: Vec<f64>,
    k: usize,
) -> Result<Graph<'a>, Box<dyn std::error::Error>> {
    let n = graph.n();
    let mut adjacency_list = vec![Vec::new(); n];

    for city in graph.cities() {
        let mut neighbors_with_alpha: Vec<(City, f64)> = graph
            .cities()
            .filter(|&other| other != city)
            .map(|other| {
                let alpha_idx = city.0 * n + other.0;
                (other, alpha_values[alpha_idx])
            })
            .collect();

        neighbors_with_alpha.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let k_th_alpha = if neighbors_with_alpha.len() > k {
            neighbors_with_alpha[k - 1].1
        } else {
            f64::INFINITY
        };

        let candidates: Vec<City> = neighbors_with_alpha
            .into_iter()
            .filter(|(_, alpha)| *alpha <= k_th_alpha)
            .map(|(neighbor, _)| neighbor)
            .collect();

        adjacency_list[city.0] = candidates;
    }

    Ok(Graph::new_list(graph.problem(), adjacency_list))
}

fn get_alpha_nearness_candidates<'a>(
    graph: &'a Graph<'a>,
    k: usize,
) -> Result<Graph<'a>, Box<dyn std::error::Error>> {
    let alpha_values = alpha_nearness::get_alpha_values(graph);
    build_candidate_graph_from_alpha_values(graph, alpha_values, k)
}

fn get_held_karp_candidates<'a>(
    graph: &'a Graph<'a>,
    k: usize,
) -> Result<Graph<'a>, Box<dyn std::error::Error>> {
    let mut graph_copy = Graph::new_matrix(graph.problem());
    graph_copy.pi = graph.pi.clone();

    let (pi, _) = held_karp::run(&mut graph_copy);

    let mut updated_graph = Graph::new_matrix(graph.problem());
    updated_graph.pi = pi;

    let alpha_values = alpha_nearness::get_alpha_values(&updated_graph);
    build_candidate_graph_from_alpha_values(graph, alpha_values, k)
}

fn get_delaunay_candidates<'a>(
    graph: &'a Graph<'a>,
) -> Result<Graph<'a>, Box<dyn std::error::Error>> {
    let adjacency_list = AdjacencyList::from_delaunay(graph.problem())?;
    Ok(Graph::new_list(graph.problem(), adjacency_list.list))
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph::get_solution_as_graph;
    use std::path::PathBuf;

    fn load_and_test<F>(problem_name: &str, solution_name: &str, test_fn: F)
    where
        F: FnOnce(&Graph, &Graph),
    {
        let benchmarks_path = PathBuf::from("../benchmarks");
        let problem_path = benchmarks_path.join(format!("data/problems/{problem_name}"));
        let solution_path = benchmarks_path.join(format!("data/solutions/{solution_name}"));

        let problem = benchmarks::read_problem_file::read_problem_file(&problem_path)
            .unwrap_or_else(|_| panic!("Failed to read {problem_name}"));
        let solution: tsp::solution::list::List<tsp::problem::TspProblem> =
            benchmarks::read_tour_file::read_tour_file(&solution_path)
                .unwrap_or_else(|_| panic!("Failed to read {solution_name}"));

        let graph = Graph::new_matrix(&problem);
        let optimal_graph = get_solution_as_graph(&problem, solution);

        test_fn(&graph, &optimal_graph);
    }

    fn calculate_edge_coverage(candidate_graph: &Graph, optimal_solution_graph: &Graph) -> f64 {
        let optimal_edges: std::collections::HashSet<_> = optimal_solution_graph.edges().collect();
        let candidate_edges: std::collections::HashSet<_> = candidate_graph.edges().collect();

        let covered_edges = optimal_edges.intersection(&candidate_edges).count();
        let total_optimal_edges = optimal_edges.len();

        if total_optimal_edges == 0 {
            0.0
        } else {
            covered_edges as f64 / total_optimal_edges as f64
        }
    }

    #[test]
    fn test_nearest_neighbor_berlin52() {
        load_and_test(
            "berlin52.tsp",
            "berlin52.opt.tour",
            |graph, optimal_graph| {
                let candidate_graph =
                    get_candidates_graph(graph, CandidateMethod::NearestNeighbor, 5)
                        .expect("Failed to get candidates");

                let coverage = calculate_edge_coverage(&candidate_graph, optimal_graph);
                println!("NearestNeighbor (k=5) coverage: {:.2}%", coverage * 100.0);

                assert!(coverage > 0.0, "Should cover some optimal edges");
            },
        );
    }

    #[test]
    fn test_alpha_nearness_berlin52() {
        load_and_test(
            "berlin52.tsp",
            "berlin52.opt.tour",
            |graph, optimal_graph| {
                let candidate_graph =
                    get_candidates_graph(graph, CandidateMethod::AlphaNearness, 5)
                        .expect("Failed to get candidates");

                let coverage = calculate_edge_coverage(&candidate_graph, optimal_graph);
                println!("AlphaNearness (k=5) coverage: {:.2}%", coverage * 100.0);

                assert!(coverage > 0.0, "Should cover some optimal edges");
            },
        );
    }

    #[test]
    fn test_held_karp_berlin52() {
        load_and_test(
            "berlin52.tsp",
            "berlin52.opt.tour",
            |graph, optimal_graph| {
                let candidate_graph = get_candidates_graph(graph, CandidateMethod::HeldKarp, 5)
                    .expect("Failed to get candidates");

                let coverage = calculate_edge_coverage(&candidate_graph, optimal_graph);
                println!("HeldKarp (k=5) coverage: {:.2}%", coverage * 100.0);

                assert!(coverage > 0.0, "Should cover some optimal edges");
            },
        );
    }

    #[test]
    fn test_delaunay_att48() {
        load_and_test("att48.tsp", "att48.opt.tour", |graph, optimal_graph| {
            let candidate_graph = get_candidates_graph(graph, CandidateMethod::Delaunay, 0)
                .expect("Failed to get candidates");

            let coverage = calculate_edge_coverage(&candidate_graph, optimal_graph);
            println!("Delaunay coverage: {:.2}%", coverage * 100.0);

            assert!(coverage > 0.0, "Should cover some optimal edges");
        });
    }
}
