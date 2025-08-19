use candidates::{CandidateMethod, get_candidates_graph};
use graph::{Graph, get_solution_as_graph};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tsp::problem::{Problem, TspProblem};

const MIN_SIZE: usize = 50;
const MAX_SIZE: usize = 200;
const K: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let benchmarks_path = PathBuf::from("../benchmarks");
    let problems_dir = benchmarks_path.join("data/problems");
    let solutions_dir = benchmarks_path.join("data/solutions");

    let mut results = Vec::new();

    for entry in std::fs::read_dir(&problems_dir)? {
        let entry = entry?;
        let problem_path = entry.path();

        if problem_path.extension().is_none_or(|ext| ext != "tsp") {
            continue;
        }

        let problem_name = problem_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let problem = match benchmarks::read_problem_file::read_problem_file(&problem_path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to read {problem_name}: {e}");
                continue;
            }
        };

        let problem_size = problem.size();

        if !(MIN_SIZE..=MAX_SIZE).contains(&problem_size) {
            continue;
        }

        println!("Processing {problem_name} (size: {problem_size})");

        let solution_path = solutions_dir.join(format!("{problem_name}.opt.tour"));
        let solution: tsp::solution::list::List<tsp::problem::TspProblem> =
            match benchmarks::read_tour_file::read_tour_file(&solution_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to read solution for {problem_name}: {e}");
                    continue;
                }
            };

        let graph = Graph::new_matrix(&problem);
        let optimal_graph = get_solution_as_graph(&problem, solution);

        let methods = [
            CandidateMethod::NearestNeighbor,
            CandidateMethod::AlphaNearness,
            CandidateMethod::HeldKarp,
            CandidateMethod::Delaunay,
        ];

        for method in methods {
            // only take either Delaunay PointsAndFunction combination or not Delaunay DistanceMatrix combination
            if matches!(method, CandidateMethod::Delaunay) {
                if matches!(problem, TspProblem::DistanceMatrix(_)) {
                    continue;
                }
            } else if !matches!(problem, TspProblem::DistanceMatrix(_)) {
                continue;
            }

            let method_name = format!("{method:?}");
            let mut percentages = Vec::new();

            for k_value in 1..=K {
                let coverage = if matches!(method, CandidateMethod::Delaunay) {
                    if k_value == 1 {
                        match get_candidates_graph(&graph, method, 0) {
                            Ok(candidate_graph) => {
                                calculate_edge_coverage(&candidate_graph, &optimal_graph)
                            }
                            Err(e) => {
                                eprintln!(
                                    "Error generating Delaunay candidates for {problem_name}: {e}"
                                );
                                0.0
                            }
                        }
                    } else {
                        percentages.last().copied().unwrap_or(0.0)
                    }
                } else {
                    match get_candidates_graph(&graph, method, k_value) {
                        Ok(candidate_graph) => {
                            calculate_edge_coverage(&candidate_graph, &optimal_graph)
                        }
                        Err(e) => {
                            eprintln!(
                                "Error generating candidates for {problem_name} with method {method:?}, k={k_value}: {e}"
                            );
                            0.0
                        }
                    }
                };

                percentages.push(coverage);
            }

            results.push((problem_name.clone(), method_name, percentages));
        }
    }

    write_csv(&results)?;

    println!("Results saved to candidates/data/benchmark_results.csv");
    Ok(())
}

fn calculate_edge_coverage(candidate_graph: &Graph, optimal_solution_graph: &Graph) -> f64 {
    let optimal_edges: HashSet<_> = optimal_solution_graph.edges().collect();
    let candidate_edges: HashSet<_> = candidate_graph.edges().collect();

    let covered_edges = optimal_edges.intersection(&candidate_edges).count();
    let total_optimal_edges = optimal_edges.len();

    if total_optimal_edges == 0 {
        0.0
    } else {
        covered_edges as f64 / total_optimal_edges as f64
    }
}

fn write_csv(results: &[(String, String, Vec<f64>)]) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = PathBuf::from("../candidates/data/benchmark_results.csv");
    let mut file = File::create(output_path)?;

    let mut header = String::from("Benchmark_Instance_Name,Method");
    for k in 1..=K {
        header.push_str(&format!(",{k}"));
    }
    writeln!(file, "{header}")?;

    for (instance, method, percentages) in results {
        let mut row = format!("{instance},{method}");
        for percentage in percentages {
            row.push_str(&format!(",{percentage:.4}"));
        }
        writeln!(file, "{row}")?;
    }

    Ok(())
}
