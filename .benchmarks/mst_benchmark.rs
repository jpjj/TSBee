use petgraph::{
    algo::min_spanning_tree,
    data::FromElements,
    graph::{NodeIndex, UnGraph},
    Graph,
};
use rand::random_range;
use std::time::{Duration, Instant};
use tsbee::{
    domain::city::City,
    penalties::{
        candidates::{
            alpha_nearness::get_alpha_candidates_v2,
            heap_mst::{
                get_min_spanning_tree_boruvka, get_min_spanning_tree_contract,
                get_min_spanning_tree_heap_optimized, get_min_spanning_tree_linear,
                get_min_spanning_tree_randomized,
            },
            utils::get_min_spanning_tree,
            Candidates,
        },
        distance::DistanceMatrix,
    },
};

fn generate_random_distance_matrix(n: usize) -> DistanceMatrix {
    let mut random_matrix: Vec<Vec<i64>> = (0..n)
        .map(|_| (0..n).map(|_| random_range(1..=1000)).collect())
        .collect();

    // Make symmetric
    for i in 0..n {
        random_matrix[i][i] = 0;
        for j in i + 1..n {
            random_matrix[j][i] = random_matrix[i][j];
        }
    }

    DistanceMatrix::new(random_matrix)
}

fn benchmark_petgraph(dm: &DistanceMatrix, n: usize) -> (i64, Duration) {
    let start = Instant::now();

    let mut graph = UnGraph::<i64, i64>::new_undirected();
    for _ in 0..n {
        graph.add_node(0);
    }

    for i in 0..n {
        let node_index_i = NodeIndex::new(i);
        for j in i + 1..n {
            let node_index_j = NodeIndex::new(j);
            graph.add_edge(node_index_i, node_index_j, dm.distance(City(i), City(j)));
        }
    }

    let mst_petgraph: Graph<i64, i64> = Graph::from_elements(min_spanning_tree(&graph));
    let score: i64 = mst_petgraph.edge_references().map(|e| e.weight()).sum();

    let duration = start.elapsed();
    (score, duration)
}

fn benchmark_heap_optimized(
    dm: &DistanceMatrix,
    n: usize,
    candidates: &Candidates,
) -> (i64, Duration) {
    let start = Instant::now();

    let mst = get_min_spanning_tree_heap_optimized(dm, candidates, n);

    let duration = start.elapsed();
    (mst.score, duration)
}

fn benchmark_boruvka(dm: &DistanceMatrix, n: usize, candidates: &Candidates) -> (i64, Duration) {
    let start = Instant::now();

    let mst = get_min_spanning_tree_boruvka(dm, candidates, n);

    let duration = start.elapsed();
    (mst.score, duration)
}

fn benchmark_linear(dm: &DistanceMatrix, n: usize, candidates: &Candidates) -> (i64, Duration) {
    let start = Instant::now();

    let mst = get_min_spanning_tree_linear(dm, candidates, n);

    let duration = start.elapsed();
    (mst.score, duration)
}

fn benchmark_randomized(dm: &DistanceMatrix, n: usize, candidates: &Candidates) -> (i64, Duration) {
    let start = Instant::now();

    let mst = get_min_spanning_tree_randomized(dm, candidates, n);

    let duration = start.elapsed();
    (mst.score, duration)
}

fn benchmark_contract(dm: &DistanceMatrix, n: usize, candidates: &Candidates) -> (i64, Duration) {
    let start = Instant::now();

    let mst = get_min_spanning_tree_contract(dm, candidates, n);

    let duration = start.elapsed();
    (mst.score, duration)
}

fn benchmark_prim_complete(dm: &DistanceMatrix, n: usize, _alpha: usize) -> (i64, Duration) {
    let start = Instant::now();

    let mst = get_min_spanning_tree(dm, n);

    let duration = start.elapsed();
    (mst.score, duration)
}

fn main() {
    println!("MST Algorithm Benchmarks");
    println!("========================\n");

    let node_counts = vec![100, 500, 1000, 2000];
    let alpha_values = vec![10, 20];
    let num_runs = 3;

    for &n in &node_counts {
        println!("\nNodes: {}", n);
        println!("-----------");

        // Generate random distance matrix for this size
        let dm = generate_random_distance_matrix(n);
        // Benchmark Petgraph
        let mut petgraph_times = Vec::new();
        let mut petgraph_score = 0;

        for _ in 0..num_runs {
            let (score, duration) = benchmark_petgraph(&dm, n);
            petgraph_score = score;
            petgraph_times.push(duration);
        }

        let avg_petgraph_time = petgraph_times.iter().sum::<Duration>() / num_runs as u32;
        println!(
            "Petgraph: {:.3}ms (score: {})",
            avg_petgraph_time.as_secs_f64() * 1000.0,
            petgraph_score
        );

        // Benchmark heap optimized with different alpha values
        for &alpha in &alpha_values {
            if alpha >= n {
                continue;
            }
            let candidates = get_alpha_candidates_v2(&dm, alpha, true);

            println!("\nα = {}", alpha);

            // Heap Optimized
            let mut heap_times = Vec::new();
            let mut heap_score = 0;

            for _ in 0..num_runs {
                let (score, duration) = benchmark_heap_optimized(&dm, n, &candidates);
                heap_score = score;
                heap_times.push(duration);
            }

            let avg_heap_time = heap_times.iter().sum::<Duration>() / num_runs as u32;
            let speedup = avg_petgraph_time.as_secs_f64() / avg_heap_time.as_secs_f64();

            println!(
                "  Heap Optimized: {:.3}ms (score: {}) - {:.2}x speedup",
                avg_heap_time.as_secs_f64() * 1000.0,
                heap_score,
                speedup
            );

            if heap_score != petgraph_score {
                println!(
                    "    WARNING: Different MST score! Expected: {}, Got: {}",
                    petgraph_score, heap_score
                );
            }

            // Boruvka
            let mut boruvka_times = Vec::new();
            let mut boruvka_score = 0;

            for _ in 0..num_runs {
                let (score, duration) = benchmark_boruvka(&dm, n, &candidates);
                boruvka_score = score;
                boruvka_times.push(duration);
            }

            let avg_boruvka_time = boruvka_times.iter().sum::<Duration>() / num_runs as u32;
            let speedup = avg_petgraph_time.as_secs_f64() / avg_boruvka_time.as_secs_f64();

            println!(
                "  Boruvka:        {:.3}ms (score: {}) - {:.2}x speedup",
                avg_boruvka_time.as_secs_f64() * 1000.0,
                boruvka_score,
                speedup
            );

            if boruvka_score != petgraph_score {
                println!(
                    "    WARNING: Different MST score! Expected: {}, Got: {}",
                    petgraph_score, boruvka_score
                );
            }

            // Linear
            let mut linear_times = Vec::new();
            let mut linear_score = 0;

            for _ in 0..num_runs {
                let (score, duration) = benchmark_linear(&dm, n, &candidates);
                linear_score = score;
                linear_times.push(duration);
            }

            let avg_linear_time = linear_times.iter().sum::<Duration>() / num_runs as u32;
            let speedup = avg_petgraph_time.as_secs_f64() / avg_linear_time.as_secs_f64();

            println!(
                "  Linear:         {:.3}ms (score: {}) - {:.2}x speedup",
                avg_linear_time.as_secs_f64() * 1000.0,
                linear_score,
                speedup
            );

            if linear_score != petgraph_score {
                println!(
                    "    WARNING: Different MST score! Expected: {}, Got: {}",
                    petgraph_score, linear_score
                );
            }

            // Randomized
            let mut randomized_times = Vec::new();
            let mut randomized_score = 0;

            for _ in 0..num_runs {
                let (score, duration) = benchmark_randomized(&dm, n, &candidates);
                randomized_score = score;
                randomized_times.push(duration);
            }

            let avg_randomized_time = randomized_times.iter().sum::<Duration>() / num_runs as u32;
            let speedup = avg_petgraph_time.as_secs_f64() / avg_randomized_time.as_secs_f64();

            println!(
                "  Randomized:     {:.3}ms (score: {}) - {:.2}x speedup",
                avg_randomized_time.as_secs_f64() * 1000.0,
                randomized_score,
                speedup
            );

            if randomized_score != petgraph_score {
                println!(
                    "    WARNING: Different MST score! Expected: {}, Got: {}",
                    petgraph_score, randomized_score
                );
            }

            // Contract
            let mut contract_times = Vec::new();
            let mut contract_score = 0;

            for _ in 0..num_runs {
                let (score, duration) = benchmark_contract(&dm, n, &candidates);
                contract_score = score;
                contract_times.push(duration);
            }

            let avg_contract_time = contract_times.iter().sum::<Duration>() / num_runs as u32;
            let speedup = avg_petgraph_time.as_secs_f64() / avg_contract_time.as_secs_f64();

            println!(
                "  Contract:       {:.3}ms (score: {}) - {:.2}x speedup",
                avg_contract_time.as_secs_f64() * 1000.0,
                contract_score,
                speedup
            );

            if contract_score != petgraph_score {
                println!(
                    "    WARNING: Different MST score! Expected: {}, Got: {}",
                    petgraph_score, contract_score
                );
            }

            // prim_complete
            let mut prim_complete_times = Vec::new();
            let mut prim_complete_score = 0;

            for _ in 0..num_runs {
                let (score, duration) = benchmark_prim_complete(&dm, n, alpha);
                prim_complete_score = score;
                prim_complete_times.push(duration);
            }

            let avg_prim_complete_time =
                prim_complete_times.iter().sum::<Duration>() / num_runs as u32;
            let speedup = avg_petgraph_time.as_secs_f64() / avg_prim_complete_time.as_secs_f64();

            println!(
                "  Prim Complete:  {:.3}ms (score: {}) - {:.2}x speedup",
                avg_prim_complete_time.as_secs_f64() * 1000.0,
                prim_complete_score,
                speedup
            );

            if prim_complete_score != petgraph_score {
                println!(
                    "    WARNING: Different MST score! Expected: {}, Got: {}",
                    petgraph_score, prim_complete_score
                );
            }
        }
    }
}
