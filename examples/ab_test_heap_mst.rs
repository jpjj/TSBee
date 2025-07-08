use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;
use tsbee::input::Input;
use tsbee::penalties::distance::DistanceMatrix;
use tsbee::solver::Solver;

/// Generate a random TSP instance with n cities
fn generate_random_tsp_instance(n: usize, seed: u64) -> DistanceMatrix {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut matrix = vec![vec![0i64; n]; n];

    // Generate random points in 2D space
    let points: Vec<(f64, f64)> = (0..n)
        .map(|_| {
            use rand::Rng;
            (rng.random::<f64>() * 1000.0, rng.random::<f64>() * 1000.0)
        })
        .collect();

    // Calculate Euclidean distances
    for i in 0..n {
        for j in i + 1..n {
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            let dist = ((dx * dx + dy * dy).sqrt()) as i64;
            matrix[i][j] = dist;
            matrix[j][i] = dist;
        }
    }

    DistanceMatrix::new(matrix)
}

#[derive(Debug)]
#[allow(dead_code)]
struct TestResult {
    instance_id: usize,
    non_heap_distance: i64,
    heap_distance: i64,
    non_heap_time_ms: f64,
    heap_time_ms: f64,
    non_heap_iterations: u64,
    heap_iterations: u64,
}

fn main() {
    println!("=== A/B Test: Heap MST vs Non-Heap MST in TSP Solver ===\n");

    // Test parameters
    let n_cities = 1000;
    let time_limit_ms = 1000; // 0.1 seconds
    let n_instances = 20;

    println!("Test Configuration:");
    println!("  Cities per instance: {}", n_cities);
    println!("  Time limit: {} ms", time_limit_ms);
    println!("  Number of instances: {}\n", n_instances);

    let mut results = Vec::new();

    // Run tests on multiple instances
    for i in 0..n_instances {
        print!("Testing instance {}...", i + 1);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let seed = 42 + i as u64;
        let dm = generate_random_tsp_instance(n_cities, seed);
        let time_limit = chrono::Duration::milliseconds(time_limit_ms);

        // Test with non-heap MST
        let input_non_heap = Input::with_heap_mst(dm.clone(), Some(time_limit), false);
        let mut solver_non_heap = Solver::new(input_non_heap);
        let start = Instant::now();
        let result_non_heap = solver_non_heap.solve(true);
        let non_heap_time = start.elapsed().as_secs_f64() * 1000.0;

        // Test with heap MST
        let input_heap = Input::with_heap_mst(dm.clone(), Some(time_limit), true);
        let mut solver_heap = Solver::new(input_heap);
        let start = Instant::now();
        let result_heap = solver_heap.solve(true);
        let heap_time = start.elapsed().as_secs_f64() * 1000.0;

        results.push(TestResult {
            instance_id: i + 1,
            non_heap_distance: result_non_heap.best_solution.distance,
            heap_distance: result_heap.best_solution.distance,
            non_heap_time_ms: non_heap_time,
            heap_time_ms: heap_time,
            non_heap_iterations: result_non_heap.stats.iterations,
            heap_iterations: result_heap.stats.iterations,
        });

        println!(" done");
    }

    // Analyze results
    println!("\n=== Results Summary ===\n");

    // Quality comparison
    let mut heap_better = 0;
    let mut non_heap_better = 0;
    let mut equal = 0;
    let mut total_improvement = 0.0;

    for result in &results {
        use std::cmp::Ordering;
        match result.heap_distance.cmp(&result.non_heap_distance) {
            Ordering::Less => {
                heap_better += 1;
                let improvement = (result.non_heap_distance - result.heap_distance) as f64
                    / result.non_heap_distance as f64
                    * 100.0;
                total_improvement += improvement;
            }
            Ordering::Greater => {
                non_heap_better += 1;
                let improvement = (result.heap_distance - result.non_heap_distance) as f64
                    / result.heap_distance as f64
                    * 100.0;
                total_improvement -= improvement;
            }
            Ordering::Equal => {
                equal += 1;
            }
        }
    }

    println!("Solution Quality:");
    println!("  Heap MST better: {} instances", heap_better);
    println!("  Non-heap MST better: {} instances", non_heap_better);
    println!("  Equal quality: {} instances", equal);

    if heap_better > 0 || non_heap_better > 0 {
        let avg_improvement = total_improvement / n_instances as f64;
        if avg_improvement > 0.0 {
            println!("  Average improvement (Heap MST): {:.3}%", avg_improvement);
        } else {
            println!(
                "  Average improvement (Non-heap MST): {:.3}%",
                -avg_improvement
            );
        }
    }

    // Iteration comparison
    let avg_heap_iterations: f64 = results
        .iter()
        .map(|r| r.heap_iterations as f64)
        .sum::<f64>()
        / n_instances as f64;
    let avg_non_heap_iterations: f64 = results
        .iter()
        .map(|r| r.non_heap_iterations as f64)
        .sum::<f64>()
        / n_instances as f64;

    println!("\nIterations (within time limit):");
    println!("  Average with Heap MST: {:.1}", avg_heap_iterations);
    println!(
        "  Average with Non-heap MST: {:.1}",
        avg_non_heap_iterations
    );
    println!(
        "  Ratio (Heap/Non-heap): {:.2}x",
        avg_heap_iterations / avg_non_heap_iterations
    );

    // Statistical significance test (simplified paired t-test)
    let differences: Vec<f64> = results
        .iter()
        .map(|r| (r.non_heap_distance - r.heap_distance) as f64)
        .collect();

    let mean_diff = differences.iter().sum::<f64>() / n_instances as f64;
    let variance = differences
        .iter()
        .map(|d| (d - mean_diff).powi(2))
        .sum::<f64>()
        / (n_instances - 1) as f64;
    let std_error = (variance / n_instances as f64).sqrt();

    if std_error > 0.0 {
        let t_statistic = mean_diff / std_error;
        println!("\nStatistical Analysis:");
        println!("  Mean distance difference: {:.2}", mean_diff);
        println!("  t-statistic: {:.2}", t_statistic);

        // Simplified significance check (t > 2 roughly corresponds to p < 0.05 for moderate sample sizes)
        if t_statistic.abs() > 2.0 {
            if t_statistic > 0.0 {
                println!("  Result: Heap MST is statistically significantly better");
            } else {
                println!("  Result: Non-heap MST is statistically significantly better");
            }
        } else {
            println!("  Result: No statistically significant difference");
        }
    }

    // Show detailed results for first few instances
    println!("\n=== Detailed Results (first 5 instances) ===");
    println!(
        "{:<10} {:<15} {:<15} {:<15} {:<15}",
        "Instance", "Non-heap Dist", "Heap Dist", "Non-heap Iter", "Heap Iter"
    );
    println!("{:-<70}", "");

    for (i, result) in results.iter().take(5).enumerate() {
        println!(
            "{:<10} {:<15} {:<15} {:<15} {:<15}",
            i + 1,
            result.non_heap_distance,
            result.heap_distance,
            result.non_heap_iterations,
            result.heap_iterations
        );
    }
}
