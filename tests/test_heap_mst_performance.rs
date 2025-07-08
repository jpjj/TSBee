use rand::{rngs::StdRng, SeedableRng};
use tsbee::domain::city::City;
use tsbee::penalties::{candidates::held_karp::BoundCalculator, distance::DistanceMatrix};

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

fn compare_held_karp_implementations(dm: &DistanceMatrix, upper_bound: i64) -> (i64, i64) {
    let n = dm.len();
    let max_iterations = 50;
    let max_time = chrono::Duration::milliseconds(100);

    // Test old implementation
    let mut old_calculator =
        BoundCalculator::new(dm.clone(), upper_bound, max_iterations, max_time);
    let old_result = old_calculator.run();

    // Test new implementation with candidates
    let candidates = if n <= 3 {
        tsbee::penalties::candidates::alpha_nearness::get_nn_candidates(dm, n.saturating_sub(1))
    } else {
        tsbee::penalties::candidates::alpha_nearness::get_alpha_candidates_v2(dm, n.min(15), true)
    };

    let mut new_calculator = BoundCalculator::with_candidates(
        dm.clone(),
        candidates,
        upper_bound,
        max_iterations,
        max_time,
    );
    let new_result = new_calculator.run();

    (old_result.min_one_tree.score, new_result.min_one_tree.score)
}

fn get_simple_upper_bound(dm: &DistanceMatrix) -> i64 {
    let n = dm.len();
    let mut visited = vec![false; n];
    let mut tour_length = 0i64;
    let mut current = 0;
    visited[0] = true;

    for _ in 1..n {
        let mut nearest = None;
        let mut min_dist = i64::MAX;

        for (j, &is_visited) in visited.iter().enumerate().take(n) {
            if !is_visited {
                let dist = dm.distance(City(current), City(j));
                if dist < min_dist {
                    min_dist = dist;
                    nearest = Some(j);
                }
            }
        }

        if let Some(next) = nearest {
            tour_length += min_dist;
            current = next;
            visited[next] = true;
        }
    }
    tour_length += dm.distance(City(current), City(0));
    tour_length
}

#[test]
fn test_heap_mst_correctness() {
    let test_sizes = vec![10, 20, 30, 50];
    let iterations = 10;

    for &size in &test_sizes {
        for i in 0..iterations {
            let seed = 42 + i as u64;
            let dm = generate_random_tsp_instance(size, seed);
            let upper_bound = get_simple_upper_bound(&dm);

            let (old_score, new_score) = compare_held_karp_implementations(&dm, upper_bound);

            // The heap-based MST uses candidate edges which may not include all optimal edges
            // So we allow a small difference in the final bounds
            let diff = (old_score - new_score).abs();
            let tolerance = (old_score.abs() as f64 * 0.05).max(50.0) as i64; // 5% tolerance or at least 50

            assert!(
                diff <= tolerance,
                "MST scores differ too much for size {}, seed {}: old={}, new={}, diff={}, tolerance={}",
                size, seed, old_score, new_score, diff, tolerance
            );
        }
    }
}

#[test]
fn test_heap_mst_edge_cases() {
    // Test very small instances where candidates should include all edges
    for n in 3..=5 {
        let dm = generate_random_tsp_instance(n, 999);
        let upper_bound = get_simple_upper_bound(&dm);

        let (old_score, new_score) = compare_held_karp_implementations(&dm, upper_bound);

        // For very small instances, the difference should be minimal
        let diff = (old_score - new_score).abs();

        assert!(
            diff <= 10,
            "MST scores differ for small instance size {}: old={}, new={}, diff={}",
            n,
            old_score,
            new_score,
            diff
        );
    }
}
