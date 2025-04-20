/// Implement the 2-opt algorithm for the TSP
fn two_opt(distance_matrix: &[Vec<i64>], time_limit: Option<f64>) -> (Vec<usize>, u64, usize) {
    let n = distance_matrix.len();

    // Start with a simple tour: 0, 1, 2, ..., n-1
    let mut best_tour: Vec<usize> = (0..n).collect();
    let mut best_distance = calculate_tour_distance(&best_tour, distance_matrix);

    let start_time = Instant::now();
    let mut iterations = 0;
    let mut improved = true;

    while improved {
        improved = false;
        iterations += 1;

        // Check if we've exceeded the time limit
        if let Some(limit) = time_limit {
            if start_time.elapsed().as_secs_f64() > limit {
                break;
            }
        }

        // Try all possible 2-opt swaps
        for i in 0..n - 2 {
            for j in i + 2..n {
                // Skip if we'd create a loop smaller than the entire tour
                if j - i == n - 1 {
                    continue;
                }

                // Calculate the change in distance if we were to reverse the segment
                let current_distance = distance_matrix[best_tour[i]][best_tour[i + 1]]
                    + distance_matrix[best_tour[j]][best_tour[(j + 1) % n]];

                let new_distance = distance_matrix[best_tour[i]][best_tour[j]]
                    + distance_matrix[best_tour[i + 1]][best_tour[(j + 1) % n]];

                // If the new path would be shorter
                if new_distance < current_distance {
                    // Reverse the segment
                    best_tour[i + 1..=j].reverse();

                    // Update the best distance
                    best_distance = calculate_tour_distance(&best_tour, distance_matrix);

                    improved = true;
                    break;
                }
            }

            if improved {
                break;
            }
        }
    }

    (best_tour, best_distance, iterations)
}
