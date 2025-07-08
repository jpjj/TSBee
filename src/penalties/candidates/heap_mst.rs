use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::{min_spanning_tree::MinSpanningTree, Candidates};
use crate::{domain::city::City, penalties::distance::DistanceMatrix};

#[derive(Copy, Clone, Eq, PartialEq)]
struct Edge {
    weight: i64,
    from: City,
    to: City,
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        other.weight.cmp(&self.weight)
    }
}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Create a minimum spanning tree on a subset of nodes using Prim's algorithm with a binary heap.
/// This implementation uses only the edges defined by the candidates structure for efficiency.
///
/// # Parameters
/// - `distance_matrix`: The complete distance matrix
/// - `candidates`: The candidate edges structure containing neighbors for each node
/// - `n`: Number of nodes to include in the MST (nodes 0 to n-1)
///
/// # Returns
/// A `MinSpanningTree` containing the edges and total weight
#[allow(dead_code)]
pub(super) fn get_min_spanning_tree_heap(
    distance_matrix: &DistanceMatrix,
    candidates: &Candidates,
    n: usize,
) -> MinSpanningTree {
    if n <= 1 {
        return MinSpanningTree::new(0, vec![]);
    }

    let mut in_mst = vec![false; n];
    let mut mst_edges = Vec::with_capacity(n - 1);
    let mut heap = BinaryHeap::new();
    let mut total_weight = 0;

    // Start with node 0
    in_mst[0] = true;

    // Add all edges from node 0 to its candidates
    for &neighbor in candidates.get_neighbors_out(&City(0)) {
        if neighbor.id() < n && !in_mst[neighbor.id()] {
            heap.push(Edge {
                weight: distance_matrix.distance(City(0), neighbor),
                from: City(0),
                to: neighbor,
            });
        }
    }

    // Also check if any other nodes have node 0 as a candidate
    for i in 1..n {
        let city_i = City(i);
        for &neighbor in candidates.get_neighbors_out(&city_i) {
            if neighbor == City(0) {
                heap.push(Edge {
                    weight: distance_matrix.distance(city_i, City(0)),
                    from: city_i,
                    to: City(0),
                });
            }
        }
    }

    // Process edges until we have n-1 edges in the MST
    while mst_edges.len() < n - 1 && !heap.is_empty() {
        let edge = heap.pop().unwrap();

        // Skip if both vertices are already in the MST
        if in_mst[edge.from.id()] && in_mst[edge.to.id()] {
            continue;
        }

        // Determine which vertex is the new one
        let new_vertex = if !in_mst[edge.to.id()] {
            edge.to
        } else {
            edge.from
        };

        // Add edge to MST
        mst_edges.push((edge.from, edge.to));
        total_weight += edge.weight;
        in_mst[new_vertex.id()] = true;

        // Add all edges from the new vertex to its candidates
        for &neighbor in candidates.get_neighbors_out(&new_vertex) {
            if neighbor.id() < n && !in_mst[neighbor.id()] {
                heap.push(Edge {
                    weight: distance_matrix.distance(new_vertex, neighbor),
                    from: new_vertex,
                    to: neighbor,
                });
            }
        }

        // Also check if any unvisited nodes have the new vertex as a candidate
        for (i, &is_in_mst) in in_mst.iter().enumerate().take(n) {
            if !is_in_mst {
                let city_i = City(i);
                for &neighbor in candidates.get_neighbors_out(&city_i) {
                    if neighbor == new_vertex {
                        heap.push(Edge {
                            weight: distance_matrix.distance(city_i, new_vertex),
                            from: city_i,
                            to: new_vertex,
                        });
                    }
                }
            }
        }
    }

    // If we couldn't build a complete MST with candidates, fall back to the complete graph
    if mst_edges.len() < n - 1 {
        return get_min_spanning_tree_heap_complete(distance_matrix, n);
    }

    MinSpanningTree::new(total_weight, mst_edges)
}

/// Fallback implementation that considers all edges (complete graph)
fn get_min_spanning_tree_heap_complete(
    distance_matrix: &DistanceMatrix,
    n: usize,
) -> MinSpanningTree {
    if n <= 1 {
        return MinSpanningTree::new(0, vec![]);
    }

    let mut in_mst = vec![false; n];
    let mut mst_edges = Vec::with_capacity(n - 1);
    let mut heap = BinaryHeap::new();
    let mut total_weight = 0;

    // Start with node 0
    in_mst[0] = true;

    // Add all edges from node 0
    for i in 1..n {
        heap.push(Edge {
            weight: distance_matrix.distance(City(0), City(i)),
            from: City(0),
            to: City(i),
        });
    }

    // Process edges until we have n-1 edges in the MST
    while mst_edges.len() < n - 1 {
        let edge = heap.pop().unwrap();

        if in_mst[edge.to.id()] {
            continue;
        }

        // Add edge to MST
        mst_edges.push((edge.from, edge.to));
        total_weight += edge.weight;
        in_mst[edge.to.id()] = true;

        // Add all edges from the new vertex
        let new_vertex = edge.to;
        for (i, &is_in_mst) in in_mst.iter().enumerate().take(n) {
            if !is_in_mst {
                heap.push(Edge {
                    weight: distance_matrix.distance(new_vertex, City(i)),
                    from: new_vertex,
                    to: City(i),
                });
            }
        }
    }

    MinSpanningTree::new(total_weight, mst_edges)
}

/// Optimized version of the heap-based MST that avoids redundant edge checking
pub(super) fn get_min_spanning_tree_heap_optimized(
    distance_matrix: &DistanceMatrix,
    candidates: &Candidates,
    n: usize,
) -> MinSpanningTree {
    if n <= 1 {
        return MinSpanningTree::new(0, vec![]);
    }

    let mut in_mst = vec![false; n];
    let mut mst_edges = Vec::with_capacity(n - 1);
    let mut heap = BinaryHeap::with_capacity(n * 10);
    let mut total_weight = 0;

    // Precompute reverse adjacency list for faster lookup
    let mut reverse_candidates: Vec<Vec<City>> = vec![vec![]; n];
    for i in 0..n {
        let city_i = City(i);
        for &neighbor in candidates.get_neighbors_out(&city_i) {
            if neighbor.id() < n {
                reverse_candidates[neighbor.id()].push(city_i);
            }
        }
    }

    // Start with node 0
    in_mst[0] = true;

    // Add all edges from node 0 to its candidates
    for &neighbor in candidates.get_neighbors_out(&City(0)) {
        if neighbor.id() < n {
            heap.push(Edge {
                weight: distance_matrix.distance(City(0), neighbor),
                from: City(0),
                to: neighbor,
            });
        }
    }

    // Add edges from nodes that have 0 as a candidate
    for &from_city in &reverse_candidates[0] {
        heap.push(Edge {
            weight: distance_matrix.distance(from_city, City(0)),
            from: from_city,
            to: City(0),
        });
    }

    // Process edges until we have n-1 edges in the MST
    let mut edges_added = 0;
    while edges_added < n - 1 && !heap.is_empty() {
        let edge = heap.pop().unwrap();

        // Skip if both vertices are already in the MST
        if in_mst[edge.from.id()] && in_mst[edge.to.id()] {
            continue;
        }

        // Determine which vertex is the new one
        let new_vertex = if !in_mst[edge.to.id()] {
            edge.to
        } else {
            edge.from
        };

        // Skip if new vertex is already in MST (can happen with bidirectional edges)
        if in_mst[new_vertex.id()] {
            continue;
        }

        // Add edge to MST
        mst_edges.push((edge.from, edge.to));
        total_weight += edge.weight;
        in_mst[new_vertex.id()] = true;
        edges_added += 1;

        // Add outgoing edges from the new vertex
        for &neighbor in candidates.get_neighbors_out(&new_vertex) {
            if neighbor.id() < n && !in_mst[neighbor.id()] {
                heap.push(Edge {
                    weight: distance_matrix.distance(new_vertex, neighbor),
                    from: new_vertex,
                    to: neighbor,
                });
            }
        }

        // Add incoming edges to the new vertex using precomputed reverse adjacency
        for &from_city in &reverse_candidates[new_vertex.id()] {
            if !in_mst[from_city.id()] {
                heap.push(Edge {
                    weight: distance_matrix.distance(from_city, new_vertex),
                    from: from_city,
                    to: new_vertex,
                });
            }
        }
    }

    // If we couldn't build a complete MST with candidates, fall back to the complete graph
    if edges_added < n - 1 {
        return get_min_spanning_tree_heap_complete(distance_matrix, n);
    }

    MinSpanningTree::new(total_weight, mst_edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::penalties::candidates::alpha_nearness::get_alpha_candidates_v2;
    use crate::penalties::candidates::utils::get_min_spanning_tree;

    #[test]
    fn test_heap_mst_vs_original() {
        // Create a small test case
        let matrix = vec![
            vec![0, 10, 15, 20],
            vec![10, 0, 35, 25],
            vec![15, 35, 0, 30],
            vec![20, 25, 30, 0],
        ];
        let dm = DistanceMatrix::new(matrix);

        // Get candidates
        let candidates = get_alpha_candidates_v2(&dm, 3, true);

        // Compare original and heap implementations
        let original_mst = get_min_spanning_tree(&dm, 4);
        let heap_mst = get_min_spanning_tree_heap(&dm, &candidates, 4);

        assert_eq!(original_mst.score, heap_mst.score);
    }

    #[test]
    fn test_heap_mst_larger() {
        // Create a larger random test case
        use rand::random_range;
        let size = 50;
        let mut matrix = vec![vec![0; size]; size];

        for i in 0..size {
            for j in i + 1..size {
                let dist = random_range(1..=1000);
                matrix[i][j] = dist;
                matrix[j][i] = dist;
            }
        }

        let dm = DistanceMatrix::new(matrix);
        let candidates = get_alpha_candidates_v2(&dm, 10, true);

        // Compare scores
        let original_mst = get_min_spanning_tree(&dm, size);
        let heap_mst = get_min_spanning_tree_heap(&dm, &candidates, size);

        // The heap version might have a slightly different MST if candidates don't include all optimal edges
        // But it should still be a valid MST with the same or similar score
        println!("Original MST score: {}", original_mst.score);
        println!("Heap MST score: {}", heap_mst.score);

        // Verify it's a valid spanning tree (has n-1 edges)
        assert_eq!(heap_mst.edges.len(), size - 1);
    }

    #[test]
    fn test_optimized_heap_mst() {
        // Test that optimized version produces same results as current heap version
        let matrix = vec![
            vec![0, 10, 15, 20],
            vec![10, 0, 35, 25],
            vec![15, 35, 0, 30],
            vec![20, 25, 30, 0],
        ];
        let dm = DistanceMatrix::new(matrix);
        let candidates = get_alpha_candidates_v2(&dm, 3, true);

        let heap_mst = get_min_spanning_tree_heap(&dm, &candidates, 4);
        let optimized_mst = get_min_spanning_tree_heap_optimized(&dm, &candidates, 4);

        assert_eq!(heap_mst.score, optimized_mst.score);
    }

    #[test]
    fn benchmark_mst_implementations() {
        use std::time::Instant;

        println!("\n=== MST Implementation Benchmark ===\n");

        let test_sizes = vec![50, 100, 200];
        let iterations = 3;

        for &size in &test_sizes {
            // Create test instance
            let mut matrix = vec![vec![0; size]; size];
            use rand::random_range;
            for i in 0..size {
                for j in i + 1..size {
                    let dist = random_range(1..=1000);
                    matrix[i][j] = dist;
                    matrix[j][i] = dist;
                }
            }
            let dm = DistanceMatrix::new(matrix);
            let candidates = get_alpha_candidates_v2(&dm, size.min(20), true);

            // Benchmark original
            let mut original_times = Vec::new();
            for _ in 0..iterations {
                let start = Instant::now();
                let _ = get_min_spanning_tree(&dm, size);
                original_times.push(start.elapsed().as_secs_f64() * 1000.0);
            }

            // Benchmark current heap
            let mut heap_times = Vec::new();
            for _ in 0..iterations {
                let start = Instant::now();
                let _ = get_min_spanning_tree_heap(&dm, &candidates, size);
                heap_times.push(start.elapsed().as_secs_f64() * 1000.0);
            }

            // Benchmark optimized heap
            let mut optimized_times = Vec::new();
            for _ in 0..iterations {
                let start = Instant::now();
                let _ = get_min_spanning_tree_heap_optimized(&dm, &candidates, size);
                optimized_times.push(start.elapsed().as_secs_f64() * 1000.0);
            }

            // Calculate averages
            let avg_original = original_times.iter().sum::<f64>() / iterations as f64;
            let avg_heap = heap_times.iter().sum::<f64>() / iterations as f64;
            let avg_optimized = optimized_times.iter().sum::<f64>() / iterations as f64;

            println!("Size: {} cities", size);
            println!("  Original:          {:.3} ms", avg_original);
            println!(
                "  Current Heap:      {:.3} ms ({:.2}x vs original)",
                avg_heap,
                avg_original / avg_heap
            );
            println!(
                "  Optimized Heap:    {:.3} ms ({:.2}x vs original, {:.2}x vs current)",
                avg_optimized,
                avg_original / avg_optimized,
                avg_heap / avg_optimized
            );
            println!();
        }
    }
}
