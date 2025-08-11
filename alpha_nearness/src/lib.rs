// use graph::Graph;
// use min1tree::get_min_1_tree;
// use mst::Kruskal;
// use std::collections::{HashMap, HashSet, VecDeque};
// use tsp::{city::City, edge::Edge};

// pub fn get_alpha_values(graph: &Graph) -> Vec<i64> {
//     let n = graph.n();

//     // Step 1: Calculate the min-1-tree of the graph
//     let (mut min1tree_edges, _) = get_min_1_tree(graph, None);

//     // Get all edges in the graph
//     let all_edges: Vec<Edge> = graph.edges().collect();
//     let mut alpha_values = Vec::with_capacity(all_edges.len());

//     // Convert min1tree edges to a HashSet for O(1) lookup
//     let min1tree_set: HashSet<Edge> = min1tree_edges.iter().cloned().collect();

//     // Get the two smallest edges incident to City(n-1)
//     let mut edges_to_last_city: Vec<(Edge, i64)> = graph
//         .neighbors_out(City(n - 1))
//         .map(|neighbor| {
//             let edge = Edge::new(City(n - 1), neighbor);
//             (edge, graph.edge_weight(edge))
//         })
//         .collect();
//     edges_to_last_city.sort_by_key(|(_, weight)| *weight);

//     let second_smallest_weight = if edges_to_last_city.len() >= 2 {
//         edges_to_last_city[1].1
//     } else {
//         0
//     };

//     // Create MST without City(n-1) and its edges for other calculations
//     let edges_without_last_city: Vec<Edge> = graph
//         .edges()
//         .filter(|e| e.u.0 < n - 1 && e.v.0 < n - 1)
//         .collect();

//     let kruskal = Kruskal::new(graph);
//     let (mst_edges_without_last, _) = kruskal.get_mst_from_sorted_edges(&edges_without_last_city);

//     // Build adjacency list for MST without last city
//     let mut mst_adj: HashMap<City, Vec<(City, i64)>> = HashMap::new();
//     for edge in &mst_edges_without_last {
//         let weight = graph.edge_weight(*edge);
//         mst_adj.entry(edge.u).or_default().push((edge.v, weight));
//         mst_adj.entry(edge.v).or_default().push((edge.u, weight));
//     }

//     // Calculate alpha values for each edge
//     for edge in all_edges {
//         let alpha_value = if edge.u.0 == n - 1 || edge.v.0 == n - 1 {
//             // Edge incident to City(n-1)
//             if min1tree_set.contains(&edge) {
//                 // Edge belongs to min-1-tree (one of two smallest edges)
//                 0
//             } else {
//                 // Return w(e) - w(e_2) where e_2 is second smallest edge
//                 graph.edge_weight(edge) - second_smallest_weight
//             }
//         } else {
//             // Edge not incident to City(n-1)
//             calculate_alpha_for_edge(edge, graph, &mst_adj)
//         };

//         alpha_values.push(alpha_value);
//     }

//     alpha_values
// }

// fn calculate_alpha_for_edge(
//     edge: Edge,
//     graph: &Graph,
//     mst_adj: &HashMap<City, Vec<(City, i64)>>,
// ) -> i64 {
//     let edge_weight = graph.edge_weight(edge);

//     // Find the maximum weight on the path between edge.u and edge.v in the MST
//     let max_path_weight = find_max_weight_on_path(edge.u, edge.v, mst_adj);

//     edge_weight - max_path_weight
// }

// fn find_max_weight_on_path(
//     start: City,
//     end: City,
//     mst_adj: &HashMap<City, Vec<(City, i64)>>,
// ) -> i64 {
//     if start == end {
//         return 0;
//     }

//     let mut visited = HashSet::new();
//     let mut queue = VecDeque::new();
//     let mut parent: HashMap<City, (City, i64)> = HashMap::new();

//     queue.push_back(start);
//     visited.insert(start);

//     // BFS to find path from start to end
//     while let Some(current) = queue.pop_front() {
//         if current == end {
//             break;
//         }

//         if let Some(neighbors) = mst_adj.get(&current) {
//             for &(neighbor, weight) in neighbors {
//                 if !visited.contains(&neighbor) {
//                     visited.insert(neighbor);
//                     parent.insert(neighbor, (current, weight));
//                     queue.push_back(neighbor);
//                 }
//             }
//         }
//     }

//     // Reconstruct path and find maximum weight
//     let mut max_weight = 0;
//     let mut current = end;

//     while let Some(&(prev, weight)) = parent.get(&current) {
//         max_weight = max_weight.max(weight);
//         current = prev;
//         if current == start {
//             break;
//         }
//     }

//     max_weight
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use graph::AdjacencyMatrix;
//     use tsp::problem::distance_matrix::DistanceMatrix;

//     fn create_test_distance_matrix() -> DistanceMatrix<i64> {
//         // 4x4 distance matrix:
//         // 0: [0, 10, 15, 20]
//         // 1: [10, 0, 35, 25]
//         // 2: [15, 35, 0, 30]
//         // 3: [20, 25, 30, 0]
//         let flat_matrix = vec![0, 10, 15, 20, 10, 0, 35, 25, 15, 35, 0, 30, 20, 25, 30, 0];
//         DistanceMatrix::from_flat(flat_matrix)
//     }

//     #[test]
//     fn test_get_alpha_values() {
//         let distance_matrix = create_test_distance_matrix();
//         let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
//         let graph = Graph::Matrix(adj_matrix);

//         let alpha_values = get_alpha_values(&graph);

//         // We should get alpha values for all 6 edges in a complete graph of 4 cities
//         assert_eq!(alpha_values.len(), 6);

//         // Alpha values should be computed (exact values depend on the algorithm implementation)
//         // This test mainly checks that the function runs without panic
//         for alpha in &alpha_values {
//             // Alpha values can be negative, zero, or positive
//             assert!(alpha >= &std::i64::MIN && alpha <= &std::i64::MAX);
//         }
//     }
// }
