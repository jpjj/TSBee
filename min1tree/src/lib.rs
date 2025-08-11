use graph::Graph;
use mst::Kruskal;
use tsp::{city::City, edge::Edge};

pub fn get_2_smallest_args<T>(vector: &[T]) -> Option<(usize, usize)>
where
    T: PartialOrd + Copy,
{
    if vector.len() < 2 {
        return None;
    }
    let mut idx_min0;
    let mut idx_min1;
    let (mut min0, mut min1) = if vector[0] < vector[1] {
        idx_min0 = 0;
        idx_min1 = 1;
        (vector[0], vector[1])
    } else {
        idx_min0 = 1;
        idx_min1 = 0;
        (vector[1], vector[0])
    };

    for (idx, &val) in vector[2..].iter().enumerate() {
        let idx = idx + 2;
        if val < min0 {
            min1 = min0;
            min0 = val;
            idx_min1 = idx_min0;
            idx_min0 = idx;
        } else if val < min1 {
            min1 = val;
            idx_min1 = idx;
        }
    }

    Some((idx_min0, idx_min1))
}

pub fn get_min_1_tree_edges<'a>(
    graph: &'a Graph<'a>,
    edges: Option<&mut [Edge]>,
) -> (Vec<Edge>, i64) {
    // Important: edges shall not contain any incident edges to City n-1!
    let kruskal = Kruskal::new(graph);
    let n = graph.n();
    let (mut mst_edges, mut total_weight) = match edges {
        Some(edges_slice) => kruskal.get_mst_from_sorted_edges(edges_slice),
        None => {
            // only edges not incident to City n-1
            let edges_slice: Vec<Edge> = graph.edges().filter(|e| e.v.0 < n - 1).collect();
            kruskal.get_mst_from_sorted_edges(&edges_slice)
        }
    };
    let weights: Vec<_> = graph
        .neighbors_out(City(n - 1))
        .map(|c| graph.weight(City(n - 1), c))
        .collect();
    let two_closest_neighors = get_2_smallest_args(&weights).expect(
        "Error while getting two smallest incident edges to City n-1 in Min-1-Tree Calculation",
    );
    let edge0 = Edge::new(City(n - 1), City(two_closest_neighors.0));
    let edge1 = Edge::new(City(n - 1), City(two_closest_neighors.1));
    mst_edges.push(edge0);
    mst_edges.push(edge1);
    total_weight += graph.edge_weight(edge0) + graph.edge_weight(edge1);
    (mst_edges, total_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph::AdjacencyMatrix;
    use tsp::problem::distance_matrix::DistanceMatrix;

    fn create_test_distance_matrix() -> DistanceMatrix<i64> {
        // 4x4 distance matrix:
        // 0: [0, 10, 15, 20]
        // 1: [10, 0, 35, 25]
        // 2: [15, 35, 0, 30]
        // 3: [20, 25, 30, 0]
        let flat_matrix = vec![0, 10, 15, 20, 10, 0, 35, 25, 15, 35, 0, 30, 20, 25, 30, 0];
        DistanceMatrix::from_flat(flat_matrix)
    }

    #[test]
    fn test_get_min_1_tree_edges() {
        // MST should be 0-1, 0-2, two additional edges should be 0-3, 1-3
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        let (edges, total_weight) = get_min_1_tree_edges(&graph, None);
        assert_eq!(edges.len(), 4);

        // Total weight should be positive
        assert_eq!(total_weight, 70);
        let expected_edges = vec![
            Edge::new(City(0), City(1)),
            Edge::new(City(0), City(2)),
            Edge::new(City(0), City(3)),
            Edge::new(City(1), City(3)),
        ];
        assert_eq!(edges, expected_edges);
    }

    #[test]
    fn test_get_min_1_tree_edges_with_sorted_edges() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        // Create sorted edges excluding those incident to last city (City(3))
        let mut edges_slice: Vec<Edge> = graph.edges().filter(|e| e.u.0 < 3 && e.v.0 < 3).collect();
        edges_slice.sort_by_key(|e| graph.edge_weight(*e));

        let (edges, total_weight) = get_min_1_tree_edges(&graph, Some(&mut edges_slice));

        assert_eq!(edges.len(), 4);

        // Total weight should be positive
        assert_eq!(total_weight, 70);
        let expected_edges = vec![
            Edge::new(City(0), City(1)),
            Edge::new(City(0), City(2)),
            Edge::new(City(0), City(3)),
            Edge::new(City(1), City(3)),
        ];
        assert_eq!(edges, expected_edges);
    }

    #[test]
    fn test_get_2_smallest_args() {
        let values = vec![15, 10, 30, 25, 5];
        let result = get_2_smallest_args(&values);

        assert_eq!(result, Some((4, 1))); // indices of values 5 and 10
    }

    #[test]
    fn test_get_2_smallest_args_empty() {
        let values: Vec<i32> = vec![];
        let result = get_2_smallest_args(&values);

        assert_eq!(result, None);
    }

    #[test]
    fn test_get_2_smallest_args_single_element() {
        let values = vec![42];
        let result = get_2_smallest_args(&values);

        assert_eq!(result, None);
    }

    #[test]
    fn test_get_2_smallest_args_two_elements() {
        let values = vec![20, 10];
        let result = get_2_smallest_args(&values);

        assert_eq!(result, Some((1, 0))); // 10 at index 1, 20 at index 0
    }
}
