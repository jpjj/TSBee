use graph::Graph;
use mst::Kruskal;
use tsp::{city::City, edge::Edge};

pub struct Min1Tree {
    pub mst_edges: Vec<Edge>,
    pub smallest_edge_last_city: Edge,
    pub second_smallest_edge_last_city: Edge,
    pub total_weight: i64,
}

impl Min1Tree {
    fn new(
        mst_edges: Vec<Edge>,
        smallest_edge_last_city: Edge,
        second_smallest_edge_last_city: Edge,
        total_weight: i64,
    ) -> Self {
        Min1Tree {
            mst_edges,
            smallest_edge_last_city,
            second_smallest_edge_last_city,
            total_weight,
        }
    }
}

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

pub fn get_min_1_tree<'a>(graph: &'a Graph<'a>, edges: Option<&mut [Edge]>) -> Min1Tree {
    // Important: edges shall not contain any incident edges to City n-1!
    let kruskal = Kruskal::new(graph);
    let n = graph.n();
    let (mst_edges, mut total_weight) = match edges {
        Some(edges_slice) => kruskal.get_mst_from_sorted_edges(edges_slice),
        None => {
            // only edges not incident to City n-1
            let edges_slice: Vec<Edge> = graph.edges().filter(|e| e.v.0 < n - 1).collect();
            kruskal.get_mst_from_sorted_edges(&edges_slice)
        }
    };
    let weights: Vec<_> = graph
        .neighbors(City(n - 1))
        .map(|c| graph.weight(City(n - 1), c))
        .collect();
    let two_closest_neighors = get_2_smallest_args(&weights).expect(
        "Error while getting two smallest incident edges to City n-1 in Min-1-Tree Calculation",
    );
    let edge0 = Edge::new(City(n - 1), City(two_closest_neighors.0));
    let edge1 = Edge::new(City(n - 1), City(two_closest_neighors.1));
    total_weight += graph.edge_weight(edge0) + graph.edge_weight(edge1);
    Min1Tree::new(mst_edges, edge0, edge1, total_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph::AdjacencyMatrix;
    use tsp::problem::{TspProblem, distance_matrix::DistanceMatrix};

    fn create_test_distance_matrix() -> TspProblem {
        // 4x4 distance matrix:
        // 0: [0, 10, 15, 20]
        // 1: [10, 0, 35, 25]
        // 2: [15, 35, 0, 30]
        // 3: [20, 25, 30, 0]
        let flat_matrix = vec![0, 10, 15, 20, 10, 0, 35, 25, 15, 35, 0, 30, 20, 25, 30, 0];
        TspProblem::DistanceMatrix(DistanceMatrix::from_flat(flat_matrix))
    }

    #[test]
    fn test_get_min_1_tree_edges() {
        // MST should be 0-1, 0-2, two additional edges should be 0-3, 1-3
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        let min1_tree = get_min_1_tree(&graph, None);
        assert_eq!(min1_tree.mst_edges.len(), 2);

        // Total weight should be positive
        assert_eq!(min1_tree.total_weight, 70);
        let expected_edges = vec![Edge::new(City(0), City(1)), Edge::new(City(0), City(2))];
        assert_eq!(min1_tree.mst_edges, expected_edges);
        assert_eq!(
            min1_tree.smallest_edge_last_city,
            Edge::new(City(0), City(3)),
        );
        assert_eq!(
            min1_tree.second_smallest_edge_last_city,
            Edge::new(City(1), City(3)),
        );
    }

    #[test]
    fn test_get_min_1_tree_edges_with_sorted_edges() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix);

        // Create sorted edges excluding those incident to last city (City(3))
        let mut edges_slice: Vec<Edge> = graph.edges().filter(|e| e.u.0 < 3 && e.v.0 < 3).collect();
        edges_slice.sort_by_key(|e| graph.edge_weight(*e));

        let min1_tree = get_min_1_tree(&graph, Some(&mut edges_slice));

        assert_eq!(min1_tree.mst_edges.len(), 2);

        // Total weight should be positive
        assert_eq!(min1_tree.total_weight, 70);
        let expected_edges = vec![Edge::new(City(0), City(1)), Edge::new(City(0), City(2))];
        assert_eq!(min1_tree.mst_edges, expected_edges);
        assert_eq!(
            min1_tree.smallest_edge_last_city,
            Edge::new(City(0), City(3)),
        );
        assert_eq!(
            min1_tree.second_smallest_edge_last_city,
            Edge::new(City(1), City(3)),
        );
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
