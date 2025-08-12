pub mod utils;
use graph::Graph;
use mst::Kruskal;
use tsp::{city::City, edge::Edge};

use utils::get_2_smallest_args;

pub struct Min1Tree<'a> {
    pub graph: &'a Graph<'a>,
    pub mst_edges: Vec<Edge>,
    pub smallest_edge_last_city: Edge,
    pub second_smallest_edge_last_city: Edge,
    pub total_weight: f64,
}

impl<'a> Min1Tree<'a> {
    fn new(
        graph: &'a Graph,
        mst_edges: Vec<Edge>,
        smallest_edge_last_city: Edge,
        second_smallest_edge_last_city: Edge,
        total_weight: f64,
    ) -> Self {
        Min1Tree {
            graph,
            mst_edges,
            smallest_edge_last_city,
            second_smallest_edge_last_city,
            total_weight,
        }
    }
    pub fn degrees(&self) -> Vec<i32> {
        let n = self.graph.n();
        let mut degrees = vec![0; n];
        for edge in &self.mst_edges {
            degrees[edge.u.0] += 1;
            degrees[edge.v.0] += 1;
        }
        // Add the two edges from the last city
        degrees[n - 1] = 2;
        degrees[self.smallest_edge_last_city.u.0] += 1;
        degrees[self.second_smallest_edge_last_city.u.0] += 1;
        degrees
    }
}

pub fn get_min_1_tree<'a>(graph: &'a Graph<'a>, edges: Option<&mut [Edge]>) -> Min1Tree<'a> {
    // Important: edges shall not contain any incident edges to City n-1!
    let kruskal = Kruskal::new(graph);
    let n = graph.n();
    let (mst_edges, mut total_weight) = match edges {
        Some(_) => kruskal.get_mst(edges),
        None => {
            let mut actual_edges = graph
                .edges()
                .filter(|e| e.v.0 < n - 1)
                .collect::<Vec<Edge>>();
            kruskal.get_mst(Some(&mut actual_edges))
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
    Min1Tree::new(graph, mst_edges, edge0, edge1, total_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph::Graph;
    use tsp::problem::{TspProblem, distance_matrix::DistanceMatrix};

    fn create_test_distance_matrix() -> TspProblem {
        // 4x4 distance matrix:
        // 0: [0, 10, 15, 20]
        // 1: [10, 0, 35, 25]
        // 2: [15, 35, 0, 30]
        // 3: [20, 25, 30, 0]
        let flat_matrix = vec![
            0.0, 10.0, 15.0, 20.0, 10.0, 0.0, 35.0, 25.0, 15.0, 35.0, 0.0, 30.0, 20.0, 25.0, 30.0,
            0.0,
        ];
        TspProblem::DistanceMatrix(DistanceMatrix::from_flat(flat_matrix))
    }

    #[test]
    fn test_get_min_1_tree_edges() {
        // MST should be 0-1, 0-2, two additional edges should be 0-3, 1-3
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::new_matrix(&distance_matrix);

        let min1_tree = get_min_1_tree(&graph, None);
        assert_eq!(min1_tree.mst_edges.len(), 2);

        // Total weight should be positive
        assert_eq!(min1_tree.total_weight, 70.0);
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

        assert_eq!(min1_tree.degrees(), vec![3, 2, 1, 2]);
    }

    #[test]
    fn test_get_min_1_tree_edges_with_sorted_edges() {
        let distance_matrix = create_test_distance_matrix();
        let graph = Graph::new_matrix(&distance_matrix);

        // Create sorted edges excluding those incident to last city (City(3))
        let mut edges_slice: Vec<Edge> = graph.edges().filter(|e| e.u.0 < 3 && e.v.0 < 3).collect();
        edges_slice.sort_by(|a, b| {
            graph
                .edge_weight(*a)
                .partial_cmp(&graph.edge_weight(*b))
                .unwrap()
        });

        let min1_tree = get_min_1_tree(&graph, Some(&mut edges_slice));

        assert_eq!(min1_tree.mst_edges.len(), 2);

        // Total weight should be positive
        assert_eq!(min1_tree.total_weight, 70.0);
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
}
