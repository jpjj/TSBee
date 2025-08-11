use graph::{AdjacencyList, Graph};
use min1tree::get_min_1_tree;
use tsp::city::City;

#[inline]
fn add_entry(flat_matrix: &mut [i64], entry: i64, idx1: usize, idx2: usize, n: usize) {
    flat_matrix[idx1 * n + idx2] = entry;
    flat_matrix[idx2 * n + idx1] = entry;
}

pub fn get_alpha_values(graph: &Graph) -> Vec<i64> {
    // Step 1: Calculate the min-1-tree of the graph
    let min1_tree = get_min_1_tree(graph, None);

    // Step 2: Calculate alpha values for edges incident to city n-1
    let n = graph.n();
    let mut alpha_values = vec![0; n * n];

    for city_idx in 0..=n - 2 {
        let entry = if city_idx == min1_tree.smallest_edge_last_city.u.0 {
            0
        } else {
            graph.weight(City(n - 1), City(city_idx))
                - graph.edge_weight(min1_tree.second_smallest_edge_last_city)
        };
        add_entry(&mut alpha_values, entry, n - 1, city_idx, n);
    }

    let mst_graph = Graph::List(
        AdjacencyList::from_edges(graph.problem(), min1_tree.mst_edges.clone()),
        graph.state().clone(),
    );

    // Calculate all the others
    for c in 0..=n - 2 {
        let city_c = City(c);
        let mut visited = vec![false; n - 1];
        let mut max_weight = vec![i64::MIN; n - 1];
        let mut stack = Vec::with_capacity(n - 1);
        stack.push(c);
        visited[c] = true;
        while let Some(u) = stack.pop() {
            let city_u = City(u);
            for city_v in mst_graph.neighbors(City(u)) {
                let v = city_v.0;
                if !visited[v] {
                    visited[v] = true;
                    stack.push(v);
                    max_weight[v] = std::cmp::max(max_weight[u], mst_graph.weight(city_u, city_v));
                    add_entry(
                        &mut alpha_values,
                        std::cmp::max(0, mst_graph.weight(city_c, city_v) - max_weight[v]), // is 0 if edge belongs to MST.
                        c,
                        v,
                        n,
                    );
                }
            }
        }
    }

    alpha_values
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph::{AdjacencyMatrix, WithoutPi};
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
    fn test_get_alpha_values() {
        let distance_matrix = create_test_distance_matrix();
        let adj_matrix = AdjacencyMatrix::new(&distance_matrix);
        let graph = Graph::Matrix(adj_matrix, WithoutPi);

        let alpha_values = get_alpha_values(&graph);

        // X are the edges of the MST, Y the additional edges to the last city, 0 are the diagonal entries:
        // 0: [ 0, X, X, Y]
        // 1: [ X, 0,  , Y]
        // 2: [ X,  , 0,  ]
        // 3: [ Y, Y,  , 0]

        // We arrive at the following entries
        // 0: [ 0, 0, 0, 0]
        // 1: [ 0, 0,20, 0]
        // 2: [ 0,20, 0, 5]
        // 3: [ 0, 0, 5, 0]

        // reason: 20 is 35 - 15, where 15 is the greatest enty in the MST from city 1 to city 2.
        //          5 is 30 - 25, where 25 is the second smallest incident edge weight of the last city 3.
        let expected_values = vec![0, 0, 0, 0, 0, 0, 20, 0, 0, 20, 0, 5, 0, 0, 5, 0];
        assert_eq!(expected_values, alpha_values);
    }
}
