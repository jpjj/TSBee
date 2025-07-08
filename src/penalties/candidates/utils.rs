use crate::{domain::city::City, penalties::distance::DistanceMatrix};

use super::min_spanning_tree::MinSpanningTree;

/// function getting the indices of the k smallest entries of some slice. The returned vector is ordered in ascending order
/// this function runs in O(nlog(k)), where n is the length of the slice
/// maybe i is the index to be skipped. This is important for the diogonal of the distance matrix.
pub(super) fn get_k_argmins_ordered<T: Ord>(
    slice: &[T],
    k: usize,
    maybe_i: Option<usize>,
) -> Vec<usize> {
    // Create (index, value) pairs
    let mut indexed: Vec<_> = slice
        .iter()
        .enumerate()
        .filter(|(idx, _)| match maybe_i {
            Some(i) => *idx != i,
            None => true,
        })
        .collect();

    // this part is the bottleneck that runs in O(nlog(k))
    if k < indexed.len() {
        // Use select_nth_unstable by comparing the values
        indexed.select_nth_unstable_by(k, |a, b| a.1.cmp(b.1));
        indexed.truncate(k);
    }

    // Sort by value
    indexed.sort_unstable_by(|a, b| a.1.cmp(b.1));

    // Extract and return just the indices
    indexed.into_iter().map(|(idx, _)| idx).collect()
}

/// create a min spanning tree on the distance matrix. n says how many of the citiesof the distance matrix should be used.
/// the last vertex might not be used to later add two vertices.
pub fn get_min_spanning_tree(distance_matrix: &DistanceMatrix, n: usize) -> MinSpanningTree {
    let mut min_cost: Vec<i64> = (0..n)
        .map(|i| distance_matrix.distance(City(0), City(i)))
        .collect();
    let mut argmin_cost: Vec<City> = (0..n).map(|_| City(0)).collect();
    let mut min_tree_edges: Vec<(City, City)> = (0..n - 1).map(|_| (City(0), City(0))).collect();
    let mut nodes_to_consider: Vec<City> = (0..n).map(City).collect();
    nodes_to_consider.swap(0, n - 1);
    for k in 1..n {
        if let Some((next_node_idx, &next_node)) = nodes_to_consider
            .iter()
            .enumerate()
            .take(n - k)
            .min_by_key(|(_, &node)| min_cost[node.id()])
        {
            min_tree_edges[k - 1] = (argmin_cost[next_node.id()], next_node);
            nodes_to_consider.swap(next_node_idx, n - k - 1);

            nodes_to_consider.iter().take(n - k - 1).for_each(|&node| {
                let edge_weight = distance_matrix.distance(next_node, node);
                if edge_weight < min_cost[node.id()] {
                    min_cost[node.id()] = edge_weight;
                    argmin_cost[node.id()] = next_node;
                }
            });
        }
    }

    let score = min_tree_edges
        .iter()
        .map(|&(c1, c2)| distance_matrix.distance(c1, c2))
        .sum();
    MinSpanningTree::new(score, min_tree_edges)
}

#[cfg(test)]
mod tests {
    use crate::penalties::candidates::utils::get_k_argmins_ordered;
    use petgraph::algo::min_spanning_tree;
    use petgraph::data::FromElements;
    use petgraph::graph::NodeIndex;
    use petgraph::graph::UnGraph;
    use petgraph::Graph;
    use rand::random_range;

    use super::*;
    #[test]
    fn test_get_k_argmins_ordered() {
        assert_eq!(
            vec![0, 3, 1],
            get_k_argmins_ordered(&[2, 4, 6, 3, 5], 3, None)
        );
        assert_eq!(
            vec![3, 1, 4],
            get_k_argmins_ordered(&[2, 4, 6, 3, 5], 3, Some(0))
        );
    }

    #[test]
    fn compare_with_petgraph() {
        let number_nodes = 100;
        let mut random_matrix: Vec<Vec<i64>> = (0..number_nodes)
            .map(|_| (0..number_nodes).map(|_| random_range(1..=100)).collect())
            .collect();
        for i in 0..number_nodes {
            for j in i + 1..number_nodes {
                random_matrix[i][j] = random_matrix[j][i]
            }
        }
        let dm = DistanceMatrix::new(random_matrix);
        let mst = get_min_spanning_tree(&dm, number_nodes);

        let mut graph = UnGraph::<i64, i64>::new_undirected();
        for _ in 0..number_nodes {
            graph.add_node(0);
        }

        for i in 0..number_nodes {
            let node_index_i = NodeIndex::new(i);
            for j in i + 1..number_nodes {
                let node_index_j = NodeIndex::new(j);
                graph.add_edge(node_index_i, node_index_j, dm.distance(City(i), City(j)));
            }
        }
        let mst_petgraph: Graph<i64, i64> = Graph::from_elements::<_>(min_spanning_tree(&graph));
        let petgraph_score: i64 = mst_petgraph.edge_references().map(|e| e.weight()).sum();
        assert_eq!(mst.score, petgraph_score);
    }
}
