use std::cmp::{max, min};

use crate::{domain::city::City, penalties::distance::DistanceMatrix};

use super::{
    candidate_set::get_nn_candidates,
    min_spanning_tree::MinSpanningTree,
    utils::{get_k_argmins_ordered, get_min_spanning_tree},
    Candidates,
};

fn get_topo_order(spanning_tree: MinSpanningTree, n: usize) -> (Vec<City>, Vec<City>) {
    let topo_order: Vec<City> = (0..n - 1)
        .map(|i| match i {
            0 => City(0),
            j => spanning_tree.edges[j - 1].1,
        })
        .collect();
    let mut pred = vec![City(0); n - 1];
    for (city1, city2) in spanning_tree.edges {
        pred[city2.id()] = city1;
    }
    (topo_order, pred)
}

fn get_beta_values(
    dm: &DistanceMatrix,
    topo_order: Vec<City>,
    pred: Vec<City>,
    n: usize,
    two_nearest_neighbors: Vec<usize>,
) -> Vec<i64> {
    let mut beta_values: Vec<i64> = vec![0; n * n];
    for (i, c1) in topo_order.iter().enumerate() {
        for (_, c2) in topo_order[i + 1..].iter().enumerate() {
            beta_values[c1.id() * n + c2.id()] = max(
                beta_values[c1.id() * n + pred[c2.id()].id()],
                dm.distance(*c2, pred[c2.id()]),
            );
            beta_values[c2.id() * n + c1.id()] = beta_values[c1.id() * n + c2.id()];
        }
    }
    // take care of n-1 node
    // we have to subtract the nearest distance if we have the nearest neighbor. Nothing changes when we have to have this guy
    // for the others, we have to subtract the second closest later.
    for i in 0..n - 1 {
        beta_values[(n - 1) * n + i] = if i == two_nearest_neighbors[0] {
            dm.distance(City(n - 1), City(two_nearest_neighbors[0]))
        } else {
            dm.distance(City(n - 1), City(two_nearest_neighbors[1]))
        };
        beta_values[i * n + n - 1] = beta_values[(n - 1) * n + i]
    }
    beta_values
}

fn get_alpha_values(dm: &DistanceMatrix, beta_values: Vec<i64>, n: usize) -> Vec<i64> {
    let mut alpha_values: Vec<i64> = beta_values;
    for i in 0..n {
        for j in i + 1..n {
            alpha_values[i * n + j] = dm.distance(City(i), City(j)) - alpha_values[i * n + j];
            alpha_values[j * n + i] = alpha_values[i * n + j];
        }
    }
    alpha_values
}

/// 1. Create min-spanning tree von G \ {n - 1}
/// 2. Add the two longest edges
/// 3. calculate alpha nearness matrix
/// 4. given that matrix, fill the candidates.
pub fn get_alpha_candidates(distance_matrix: &DistanceMatrix, k: usize) -> Candidates {
    let n = distance_matrix.len();
    // 1.
    let spanning_tree = get_min_spanning_tree(distance_matrix, n - 1);
    // 2
    let two_nearest_neighbors = get_k_argmins_ordered(distance_matrix.row(n - 1), 2, Some(n - 1));
    // // 3.1 create topo order and predecessors
    let (topo_order, pred) = get_topo_order(spanning_tree, n);

    // 3.2 create beta values
    // the topo order ensures that beta_values[i][pred[j].id()] always has been computed before.
    // for the very first iterations, of the inner for loop, i = pred[j], so the maximum is always the second value.
    // this is fine, since it belongs to the min span tree. Hence, the subtraction later becomes 0.
    let beta_values = get_beta_values(&distance_matrix, topo_order, pred, n, two_nearest_neighbors);

    let alpha_values = get_alpha_values(&distance_matrix, beta_values, n);
    let alpha_distance_matrix = DistanceMatrix::from_flat(alpha_values);
    get_nn_candidates(&alpha_distance_matrix, k)
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_alpha_candidates() {
        let points = vec![
            (0, 0),  // 0
            (0, 1),  // 1
            (1, 0),  // 2
            (1, 1),  // 3
            (10, 0), // 4
            (10, 1), // 5
            (11, 0), // 6
            (11, 1), // 7
        ];
        let distance_matrix = DistanceMatrix::new_euclidian(points);
        let k = 3;
        let candidates = get_alpha_candidates(&distance_matrix, k);
        assert_eq!(
            candidates.get_neighbors_out(&City(0)),
            vec![City(1), City(2), City(3)]
        );
        assert_eq!(
            candidates.get_neighbors_out(&City(1)),
            vec![City(0), City(3), City(2)]
        );
        assert_eq!(
            candidates.get_neighbors_out(&City(2)),
            vec![City(0), City(3), City(4)]
        );
        assert_eq!(
            candidates.get_neighbors_out(&City(3)),
            vec![City(1), City(2), City(5)]
        );
        assert_eq!(
            candidates.get_neighbors_out(&City(4)),
            vec![City(2), City(5), City(6)]
        );
        assert_eq!(
            candidates.get_neighbors_out(&City(5)),
            vec![City(3), City(4), City(7)]
        );
        assert_eq!(
            candidates.get_neighbors_out(&City(6)),
            vec![City(4), City(7), City(5)]
        );
        assert_eq!(
            candidates.get_neighbors_out(&City(7)),
            vec![City(5), City(6), City(4)]
        );
    }
}
