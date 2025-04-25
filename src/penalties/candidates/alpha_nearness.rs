use std::cmp::{max, min};

use crate::{domain::city::City, penalties::distance::DistanceMatrix};

use super::{
    candidate_set::{self, get_nn_candidates},
    utils::{get_k_argmins_ordered, get_min_spanning_tree},
    Candidates,
};

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

    // 3.2 create beta values
    // the topo order ensures that beta_values[i][pred[j].id()] always has been computed before.
    // for the very first iterations, of the inner for loop, i = pred[j], so the maximum is always the second value.
    // this is fine, since it belongs to the min span tree. Hence, the subtraction later becomes 0.
    let mut beta_values: Vec<Vec<i64>> = vec![vec![0; n]; n];
    for (i, city_i) in topo_order.iter().enumerate() {
        for (p, city_j) in topo_order[i + 1..].iter().enumerate() {
            let j = i + p + 1;
            beta_values[city_i.id()][city_j.id()] = max(
                beta_values[city_i.id()][pred[city_j.id()].id()],
                distance_matrix.distance(City(city_j.id()), pred[city_j.id()]),
            );
            beta_values[city_j.id()][city_i.id()] = beta_values[city_i.id()][city_j.id()];
        }
    }
    // take care of n-1 node
    // we have to subtract the nearest distance if we have the nearest neighbor. Nothing changes when we have to have this guy
    // for the others, we have to subtract the second closest later.
    for i in 0..n - 1 {
        beta_values[n - 1][i] = if i == two_nearest_neighbors[0] {
            distance_matrix.distance(City(n - 1), City(two_nearest_neighbors[0]))
        } else {
            distance_matrix.distance(City(n - 1), City(two_nearest_neighbors[1]))
        };
        beta_values[i][n - 1] = beta_values[n - 1][i]
    }
    let mut alpha_values: Vec<Vec<i64>> = vec![vec![0; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            alpha_values[i][j] = distance_matrix.distance(City(i), City(j)) - beta_values[i][j];
            alpha_values[j][i] = alpha_values[i][j];
        }
    }
    let alpha_distance_matrix = DistanceMatrix::new(alpha_values);
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
