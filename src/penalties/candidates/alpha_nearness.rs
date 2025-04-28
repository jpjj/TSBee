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

    // refactor this. You only want the beta values for one i at a time and transform them into candidates.
    // so we do not have to build a giant matrix here.
    let beta_values = get_beta_values(&distance_matrix, topo_order, pred, n, two_nearest_neighbors);

    let alpha_values = get_alpha_values(&distance_matrix, beta_values, n);
    let alpha_distance_matrix = DistanceMatrix::from_flat(alpha_values);
    let mut cans = get_nn_candidates(&alpha_distance_matrix, k);
    cans.sort(distance_matrix);
    return cans;
}

/// 1. Create min-spanning tree von G \ {n - 1}
/// 2. Add the two longest edges
/// 3. calculate alpha nearness matrix
/// 4. given that matrix, fill the candidates.
pub fn get_alpha_candidates_v2(distance_matrix: &DistanceMatrix, k: usize) -> Candidates {
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

    // refactor this. You only want the beta values for one i at a time and transform them into candidates.
    // so we do not have to build a giant matrix here.
    let mut candidates = vec![Vec::with_capacity(k); n];
    let mut beta_computed_by = vec![n - 1; n]; // n -1 because this is the missing node.
    let mut beta_i_values = vec![i64::MIN; n];
    let second_shortest_distance =
        distance_matrix.distance(City(n - 1), City(two_nearest_neighbors[1]));
    for city_i in topo_order.iter() {
        beta_i_values[city_i.id()] = 0;
        let mut city_k: City = *city_i;
        while pred[city_k.id()] != city_k {
            let city_j = pred[city_k.id()];
            beta_i_values[city_j.id()] = max(
                beta_i_values[city_k.id()],
                distance_matrix.distance(city_k, city_j),
            );
            beta_computed_by[city_j.id()] = city_i.id();
            city_k = city_j;
        }
        for city_j in topo_order.iter() {
            if city_i != city_j && beta_computed_by[city_j.id()] != city_i.id() {
                beta_i_values[city_j.id()] = max(
                    beta_i_values[pred[city_j.id()].id()],
                    distance_matrix.distance(*city_j, pred[city_j.id()]),
                )
            }
        }
        let mut alpha_i_values: Vec<i64> = distance_matrix
            .row(city_i.id())
            .iter()
            .enumerate()
            .take(n - 1)
            .map(|(j, x)| x - beta_i_values[j])
            .collect();
        // HIER FEHLEN NOCH DIE ALPHA BZW BETA VALUES VON N-1
        let last_alpha_value = if city_i.id() == two_nearest_neighbors[0] {
            0
        } else {
            distance_matrix.distance(*city_i, City(two_nearest_neighbors[1]))
                - second_shortest_distance
        };
        alpha_i_values.push(last_alpha_value);
        candidates[city_i.id()] = get_k_argmins_ordered(&alpha_i_values, k, Some(city_i.id()))
            .iter()
            .map(|x| City(*x))
            .collect();
    }

    let alpha_i_values: Vec<i64> = distance_matrix
        .row(n - 1)
        .iter()
        .enumerate()
        .map(|(j, x)| max(0, x - second_shortest_distance))
        .collect();
    candidates[n - 1] = get_k_argmins_ordered(&alpha_i_values, k, Some(n - 1))
        .iter()
        .map(|x| City(*x))
        .collect();

    let mut cans = Candidates::new(candidates);
    cans.sort(distance_matrix);
    return cans;
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, vec};

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
            vec![City(5), City(6), City(2)]
        );
        assert_eq!(
            candidates.get_neighbors_out(&City(5)),
            vec![City(4), City(7), City(3),]
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

    #[test]
    fn test_both_versions() {
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
        let n = points.len();
        let distance_matrix = DistanceMatrix::new_euclidian(points);
        let k = 3;

        let cans1 = get_alpha_candidates(&distance_matrix, k);
        let cans2 = get_alpha_candidates_v2(&distance_matrix, k);
        for i in 0..n {
            assert_eq!(
                cans1
                    .get_neighbors_out(&City(i))
                    .iter()
                    .map(|x| x.id())
                    .collect::<HashSet<_>>(),
                cans2
                    .get_neighbors_out(&City(i))
                    .iter()
                    .map(|x| x.id())
                    .collect::<HashSet<_>>()
            );
        }
    }

    #[test]
    fn test_both_versions2() {
        let points = vec![
            (565, 575),
            (25, 185),
            (345, 750),
            (945, 685),
            (845, 655),
            (880, 660),
            (25, 230),
            (525, 1000),
            (580, 1175),
            (650, 1130),
            (1605, 620),
            (1220, 580),
            (1465, 200),
            (1530, 5),
            (845, 680),
            (725, 370),
            (145, 665),
            (415, 635),
            (510, 875),
            (560, 365),
            (300, 465),
            (520, 585),
            (480, 415),
            (835, 625),
            (975, 580),
            (1215, 245),
            (1320, 315),
            (1250, 400),
            (660, 180),
            (410, 250),
            (420, 555),
            (575, 665),
            (1150, 1160),
            (700, 580),
            (685, 595),
            (685, 610),
            (770, 610),
            (795, 645),
            (720, 635),
            (760, 650),
            (475, 960),
            (95, 260),
            (875, 920),
            (700, 500),
            (555, 815),
            (830, 485),
            (1170, 65),
            (830, 610),
            (605, 625),
            (595, 360),
            (1340, 725),
            (1740, 245),
        ];
        let n = points.len();
        let distance_matrix = DistanceMatrix::new_euclidian(points);
        let k = 11;

        let cans1 = get_alpha_candidates(&distance_matrix, k);
        let cans2 = get_alpha_candidates_v2(&distance_matrix, k);
        for i in 0..n {
            assert_eq!(
                cans1
                    .get_neighbors_out(&City(i))
                    .iter()
                    .map(|x| x.id())
                    .collect::<HashSet<_>>(),
                cans2
                    .get_neighbors_out(&City(i))
                    .iter()
                    .map(|x| x.id())
                    .collect::<HashSet<_>>()
            );
        }
    }
}
