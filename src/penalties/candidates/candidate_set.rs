use crate::{domain::city::City, penalties::distance::DistanceMatrix};

use super::{utils::get_k_argmins_ordered, Candidates};

pub fn get_nn_candidates(distance_matrix: &DistanceMatrix, k: usize) -> Candidates {
    let n = distance_matrix.len();
    let mut candidates = vec![Vec::with_capacity(k); n];

    for i in 0..n {
        candidates[i] = get_k_argmins_ordered(distance_matrix.row(i), k, Some(i))
            .iter()
            .map(|x| City(*x))
            .collect();
    }
    Candidates::new(candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2nn_candidates() {
        let matrix = vec![vec![0, 1, 2], vec![30, 0, 40], vec![500, 600, 0]];
        let distance_matrix = DistanceMatrix::new(matrix);
        let k = 2;
        let candidates = get_nn_candidates(&distance_matrix, k);
        assert_eq!(candidates.get_neighbors(&City(0)), vec![City(1), City(2)]);
        assert_eq!(candidates.get_neighbors(&City(1)), vec![City(0), City(2)]);
        assert_eq!(candidates.get_neighbors(&City(2)), vec![City(0), City(1)]);
    }

    #[test]
    fn test_1nn_candidate() {
        let matrix = vec![vec![0, 1, 2], vec![30, 0, 40], vec![500, 600, 0]];
        let distance_matrix = DistanceMatrix::new(matrix);
        let k = 1;
        let candidates = get_nn_candidates(&distance_matrix, k);
        assert_eq!(candidates.get_neighbors(&City(0)), vec![City(1)]);
        assert_eq!(candidates.get_neighbors(&City(1)), vec![City(0)]);
        assert_eq!(candidates.get_neighbors(&City(2)), vec![City(0)]);
    }
}
