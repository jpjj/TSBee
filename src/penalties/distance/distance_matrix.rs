use crate::domain::city::City;

use super::utils::{flatten, is_symmetric};
#[derive(Clone)]
pub struct DistanceMatrix {
    n: usize,
    flat_matrix: Vec<u64>,
    symmetric: bool,
}

impl DistanceMatrix {
    pub fn new(matrix: Vec<Vec<u64>>) -> DistanceMatrix {
        let n = matrix.len();

        let flat_matrix = flatten(matrix);
        let symmetric = is_symmetric(&flat_matrix, n);

        DistanceMatrix {
            n,
            flat_matrix,
            symmetric,
        }
    }

    pub fn row(&self, i: usize) -> &[u64] {
        &self.flat_matrix[i * self.n..(i + 1) * self.n]
    }
    pub fn column(&self, j: usize) -> Vec<u64> {
        (0..self.n)
            .map(|i| self.flat_matrix[i * self.n + j])
            .collect()
    }

    pub fn distance(&self, i: City, j: City) -> u64 {
        self.flat_matrix[i.id() * self.n + j.id()]
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_symmetric(&self) -> bool {
        self.symmetric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_matrix() {
        let matrix = vec![vec![0, 1, 2], vec![30, 0, 40], vec![500, 600, 0]];
        let distance_matrix = DistanceMatrix::new(matrix);
        assert_eq!(distance_matrix.distance(City(0), City(1)), 1);
        assert_eq!(distance_matrix.distance(City(1), City(2)), 40);
        assert_eq!(distance_matrix.distance(City(2), City(0)), 500);
        assert!(!distance_matrix.is_symmetric());
    }
    #[test]
    fn test_distance_matrix2() {
        let matrix = vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]];
        let distance_matrix = DistanceMatrix::new(matrix);
        assert!(distance_matrix.is_symmetric());
    }
}
