use crate::domain::city::City;

use super::utils::{euclid_distance, flatten, is_symmetric};
#[derive(Clone)]
pub struct DistanceMatrix {
    n: usize,
    flat_matrix: Vec<i64>,
    symmetric: Option<bool>,
}

impl DistanceMatrix {
    pub fn new(matrix: Vec<Vec<i64>>) -> DistanceMatrix {
        let n = matrix.len();

        let flat_matrix = flatten(matrix);

        DistanceMatrix {
            n,
            flat_matrix,
            symmetric: None,
        }
    }

    pub fn new_euclidian(points: Vec<(i64, i64)>) -> DistanceMatrix {
        let n = points.len();
        let matrix = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| euclid_distance(points[i], points[j]))
                    .collect()
            })
            .collect();
        Self::new(matrix)
    }

    pub fn from_flat(flat_matrix: Vec<i64>) -> DistanceMatrix {
        let n_squared = flat_matrix.len();
        let n = n_squared.isqrt();
        assert_eq!(n * n, n_squared);
        DistanceMatrix {
            n,
            flat_matrix,
            symmetric: None,
        }
    }

    pub fn row(&self, i: usize) -> &[i64] {
        &self.flat_matrix[i * self.n..(i + 1) * self.n]
    }
    pub fn column(&self, j: usize) -> Vec<i64> {
        (0..self.n)
            .map(|i| self.flat_matrix[i * self.n + j])
            .collect()
    }

    pub fn distance(&self, i: City, j: City) -> i64 {
        self.flat_matrix[i.id() * self.n + j.id()]
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_symmetric(&self) -> bool {
        match self.symmetric {
            None => is_symmetric(&self.flat_matrix, self.n),
            Some(symmetric) => symmetric,
        }
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
