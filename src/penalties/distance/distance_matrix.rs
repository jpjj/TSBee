use crate::{domain::city::City, penalties::distance::utils::att_distance};

use super::utils::{euclid_distance, flatten, is_symmetric};
#[derive(Clone)]
pub struct DistanceMatrix {
    n: usize,
    flat_matrix: Vec<i64>,
    pi: Option<Vec<i64>>,
    symmetric: Option<bool>,
}

impl DistanceMatrix {
    pub fn new(matrix: Vec<Vec<i64>>) -> DistanceMatrix {
        let n = matrix.len();

        let flat_matrix = flatten(matrix);

        DistanceMatrix {
            n,
            flat_matrix,
            pi: None,
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

    pub fn new_att(points: Vec<(i64, i64)>) -> DistanceMatrix {
        let n = points.len();
        let matrix = (0..n)
            .map(|i| (0..n).map(|j| att_distance(points[i], points[j])).collect())
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
            pi: None,
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
        match &self.pi {
            None => self.flat_matrix[i.id() * self.n + j.id()],
            Some(pi) => self.flat_matrix[i.id() * self.n + j.id()] + pi[i.id()] + pi[j.id()],
        }
    }

    pub fn update_pi(&mut self, pi: Vec<i64>) {
        self.pi = Some(pi)
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
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

    pub fn sum_of_abs_distance(&self) -> i64 {
        self.flat_matrix.iter().map(|x| x.abs()).sum()
    }
    /// Creates out of an asymmtric distance matrix representing a tsp instance a 2n x 2n symmetric distance matrix that
    /// represents an equivalent problem.
    /// Jonker, R., & Volgenant, A. (1983). Transforming asymmetric into symmetric traveling salesman problems. Operations Research Letters, 2(4), 161–163
    pub fn symmetrize(self) -> Self {
        // cities 0..n are the city entries. I
        // cities n..2n are the city exits. O
        // 1. it holds: d(I(x),O(x)) = - inf
        // 2. it holds: d(I(x),I(y)) = inf
        // 3. it holds: d(O(x),O(y)) = inf
        // 4. it holds: d(O(x),I(y)) = d(x,y)
        // 5. it holds: d(x,x) = 0
        // These rule enforce that
        // 1. All entries and exits of a city are next to eachother
        // there are never two entries or two exits next to eachother
        // It is always like this in the sequence: (I,O,I,O,I,O) or starting with O.
        // instead of +/- inf, we will use the sum of the abs of all distances.
        let sum_of_abs_distance: i64 = self.sum_of_abs_distance();
        // create flat_matrix and apply 2 + 3
        let n = self.n;
        let mut flat_matrix = vec![sum_of_abs_distance; n * n * 4];
        for i in 0..n {
            for j in n..(2 * n) {
                flat_matrix[i * 2 * n + j] = self.flat_matrix[i * n + j - n];
                flat_matrix[j * 2 * n + i] = self.flat_matrix[i * n + j - n];
            }
        }
        for i in 0..n {
            flat_matrix[i * 2 * n + i + n] = -sum_of_abs_distance; // apply 1 I/O
            flat_matrix[(i + n) * 2 * n + i] = -sum_of_abs_distance; // apply 1 O/I
            flat_matrix[i * 2 * n + i] = 0; // apply 5 for entries
            flat_matrix[(i + n) * 2 * n + i + n] = 0; // apply 5 for exits
        }
        Self::from_flat(flat_matrix)
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

    #[test]
    fn test_symmetrize() {
        let matrix = vec![vec![0, 1, 10], vec![1, 0, 3], vec![2, 3, 0]];
        let distance_matrix = DistanceMatrix::new(matrix);
        assert!(!distance_matrix.is_symmetric());
        let symmetrized_matrix = distance_matrix.symmetrize();
        let expected_flat_matrix: Vec<i64> = vec![
            0, 20, 20, -20, 1, 10, 20, 0, 20, 1, -20, 3, 20, 20, 0, 2, 3, -20, -20, 1, 2, 0, 20,
            20, 1, -20, 3, 20, 0, 20, 10, 3, -20, 20, 20, 0,
        ];
        assert_eq!(expected_flat_matrix, symmetrized_matrix.flat_matrix);
    }
}
