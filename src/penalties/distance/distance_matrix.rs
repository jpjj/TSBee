pub struct DistanceMatrix {
    n: usize,
    matrix: Vec<u64>,
    symmetric: bool,
}

impl DistanceMatrix {
    pub fn new(matrix: Vec<Vec<u64>>) -> DistanceMatrix {
        let n = matrix.len();
        let symmetric = matrix.iter().enumerate().all(|(i, row)| {
            row.iter()
                .enumerate()
                .all(|(j, &value)| value == matrix[j][i])
        });
        let matrix = matrix
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .collect::<Vec<u64>>();
        DistanceMatrix {
            n,
            matrix,
            symmetric,
        }
    }

    pub fn row(&self, i: usize) -> &[u64] {
        &self.matrix[i * self.n..(i + 1) * self.n]
    }
    pub fn column(&self, j: usize) -> Vec<u64> {
        (0..self.n).map(|i| self.matrix[i * self.n + j]).collect()
    }

    pub fn distance(&self, i: usize, j: usize) -> u64 {
        self.matrix[i * self.n + j]
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
        assert_eq!(distance_matrix.distance(0, 1), 1);
        assert_eq!(distance_matrix.distance(1, 2), 40);
        assert_eq!(distance_matrix.distance(2, 0), 500);
        assert!(!distance_matrix.is_symmetric());
    }
    #[test]
    fn test_distance_matrix2() {
        let matrix = vec![vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]];
        let distance_matrix = DistanceMatrix::new(matrix);
        assert!(distance_matrix.is_symmetric());
    }
}
