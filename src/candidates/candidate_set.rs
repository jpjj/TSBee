use crate::penalties::distance::DistanceMatrix;

fn k_argminimum<T: Ord>(slice: &[T], cur_idx: usize, k: usize) -> Vec<usize> {
    // Create (index, value) pairs, filter out current index
    let mut indexed: Vec<_> = slice
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != cur_idx)
        .collect();

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

pub fn get_nn_candidates(distance_matrix: &DistanceMatrix, k: usize) -> Vec<Vec<usize>> {
    let n = distance_matrix.len();
    let mut candidates = vec![Vec::with_capacity(k); n];
    for i in 0..n {
        candidates[i] = k_argminimum(distance_matrix.row(i), i, k);
    }
    candidates
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
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0], vec![1, 2]);
        assert_eq!(candidates[1], vec![0, 2]);
        assert_eq!(candidates[2], vec![0, 1]);
    }

    #[test]
    fn test_1nn_candidate() {
        let matrix = vec![vec![0, 1, 2], vec![30, 0, 40], vec![500, 600, 0]];
        let distance_matrix = DistanceMatrix::new(matrix);
        let k = 1;
        let candidates = get_nn_candidates(&distance_matrix, k);
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0], vec![1]);
        assert_eq!(candidates[1], vec![0]);
        assert_eq!(candidates[2], vec![0]);
    }
}
