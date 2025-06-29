/// given the index of flat matrix, get the index of the inverse element
fn get_reverse_index(m: usize, n: usize) -> usize {
    let i = m / n;
    let j = m % n;
    j * n + i
}

/// given a flat matrix and dimension n,check that it is symmetric
pub(super) fn is_symmetric(flat_matrix: &[i64], n: usize) -> bool {
    flat_matrix.len() == n * n
        && flat_matrix
            .iter()
            .enumerate()
            .all(|(m, val)| *val == flat_matrix[get_reverse_index(m, n)])
}

pub(super) fn flatten(matrix: Vec<Vec<i64>>) -> Vec<i64> {
    matrix
        .iter()
        .flat_map(|row| row.iter())
        .copied()
        .collect::<Vec<i64>>()
}

pub(super) fn euclid_distance(a: (i64, i64), b: (i64, i64)) -> i64 {
    (1_000_000 * ((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1))).isqrt()
}

pub(super) fn att_distance(a: (i64, i64), b: (i64, i64)) -> i64 {
    let xd = a.0 - b.0;
    let yd = a.1 - b.1;
    let r = ((xd * xd + yd * yd) as f64 / 10.0).sqrt();
    (1_000_000.0 * r.ceil()) as i64
}

#[cfg(test)]
mod tests {
    use crate::penalties::distance::utils::is_symmetric;

    #[test]
    fn test_symmetric() {
        let matrix = vec![0, 1, 2, 1, 0, 3, 2, 3, 0];
        let n = 3;
        assert!(is_symmetric(&matrix, n))
    }

    #[test]
    fn test_not_symmetric() {
        let matrix = vec![0, 1, 2, 1, 0, 3, 2, 5, 0];
        let n = 3;
        assert!(!is_symmetric(&matrix, n))
    }

    #[test]
    fn test_not_quadratic() {
        let matrix = vec![0, 1, 2, 1, 0, 3, 2, 5];
        let n = 3;
        assert!(!is_symmetric(&matrix, n))
    }
}
