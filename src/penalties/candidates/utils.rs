/// function getting the indices of the k smallest entries of some slice. The returned vector is ordered in ascending order
/// this function runs in O(nlog(k)), where n is the length of the slice
/// maybe i is the index to be skipped. This is important for the diogonal of the distance matrix.
pub(super) fn get_k_argmins_ordered<T: Ord>(
    slice: &[T],
    k: usize,
    maybe_i: Option<usize>,
) -> Vec<usize> {
    // Create (index, value) pairs
    let mut indexed: Vec<_> = slice
        .iter()
        .enumerate()
        .filter(|(idx, _)| match maybe_i {
            Some(i) => *idx != i,
            None => true,
        })
        .collect();

    // this part is the bottleneck that runs in O(nlog(k))
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

#[cfg(test)]
mod tests {
    use crate::penalties::candidates::utils::get_k_argmins_ordered;

    #[test]
    fn test_get_k_argmins_ordered() {
        assert_eq!(
            vec![0, 3, 1],
            get_k_argmins_ordered(&[2, 4, 6, 3, 5], 3, None)
        );
        assert_eq!(
            vec![3, 1, 4],
            get_k_argmins_ordered(&[2, 4, 6, 3, 5], 3, Some(0))
        );
    }
}
