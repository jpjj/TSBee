pub fn get_2_smallest_args<T>(vector: &[T]) -> Option<(usize, usize)>
where
    T: PartialOrd + Copy,
{
    if vector.len() < 2 {
        return None;
    }
    let mut idx_min0;
    let mut idx_min1;
    let (mut min0, mut min1) = if vector[0] < vector[1] {
        idx_min0 = 0;
        idx_min1 = 1;
        (vector[0], vector[1])
    } else {
        idx_min0 = 1;
        idx_min1 = 0;
        (vector[1], vector[0])
    };

    for (idx, &val) in vector[2..].iter().enumerate() {
        let idx = idx + 2;
        if val < min0 {
            min1 = min0;
            min0 = val;
            idx_min1 = idx_min0;
            idx_min0 = idx;
        } else if val < min1 {
            min1 = val;
            idx_min1 = idx;
        }
    }

    Some((idx_min0, idx_min1))
}

#[cfg(test)]
mod tests {
    use crate::utils::get_2_smallest_args;

    #[test]
    fn test_get_2_smallest_args() {
        let values = vec![15, 10, 30, 25, 5];
        let result = get_2_smallest_args(&values);

        assert_eq!(result, Some((4, 1))); // indices of values 5 and 10
    }

    #[test]
    fn test_get_2_smallest_args_empty() {
        let values: Vec<i32> = vec![];
        let result = get_2_smallest_args(&values);

        assert_eq!(result, None);
    }

    #[test]
    fn test_get_2_smallest_args_single_element() {
        let values = vec![42];
        let result = get_2_smallest_args(&values);

        assert_eq!(result, None);
    }

    #[test]
    fn test_get_2_smallest_args_two_elements() {
        let values = vec![20, 10];
        let result = get_2_smallest_args(&values);

        assert_eq!(result, Some((1, 0))); // 10 at index 1, 20 at index 0
    }
}
