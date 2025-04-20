use std::{cmp::Ordering, ops::SubAssign};

use crate::domain::route::Route;

#[derive(Clone, Debug)]
pub struct Solution {
    pub route: Route,
    pub distance: i64,
}

impl Solution {
    /// apply two-opt move. In order to not recalculate the distance, it most be ensured that the delta distance
    /// is the actual difference between the old and the new sequence.
    pub(super) fn apply_two_opt(&mut self, idx1: usize, idx2: usize, delta_distance: i64) {
        let n = self.route.len();
        self.route.sequence.rotate_left(idx1);
        self.route.sequence[(n + idx2 - idx1) % n..].reverse();
        self.distance.sub_assign(delta_distance);
    }
}
impl PartialOrd for Solution {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Solution {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}

impl PartialEq for Solution {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Solution {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_solutions() {
        let sol1 = Solution {
            route: Route::from_iter(vec![0, 1, 2]),
            distance: 0,
        };
        let sol2 = Solution {
            route: Route::from_iter(vec![0, 1, 3]),
            distance: 0,
        };
        let sol3 = Solution {
            route: Route::from_iter(vec![0, 1, 4]),
            distance: 1,
        };
        assert!(sol1 == sol2);
        assert!(sol1 < sol3);
    }
}
