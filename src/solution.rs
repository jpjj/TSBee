use std::{cmp::Ordering, ops::SubAssign};

use crate::domain::route::Route;

/// Represents a solution to the Traveling Salesman Problem.
///
/// A solution consists of a route (the order in which cities are visited) and
/// the total distance of that route. Solutions can be compared based on their
/// distance, with shorter distances being better.
///
/// # Fields
///
/// * `route` - The sequence of cities forming the tour
/// * `distance` - The total distance of the tour (including return to start)
///
/// # Ordering
///
/// Solutions are ordered by distance, allowing easy identification of better solutions.
/// Two solutions with the same distance are considered equal, regardless of the
/// actual route taken.
#[derive(Clone, Debug)]
pub struct Solution {
    pub route: Route,
    pub distance: i64,
}

impl Solution {
    /// Applies a 2-opt move to improve the solution.
    ///
    /// A 2-opt move removes two edges from the tour and reconnects the path by
    /// reversing one of the segments. This is done in-place for efficiency.
    ///
    /// # Arguments
    ///
    /// * `idx1` - Position of the first city in the edge to remove
    /// * `idx2` - Position of the second city in the edge to remove
    /// * `delta_distance` - The improvement in distance (positive value means improvement)
    ///
    /// # Algorithm
    ///
    /// 1. Rotate the tour so that idx1 is at position 0
    /// 2. Reverse the segment from idx2 to the end
    /// 3. Update the total distance
    ///
    /// # Example
    ///
    /// Tour: A-B-C-D-E-F-A, remove edges (B,C) and (E,F)
    /// Result: A-B-F-E-D-C-A (segment C-D-E is reversed)
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
