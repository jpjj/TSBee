use std::cmp::Ordering;

use crate::domain::route::Route;

#[derive(Clone)]
pub struct Solution {
    pub route: Route,
    pub distance: u64,
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
