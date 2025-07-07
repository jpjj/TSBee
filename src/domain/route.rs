use std::ops::Index;

use super::city::City;

/// Represents a tour in the Traveling Salesman Problem.
///
/// A route is an ordered sequence of cities that forms a complete tour.
/// The tour is implicitly closed - after visiting the last city, the
/// salesman returns to the first city.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    /// The ordered sequence of cities in the tour
    pub sequence: Vec<City>,
}

impl Route {
    /// Creates a new route from a sequence of cities.
    ///
    /// # Arguments
    ///
    /// * `sequence` - A vector of cities representing the tour order
    pub fn new(sequence: Vec<City>) -> Route {
        Route { sequence }
    }

    /// Checks if the route contains no cities.
    ///
    /// # Returns
    ///
    /// `true` if the route is empty, `false` otherwise
    pub fn is_empty(&self) -> bool {
        self.sequence.len() == 0
    }

    /// Returns the number of cities in the route.
    ///
    /// # Returns
    ///
    /// The number of cities in the tour
    pub fn len(&self) -> usize {
        self.sequence.len()
    }
}
impl Index<usize> for Route {
    type Output = City;

    fn index(&self, index: usize) -> &Self::Output {
        &self.sequence[index]
    }
}

impl FromIterator<usize> for Route {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Route::new(iter.into_iter().map(City).collect())
    }
}

impl IntoIterator for Route {
    type Item = usize;
    type IntoIter = std::iter::Map<std::vec::IntoIter<City>, fn(City) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.sequence.into_iter().map(|city| city.0)
    }
}
