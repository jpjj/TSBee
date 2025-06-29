use std::ops::Index;

use super::city::City;

#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub sequence: Vec<City>,
}

impl Route {
    pub fn new(sequence: Vec<City>) -> Route {
        Route { sequence }
    }
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
