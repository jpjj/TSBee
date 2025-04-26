use crate::domain::city::City;

use super::distance::DistanceMatrix;

pub mod alpha_nearness;
pub mod candidate_set;
pub mod held_karp;
mod min_one_tree;
mod min_spanning_tree;
pub mod utils;
pub(crate) struct Candidates {
    candidates: Vec<Vec<City>>,
}

impl Candidates {
    fn new(candidates: Vec<Vec<City>>) -> Self {
        Self { candidates }
    }

    /// get the neighbors of a city
    pub(crate) fn get_neighbors_out(&self, city: &City) -> &[City] {
        return &self.candidates[city.0];
    }

    pub(crate) fn sort(&mut self, dm: &DistanceMatrix) {
        self.candidates
            .iter_mut()
            .enumerate()
            .for_each(|(i, cans)| cans.sort_by_key(|c| dm.distance(City(i), *c)));
    }
}
