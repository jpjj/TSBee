use crate::domain::city::City;

pub mod candidate_set;
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
}
