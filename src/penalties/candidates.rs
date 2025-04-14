use crate::domain::city::City;

pub mod candidate_set;
pub mod utils;

pub(crate) struct Candidates {
    candidates: Vec<Vec<City>>,
    inverse_candidates:
}

impl Candidates {
    fn new(candidates: Vec<Vec<City>>) -> Self {
        Self { candidates }
    }

    pub(crate) fn get_neighbors(&self, city: &City) -> &[City] {
        return &self.candidates[city.0];
    }
}
