use crate::domain::city::City;

pub mod candidate_set;
pub mod utils;

pub(crate) struct Candidates {
    candidates: Vec<Vec<City>>,
    inverse_candidates: Vec<Vec<City>>,
}

impl Candidates {
    fn new(candidates: Vec<Vec<City>>) -> Self {
        let n = candidates.len();
        let mut inverse_candidates = vec![vec![]; n];
        for (city_id, cans) in candidates.iter().enumerate() {
            let city_1 = City(city_id);
            for city_2 in cans {
                inverse_candidates[city_2.id()].push(city_1);
            }
        }
        Self {
            candidates,
            inverse_candidates,
        }
    }

    /// get the neighbors of a city
    pub(crate) fn get_neighbors_out(&self, city: &City) -> &[City] {
        return &self.candidates[city.0];
    }

    /// get all cities that have city in their candidate list
    pub(crate) fn get_neighbors_in(&self, city: &City) -> &[City] {
        return &self.inverse_candidates[city.0];
    }
}
