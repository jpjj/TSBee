use tsp::{city::City, problem::Problem, problem::TspProblem};

pub trait Candidates {
    fn get_candidates(&self, c: &City, k: usize) -> impl Iterator<Item = City>;

    fn sort_by<F>(&mut self, from: &City, compare: F)
    where
        F: FnMut(&City, &City) -> std::cmp::Ordering;
}

pub struct NearestNeighbors {
    candidates: Vec<Vec<City>>,
}

impl NearestNeighbors {
    pub fn get_k(tsp_problem: &TspProblem, k: Option<usize>) -> Self {
        let n = tsp_problem.size();
        let mut candidates = Vec::with_capacity(n);

        for from_city in tsp_problem.cities() {
            let mut city_candidates: Vec<City> =
                tsp_problem.cities().filter(|&c| c != from_city).collect();
            city_candidates.sort_by(|a, b| {
                let dist_a = tsp_problem.distance(from_city, *a);
                let dist_b = tsp_problem.distance(from_city, *b);
                dist_a.cmp(&dist_b)
            });
            if let Some(actual_k) = k {
                city_candidates.truncate(actual_k);
            }
            candidates.push(city_candidates);
        }
        NearestNeighbors { candidates }
    }
}

impl Candidates for NearestNeighbors {
    fn get_candidates(&self, c: &City, k: usize) -> impl Iterator<Item = City> {
        let idx = c.0;
        if idx < self.candidates.len() {
            let available = &self.candidates[idx];
            let limit = k.min(available.len());
            available[..limit].iter().cloned()
        } else {
            [].iter().cloned()
        }
    }

    fn sort_by<F>(&mut self, from: &City, mut compare: F)
    where
        F: FnMut(&City, &City) -> std::cmp::Ordering,
    {
        let idx = from.0;
        if idx < self.candidates.len() {
            self.candidates[idx].sort_by(&mut compare);
        }
    }
}
