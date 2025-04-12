use std::{
    cmp::{max, min},
    ops::SubAssign,
};

use crate::{
    domain::city::City,
    penalties::{
        candidates::Candidates,
        distance::{self, DistanceMatrix, DistancePenalizer},
    },
    solution::Solution,
};

pub(super) struct LocalSearch<'a> {
    distance_matrix: &'a DistanceMatrix,
    candidates: &'a Candidates,
    current_solution: &'a Solution,
    dont_look_bits: &'a mut Vec<bool>,
}

impl<'a> LocalSearch<'a> {
    pub(crate) fn new(
        distance_matrix: &'a DistanceMatrix,
        candidates: &'a Candidates,
        current_solution: &'a Solution,
        dont_look_bits: &'a mut Vec<bool>,
    ) -> Self {
        LocalSearch {
            distance_matrix,
            candidates,
            current_solution,
            dont_look_bits,
        }
    }

    pub(crate) fn execute_2opt(&mut self) -> Solution {
        // There will be four different cities involded:
        // route[i], route[i+1]
        // route[j] and route[j+1]

        // j is a neighbor of i.
        // a = (route[i], route[i+1]) will be removed
        // b = (route[i], route[j]) will e added
        // c = (route[j], route[j+1]) will be removed
        // d = (route[i+1], route[j+1]) will be added

        // given the positive gain criteria, we will have
        // if -dist(a) + dist(b) positive, we can already break
        // reason for this is that If there is a 2Opt improvement, there will be some
        // sequence of a,b,c,d, such that the partial sum (b-a) is negative.

        // Wenn das läuft, don't look bits einfügen.
        // geht über mehrere 2Opt iterations
        // erst alle scharf gestellt
        // wenn two-opt move gemacht wird, kann für die 4 (wenn man nachdenkt, vielleicht sogar weniger)
        // beteiligten Cities die bits wieder scharf gestellt werden.
        // das sollte gewaltig dabei helfen, die finale no improvement runde zu verkürzen.
        // idee ist ja prinzipiell, wenn einmal alles angeschaut wurde O(nk), kann pro verbesserung nur 4 neue Optionen hinzukommen.
        let mut current_solution = self.current_solution.clone();
        let sequence = current_solution.route.sequence.clone();
        let n = current_solution.route.len();
        let mut succ: Vec<City> = sequence.clone();
        succ.rotate_left(1);
        let mut city_to_route_pos = vec![0; n];
        for (i, city_i) in current_solution.route.clone().into_iter().enumerate() {
            city_to_route_pos[city_i] = i;
        }
        for (i, &city_i) in sequence.iter().enumerate() {
            if !self.dont_look_bits[city_i.id()] {
                continue;
            }
            // self.dont_look_bits[city_i.id()] = false;
            // we remove edge a
            let city_i_succ = succ[i];
            let dist_a = self.distance_matrix.distance(city_i, city_i_succ) as i64;
            for &city_j in self.candidates.get_neighbors(&city_i) {
                let j = city_to_route_pos[city_j.id()];
                let city_j_succ = succ[j];

                let dist_b = self.distance_matrix.distance(city_i, city_j) as i64;
                if dist_b - dist_a > 0 {
                    // all other cities j are even further away from city i, because candidate lists are in ascending order of distance
                    // hence, we will stay positive, and according to the positive gain criterion, we might as well stop.
                    break;
                }
                let dist_c = self.distance_matrix.distance(city_j, city_j_succ) as i64;
                let dist_d = self.distance_matrix.distance(city_i_succ, city_j_succ) as i64;

                let dist_delta = dist_b + dist_d - dist_a - dist_c;
                if dist_delta < 0 {
                    current_solution.route.sequence[min(i, j) + 1..=max(i, j)].reverse();
                    current_solution.distance.sub_assign((-dist_delta) as u64);
                    self.dont_look_bits[city_i.id()] = true;
                    self.dont_look_bits[city_j.id()] = true;
                    self.dont_look_bits[city_i_succ.id()] = true;
                    self.dont_look_bits[city_j_succ.id()] = true;
                    self.dont_look_bits[sequence[(i + n - 1) % n].id()] = true;
                    self.dont_look_bits[sequence[(j + n - 1) % n].id()] = true;
                    return current_solution;
                }
            }
        }
        current_solution
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        domain::route::Route,
        penalties::{candidates::candidate_set::get_nn_candidates, distance::DistancePenalizer},
    };

    use super::*;

    #[test]
    fn test_2opt_move() {
        // test of having 4 cities on the line: 0 - 1 -- 2 --- 3, with increasing distances
        let dm = DistanceMatrix::new(vec![
            vec![0, 1, 3, 6],
            vec![1, 0, 2, 5],
            vec![3, 2, 0, 3],
            vec![6, 5, 3, 0],
        ]);
        let candidates = get_nn_candidates(&dm, 2);
        let route = Route::from_iter(vec![0, 2, 1, 3]);
        let penalizer = DistancePenalizer::new(dm);
        let solution = penalizer.penalize(&route);
        assert_eq!(solution.distance, 16);
        let mut dont_look_bits: Vec<bool> = (0..4).map(|_| true).collect();
        let mut local_search = LocalSearch::new(
            &penalizer.distance_matrix,
            &candidates,
            &solution,
            &mut dont_look_bits,
        );
        let new_solution = local_search.execute_2opt();
        assert_eq!(new_solution.route, Route::from_iter(vec![0, 1, 2, 3]));
        assert_eq!(new_solution.distance, 12);
    }
}
