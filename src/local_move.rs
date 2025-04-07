use std::{
    cmp::{max, min},
    ops::SubAssign,
};

use crate::{
    domain::city::City,
    penalties::{candidates::Candidates, distance::DistanceMatrix},
    solution::Solution,
};

pub(super) struct LocalMove<'a> {
    distance_matrix: &'a DistanceMatrix,
    candidates: &'a Candidates,
    current_solution: &'a Solution,
}

impl<'a> LocalMove<'a> {
    pub(crate) fn new(
        distance_matrix: &'a DistanceMatrix,
        candidates: &'a Candidates,
        current_solution: &'a Solution,
    ) -> Self {
        LocalMove {
            distance_matrix,
            candidates,
            current_solution,
        }
    }

    pub(crate) fn execute_2opt(&self) -> Solution {
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
        let succ: Vec<City> = sequence
            .iter()
            .enumerate()
            .map(|(i, city_i)| sequence[(i + 1) % n])
            .collect();
        let pred: Vec<City> = sequence
            .iter()
            .enumerate()
            .map(|(i, city_i)| sequence[(i - 1) % n])
            .collect();
        let mut city_to_route_pos = vec![0; n];
        for (i, city_i) in current_solution.route.clone().into_iter().enumerate() {
            city_to_route_pos[city_i] = i;
        }
        for (i, &city_i) in sequence.iter().enumerate() {
            // we remove edge a
            let city_i_plus_1 = succ[city_i.id()];
            let dist_a = self.distance_matrix.distance(city_i, city_i_plus_1) as i64;
            for &city_j in self.candidates.get_neighbors(&city_i) {
                let j = city_to_route_pos[city_j.id()];
                let city_j_plus_1 = succ[city_j.id()];

                let dist_b = self.distance_matrix.distance(city_i, city_j) as i64;
                if dist_a - dist_b > 0 {
                    // all other cities j are even further away from city i, because candidate lits are ordered
                    // hence, we will stay positive, and according to the positive gain criterion, we might as well stop.
                    break;
                }
                let dist_c = self.distance_matrix.distance(city_j, city_j_plus_1) as i64;
                let dist_d = self.distance_matrix.distance(city_i_plus_1, city_j_plus_1) as i64;

                let dist_delta = dist_b + dist_d - dist_a - dist_c;
                if dist_delta < 0 {
                    current_solution.route.sequence[min(j, i + 1)..=max(j, i + 1)].reverse();
                    current_solution.distance.sub_assign((-dist_delta) as u64);
                    return current_solution;
                }
            }
        }
        current_solution
    }
}
