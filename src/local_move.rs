use std::ops::SubAssign;

use crate::{
    domain::city::City,
    penalties::{
        candidates::Candidates,
        distance::{DistanceMatrix, DistancePenalizer},
    },
    solution::Solution,
};

/// returns (n + pos_a - pos_b) % n
/// Here, pos_a and pos_b are between 0 and n, two indices on a length n vector.
/// Returns relative position given of pos_a giben pos_b would be 0
fn get_rel_pos(pos_a: usize, pos_b: usize, n: usize) -> usize {
    return (n + pos_a - pos_b) % n;
}

pub(super) struct LocalSearch<'a> {
    distance_matrix: &'a DistanceMatrix,
    candidates: &'a Candidates,
    current_solution: &'a Solution,
    pub dont_look_bits: &'a mut Vec<bool>,
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

    fn assert_correct_change(&self, current_solution: &Solution) {
        let distance_penalizer = DistancePenalizer::new(self.distance_matrix.clone());
        assert_eq!(
            *current_solution,
            distance_penalizer.penalize(&current_solution.route.clone())
        );
    }

    pub(crate) fn execute_3opt(&mut self, dlb: bool) -> Solution {
        // preparation
        let mut current_solution = self.current_solution.clone();
        let sequence = current_solution.route.sequence.clone();
        let n = current_solution.route.len();
        let mut succ: Vec<City> = sequence.clone();
        succ.rotate_left(1);
        let mut pred: Vec<City> = sequence.clone();
        pred.rotate_right(1);
        let mut city_to_route_pos = vec![0; n];
        for (i, city_i) in sequence.iter().enumerate() {
            city_to_route_pos[city_i.id()] = i;
        }

        for (c1_pos, c1) in sequence.iter().enumerate() {
            if dlb && !self.dont_look_bits[c1.id()] {
                continue;
            }
            self.dont_look_bits[c1.id()] = false;
            let c2 = pred[c1_pos];
            let c2_pos = (n - 1 + c1_pos) % n;
            let dist1 = self.distance_matrix.distance(*c1, c2);

            for c3 in self.candidates.get_neighbors_out(&c2) {
                let c3_pos = city_to_route_pos[c3.id()];
                let dist2 = self.distance_matrix.distance(c2, *c3);

                // positive gain criterion
                if dist1 <= dist2 {
                    // we add something that is bigger than what we would remove so far.
                    // since candidates are ordered by length, we break and go to next candidate.
                    break;
                }
                for c4 in [succ[c3_pos], pred[c3_pos]] {
                    if c4 == succ[c3_pos] {
                        let c4_pos = (c3_pos + 1) % n;
                        let dist3 = self.distance_matrix.distance(*c3, c4);

                        // check whether you can do a 2-optmove:
                        let dist4 = self.distance_matrix.distance(c4, *c1);

                        // edges we remove are longer than the edges we add.
                        if dist1 + dist3 > dist2 + dist4 {
                            current_solution.apply_two_opt(
                                c1_pos,
                                c4_pos,
                                dist1 + dist3 - (dist2 + dist4),
                            );
                            // self.assert_correct_change(&current_solution);
                            self.dont_look_bits[c1.id()] = true;
                            self.dont_look_bits[c2.id()] = true;
                            self.dont_look_bits[c3.id()] = true;
                            self.dont_look_bits[c4.id()] = true;

                            return current_solution;
                        }

                        // 2opt did not work, get c5
                        for c5 in self.candidates.get_neighbors_out(&c4) {
                            if c5 == c3 {
                                continue; // wäre ein anderer 2Opt move, unnötig wahrscheinlich
                            }
                            let dist4 = self.distance_matrix.distance(c4, *c5);
                            // positive gain criterion:
                            if dist1 + dist3 <= dist2 + dist4 {
                                // like before, dist4 will only grow after
                                break;
                            }
                            let c5_pos = city_to_route_pos[c5.id()];
                            let c5_rel_pos = get_rel_pos(c5_pos, c1_pos, n);
                            if 0 < c5_rel_pos && c5_rel_pos < get_rel_pos(c3_pos, c1_pos, n) {
                                // in this case, we have to get the successor
                                let c6 = succ[c5_pos];
                                let c6_pos = (c5_pos + 1) % n;
                                let dist5 = self.distance_matrix.distance(*c5, c6);
                                let dist6 = self.distance_matrix.distance(c6, *c1);
                                if dist1 + dist3 + dist5 > dist2 + dist4 + dist6 {
                                    // modify route of current_solution
                                    current_solution.route.sequence.rotate_left(c1_pos);
                                    // now, city1 is at 0, city2 is at n-1
                                    // first, we need to shift the slice from c6_pos until end by
                                    current_solution.route.sequence
                                        [get_rel_pos(c6_pos, c1_pos, n)..]
                                        .rotate_left((n + c3_pos - c6_pos + 1) % n);
                                    // we swapped two sequences, now we reverse the one further behind.
                                    current_solution.route.sequence
                                        [(n - c3_pos + c6_pos - 1) % n..]
                                        .reverse();
                                    // modify distance of current solution
                                    current_solution.distance.sub_assign(
                                        dist1 + dist3 + dist5 - (dist2 + dist4 + dist6),
                                    );
                                    // self.assert_correct_change(&current_solution);

                                    self.dont_look_bits[c1.id()] = true;
                                    self.dont_look_bits[c2.id()] = true;
                                    self.dont_look_bits[c3.id()] = true;
                                    self.dont_look_bits[c4.id()] = true;
                                    self.dont_look_bits[c5.id()] = true;
                                    self.dont_look_bits[c6.id()] = true;
                                    return current_solution;
                                }
                            } else {
                                // so c6 must be the predecessor
                                let c6 = pred[c5_pos];
                                let c6_pos = (n + c5_pos - 1) % n;
                                let dist5 = self.distance_matrix.distance(*c5, c6);
                                let dist6 = self.distance_matrix.distance(c6, *c1);
                                if dist1 + dist3 + dist5 > dist2 + dist4 + dist6 {
                                    // modify route of current_solution
                                    current_solution.route.sequence.rotate_left(c1_pos);
                                    // now, city1 is at 0, city2 is at n-1
                                    // we swapped two sequences, now we reverse one.
                                    current_solution.route.sequence[(n + c5_pos - c1_pos) % n..]
                                        .reverse();
                                    // we need to shift the slice from c4_pos until end by
                                    current_solution.route.sequence[(n + c4_pos - c1_pos) % n..]
                                        .rotate_left((n + c6_pos - c4_pos + 1) % n);
                                    // modify distance of current solution
                                    current_solution.distance.sub_assign(
                                        dist1 + dist3 + dist5 - (dist2 + dist4 + dist6),
                                    );
                                    // self.assert_correct_change(&current_solution);

                                    self.dont_look_bits[c1.id()] = true;
                                    self.dont_look_bits[c2.id()] = true;
                                    self.dont_look_bits[c3.id()] = true;
                                    self.dont_look_bits[c4.id()] = true;
                                    self.dont_look_bits[c5.id()] = true;
                                    self.dont_look_bits[c6.id()] = true;
                                    return current_solution;
                                }
                            }
                        }
                    } else {
                        // c4 = pred[c3_pos]
                        let c4_pos = (n + c3_pos - 1) % n;
                        let dist3 = self.distance_matrix.distance(*c3, c4);
                        for c5 in self.candidates.get_neighbors_out(&c4) {
                            let dist4 = self.distance_matrix.distance(c4, *c5);
                            // positive gain criterion:
                            if dist1 + dist3 <= dist2 + dist4 {
                                // like before, dist4 will only grow
                                break;
                            }
                            let c5_pos = city_to_route_pos[c5.id()];
                            let rel_pos = (n + c5_pos - c3_pos) % n;
                            // check whether c5 is betwwen c3 and c2. Otherwise, we would get no roundtrip
                            if rel_pos <= (n + c2_pos - c3_pos) % n {
                                // in this case, we have to get the successor or succesor
                                for c6 in [succ[c5_pos], pred[c5_pos]] {
                                    let dist5 = self.distance_matrix.distance(*c5, c6);
                                    let dist6 = self.distance_matrix.distance(c6, *c1);
                                    if dist1 + dist3 + dist5 > dist2 + dist4 + dist6 {
                                        if c6 == succ[c5_pos] {
                                            if *c5 == c2 {
                                                // überlappung, beide haben succ
                                                continue;
                                            }
                                            let c6_pos = (c5_pos + 1) % n;
                                            // modify route of current_solution
                                            current_solution.route.sequence.rotate_left(c1_pos);
                                            // now, city1 is at 0, city2 is at n-1
                                            // we can reverse from c3 to including c5 and c6 until the end
                                            current_solution.route.sequence[(n + c3_pos - c1_pos)
                                                % n
                                                ..=(n + c5_pos - c1_pos) % n]
                                                .reverse();
                                            current_solution.route.sequence
                                                [(n + c6_pos - c1_pos) % n..]
                                                .reverse();
                                            // modify distance of current solution
                                            current_solution.distance.sub_assign(
                                                dist1 + dist3 + dist5 - (dist2 + dist4 + dist6),
                                            );
                                            // self.assert_correct_change(&current_solution);
                                            self.dont_look_bits[c1.id()] = true;
                                            self.dont_look_bits[c2.id()] = true;
                                            self.dont_look_bits[c3.id()] = true;
                                            self.dont_look_bits[c4.id()] = true;
                                            self.dont_look_bits[c5.id()] = true;
                                            self.dont_look_bits[c6.id()] = true;
                                            return current_solution;
                                        } else {
                                            if c5 == c3 {
                                                // überlappung, beide haben pred
                                                continue;
                                            }
                                            // so c6 must be the predecessor
                                            let c6_pos = (n + c5_pos - 1) % n;
                                            // modify route of current_solution
                                            current_solution.route.sequence.rotate_left(c1_pos);
                                            // now, city1 is at 0, city2 is at n-1
                                            // we only need to shift the slice from c4_pos until end by
                                            current_solution.route.sequence
                                                [(n + c3_pos - c1_pos) % n..]
                                                .rotate_left((n + c6_pos - c3_pos + 1) % n);
                                            // modify distance of current solution
                                            current_solution.distance.sub_assign(
                                                dist1 + dist3 + dist5 - (dist2 + dist4 + dist6),
                                            );
                                            // self.assert_correct_change(&current_solution);
                                            self.dont_look_bits[c1.id()] = true;
                                            self.dont_look_bits[c2.id()] = true;
                                            self.dont_look_bits[c3.id()] = true;
                                            self.dont_look_bits[c4.id()] = true;
                                            self.dont_look_bits[c5.id()] = true;
                                            self.dont_look_bits[c6.id()] = true;
                                            return current_solution;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        current_solution
    }
}
