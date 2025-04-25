use crate::domain::city::City;

pub(super) struct MinSpanningTree {
    pub score: i64,
    pub edges: Vec<(City, City)>,
}

impl MinSpanningTree {
    pub fn new(score: i64, edges: Vec<(City, City)>) -> Self {
        Self { score, edges }
    }
}
