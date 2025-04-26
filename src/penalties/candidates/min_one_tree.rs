use crate::domain::city::City;

#[derive(Clone)]
pub struct MinOneTree {
    pub score: i64,
    pub edges: Vec<(City, City)>,
}

impl MinOneTree {
    pub fn new(score: i64, edges: Vec<(City, City)>) -> Self {
        Self { score, edges }
    }
    pub fn get_degrees(&self) -> Vec<u32> {
        // one tree has as many edges as nodes
        let n = self.edges.len();
        let mut degrees = vec![0; n];
        for (a, b) in &self.edges {
            degrees[a.id()] += 1;
            degrees[b.id()] += 1;
        }
        return degrees;
    }
}
