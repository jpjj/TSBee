use crate::city::City;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    u: City,
    v: City,
}

impl Edge {
    pub fn new(u: City, v: City) -> Self {
        let (u, v) = if u.0 < v.0 { (u, v) } else { (v, u) };
        Self { u, v }
    }
}

#[cfg(test)]
mod tests {
    use crate::{city::City, edge::Edge};

    #[test]
    fn test_edge() {
        let e1 = Edge::new(City(0), City(1));
        let e2 = Edge::new(City(1), City(0));
        assert_eq!(e1, e2);
    }
}
