use std::iter::Sum;
use tsp::{
    city::City,
    edge::Edge,
    problem::{Problem, TspProblem, distance_matrix::DistanceMatrix},
};

pub trait Graph {
    type Weight;

    fn weight(&self, c1: City, c2: City) -> Self::Weight;

    fn edges(&self) -> impl Iterator<Item = Edge>;

    fn complete_weight(&self) -> Self::Weight
    where
        <Self as Graph>::Weight: Sum,
    {
        self.edges().map(|e| self.weight(e.u, e.v)).sum()
    }

    fn n(&self) -> usize;

    fn m(&self) -> usize;

    fn cities(&self) -> impl Iterator<Item = City> {
        (0..self.n()).map(City)
    }

    fn neighbors_out(&self, c: City) -> impl Iterator<Item = City>;

    fn neighbors_in(&self, c: City) -> impl Iterator<Item = City>;
}

pub struct AdjacencyMatrix<'a> {
    problem: &'a DistanceMatrix<i64>,
}

impl<'a> AdjacencyMatrix<'a> {
    pub fn new(problem: &'a DistanceMatrix<i64>) -> Self {
        Self { problem }
    }
}

pub struct AdjacencyList<'a> {
    problem: &'a TspProblem,
    list: Vec<Vec<City>>,
}

impl<'a> AdjacencyList<'a> {
    pub fn new(problem: &'a TspProblem, list: Vec<Vec<City>>) -> Self {
        Self { problem, list }
    }
}

pub enum Graphs<'a> {
    Matrix(AdjacencyMatrix<'a>),
    List(AdjacencyList<'a>),
}

impl<'a> Graph for Graphs<'a> {
    type Weight = i64;

    fn weight(&self, c1: City, c2: City) -> i64 {
        match self {
            Self::Matrix(am) => am.problem.distance(c1, c2),
            Self::List(al) => al.problem.distance(c1, c2),
        }
    }

    fn n(&self) -> usize {
        match self {
            Self::Matrix(am) => am.problem.size(),
            Self::List(al) => al.problem.size(),
        }
    }

    fn m(&self) -> usize {
        match self {
            Self::Matrix(am) => {
                let n = am.problem.size();
                n * (n - 1) / 2
            }
            Self::List(al) => al.list.iter().map(|x| x.len()).sum::<usize>() / 2,
        }
    }

    fn neighbors_out(&self, c: City) -> impl Iterator<Item = City> {
        let n = self.n();
        let neighbors: Vec<City> = match self {
            Self::Matrix(_) => (0..n).filter(|&i| i != c.0).map(City).collect(),
            Self::List(al) => al.list[c.0].to_vec(),
        };
        neighbors.into_iter()
    }

    fn neighbors_in(&self, c: City) -> impl Iterator<Item = City> {
        let n = self.n();
        let neighbors: Vec<City> = match self {
            Self::Matrix(_) => (0..n).filter(|&i| i != c.0).map(City).collect(),
            Self::List(al) => (0..n)
                .filter(|&i| al.list[i].contains(&c))
                .map(City)
                .collect(),
        };
        neighbors.into_iter()
    }

    fn edges(&self) -> impl Iterator<Item = Edge> {
        let edges: Vec<Edge> = match self {
            Self::Matrix(am) => {
                let n = am.problem.size();
                (0..n)
                    .flat_map(|i| (i + 1..n).map(move |j| Edge::new(City(i), City(j))))
                    .collect()
            }
            Self::List(al) => al
                .list
                .iter()
                .enumerate()
                .flat_map(|(i, neighbors)| {
                    neighbors.iter().filter_map(move |&neighbor| {
                        if neighbor.0 > i {
                            Some(Edge::new(City(i), neighbor))
                        } else {
                            None
                        }
                    })
                })
                .collect(),
        };
        edges.into_iter()
    }
}
