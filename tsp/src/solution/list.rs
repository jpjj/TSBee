use std::{collections::HashSet, iter::Sum, marker::PhantomData, ops::Index};

use crate::{city::City, edge::Edge, problem::Problem, solution::Solution};

pub struct List<P> {
    n: usize,
    vector: Vec<City>,
    _phantom_data: PhantomData<P>,
}

impl<T> List<T> {
    pub fn new(vector: Vec<City>) -> Self {
        let n = vector.len();
        Self {
            n,
            vector,
            _phantom_data: PhantomData,
        }
    }
}

impl<P> Index<usize> for List<P> {
    type Output = City;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.vector[idx]
    }
}

impl<P> Solution<P> for List<P>
where
    P: Problem,
    P::Distance: Sum,
{
    type Distance = P::Distance;

    fn distance(&self, problem: &P) -> Self::Distance {
        self.vector
            .windows(2)
            .map(|pair| problem.distance(pair[0], pair[1]))
            .chain(std::iter::once(if self.vector.is_empty() {
                panic!("Cannot calculate distance for empty tour")
            } else {
                problem.distance(*self.vector.last().unwrap(), self.vector[0])
            }))
            .sum()
    }

    fn size(&self) -> usize {
        self.n
    }

    fn get_edges(&self) -> std::collections::HashSet<crate::edge::Edge> {
        let mut set = HashSet::with_capacity(self.size());
        self.vector.windows(2).for_each(|pair| {
            set.insert(Edge::new(pair[0], pair[1]));
        });
        set.insert(Edge::new(self.vector[0], self.vector[self.size() - 1]));
        set
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        city::City,
        edge::Edge,
        problem::{Problem, distance_matrix::DistanceMatrix},
        solution::{Solution, list::List},
    };
    #[test]
    fn test_list() {
        let problem = DistanceMatrix::from_flat(vec![0, 1, 2, 1, 0, 3, 2, 3, 0]);
        let solution = List::<DistanceMatrix<i32>>::new((0..3).map(City).collect());
        assert_eq!(solution.distance(&problem), 6);
        assert_eq!(solution.size(), problem.size());
        assert!(solution.get_edges().contains(&Edge::new(City(2), City(0))));
    }
}
