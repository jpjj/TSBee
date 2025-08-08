use crate::problem::Problem;
use crate::solution::Solution;

pub struct Instance<P: Problem, S: Solution<P>> {
    pub problem: P,
    pub solution: S,
}

impl<P: Problem, S: Solution<P, Distance = P::Distance>> Instance<P, S> {
    pub fn new(problem: P, solution: S) -> Self {
        Instance { problem, solution }
    }

    pub fn objective_value(&self) -> P::Distance {
        self.solution.distance(&self.problem)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        city::City,
        instance::Instance,
        problem::euclidean::{Euclidian, Point},
        solution::list::List,
    };

    #[test]
    fn test_instance() {
        let problem = Euclidian::<i32, f64>::new(vec![Point(0, 0), Point(0, 3), Point(4, 3)]);
        let solution = List::<Euclidian<i32, f64>>::new((0..3).map(City).collect());
        let instance = Instance::new(problem, solution);
        assert_eq!(instance.objective_value(), 12.0);
    }
}
