use crate::problem::{Problem, TspProblem};
use crate::solution::Solution;
use crate::solution::list::List;

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

pub type TspInstance = Instance<TspProblem, List<TspProblem>>;

#[cfg(test)]
mod tests {
    use crate::{
        city::City,
        instance::Instance,
        problem::points_and_function::{Point, PointsAndFunction, euc_2d::Euc2d},
        solution::list::List,
    };

    #[test]
    fn test_instance() {
        let problem = PointsAndFunction::<f64, f64, Euc2d>::new(vec![
            Point(0.0, 0.0),
            Point(0.0, 3.0),
            Point(4.0, 3.0),
        ]);
        let solution = List::<PointsAndFunction<f64, f64, Euc2d>>::new((0..3).map(City).collect());
        let instance = Instance::new(problem, solution);
        assert_eq!(instance.objective_value(), 12.0);
    }
}
