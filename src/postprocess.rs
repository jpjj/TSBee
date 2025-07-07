use pyo3::pyclass;

use crate::solver::solution_report::SolutionReport;

/// Solution data returned to Python containing the results of the TSP solver.
///
/// This class represents the solution found by the TSP solver and provides
/// read-only access to the solution details.
///
/// Attributes:
///     distance (int): The total distance of the tour found. This is the sum
///         of distances between consecutive cities in the tour, including the
///         return from the last city to the first.
///     iterations (int): The number of iterations performed by the solver.
///         Each iteration attempts to improve the current tour using local search.
///     time (float): The total time taken by the solver in seconds. This includes
///         initialization, local search, and any perturbations.
///     tour (List[int]): The order in which cities should be visited, represented
///         as a list of city indices (0-based). The tour implicitly returns to
///         the first city after visiting the last one.
///
/// Example:
///     >>> solution = tsp_solve.solve(distance_matrix)
///     >>> print(f"Best distance: {solution.distance}")
///     >>> print(f"Found in {solution.iterations} iterations")
///     >>> print(f"Time taken: {solution.time:.2f} seconds")
///     >>> print(f"Tour: {' -> '.join(map(str, solution.tour))}")
#[pyclass]
pub struct PySolution {
    #[pyo3(get)]
    distance: i64,
    #[pyo3(get)]
    iterations: u64,
    #[pyo3(get)]
    time: f64,
    #[pyo3(get)]
    tour: Vec<usize>,
}

impl From<SolutionReport> for PySolution {
    fn from(solution_report: SolutionReport) -> Self {
        let tour: Vec<usize> = Vec::from_iter(solution_report.best_solution.route);
        PySolution {
            distance: solution_report.best_solution.distance,
            iterations: solution_report.stats.iterations,
            time: solution_report.stats.time_taken.num_milliseconds() as f64 / 1000.0,
            tour,
        }
    }
}
