use pyo3::pyclass;

use crate::solver::SolutionReport;

/// Solution data returned to Python
#[pyclass]
pub struct PySolution {
    #[pyo3(get)]
    distance: u64,
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
