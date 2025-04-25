mod domain;
mod input;
mod local_move;
pub mod penalties;
mod postprocess;
mod preprocess;
mod solution;
mod solver;
use postprocess::PySolution;
use pyo3::prelude::*;
use solver::Solver;

/// Solving the Traveling Salesman Problem using the 2-opt algorithm
#[pyfunction]
#[pyo3(signature = (distance_matrix, time_limit=None))]
fn solve(distance_matrix: Vec<Vec<i64>>, time_limit: Option<f64>) -> PyResult<PySolution> {
    let raw_input = preprocess::RawInput::new(distance_matrix, time_limit);

    // Validate the input
    raw_input.validate()?;

    let input = raw_input.into();
    let mut solver = Solver::new(input);

    let solution_report = solver.solve(false);

    let py_solution = solution_report.into();

    Ok(py_solution)
}

/// A Python module implemented in Rust for solving the Traveling Salesman Problem.
#[pymodule]
fn tsp_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
