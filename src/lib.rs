mod candidates;
mod input;
mod penalties;
mod route;
mod solution;
mod solver;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use solver::Solver;

/// Solution data returned to Python
#[pyclass]
struct PySolution {
    #[pyo3(get)]
    distance: u64,
    #[pyo3(get)]
    iterations: u64,
    #[pyo3(get)]
    time: f64,
    #[pyo3(get)]
    tour: Vec<usize>,
}

/// Validates that the distance matrix is non-empty and square
fn validate_distance_matrix(distance_matrix: &[Vec<u64>]) -> PyResult<()> {
    if distance_matrix.is_empty() {
        return Err(PyValueError::new_err("Distance matrix cannot be empty"));
    }

    let n = distance_matrix.len();

    // Check if the matrix is square
    for row in distance_matrix {
        if row.len() != n {
            return Err(PyValueError::new_err("Distance matrix must be square"));
        }
    }

    Ok(())
}

/// Solving the Traveling Salesman Problem using the 2-opt algorithm
#[pyfunction]
#[pyo3(signature = (distance_matrix, time_limit=None))]
fn solve(distance_matrix: Vec<Vec<u64>>, time_limit: Option<f64>) -> PyResult<PySolution> {
    // Validate the input distance matrix
    validate_distance_matrix(&distance_matrix)?;

    let input = input::get_input_from_raw(distance_matrix, time_limit);
    let mut solver = Solver::new(input);

    solver.solve();
    let distance = solver.best_distance();
    let iterations = solver.iterations();
    let solution_time = solver.time_taken().num_seconds() as f64;
    let tour = solver.best_route();

    Ok(PySolution {
        distance,
        iterations,
        time: solution_time,
        tour,
    })
}

/// A Python module implemented in Rust for solving the Traveling Salesman Problem.
#[pymodule]
fn tsp_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
