use pyo3::{exceptions::PyValueError, prelude::*};

/// Validates that the distance matrix is non-empty and square
fn validate_distance_matrix(distance_matrix: &[Vec<f64>]) -> PyResult<()> {
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

/// Solving the Traveling Salesman Problem with Time Windows.
#[pyfunction]
#[pyo3(signature = (distance_matrix, time_limit=None))]
fn solve(distance_matrix: Vec<Vec<f64>>, time_limit: Option<f64>) -> PyResult<Vec<usize>> {
    // Validate the input distance matrix
    validate_distance_matrix(&distance_matrix)?;

    let n = distance_matrix.len();

    // For now, just return the default tour (0, 1, 2, ..., n-1)
    // In a real implementation, you would call your TSP algorithm here
    // and use the time_limit parameter to control execution time

    let default_tour: Vec<usize> = (0..n).collect();

    Ok(default_tour)
}

/// A Python module implemented in Rust.
#[pymodule]
fn tsp_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
