use pyo3::{exceptions::PyValueError, PyResult};

use crate::{input::Input, penalties::distance::DistanceMatrix};

#[allow(dead_code)]
pub struct RawInput {
    distance_matrix: Vec<Vec<i64>>,
    time_limit: Option<f64>,
}

#[allow(dead_code)]
impl RawInput {
    pub(super) fn new(distance_matrix: Vec<Vec<i64>>, time_limit: Option<f64>) -> Self {
        Self {
            distance_matrix,
            time_limit,
        }
    }
    pub(super) fn validate(&self) -> PyResult<()> {
        validate_distance_matrix(&self.distance_matrix)?;
        Ok(())
    }
}

/// Validates that the distance matrix is non-empty and square
fn validate_distance_matrix(distance_matrix: &[Vec<i64>]) -> PyResult<()> {
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

    // Check if the matrix has zero entries on diagonal
    for (i, row) in distance_matrix.iter().enumerate() {
        if 0 != row[i] {
            return Err(PyValueError::new_err(
                "Distance matrix must have zeros on diagonal",
            ));
        }
    }

    Ok(())
}

impl From<RawInput> for Input {
    fn from(val: RawInput) -> Self {
        Input::new(
            DistanceMatrix::new(val.distance_matrix),
            val.time_limit
                .map(|t| chrono::Duration::microseconds((t * 1_000_000.0) as i64)),
        )
    }
}
