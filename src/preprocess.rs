use pyo3::{exceptions::PyValueError, PyResult};

use crate::{input::Input, penalties::distance::DistanceMatrix};

pub(super) struct RawInput {
    distance_matrix: Vec<Vec<i64>>,
    time_limit: Option<f64>,
}

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

    Ok(())
}

impl Into<Input> for RawInput {
    fn into(self) -> Input {
        Input::new(
            DistanceMatrix::new(self.distance_matrix),
            self.time_limit.map(|t| chrono::Duration::seconds(t as i64)),
        )
    }
}
