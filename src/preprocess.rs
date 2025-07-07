use pyo3::{exceptions::PyValueError, PyResult};

use crate::{input::Input, penalties::distance::DistanceMatrix};

#[allow(dead_code)]
pub struct RawInput {
    distance_matrix: Vec<Vec<f64>>,
    time_limit: Option<f64>,
    scale_factor: f64,
}

#[allow(dead_code)]
impl RawInput {
    pub(super) fn new(
        distance_matrix: Vec<Vec<f64>>,
        time_limit: Option<f64>,
        scale_factor: f64,
    ) -> Self {
        Self {
            distance_matrix,
            time_limit,
            scale_factor,
        }
    }
    pub(super) fn validate(&self) -> PyResult<()> {
        validate_distance_matrix(&self.distance_matrix)?;
        Ok(())
    }

    /// Converts the float distance matrix to integers by scaling
    fn scale_to_integers(&self) -> Vec<Vec<i64>> {
        self.distance_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&val| (val * self.scale_factor).round() as i64)
                    .collect()
            })
            .collect()
    }
}

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

    // Check if the matrix has zero entries on diagonal
    for (i, row) in distance_matrix.iter().enumerate() {
        if row[i] != 0.0 {
            return Err(PyValueError::new_err(
                "Distance matrix must have zeros on diagonal",
            ));
        }
    }

    Ok(())
}

impl From<RawInput> for Input {
    fn from(val: RawInput) -> Self {
        let scaled_matrix = val.scale_to_integers();
        Input::new(
            DistanceMatrix::new(scaled_matrix),
            val.time_limit
                .map(|t| chrono::Duration::microseconds((t * 1_000_000.0) as i64)),
        )
    }
}
