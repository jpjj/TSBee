pub mod domain;
pub mod input;
mod local_move;
pub mod penalties;
mod postprocess;
mod preprocess;
mod solution;
pub mod solver;
use postprocess::PySolution;
use pyo3::prelude::*;
use solver::Solver;

use crate::input::Input;

/// Solves the Traveling Salesman Problem using an advanced 3-opt local search algorithm.
///
/// This function finds a near-optimal solution to the TSP by iteratively improving
/// an initial tour using 3-opt moves, alpha-nearness candidate generation, and
/// Don't Look Bits (DLB) optimization.
///
/// Args:
///     distance_matrix (List[List[int]]): A symmetric N×N matrix where element [i][j]
///         represents the distance from city i to city j. Distances must be integers.
///         The matrix must be symmetric: distance[i][j] == distance[j][i].
///     time_limit (Optional[float]): Maximum time in seconds to run the solver.
///         If None, the solver runs until convergence. Default is None.
///
/// Returns:
///     PySolution: An object containing:
///         - distance (int): Total distance of the best tour found
///         - tour (List[int]): Order of cities in the best tour (0-indexed)
///         - iterations (int): Number of iterations performed
///         - time (float): Time taken in seconds
///
/// Raises:
///     ValueError: If the distance matrix is invalid (not square, not symmetric,
///         contains negative values, or has invalid diagonal).
///
/// Example:
///     >>> import tsp_solve
///     >>> # Distance matrix for 4 cities
///     >>> distances = [
///     ...     [0, 10, 15, 20],
///     ...     [10, 0, 35, 25],
///     ...     [15, 35, 0, 30],
///     ...     [20, 25, 30, 0]
///     ... ]
///     >>> solution = tsp_solve.solve(distances, time_limit=10.0)
///     >>> print(f"Best tour: {solution.tour}")
///     >>> print(f"Total distance: {solution.distance}")
#[pyfunction]
#[pyo3(signature = (distance_matrix, time_limit=None))]
fn solve(distance_matrix: Vec<Vec<i64>>, time_limit: Option<f64>) -> PyResult<PySolution> {
    let raw_input = preprocess::RawInput::new(distance_matrix, time_limit);

    // Validate the input
    raw_input.validate()?;

    let mut input: Input = raw_input.into();
    let symmetric_instance = input.distance_matrix.is_symmetric();
    let mut big_m = 0;
    if !symmetric_instance {
        big_m = input.distance_matrix.sum_of_abs_distance();
        input.distance_matrix = input.distance_matrix.symmetrize();
    }

    let mut solver = Solver::new(input);
    let mut solution_report = solver.solve(false);

    if !symmetric_instance {
        solution_report = solution_report.desymmetrize(big_m);
    }
    let py_solution = solution_report.into();

    Ok(py_solution)
}

/// A high-performance Traveling Salesman Problem solver.
///
/// This module provides a Rust-powered TSP solver that uses advanced local search
/// algorithms to find near-optimal solutions efficiently. The solver implements
/// a 3-opt local search with several optimizations:
///
/// - **3-opt moves**: Considers removing three edges and reconnecting the tour
/// - **Alpha-nearness**: Intelligent candidate generation to reduce search space
/// - **Don't Look Bits (DLB)**: Avoids redundant searches for efficiency
/// - **Double-bridge perturbation**: Helps escape local optima
///
/// The solver is designed for symmetric TSP instances where the distance from
/// city A to city B equals the distance from city B to city A.
///
/// Example:
///     >>> import tsp_solve
///     >>> import numpy as np
///     >>>
///     >>> # Generate random city coordinates
///     >>> np.random.seed(42)
///     >>> n_cities = 50
///     >>> coords = np.random.rand(n_cities, 2) * 100
///     >>>
///     >>> # Calculate distance matrix
///     >>> distances = np.zeros((n_cities, n_cities), dtype=int)
///     >>> for i in range(n_cities):
///     ...     for j in range(n_cities):
///     ...         if i != j:
///     ...             dist = np.linalg.norm(coords[i] - coords[j])
///     ...             distances[i, j] = int(dist * 100)  # Scale to integers
///     >>>
///     >>> # Solve TSP
///     >>> solution = tsp_solve.solve(distances.tolist())
///     >>> print(f"Tour length: {solution.distance}")
///     >>> print(f"Tour: {solution.tour[:10]}...")  # First 10 cities
#[pymodule]
fn tsp_solve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
