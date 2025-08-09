pub mod parse_tsp_file;

use parse_tsp_file::parse_tsp_file;
use std::path::Path;
use tsp::problem::TspProblem;
use tsp::problem::distance_matrix::DistanceMatrix;
use tsp::problem::points_and_function::{Point, PointsAndFunction};

use crate::read_problem_file::parse_tsp_file::EdgeWeightType;

pub fn read_problem_file(path: &Path) -> Result<TspProblem, Box<dyn std::error::Error>> {
    let tsp_data = parse_tsp_file(path)?;

    let problem = match tsp_data.edge_weight_type {
        EdgeWeightType::Euc2D | EdgeWeightType::Ceil2D => {
            if let Some(coords) = tsp_data.node_coords {
                let points: Vec<Point<f64>> =
                    coords.into_iter().map(|(x, y)| Point(x, y)).collect();
                TspProblem::Euclidean(PointsAndFunction::new(points))
            } else {
                return Err("Expected node coordinates for Euclidean problem".into());
            }
        }
        EdgeWeightType::Att => {
            if let Some(coords) = tsp_data.node_coords {
                let points: Vec<Point<f64>> =
                    coords.into_iter().map(|(x, y)| Point(x, y)).collect();
                TspProblem::Att(PointsAndFunction::new(points))
            } else {
                return Err("Expected node coordinates for Att problem".into());
            }
        }
        EdgeWeightType::Geo => {
            if let Some(coords) = tsp_data.node_coords {
                let points: Vec<Point<f64>> =
                    coords.into_iter().map(|(x, y)| Point(x, y)).collect();
                TspProblem::Geo(PointsAndFunction::new(points))
            } else {
                return Err("Expected node coordinates for Att problem".into());
            }
        }
        EdgeWeightType::Explicit => {
            if let Some(weights) = tsp_data.edge_weights {
                let flat_weights: Vec<i64> = weights.into_iter().flatten().collect();
                TspProblem::DistanceMatrix(DistanceMatrix::from_flat(flat_weights))
            } else {
                return Err("Expected edge weights for explicit problem".into());
            }
        }
        _ => {
            return Err(format!(
                "Unsupported edge weight type: {:?}",
                tsp_data.edge_weight_type
            )
            .into());
        }
    };
    Ok(problem)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::read_problem_file::read_problem_file;
    use test_case::test_case;
    use tsp::{
        city::City,
        problem::{Problem, TspProblem},
    };

    #[test_case("fri26", 26, 83, 42; "fri26")]
    #[test_case("gr96", 96, 1756, 814; "gr96")]
    fn test_read_problem_file(name: &str, size: usize, entry_0_1: i64, entry_2_4: i64) {
        let path = PathBuf::from(format!("data/problems/{name}.tsp"));
        assert!(path.exists());
        let problem: TspProblem = read_problem_file(&path).unwrap();
        assert_eq!(problem.size(), size);
        assert_eq!(problem.distance(City(0), City(1)), entry_0_1);
        assert_eq!(problem.distance(City(2), City(4)), entry_2_4);
    }
}
