//! This script is used to read TSPLIB problem files and return TspFileData struct.
//! The definition of these files can be found here: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub enum ProblemType {
    TSP,
    ATSP,
    SOP,
    HCP,
    CVRP,
    TOUR,
}

#[derive(Debug, Clone)]
pub enum EdgeWeightType {
    Explicit,
    Euc2D,
    Euc3D,
    MAX2D,
    MAX3D,
    MAN2D,
    MAN3D,
    Ceil2D,
    Geo,
    Att,
    XRay1,
    XRay2,
    Special,
}

#[derive(Debug, Clone)]
pub enum EdgeWeightFormat {
    Function,
    FullMatrix,
    UpperRow,
    LowerRow,
    UpperDiagRow,
    LowerDiagRow,
    UpperCol,
    LowerCol,
    UpperDiagCol,
    LowerDiagCol,
}

#[derive(Debug, Clone)]
pub enum EdgeDataFormat {
    EdgeList,
    AdjList,
}

#[derive(Debug, Clone)]
pub enum NodeCoordType {
    TwoDCoords,
    ThreeDCoords,
    NoCoords,
}

#[derive(Debug)]
pub struct TspFileData {
    pub name: String,
    pub dimension: usize,
    pub edge_weight_type: EdgeWeightType,
    pub edge_weight_format: Option<EdgeWeightFormat>,
    pub node_coords: Option<Vec<(f64, f64)>>,
    pub edge_weights: Option<Vec<Vec<i64>>>,
}

pub fn parse_tsp_file(path: &Path) -> Result<TspFileData, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut name = String::new();
    let mut dimension = 0;
    let mut edge_weight_type = EdgeWeightType::Euc2D;
    let mut edge_weight_format = None;
    let mut node_coords = Vec::new();
    let mut edge_weights = Vec::new();

    let mut in_coord_section = false;
    let mut in_weight_section = false;
    let mut in_display_section = false;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty() || in_display_section {
            continue;
        }

        if line == "EOF" {
            break;
        }

        if line == "NODE_COORD_SECTION" {
            in_coord_section = true;
            in_weight_section = false;
            continue;
        }

        if line == "EDGE_WEIGHT_SECTION" {
            in_weight_section = true;
            in_coord_section = false;
            continue;
        }

        if line == "DISPLAY_DATA_SECTION" {
            in_display_section = true;
            in_coord_section = false;
            in_weight_section = false;
            continue;
        }

        if in_coord_section {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let (Ok(_), Ok(x), Ok(y)) = (
                    parts[0].parse::<usize>(),
                    parts[1].parse::<f64>(),
                    parts[2].parse::<f64>(),
                ) {
                    node_coords.push((x, y));
                }
            }
        } else if in_weight_section {
            let weights: Vec<i64> = line
                .split_whitespace()
                .filter_map(|s| s.parse::<i64>().ok())
                .collect();
            if !weights.is_empty() {
                edge_weights.extend(weights);
            }
        } else if line.contains(':') {
            let parts: Vec<&str> = line.splitn(2, ':').collect();
            if parts.len() == 2 {
                let key = parts[0].trim();
                let value = parts[1].trim();

                match key {
                    "NAME" => name = value.to_string(),
                    "DIMENSION" => dimension = value.parse().unwrap_or(0),
                    "EDGE_WEIGHT_TYPE" => {
                        edge_weight_type = match value {
                            "EUC_2D" => EdgeWeightType::Euc2D,
                            "CEIL_2D" => EdgeWeightType::Ceil2D,
                            "ATT" => EdgeWeightType::Att,
                            "GEO" => EdgeWeightType::Geo,
                            "EXPLICIT" => EdgeWeightType::Explicit,
                            _ => EdgeWeightType::Euc2D,
                        }
                    }
                    "EDGE_WEIGHT_FORMAT" => {
                        edge_weight_format = Some(match value {
                            "UPPER_ROW" => EdgeWeightFormat::UpperRow,
                            "LOWER_ROW" => EdgeWeightFormat::LowerRow,
                            "UPPER_DIAG_ROW" => EdgeWeightFormat::UpperDiagRow,
                            "LOWER_DIAG_ROW" => EdgeWeightFormat::LowerDiagRow,
                            "FULL_MATRIX" => EdgeWeightFormat::FullMatrix,
                            "FUNCTION" => EdgeWeightFormat::Function,
                            _ => EdgeWeightFormat::FullMatrix,
                        })
                    }
                    _ => {}
                }
            }
        }
    }

    let matrix = if !edge_weights.is_empty() {
        Some(build_distance_matrix(
            dimension,
            &edge_weights,
            &edge_weight_format,
        )?)
    } else {
        None
    };

    Ok(TspFileData {
        name,
        dimension,
        edge_weight_type,
        edge_weight_format,
        node_coords: if node_coords.is_empty() {
            None
        } else {
            Some(node_coords)
        },
        edge_weights: matrix,
    })
}

pub fn build_distance_matrix(
    dimension: usize,
    weights: &[i64],
    format: &Option<EdgeWeightFormat>,
) -> Result<Vec<Vec<i64>>, Box<dyn std::error::Error>> {
    let mut matrix = vec![vec![0; dimension]; dimension];
    let mut no_matching_format = false;
    let n = dimension;
    match format {
        Some(EdgeWeightFormat::FullMatrix) => {
            matrix = weights.chunks(n).map(|x| x.to_vec()).collect();
        }
        Some(EdgeWeightFormat::UpperRow) => {
            let mut idx = 0;
            for i in 0..dimension {
                for j in i + 1..dimension {
                    if idx < weights.len() {
                        matrix[i][j] = weights[idx];
                        matrix[j][i] = weights[idx];
                        idx += 1;
                    }
                }
            }
        }
        Some(EdgeWeightFormat::LowerRow) => {
            let mut idx = 0;
            for i in 1..dimension {
                for j in 0..i {
                    if idx < weights.len() {
                        matrix[i][j] = weights[idx];
                        matrix[j][i] = weights[idx];
                        idx += 1;
                    }
                }
            }
        }
        Some(EdgeWeightFormat::UpperDiagRow) => {
            let mut idx = 0;
            for i in 0..dimension {
                for j in i..dimension {
                    if idx < weights.len() {
                        matrix[i][j] = weights[idx];
                        matrix[j][i] = weights[idx];
                        idx += 1;
                    }
                }
            }
        }
        Some(EdgeWeightFormat::LowerDiagRow) => {
            let mut idx = 0;
            for i in 0..dimension {
                for j in 0..=i {
                    if idx < weights.len() {
                        matrix[i][j] = weights[idx];
                        matrix[j][i] = weights[idx];
                        idx += 1;
                    }
                }
            }
        }
        _ => {
            no_matching_format = true;
        }
    }
    if no_matching_format {
        return Err("No suitalbe edge weight format".into());
    }
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use crate::read_problem_file::parse_tsp_file::EdgeWeightType;

    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_euclidean_tsp() {
        let path = PathBuf::from("data/problems/berlin52.tsp");
        assert!(path.exists());
        let data = parse_tsp_file(&path).unwrap();
        assert_eq!(data.dimension, 52);
        assert_eq!(data.name, "berlin52");
        assert!(matches!(data.edge_weight_type, EdgeWeightType::Euc2D));
        assert!(data.node_coords.is_some());
    }
}
