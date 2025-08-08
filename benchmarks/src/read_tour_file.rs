use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use tsp::city::City;
use tsp::solution::list::List;

pub fn read_tour_file<P>(path: &Path) -> Result<List<P>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut tour = Vec::new();
    let mut in_tour_section = false;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line == "TOUR_SECTION" {
            in_tour_section = true;
            continue;
        }

        if line == "-1" || line == "EOF" {
            break;
        }

        if in_tour_section {
            if let Ok(city) = line.parse::<i32>() {
                if city > 0 {
                    tour.push(City((city - 1) as usize));
                }
            }
        }
    }

    Ok(List::new(tour))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tsp::{
        city::City,
        problem::distance_matrix::DistanceMatrix,
        solution::{Solution, list::List},
    };

    use crate::read_tour_file::read_tour_file;

    #[test]
    fn test_read_tour_file() {
        let path = PathBuf::from("data/problems/a280.opt.tour");
        if path.exists() {
            let solution: List<DistanceMatrix<i32>> = read_tour_file(&path).unwrap();
            assert_eq!(solution.size(), 280);
            assert_eq!(solution[0], City(0));
            assert_eq!(solution[10], City(235));
        }
    }
}
