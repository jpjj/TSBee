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
            let row_of_cities: Vec<&str> = line.split_whitespace().collect();
            for new_city in row_of_cities.iter() {
                if let Ok(city) = new_city.parse::<i32>() {
                    if city > 0 {
                        tour.push(City((city - 1) as usize));
                    }
                }
            }
        }
    }

    Ok(List::new(tour))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::read_tour_file::read_tour_file;
    use test_case::test_case;
    use tsp::{
        city::City,
        problem::distance_matrix::DistanceMatrix,
        solution::{Solution, list::List},
    };

    #[test_case("a280", 280, 0, 279; "a280")]
    #[test_case("gr24", 24, 15, 0; "gr24")]
    fn test_read_tour_file(name: &str, size: usize, first_city_idx: usize, last_city_idx: usize) {
        let path = PathBuf::from(format!("data/solutions/{name}.opt.tour"));
        assert!(path.exists());
        let solution: List<DistanceMatrix<i32>> = read_tour_file(&path).unwrap();
        assert_eq!(solution.size(), size);
        assert_eq!(solution[0], City(first_city_idx));
        assert_eq!(solution[size - 1], City(last_city_idx));
    }
}
