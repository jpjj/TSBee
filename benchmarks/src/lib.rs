pub mod read_best_values_file;
pub mod read_problem_file;
pub mod read_tour_file;

use std::fs;
use std::path::{Path, PathBuf};
use tsp::instance::{Instance, TspInstance};
use tsp::problem::TspProblem;
use tsp::solution::list::List;

pub fn get_all_instance_names() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let problems_dir = Path::new("data/problems");

    let mut instance_names = Vec::new();

    for entry in fs::read_dir(problems_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("tsp") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                instance_names.push(stem.to_string());
            }
        }
    }

    instance_names.sort();
    Ok(instance_names)
}

pub fn load_instance(instance_name: &str) -> Result<TspInstance, Box<dyn std::error::Error>> {
    let problem_path = PathBuf::from(format!("data/problems/{instance_name}.tsp"));
    let solution_path = PathBuf::from(format!("data/solutions/{instance_name}.opt.tour"));

    let problem = read_problem_file::read_problem_file(&problem_path)?;
    let solution: List<TspProblem> = read_tour_file::read_tour_file(&solution_path)?;

    Ok(Instance::new(problem, solution))
}

#[cfg(test)]
mod tests {

    use super::*;
    use tsp::problem::Problem;
    use tsp::solution::Solution;

    #[test]
    fn test_load_instance() {
        let name = String::from("a280");
        let result = load_instance(&name);
        assert!(result.is_ok());
        let instance = result.unwrap();
        assert_eq!(instance.problem.size(), instance.solution.size());
    }
}
