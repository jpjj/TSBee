mod domain;
mod input;
mod local_move;
pub mod penalties;
mod postprocess;
mod preprocess;
mod solution;
mod solver;
use input::Input;
use penalties::distance::DistanceMatrix;
use rand::{self, rngs::StdRng, Rng, SeedableRng};
use solver::Solver;
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(43);
    // let mut solutions = vec![];
    let number_trials = 10;
    let problem_size = 3000;
    let square_width = 1_000_000;
    for _ in 0..number_trials {
        // if i % 100 == 0 {
        //     println!("{}", i);
        // }
        let city_coordinates = (0..problem_size)
            .map(|_| {
                (
                    rng.random_range(0..square_width),
                    rng.random_range(0..square_width),
                )
            })
            .collect::<Vec<(i64, i64)>>();
        let distance_matrix = DistanceMatrix::new_euclidian(city_coordinates);
        let input: Input = Input::new(distance_matrix, Some(chrono::TimeDelta::seconds(2)));
        let mut solver = Solver::new(input);

        let start = Instant::now();
        let sol1 = solver.solve(false);
        let duration = Instant::now().checked_duration_since(start);
        println!("Time elapsed: {:?}", duration);
        println!("Iterations in total: {:?}", sol1.stats.iterations);

        println!(
            "Iterations since last improvement: {:?}",
            sol1.stats.iterations_since_last_improvement
        );
        if let Some(result) = sol1.stats.held_karp_result {
            let precent_gap =
                (sol1.best_solution.distance as f64 / result.min_one_tree.score as f64 - 1.0)
                    * 100.0;
            println!("Held-karp-gap: {:.2}.", precent_gap);
        }
    }

    Ok(())
}
