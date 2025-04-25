mod domain;
mod input;
mod local_move;
pub mod penalties;
mod postprocess;
mod preprocess;
mod solution;
mod solver;
use csv::Writer;
use input::Input;
use rand::{self, rngs::StdRng, Rng, SeedableRng};
use solver::Solver;
use std::time::Instant;
use std::{error::Error, fs::File};

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = StdRng::seed_from_u64(43);
    // let mut solutions = vec![];
    let number_trials = 10;
    let problem_size = 1000;
    let square_width = 1000;
    for i in 0..number_trials {
        // if i % 100 == 0 {
        //     println!("{}", i);
        // }
        let mut city_coordinates = (0..problem_size)
            .into_iter()
            .map(|_| {
                (
                    rng.random_range(0..square_width),
                    rng.random_range(0..square_width),
                )
            })
            .collect::<Vec<(i32, i32)>>();
        // city_coordinates = cities.clone();
        // if i != 9665 {
        //     continue;
        // }
        let distance_matrix = city_coordinates
            .iter()
            .map(|(x, y)| {
                city_coordinates
                    .iter()
                    .map(|(a, b)| (((x - a) * (x - a) + (y - b) * (y - b)) as i64).isqrt())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let raw_input = preprocess::RawInput::new(distance_matrix.clone(), Some(1.0));
        let input: Input = raw_input.into();
        let mut solver = Solver::new(input);

        let start = Instant::now();
        let sol1 = solver.solve(true);
        let duration = Instant::now().checked_duration_since(start);
        println!("Time elapsed dlb: {:?}", duration);
        println!("Iterations in total dlb: {:?}", sol1.stats.iterations);

        println!(
            "Iterations since last improvement dlb: {:?}",
            sol1.stats.iterations_since_last_improvement
        );
        let raw_input = preprocess::RawInput::new(distance_matrix.clone(), Some(1.0));

        let input: Input = raw_input.into();
        let mut solver = Solver::new(input);

        let start = Instant::now();
        let sol2 = solver.solve(false);

        let duration = Instant::now().checked_duration_since(start);
        println!("Time elapsed no dlb: {:?}", duration);
        println!("Iterations in total no dlb: {:?}", sol2.stats.iterations);

        println!(
            "Iterations since last improvement no dlb: {:?}",
            sol2.stats.iterations_since_last_improvement
        );
        if sol1.best_solution != sol2.best_solution {
            let precent_gap =
                (sol1.best_solution.distance as f64 / sol2.best_solution.distance as f64 - 1.0)
                    * 100.0;
            println!(
                "different solutions detected at iteration {}: {:.2}.",
                i, precent_gap
            );
        }
    }
    // let distances = solutions
    //     .into_iter()
    //     .map(|s| s.best_solution.distance)
    //     .collect::<Vec<i64>>();
    // println!("{}", distances.into_iter().sum::<i64>());
    // Create a file to write to
    // let file = File::create("nodontlookbits100k.csv")?;
    // let mut writer = Writer::from_writer(file);

    // // Write each element as a row
    // for num in distances {
    //     writer.write_record(&[num.to_string()])?;
    // }

    // // Flush the writer to ensure all data is written
    // writer.flush()?;

    // println!("CSV file created successfully!");
    Ok(())
    // with don't look bits it is 24781
}
