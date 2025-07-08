use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

use chrono::TimeDelta;
use tsbee::domain::city::City;
use tsbee::domain::route::Route;
use tsbee::penalties::candidates::alpha_nearness::get_alpha_candidates;
use tsbee::penalties::candidates::held_karp::BoundCalculator;
use tsbee::penalties::candidates::Candidates;
use tsbee::penalties::distance::{DistanceMatrix, DistancePenalizer};

fn read_att532() -> Vec<(i64, i64)> {
    let file_path = "./tests/data/ATT532/att532.tsp";
    let mut coordinates = Vec::new();

    // Open the file
    let file = match File::open(file_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error opening file {}: {}", file_path, e);
            return coordinates;
        }
    };

    let reader = BufReader::new(file);

    // Skip header lines and process only coordinate data
    let mut in_coord_section = false;

    for line in reader.lines() {
        let line = match line {
            Ok(line) => line.trim().to_string(),
            Err(e) => {
                eprintln!("Error reading line: {}", e);
                continue;
            }
        };

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Check if we've reached the coordinate section
        if line == "NODE_COORD_SECTION" {
            in_coord_section = true;
            continue;
        }

        // Check if we've reached the end
        if line == "EOF" {
            break;
        }

        // Parse coordinates if we're in the coordinate section
        if in_coord_section {
            let parts: Vec<&str> = line.split_whitespace().collect();

            // Each line should have 3 parts: node_id x_coord y_coord
            if parts.len() == 3 {
                // Parse x and y coordinates (skip the node ID)
                match (parts[1].parse::<i64>(), parts[2].parse::<i64>()) {
                    (Ok(x), Ok(y)) => {
                        coordinates.push((x, y));
                    }
                    _ => {
                        eprintln!("Error parsing coordinates from line: {}", line);
                    }
                }
            }
        }
    }

    coordinates
}

fn read_att532_tour() -> Vec<usize> {
    let file_path = "./tests/data/ATT532/att532.tour";
    let mut tour: Vec<usize> = Vec::new();

    // Open the file
    let file = match File::open(file_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error opening file {}: {}", file_path, e);
            return tour;
        }
    };

    let reader = BufReader::new(file);

    // Skip header lines and process only coordinate data
    let mut in_tour_section = false;

    for line in reader.lines() {
        let line = match line {
            Ok(line) => line.trim().to_string(),
            Err(e) => {
                eprintln!("Error reading line: {}", e);
                continue;
            }
        };

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Check if we've reached the coordinate section
        if line == "TOUR_SECTION" {
            in_tour_section = true;
            continue;
        }

        // Check if we've reached the end
        if line == "EOF" {
            break;
        }

        // Parse coordinates if we're in the coordinate section
        if in_tour_section {
            let parts: Vec<&str> = line.split_whitespace().collect();

            // Each line should have 3 parts: node_id x_coord y_coord
            if parts.len() == 1 {
                // Parse x and y coordinates (skip the node ID)
                match parts[0].parse::<usize>() {
                    Ok(x) => {
                        tour.push(x - 1);
                    }
                    _ => {
                        eprintln!("Error parsing tour cities from line: {}", line);
                    }
                }
            }
        }
    }

    tour
}

#[test]
fn test_read_att532() {
    let coords = read_att532();
    assert_eq!(coords.len(), 532);
    assert_eq!(coords[0], (7810, 6053)); // First city
    assert_eq!(coords[531], (5469, 10)); // Last city (532nd city)
}

#[test]
fn test_correct_opt_sol_att532() {
    let points = read_att532();
    let dm = DistanceMatrix::new_att(points);
    let distance_penalizer = DistancePenalizer::new(dm);
    let optimal_route: Route = read_att532_tour().into_iter().collect();
    let solution = distance_penalizer.penalize(&optimal_route);
    assert_eq!(solution.distance, 27686 * 1_000_000);
}

fn get_optimal_neighors_att532() -> Vec<HashSet<usize>> {
    let optimal_route: Vec<usize> = read_att532_tour();
    let n = optimal_route.len();
    let optimal_neighbors: Vec<HashSet<usize>> = optimal_route
        .iter()
        .enumerate()
        .map(|(i, _)| {
            vec![
                optimal_route[(n + i - 1) % n],
                optimal_route[(n + i + 1) % n],
            ]
            .into_iter()
            .collect()
        })
        .collect();
    optimal_neighbors
}

fn print_rank_counter(alpha_cans: Candidates) {
    let optimal_route: Vec<usize> = read_att532_tour();
    let optimal_neighbors = get_optimal_neighors_att532();
    let mut rank_counter: Vec<f64> = vec![0.0; 100];
    for (i, city) in optimal_route.iter().enumerate() {
        let cans = alpha_cans.get_neighbors_out(&City(*city));
        let cans_len = cans.len();
        for j in 0..cans_len {
            if optimal_neighbors[i].contains(&cans[j].id()) {
                rank_counter[j] += 1.0;
            }
        }
    }
    let sumi = rank_counter.iter().sum::<f64>();
    rank_counter = rank_counter.iter().map(|x| x * 100.0 / sumi).collect();
    for (i, _) in rank_counter.iter().enumerate().take(23) {
        eprintln!("{}: {:.1}", i + 1, rank_counter[i]);
    }
    let average_rank: f64 = rank_counter
        .iter()
        .enumerate()
        .map(|(i, prob)| (i + 1) as f64 * prob)
        .sum::<f64>()
        / 100.0;
    eprint!("Average Rank: {:.1}", average_rank);
}

#[test]
fn test_correct_alpha_nearness_neighbors_att532() {
    let dm = DistanceMatrix::new_att(read_att532());
    let alpha_cans = get_alpha_candidates(&dm, 532);
    print_rank_counter(alpha_cans);
}

#[test]
fn test_correct_alpha_nearness_neighbors_improved_att532() {
    let mut dm = DistanceMatrix::new_att(read_att532());
    let upper_bound = 27686 * 1_000_000;
    let alpha_cans = get_alpha_candidates(&dm, 20);
    let mut bound_calculator = BoundCalculator::with_candidates(
        dm.clone(),
        alpha_cans,
        upper_bound,
        50000,
        TimeDelta::seconds(3),
    );
    let result = bound_calculator.run();
    dm.update_pi(result.pi.clone());
    let alpha_cans = get_alpha_candidates(&dm, 10);
    print_rank_counter(alpha_cans);
}
