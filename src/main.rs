use rand::Rng;
use std::time::Instant;

fn main() {
    // Create a vector of length 1000 with values 0..999
    let mut vec: Vec<usize> = (0..1000).collect();

    // Number of slice reversals to perform
    const NUM_REVERSALS: usize = 7424;

    // Initialize random number generator
    let mut rng = rand::rng();

    // Start timing
    let start_time = Instant::now();

    // Perform random slice reversals
    for _ in 0..NUM_REVERSALS {
        // Generate random start and end indices for the slice
        // Ensure start < end to have a valid slice
        let start = rng.random_range(0..999);
        let end = rng.random_range(start + 1..1000);

        // Reverse the slice
        vec[start..end].reverse();
    }

    // Calculate elapsed time
    let elapsed = start_time.elapsed();
    println!("Reversed slices: {:?}", &vec); // Print first 10 elements for verification

    println!("Performed {NUM_REVERSALS} random slice reversals on a vector of length 1000");
    println!("Total time: {:.6} seconds", elapsed.as_secs_f64());
    println!(
        "Average time per reversal: {:.9} seconds",
        elapsed.as_secs_f64() / NUM_REVERSALS as f64
    );
}
