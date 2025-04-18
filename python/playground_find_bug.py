import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import tsp_solve
import fast_tsp
import csv
import datetime
import os

from python.playground import (
    calculate_distance_matrix,
    calculate_tour_distance,
    generate_points,
    plot_tour,
)


def main():
    # for n = 10, lasse 1000 expermiente laufen und return das experiment,
    # wo die benutze längste  Kante von tsp_solve prozentual am längsten im gegensatz zur längsten von fast tsp ist.
    # Define problem sizes
    sizes = [15] * 1000

    # Set a fixed random seed for reproducibility
    seed = 42
    results = {
        "sizes": sizes,
        "tsp_solve_distances": [],
        "tsp_solve_iterations": [],
        "tsp_solve_times": [],
        "fast_tsp_distances": [],
        "fast_tsp_times": [],
        "points_data": [],  # Store points for each size
        "tsp_solve_tours": [],  # Store tours for each size
        "fast_tsp_tours": [],  # Store tours for each size
        "diff_percentages": [],  # Store difference percentages
    }
    for i, size in enumerate(sizes):

        # Generate points with consistent seed per size
        points = generate_points(size, seed=seed + i)
        results["points_data"].append(points)
        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(points)

        # Solve with tsp_solve
        start_time = time.time()
        tsp_solve_solution = tsp_solve.solve(distance_matrix)
        tsp_solve_time = time.time() - start_time

        # Solve with fast_tsp
        start_time = time.time()
        fast_tsp_tour = fast_tsp.find_tour(distance_matrix)
        fast_tsp_time = time.time() - start_time

        # Calculate fast_tsp distance
        fast_tsp_distance = calculate_tour_distance(fast_tsp_tour, distance_matrix)

        # Calculate difference percentage
        diff_percentage = abs(tsp_solve_solution.distance - fast_tsp_distance) / min(
            tsp_solve_solution.distance, fast_tsp_distance
        )
        results["diff_percentages"].append(diff_percentage)

        # Record results for tsp_solve
        results["tsp_solve_distances"].append(tsp_solve_solution.distance)
        results["tsp_solve_iterations"].append(tsp_solve_solution.iterations)
        results["tsp_solve_times"].append(tsp_solve_time)
        results["tsp_solve_tours"].append(tsp_solve_solution.tour)

        # Record results for fast_tsp
        results["fast_tsp_distances"].append(fast_tsp_distance)
        results["fast_tsp_times"].append(fast_tsp_time)
        results["fast_tsp_tours"].append(fast_tsp_tour)

        print(
            f"  tsp_solve - Distance: {tsp_solve_solution.distance}, Iterations: {tsp_solve_solution.iterations}, Time: {tsp_solve_time:.2f} seconds"
        )
        print(
            f"  fast_tsp - Distance: {fast_tsp_distance}, Time: {fast_tsp_time:.2f} seconds"
        )
        print(f"  Difference: {diff_percentage * 100:.2f}%")

    # Find the instance with the largest difference
    max_diff_idx = np.argmax(results["diff_percentages"])
    max_diff_size = results["sizes"][max_diff_idx]
    max_diff_points = results["points_data"][max_diff_idx]
    fig, axs = plt.subplots(1, 2, figsize=(15, 12))
    # Plot 3: tsp_solve route for the instance with the largest difference
    tsp_solve_tour = results["tsp_solve_tours"][max_diff_idx]
    plot_tour(
        axs[0],
        max_diff_points,
        tsp_solve_tour,
        f"tsp_solve Tour (n={max_diff_size}, dist={results['tsp_solve_distances'][max_diff_idx]})",
    )

    # Plot 4: fast_tsp route for the instance with the largest difference
    fast_tsp_tour = results["fast_tsp_tours"][max_diff_idx]
    plot_tour(
        axs[1],
        max_diff_points,
        fast_tsp_tour,
        f"fast_tsp Tour (n={max_diff_size}, dist={results['fast_tsp_distances'][max_diff_idx]})",
    )
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save file with timestamp
    filename = f"results/tsp_comparison_{timestamp}.png"
    print(max_diff_points)
    print(tsp_solve_tour)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    main()
