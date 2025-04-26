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


def generate_points(size, seed=42):
    """Generate random 2D points."""
    random.seed(seed)
    points = [
        (random.uniform(0, 1_000_000), random.uniform(0, 1_000_000))
        for _ in range(size)
    ]
    return points


def calculate_distance_matrix(points):
    """Calculate the Euclidean distance matrix for the given points."""
    size = len(points)
    distance_matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i != j:
                distance_matrix[i][j] = int(
                    math.sqrt(
                        (points[i][0] - points[j][0]) ** 2
                        + (points[i][1] - points[j][1]) ** 2
                    )
                )
    return distance_matrix


def run_experiment(sizes, seed=42):
    """Run the experiment for different problem sizes using both tsp_solve and fast_tsp."""
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

    for size in sizes:
        print(f"Running experiment with {size} cities...")

        # Generate points with consistent seed per size
        points = generate_points(size, seed=seed + size)
        results["points_data"].append(points)

        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(points)

        # Solve with tsp_solve
        start_time = time.time()
        tsp_solve_solution = tsp_solve.solve(distance_matrix, 10.0)
        tsp_solve_time = time.time() - start_time

        # Solve with fast_tsp
        start_time = time.time()
        fast_tsp_tour = fast_tsp.find_tour(distance_matrix)
        fast_tsp_time = time.time() - start_time

        # Calculate fast_tsp distance
        fast_tsp_distance = calculate_tour_distance(fast_tsp_tour, distance_matrix)

        # Calculate difference percentage
        diff_percentage = (tsp_solve_solution.distance - fast_tsp_distance) / min(
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

    return results


def calculate_tour_distance(tour, distance_matrix):
    """Calculate the total distance of a tour."""
    total_distance = 0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]
        total_distance += distance_matrix[from_city][to_city]
    return total_distance


def plot_results(results):
    """Plot the experimental results in a 2x2 grid."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Running time vs Problem size
    axs[0, 0].plot(
        results["sizes"], results["tsp_solve_times"], "ro-", label="tsp_solve"
    )
    axs[0, 0].plot(results["sizes"], results["fast_tsp_times"], "bo-", label="fast_tsp")
    axs[0, 0].set_xlabel("Number of Cities")
    axs[0, 0].set_ylabel("Solution Time (seconds)")
    axs[0, 0].set_title("Computation Time vs Problem Size")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Plot 2: Solution quality (distance) vs runtime
    axs[0, 1].scatter(
        results["tsp_solve_times"],
        results["tsp_solve_distances"],
        color="red",
        label="tsp_solve",
    )
    axs[0, 1].scatter(
        results["fast_tsp_times"],
        results["fast_tsp_distances"],
        color="blue",
        label="fast_tsp",
    )
    axs[0, 1].set_xlabel("Solution Time (seconds)")
    axs[0, 1].set_ylabel("Tour Distance")
    axs[0, 1].set_title("Solution Quality vs Runtime")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Find the instance with the largest difference
    max_diff_idx = np.argmax(results["diff_percentages"])
    max_diff_size = results["sizes"][max_diff_idx]
    max_diff_points = results["points_data"][max_diff_idx]

    # Plot 3: tsp_solve route for the instance with the largest difference
    tsp_solve_tour = results["tsp_solve_tours"][max_diff_idx]
    plot_tour(
        axs[1, 0],
        max_diff_points,
        tsp_solve_tour,
        f"tsp_solve Tour (n={max_diff_size}, dist={results['tsp_solve_distances'][max_diff_idx]})",
    )

    # Plot 4: fast_tsp route for the instance with the largest difference
    fast_tsp_tour = results["fast_tsp_tours"][max_diff_idx]
    plot_tour(
        axs[1, 1],
        max_diff_points,
        fast_tsp_tour,
        f"fast_tsp Tour (n={max_diff_size}, dist={results['fast_tsp_distances'][max_diff_idx]})",
    )

    # Add info about difference
    max_diff_percent = results["diff_percentages"][max_diff_idx] * 100
    fig.suptitle(
        f"TSP Solver Comparison (Max Difference: {max_diff_percent:.2f}% at n={max_diff_size})"
    )

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save file with timestamp
    filename = f"results/tsp_comparison_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Plot saved to {filename}")


def plot_tour(ax, points, tour, title):
    """Plot a TSP tour."""
    # Extract coordinates for the tour
    x = [points[city][0] for city in tour]
    y = [points[city][1] for city in tour]

    # Add the connection back to the starting city
    x.append(x[0])
    y.append(y[0])

    # Plot the route
    ax.plot(x, y, "b-", linewidth=0.8)
    ax.plot(x, y, "ro", markersize=3)

    # Highlight start point
    ax.plot(x[0], y[0], "go", markersize=6)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)


def fit_complexity(sizes, iterations):
    """Fit a polynomial curve to estimate computational complexity."""
    log_sizes = np.log(sizes)
    log_iterations = np.log(iterations)

    # Linear regression on log-log data
    coeffs = np.polyfit(log_sizes, log_iterations, 1)

    # The slope of the line is the exponent in the complexity
    complexity_exponent = coeffs[0]

    # Generate fitted curve for plotting
    fitted_curve = np.exp(coeffs[1]) * np.array(sizes) ** complexity_exponent

    return complexity_exponent, fitted_curve


def save_results_to_csv(results, filename=None):
    """Save the experimental results to a CSV file."""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename with timestamp if not provided
    if filename is None:
        filename = f"results/experiment_results_{timestamp}.csv"
    else:
        # Add the directory prefix to the provided filename
        filename = os.path.join("results", filename)

    with open(filename, "w", newline="") as csvfile:
        fieldnames = [
            "dimension",
            "tsp_solve_iterations",
            "tsp_solve_distance",
            "tsp_solve_time",
            "fast_tsp_distance",
            "fast_tsp_time",
            "difference_percentage",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, size in enumerate(results["sizes"]):
            writer.writerow(
                {
                    "dimension": size,
                    "tsp_solve_iterations": results["tsp_solve_iterations"][i],
                    "tsp_solve_distance": results["tsp_solve_distances"][i],
                    "tsp_solve_time": results["tsp_solve_times"][i],
                    "fast_tsp_distance": results["fast_tsp_distances"][i],
                    "fast_tsp_time": results["fast_tsp_times"][i],
                    "difference_percentage": results["diff_percentages"][i] * 100,
                }
            )
    print(f"Results saved to {filename}")


def main():
    # Define problem sizes
    sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    sizes = [i + 3000 for i in range(10)]

    # Set a fixed random seed for reproducibility
    seed = 42

    # Run experiments with both solvers
    results = run_experiment(sizes, seed)

    # Save results to CSV
    save_results_to_csv(results)

    # Only fit complexity for tsp_solve which has iterations
    exponent, fitted_curve = fit_complexity(
        results["sizes"], results["tsp_solve_iterations"]
    )
    print(f"Estimated complexity for tsp_solve: O(n^{exponent:.2f})")

    # Plot results
    plot_results(results)

    # Print summary table
    print("\nSummary:")
    print(
        "--------------------------------------------------------------------------------------------------"
    )
    print(
        "| Size | tsp_solve |            |           | fast_tsp |           | Difference |"
    )
    print(
        "|      | Distance  | Iterations | Time (s)  | Distance | Time (s)  | (%)        |"
    )
    print(
        "--------------------------------------------------------------------------------------------------"
    )
    for i, size in enumerate(results["sizes"]):
        print(
            f"| {size:4d} | {results['tsp_solve_distances'][i]:8d} | {results['tsp_solve_iterations'][i]:10d} | "
            f"{results['tsp_solve_times'][i]:8.2f} | {results['fast_tsp_distances'][i]:8d} | "
            f"{results['fast_tsp_times'][i]:8.2f} | {results['diff_percentages'][i] * 100:10.2f} |"
        )
    print(
        "--------------------------------------------------------------------------------------------------"
    )

    # Find instance with largest difference
    max_diff_idx = np.argmax(results["diff_percentages"])
    max_diff_size = results["sizes"][max_diff_idx]
    max_diff_percent = results["diff_percentages"][max_diff_idx] * 100

    print(f"\nLargest difference: {max_diff_percent:.2f}% at size {max_diff_size}")
    print(f"  tsp_solve distance: {results['tsp_solve_distances'][max_diff_idx]}")
    print(f"  fast_tsp distance: {results['fast_tsp_distances'][max_diff_idx]}")


if __name__ == "__main__":
    main()
