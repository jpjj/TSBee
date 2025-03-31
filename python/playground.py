import random
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import tsp_solve
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
    """Run the experiment for different problem sizes."""
    results = {"sizes": sizes, "distances": [], "iterations": [], "times": []}

    for size in sizes:
        print(f"Running experiment with {size} cities...")

        # Generate points with consistent seed per size
        points = generate_points(size, seed=seed + size)

        # Calculate distance matrix
        distance_matrix = calculate_distance_matrix(points)

        # Solve TSP
        start_time = time.time()
        solution = tsp_solve.solve(distance_matrix)
        solve_time = time.time() - start_time

        # Record results
        results["distances"].append(solution.distance)
        results["iterations"].append(solution.iterations)
        results["times"].append(solve_time)

        print(f"  Distance: {solution.distance}")
        print(f"  Iterations: {solution.iterations}")
        print(f"  Time: {solve_time:.2f} seconds")

    return results


def plot_results(results):
    """Plot the experimental results."""
    plt.figure(figsize=(8, 6))

    # Plot time vs problem size
    plt.plot(results["sizes"], results["times"], "go-")
    plt.xlabel("Number of Cities")
    plt.ylabel("Solution Time (seconds)")
    plt.title("Computation Time vs Problem Size")
    plt.grid(True)

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save file with timestamp
    filename = f"results/scaling_results_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Plot saved to {filename}")


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
        fieldnames = ["dimension", "iterations", "distance", "time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, size in enumerate(results["sizes"]):
            writer.writerow(
                {
                    "dimension": size,
                    "iterations": results["iterations"][i],
                    "distance": results["distances"][i],
                    "time": results["times"][i],
                }
            )
    print(f"Results saved to {filename}")


def main():
    # Define problem sizes
    sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Set a fixed random seed for reproducibility
    seed = 42

    # Run experiments
    results = run_experiment(sizes, seed)

    # Save results to CSV
    save_results_to_csv(results)

    # Fit complexity curve
    exponent, fitted_curve = fit_complexity(results["sizes"], results["iterations"])
    print(f"Estimated complexity: O(n^{exponent:.2f})")

    # Add fitted curve to results
    results["fitted_curve"] = fitted_curve

    # Plot results
    plot_results(results)

    # Print summary table
    print("\nSummary:")
    print("-------------------------------------------------------------------------")
    print(
        "| Size | Distance | Iterations | Time (s) | Iterations/Size^{:.2f} |".format(
            exponent
        )
    )
    print("-------------------------------------------------------------------------")
    for i, size in enumerate(results["sizes"]):
        ratio = results["iterations"][i] / (size**exponent)
        print(
            f"| {size:4d} | {results['distances'][i]:8d} | {results['iterations'][i]:10d} | {results['times'][i]:8.2f} | {ratio:21.2f} |"
        )
    print("-------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
