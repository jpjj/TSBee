"""
Example using randomly generated cities with Euclidean distances.

This example shows how to:
1. Generate random city coordinates
2. Calculate a distance matrix from coordinates
3. Solve the TSP
4. Visualize the solution
"""

import matplotlib.pyplot as plt
import numpy as np
import tsp_solve


def calculate_distance_matrix(coords):
    """Calculate Euclidean distance matrix from coordinates."""
    n = len(coords)
    distance_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate Euclidean distance
                dist = np.sqrt(
                    (coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2
                )
                # Convert to integer (required by solver)
                # Multiply by 100 to preserve precision
                distance_matrix[i][j] = int(dist * 100)

    return distance_matrix


def visualize_tour(coords, tour, title="TSP Solution"):
    """Visualize the TSP tour."""
    plt.figure(figsize=(10, 8))

    # Plot cities
    x = [coords[i][0] for i in range(len(coords))]
    y = [coords[i][1] for i in range(len(coords))]
    plt.scatter(x, y, c="red", s=100, zorder=2)

    # Label cities
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.annotate(str(i), (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Plot tour
    tour_x = [coords[tour[i]][0] for i in range(len(tour))]
    tour_y = [coords[tour[i]][1] for i in range(len(tour))]

    # Close the tour by adding the first city at the end
    tour_x.append(tour_x[0])
    tour_y.append(tour_y[0])

    plt.plot(tour_x, tour_y, "b-", linewidth=1, alpha=0.7, zorder=1)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random cities
    n_cities = 30
    coords = np.random.rand(n_cities, 2) * 100  # Cities in 100x100 grid

    print(f"Generated {n_cities} random cities")
    print("-" * 40)

    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(coords)

    # Solve TSP
    print("Solving TSP...")
    solution = tsp_solve.solve(distance_matrix.tolist())

    # Print results
    print("\nResults:")
    print(f"Total distance: {solution.distance / 100:.2f} units")
    print(f"Tour: {solution.tour[:10]}... (first 10 cities)")
    print(f"Iterations: {solution.iterations}")
    print(f"Time taken: {solution.time:.3f} seconds")

    # Calculate statistics
    avg_distance_per_city = solution.distance / 100 / n_cities
    print("\nStatistics:")
    print(f"Average distance per city: {avg_distance_per_city:.2f} units")
    print(f"Cities per second: {n_cities / solution.time:.0f}")

    # Visualize the solution
    visualize_tour(
        coords,
        solution.tour,
        f"TSP Solution ({n_cities} cities, distance: {solution.distance/100:.2f})",
    )


if __name__ == "__main__":
    main()
