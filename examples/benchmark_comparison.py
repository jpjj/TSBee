"""
Benchmark comparison example.

This example compares tsbee with other TSP solvers and demonstrates
how to measure performance across different problem sizes.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import tsbee


def generate_clustered_cities(n_cities: int, n_clusters: int = 4) -> np.ndarray:
    """Generate cities in clusters (makes TSP more challenging)."""
    coords = []
    cities_per_cluster = n_cities // n_clusters

    # Define cluster centers
    cluster_centers = [(25, 25), (75, 25), (25, 75), (75, 75)][:n_clusters]

    for i, (cx, cy) in enumerate(cluster_centers):
        # Generate cities around each cluster center
        n = cities_per_cluster if i < n_clusters - 1 else n_cities - len(coords)
        cluster_coords = np.random.randn(n, 2) * 10 + [cx, cy]
        coords.extend(cluster_coords)

    return np.array(coords)


def calculate_tour_distance(coords: np.ndarray, tour: List[int]) -> float:
    """Calculate the total distance of a tour."""
    distance = 0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        distance += np.linalg.norm(coords[tour[i]] - coords[tour[j]])
    return distance


def benchmark_solver(coords: np.ndarray, time_limit: float = None) -> Dict:
    """Benchmark tsbee on given coordinates."""
    n = len(coords)

    # Calculate distance matrix
    distance_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                distance_matrix[i][j] = int(dist * 100)

    # Solve
    start_time = time.time()
    solution = tsbee.solve(distance_matrix.tolist(), time_limit=time_limit)
    end_time = time.time()

    # Calculate actual distance
    actual_distance = calculate_tour_distance(coords, solution.tour)

    return {
        "solver": "tsbee",
        "n_cities": n,
        "distance": actual_distance,
        "time": end_time - start_time,
        "iterations": solution.iterations,
        "tour": solution.tour,
    }


def nearest_neighbor_heuristic(coords: np.ndarray) -> Tuple[List[int], float]:
    """Simple nearest neighbor heuristic for comparison."""
    n = len(coords)
    unvisited = set(range(1, n))
    tour = [0]
    current = 0

    while unvisited:
        # Find nearest unvisited city
        nearest = min(unvisited, key=lambda j: np.linalg.norm(coords[current] - coords[j]))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    distance = calculate_tour_distance(coords, tour)
    return tour, distance


def main():
    np.random.seed(42)

    print("TSP Solver Benchmark Comparison")
    print("=" * 60)

    # Test different problem sizes
    problem_sizes = [10, 20, 50, 100]

    for n_cities in problem_sizes:
        print(f"\nProblem size: {n_cities} cities")
        print("-" * 40)

        # Generate cities
        coords = generate_clustered_cities(n_cities)

        # Benchmark tsbee
        result = benchmark_solver(coords)

        print("tsbee:")
        print(f"  Distance: {result['distance']:.2f}")
        print(f"  Time: {result['time']:.3f} seconds")
        print(f"  Iterations: {result['iterations']}")

        # Compare with nearest neighbor heuristic
        start_time = time.time()
        _, nn_distance = nearest_neighbor_heuristic(coords)
        nn_time = time.time() - start_time

        print("\nNearest Neighbor:")
        print(f"  Distance: {nn_distance:.2f}")
        print(f"  Time: {nn_time:.6f} seconds")

        # Calculate improvement
        improvement = (nn_distance - result["distance"]) / nn_distance * 100
        print(f"\nImprovement over NN: {improvement:.1f}%")
        print(f"Speed ratio: {result['time'] / nn_time:.1f}x slower")

    # Performance scaling analysis
    print("\n\nPerformance Scaling Analysis")
    print("=" * 60)

    sizes = [10, 20, 30, 40, 50, 75, 100]

    times = []

    for n in sizes:
        coords = generate_clustered_cities(n)
        result = benchmark_solver(coords)
        times.append(result["time"])
        print(f"n={n:3d}: {result['time']:6.3f}s ({result['iterations']:4d} iterations)")

    # Simple analysis
    print("\nTime complexity appears to be polynomial")
    print("(3-opt has O(n³) worst-case complexity per iteration)")


if __name__ == "__main__":
    main()
