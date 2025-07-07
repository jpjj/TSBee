"""
Basic usage example for tsbee.

This example demonstrates how to solve a simple TSP instance with a small
number of cities using a pre-defined distance matrix.
"""

import tsbee


def main():
    # Example 1: Small TSP instance with 5 cities
    print("Example 1: 5-city TSP")
    print("-" * 40)

    # Distance matrix (symmetric)
    # Cities could represent: New York, Boston, Philadelphia, Washington DC, Baltimore
    distance_matrix = [
        [0, 215, 95, 225, 185],  # From New York
        [215, 0, 305, 440, 400],  # From Boston
        [95, 305, 0, 140, 100],  # From Philadelphia
        [225, 440, 140, 0, 40],  # From Washington DC
        [185, 400, 100, 40, 0],  # From Baltimore
    ]

    # Solve the TSP
    solution = tsbee.solve(distance_matrix)

    # Print results
    print(f"Total distance: {solution.distance}")
    print(f"Tour: {' -> '.join(map(str, solution.tour))}")
    print(f"Iterations: {solution.iterations}")
    print(f"Time taken: {solution.time:.3f} seconds")

    # Verify the tour visits all cities exactly once
    assert len(solution.tour) == len(distance_matrix)
    assert set(solution.tour) == set(range(len(distance_matrix)))

    print("\n")

    # Example 2: Solve with time limit
    print("Example 2: Same problem with 1-second time limit")
    print("-" * 40)

    solution_timed = tsbee.solve(distance_matrix, time_limit=1.0)

    print(f"Total distance: {solution_timed.distance}")
    print(f"Tour: {' -> '.join(map(str, solution_timed.tour))}")
    print(f"Iterations: {solution_timed.iterations}")
    print(f"Time taken: {solution_timed.time:.3f} seconds")


if __name__ == "__main__":
    main()
