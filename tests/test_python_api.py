"""
Tests for the tsbee Python API.

These tests verify that the Python bindings work correctly and that
the solver produces valid solutions for various input types.
"""

from typing import List

import pytest
import tsbee


def calculate_tour_distance(distance_matrix: List[List[int]], tour: List[int]) -> int:
    """Calculate the total distance of a tour."""
    n = len(tour)
    total_distance = 0

    for i in range(n):
        from_city = tour[i]
        to_city = tour[(i + 1) % n]
        total_distance += distance_matrix[from_city][to_city]

    return total_distance


class TestBasicFunctionality:
    """Test basic solver functionality."""

    def test_import(self):
        """Test that the module can be imported."""
        assert hasattr(tsbee, "solve")

    def test_small_symmetric_matrix(self):
        """Test solving a small 4-city problem."""
        distance_matrix = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ]

        tour = tsbee.solve(distance_matrix)

        # Check solution validity
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

    def test_with_time_limit(self):
        """Test solver with time limit."""
        n = 20
        distance_matrix = [[abs(i - j) * 10 for j in range(n)] for i in range(n)]

        tour = tsbee.solve(distance_matrix, time_limit=0.1)

        assert len(tour) == n
        assert set(tour) == set(range(n))


class TestInputValidation:
    """Test input validation and error handling."""

    def test_non_square_matrix(self):
        """Test that non-square matrices are rejected."""
        distance_matrix = [
            [0, 10, 15],
            [10, 0, 35, 25],  # Too many columns
            [15, 35, 0],
        ]

        with pytest.raises(ValueError, match="square"):
            tsbee.solve(distance_matrix)

    def test_non_zero_diagonal(self):
        """Test that non-zero diagonal values are rejected."""
        distance_matrix = [[5, 10, 15], [10, 0, 35], [15, 35, 0]]  # Non-zero diagonal

        with pytest.raises(ValueError, match="zeros on diagonal"):
            tsbee.solve(distance_matrix)

    def test_empty_matrix(self):
        """Test that empty matrices are rejected."""
        with pytest.raises(ValueError):
            tsbee.solve([])

    def test_single_city(self):
        """Test single city case."""
        distance_matrix = [[0]]
        tour = tsbee.solve(distance_matrix)

        assert tour == [0]


class TestSolutionQuality:
    """Test solution quality and correctness."""

    def test_optimal_triangle(self):
        """Test that optimal solution is found for a triangle."""
        # For 3 cities, any tour is optimal
        distance_matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]

        tour = tsbee.solve(distance_matrix)

        # Check that we get a valid tour
        assert len(tour) == 3
        assert set(tour) == {0, 1, 2}

        # Calculate expected distance (all tours have same length for 3 cities)
        expected_distance = 10 + 15 + 20
        actual_distance = calculate_tour_distance(distance_matrix, tour)
        assert actual_distance == expected_distance

    def test_solution_improvement(self):
        """Test that the solver improves upon a naive solution."""
        # Create a problem where nearest neighbor is suboptimal
        distance_matrix = [
            [0, 1, 10, 10],
            [1, 0, 10, 10],
            [10, 10, 0, 1],
            [10, 10, 1, 0],
        ]

        tour = tsbee.solve(distance_matrix)

        # Check that we get a valid tour
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

        # Optimal tour should connect the close pairs: 0-1 and 2-3
        # Total distance: 1 + 10 + 1 + 10 = 22
        actual_distance = calculate_tour_distance(distance_matrix, tour)
        assert actual_distance == 22

    def test_larger_problem(self):
        """Test a larger problem to ensure scalability."""
        n = 50
        # Create a circular layout where optimal tour visits cities in order
        import math

        distance_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    # Distance based on circular positions
                    angle_i = 2 * math.pi * i / n
                    angle_j = 2 * math.pi * j / n
                    x_i, y_i = math.cos(angle_i), math.sin(angle_i)
                    x_j, y_j = math.cos(angle_j), math.sin(angle_j)
                    dist = math.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                    row.append(int(dist * 1000))
            distance_matrix.append(row)

        tour = tsbee.solve(distance_matrix, time_limit=2.0)

        assert len(tour) == n
        assert set(tour) == set(range(n))


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_all_equal_distances(self):
        """Test case where all distances are equal."""
        n = 5
        distance_matrix = [[0 if i == j else 100 for j in range(n)] for i in range(n)]

        tour = tsbee.solve(distance_matrix)

        # Check that we get a valid tour
        assert len(tour) == n
        assert set(tour) == set(range(n))

        # All tours have the same length
        expected_distance = 100 * n
        actual_distance = calculate_tour_distance(distance_matrix, tour)
        assert actual_distance == expected_distance

    def test_very_large_distances(self):
        """Test with very large distance values."""
        # Use large but valid integer distances
        distance_matrix = [
            [0, 1000000, 2000000],
            [1000000, 0, 1500000],
            [2000000, 1500000, 0],
        ]

        tour = tsbee.solve(distance_matrix)

        # Check that we get a valid tour
        assert len(tour) == 3
        assert set(tour) == {0, 1, 2}

        # Check that the distance is correct
        actual_distance = calculate_tour_distance(distance_matrix, tour)
        assert actual_distance == 4500000

    def test_deterministic_behavior(self):
        """Test that solver produces consistent results."""
        distance_matrix = [
            [0, 10, 15, 20, 25],
            [10, 0, 35, 25, 30],
            [15, 35, 0, 30, 20],
            [20, 25, 30, 0, 15],
            [25, 30, 20, 15, 0],
        ]

        # Run multiple times
        tours = []
        for _ in range(3):
            tour = tsbee.solve(distance_matrix)
            tours.append(tuple(tour))

        # Should produce the same result each time
        assert len(set(tours)) == 1


class TestAsymmetry:
    """Test cases for the atsp."""

    """
    NAME:  br17
    TYPE: ATSP
    COMMENT: 17 city problem (Repetto)
    DIMENSION:  17
    EDGE_WEIGHT_TYPE: EXPLICIT
    EDGE_WEIGHT_FORMAT: FULL_MATRIX
    EDGE_WEIGHT_SECTION
    [
    [9999,    3,    5,   48,   48,    8,    8,    5,    5,    3,    3,    0,    3,    5,    8,    8, 5],
    [   3, 9999,    3,   48,   48,    8,    8,    5,    5,    0,    0,    3,    0,    3,    8,    8, 5],
    [   5,    3, 9999,   72,   72,   48,   48,   24,   24,    3,    3,    5,    3,    0,   48,   48, 24],
    [  48,   48,   74, 9999,    0,    6,    6,   12,   12,   48,   48,   48,   48,   74,    6,    6, 12],
    [  48,   48,   74,    0, 9999,    6,    6,   12,   12,   48,   48,   48,   48,   74,    6,    6, 12],
    [   8,    8,   50,    6,    6, 9999,    0,    8,    8,    8,    8,    8,    8,   50,    0,    0, 8],
    [   8,    8,   50,    6,    6,    0, 9999,    8,    8,    8,    8,    8,    8,   50,    0,    0, 8],
    [   5,    5,   26,   12,   12,    8,    8, 9999,    0,    5,    5,    5,    5,   26,    8,    8, 0],
    [   5,    5,   26,   12,   12,    8,    8,    0, 9999,    5,    5,    5,    5,   26,    8,    8, 0],
    [   3,    0,    3,   48,   48,    8,    8,    5,    5, 9999,    0,    3,    0,    3,    8,    8, 5],
    [   3,    0,    3,   48,   48,    8,    8,    5,    5,    0, 9999,    3,    0,    3,    8,    8, 5],
    [   0,    3,    5,   48,   48,    8,    8,    5,    5,    3,    3, 9999,    3,    5,    8,    8, 5],
    [   3,    0,    3,   48,   48,    8,    8,    5,    5,    0,    0,    3, 9999,    3,    8,    8, 5],
    [   5,    3,    0,   72,   72,   48,   48,   24,   24,    3,    3,    5,    3, 9999,   48,   48, 24],
    [   8,    8,   50,    6,    6,    0,    0,    8,    8,    8,    8,    8,    8,   50, 9999,    0, 8],
    [   8,    8,   50,    6,    6,    0,    0,    8,    8,    8,    8,    8,    8,   50,    0, 9999, 8],
    [   5,    5,   26,   12,   12,    8,    8,    0,    0,    5,    5,    5,    5,   26,    8,    8, 9999],
    ]
    EOF

    """

    # http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/atsp/
    def test_br17(self):
        """Test case where all distances are equal."""
        distance_matrix = [
            [0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5],
            [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
            [5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24],
            [48, 48, 74, 0, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
            [48, 48, 74, 0, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12],
            [8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
            [8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
            [5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 0],
            [5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 0],
            [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
            [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
            [0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5],
            [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
            [5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24],
            [8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
            [8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8],
            [5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 0],
        ]

        tour = tsbee.solve(distance_matrix, 1)

        # Check that we get a valid tour
        assert len(tour) == 17
        assert set(tour) == set(range(17))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
