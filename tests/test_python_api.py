"""
Tests for the tsp_solve Python API.

These tests verify that the Python bindings work correctly and that
the solver produces valid solutions for various input types.
"""

import pytest
import tsp_solve


class TestBasicFunctionality:
    """Test basic solver functionality."""

    def test_import(self):
        """Test that the module can be imported."""
        assert hasattr(tsp_solve, "solve")
        assert hasattr(tsp_solve, "PySolution")

    def test_small_symmetric_matrix(self):
        """Test solving a small 4-city problem."""
        distance_matrix = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ]

        solution = tsp_solve.solve(distance_matrix)

        # Check solution attributes
        assert hasattr(solution, "distance")
        assert hasattr(solution, "tour")
        assert hasattr(solution, "iterations")
        assert hasattr(solution, "time")

        # Check solution validity
        assert len(solution.tour) == 4
        assert set(solution.tour) == {0, 1, 2, 3}
        assert solution.distance > 0
        assert solution.iterations > 0
        assert solution.time >= 0

    def test_with_time_limit(self):
        """Test solver with time limit."""
        n = 20
        distance_matrix = [[abs(i - j) * 10 for j in range(n)] for i in range(n)]

        solution = tsp_solve.solve(distance_matrix, time_limit=0.1)

        assert solution.time <= 0.2  # Allow some overhead
        assert len(solution.tour) == n
        assert set(solution.tour) == set(range(n))


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
            tsp_solve.solve(distance_matrix)

    def test_non_zero_diagonal(self):
        """Test that non-zero diagonal values are rejected."""
        distance_matrix = [[5, 10, 15], [10, 0, 35], [15, 35, 0]]  # Non-zero diagonal

        with pytest.raises(ValueError, match="zeros on diagonal"):
            tsp_solve.solve(distance_matrix)

    def test_empty_matrix(self):
        """Test that empty matrices are rejected."""
        with pytest.raises(ValueError):
            tsp_solve.solve([])

    def test_single_city(self):
        """Test single city case."""
        distance_matrix = [[0]]
        solution = tsp_solve.solve(distance_matrix)

        assert solution.tour == [0]
        assert solution.distance == 0


class TestSolutionQuality:
    """Test solution quality and correctness."""

    def test_optimal_triangle(self):
        """Test that optimal solution is found for a triangle."""
        # For 3 cities, any tour is optimal
        distance_matrix = [[0, 10, 20], [10, 0, 15], [20, 15, 0]]

        solution = tsp_solve.solve(distance_matrix)

        # Calculate expected distance (all tours have same length for 3 cities)
        expected_distance = 10 + 15 + 20
        assert solution.distance == expected_distance

    def test_solution_improvement(self):
        """Test that the solver improves upon a naive solution."""
        # Create a problem where nearest neighbor is suboptimal
        distance_matrix = [
            [0, 1, 10, 10],
            [1, 0, 10, 10],
            [10, 10, 0, 1],
            [10, 10, 1, 0],
        ]

        solution = tsp_solve.solve(distance_matrix)

        # Optimal tour should connect the close pairs: 0-1 and 2-3
        # Total distance: 1 + 10 + 1 + 10 = 22
        assert solution.distance == 22

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

        solution = tsp_solve.solve(distance_matrix, time_limit=2.0)

        assert len(solution.tour) == n
        assert set(solution.tour) == set(range(n))
        assert solution.distance > 0


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_all_equal_distances(self):
        """Test case where all distances are equal."""
        n = 5
        distance_matrix = [[0 if i == j else 100 for j in range(n)] for i in range(n)]

        solution = tsp_solve.solve(distance_matrix)

        # All tours have the same length
        expected_distance = 100 * n
        assert solution.distance == expected_distance

    def test_very_large_distances(self):
        """Test with very large distance values."""
        # Use large but valid integer distances
        distance_matrix = [
            [0, 1000000, 2000000],
            [1000000, 0, 1500000],
            [2000000, 1500000, 0],
        ]

        solution = tsp_solve.solve(distance_matrix)

        assert solution.distance == 4500000

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
        solutions = []
        for _ in range(3):
            solution = tsp_solve.solve(distance_matrix)
            solutions.append(solution.distance)

        # Should produce the same result each time
        assert len(set(solutions)) == 1


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

        tsp_solve.solve(distance_matrix, 1)

        # expected_distance = 39
        # assert solution.distance == expected_distance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
