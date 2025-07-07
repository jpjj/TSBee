"""
Test script to verify TSPLIB parser functionality.
"""

from pathlib import Path

import numpy as np
from tsplib_parser import TSPLIBParser, parse_tour_file


def test_parse_instance(filepath: Path):
    """Test parsing a single instance."""
    print(f"\nTesting: {filepath.name}")
    print("-" * 40)

    try:
        parser = TSPLIBParser()
        data = parser.parse_file(filepath)

        print(f"Name: {data['name']}")
        print(f"Dimension: {data['dimension']}")
        print(f"Edge weight type: {data['edge_weight_type']}")
        print(f"Distance matrix shape: {data['distance_matrix'].shape}")

        # Check if matrix is symmetric or asymmetric
        matrix = data["distance_matrix"]
        is_symmetric = np.allclose(matrix, matrix.T)
        print(f"Symmetric: {is_symmetric}")

        # Sample some distances
        if data["dimension"] >= 3:
            print("Sample distances:")
            print(f"  d(0,1) = {matrix[0,1]}")
            print(f"  d(1,0) = {matrix[1,0]}")
            print(f"  d(0,2) = {matrix[0,2]}")
            print(f"  d(2,0) = {matrix[2,0]}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_parse_tour(filepath: Path):
    """Test parsing a tour file."""
    print(f"\nTesting tour: {filepath.name}")
    print("-" * 40)

    try:
        tour, distance = parse_tour_file(filepath)

        print(f"Tour length: {len(tour)}")
        print(f"Optimal distance: {distance}")
        print(f"First 10 cities: {tour[:10]}")
        print(f"Last 10 cities: {tour[-10:]}")

        # Check if tour is valid (visits each city exactly once)
        unique_cities = set(tour)
        print(f"Valid tour: {len(unique_cities) == len(tour)}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Run parser tests."""
    data_dir = Path(__file__).parent / "data"

    print("TSPLIB Parser Test")
    print("=" * 60)

    # Test TSP instances
    tsp_dir = data_dir / "tsp"
    if tsp_dir.exists():
        tsp_files = sorted(tsp_dir.glob("*.tsp"))
        print(f"\nFound {len(tsp_files)} TSP files")

        for tsp_file in tsp_files[:2]:  # Test first 2
            test_parse_instance(tsp_file)

            # Test corresponding tour file
            tour_file = tsp_file.with_suffix(".tour")
            if tour_file.exists():
                test_parse_tour(tour_file)

    # Test ATSP instances
    atsp_dir = data_dir / "atsp"
    if atsp_dir.exists():
        atsp_files = sorted(atsp_dir.glob("*.atsp"))
        print(f"\n\nFound {len(atsp_files)} ATSP files")

        for atsp_file in atsp_files[:2]:  # Test first 2
            test_parse_instance(atsp_file)

            # Test corresponding tour file
            tour_file = atsp_file.with_suffix(".tour")
            if tour_file.exists():
                test_parse_tour(tour_file)


if __name__ == "__main__":
    main()
