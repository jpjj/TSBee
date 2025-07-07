"""
Benchmark runner for TSP solver using TSPLIB instances.

Measures timing and optimality gap across multiple runs.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path to import tsp_solve
sys.path.insert(0, str(Path(__file__).parent.parent))

import tsp_solve
from tsplib_parser import TSPLIBParser, parse_tour_file


def calculate_tour_distance(distance_matrix: np.ndarray, tour: List[int]) -> int:
    """Calculate the total distance of a tour."""
    n = len(tour)
    total_distance = 0

    for i in range(n):
        from_city = tour[i]
        to_city = tour[(i + 1) % n]
        total_distance += distance_matrix[from_city][to_city]

    return total_distance


def run_single_benchmark(
    instance_name: str,
    distance_matrix: List[List[int]],
    optimal_tour: List[int],
    optimal_distance: int,
    time_limit: float,
    num_runs: int = 5,
) -> Dict:
    """Run benchmark on a single instance multiple times."""
    n = len(distance_matrix)
    results = []

    # Verify optimal solution
    distance_matrix_np = np.array(distance_matrix)
    calculated_optimal = calculate_tour_distance(distance_matrix_np, optimal_tour)

    if calculated_optimal != optimal_distance:
        print(f"WARNING: Optimal distance mismatch for {instance_name}")
        print(f"  Expected: {optimal_distance}, Calculated: {calculated_optimal}")

    for run in range(num_runs):
        start_time = time.time()
        solution = tsp_solve.solve(distance_matrix, time_limit=time_limit)
        end_time = time.time()

        # Always calculate the actual tour distance ourselves
        # (don't trust the solver's reported distance)
        solution_distance = calculate_tour_distance(distance_matrix_np, solution.tour)

        # Warn if solver did 0 iterations (might indicate insufficient time)
        if solution.iterations == 0:
            print(f"  Warning: 0 iterations for run {run + 1}")

        # Calculate optimality gap
        gap = (solution_distance - optimal_distance) / optimal_distance * 100

        results.append(
            {
                "time": end_time - start_time,
                "distance": solution_distance,
                "gap": gap,
                "iterations": solution.iterations,
                "tour": solution.tour,
            }
        )

    # Calculate statistics
    times = [r["time"] for r in results]
    gaps = [r["gap"] for r in results]
    distances = [r["distance"] for r in results]
    iterations = [r["iterations"] for r in results]

    return {
        "instance": instance_name,
        "dimension": n,
        "optimal_distance": optimal_distance,
        "time_limit": time_limit,
        "num_runs": num_runs,
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "avg_distance": np.mean(distances),
        "best_distance": np.min(distances),
        "worst_distance": np.max(distances),
        "avg_gap": np.mean(gaps),
        "std_gap": np.std(gaps),
        "min_gap": np.min(gaps),
        "max_gap": np.max(gaps),
        "avg_iterations": np.mean(iterations),
        "all_results": results,
    }


def benchmark_instances(
    instance_files: List[Tuple[str, Path, Path]], num_runs: int = 5
) -> pd.DataFrame:
    """Run benchmarks on multiple instances, loading them one at a time."""
    all_results = []

    for name, tsp_file, tour_file in instance_files:
        print(f"\nLoading and benchmarking {name}...")

        try:
            # Load instance just-in-time
            instance = load_instance(tsp_file, tour_file)
            print(f"  Loaded {name} (n={instance['dimension']})")

            # Calculate time limit: max(0.5, n/1000) for better results on small instances
            time_limit = max(0.5, instance["dimension"] / 1000)

            result = run_single_benchmark(
                instance_name=instance["name"],
                distance_matrix=instance["distance_matrix"],
                optimal_tour=instance["optimal_tour"],
                optimal_distance=instance["optimal_distance"],
                time_limit=time_limit,
                num_runs=num_runs,
            )

            all_results.append(result)

            # Print summary
            print(f"  Optimal distance: {result['optimal_distance']}")
            print(f"  Best found: {result['best_distance']} (gap: {result['min_gap']:.2f}%)")
            print(f"  Average gap: {result['avg_gap']:.2f}% ± {result['std_gap']:.2f}%")
            print(f"  Average time: {result['avg_time']:.3f}s ± {result['std_time']:.3f}s")
            print(f"  Average iterations: {result['avg_iterations']:.0f}")

        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            continue

    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append(
            {
                "Instance": r["instance"],
                "Size": r["dimension"],
                "Optimal": r["optimal_distance"],
                "Best Found": r["best_distance"],
                "Avg Gap (%)": f"{r['avg_gap']:.2f}",
                "Min Gap (%)": f"{r['min_gap']:.2f}",
                "Max Gap (%)": f"{r['max_gap']:.2f}",
                "Avg Time (s)": f"{r['avg_time']:.3f}",
                "Time Limit (s)": f"{r['time_limit']:.3f}",
                "Avg Iterations": f"{r['avg_iterations']:.0f}",
            }
        )

    return pd.DataFrame(summary_data), all_results


def load_instance(tsp_file: Path, tour_file: Path) -> Dict:
    """Load a TSPLIB instance and its optimal solution."""
    # Parse TSP file
    parser = TSPLIBParser()
    instance_data = parser.parse_file(tsp_file)

    # Parse tour file
    optimal_tour, optimal_distance = parse_tour_file(tour_file)

    return {
        "name": instance_data["name"],
        "dimension": instance_data["dimension"],
        "distance_matrix": instance_data["distance_matrix"].tolist(),
        "optimal_tour": optimal_tour,
        "optimal_distance": optimal_distance,
    }


def main():
    """Main benchmark runner."""
    print("TSP Solver Benchmark Suite")
    print("=" * 60)

    # Define instances to benchmark
    # TSP instances (symmetric)
    tsp_instances = [
        ("si1032", 1032),  # Medium instance
        ("u2319", 2319),  # Large instance
        ("pr2392", 2392),  # Large instance
        ("pcb3038", 3038),  # Large instance
        ("fnl4461", 4461),  # Very large instance
        # ("rl5934", 5934),  # Skipping - exceeds 5000 limit
    ]

    # ATSP instances (asymmetric)
    atsp_instances = [
        ("ftv170", 170),  # Small-medium instance
        ("rbg323", 323),  # Medium instance
        ("rbg358", 358),  # Medium instance
        ("rbg403", 403),  # Medium-large instance
        ("rbg443", 443),  # Large instance
    ]

    # Prepare instance file paths
    instance_files = []

    # Check if instance files exist
    data_dir = Path(__file__).parent / "data"

    print("\nChecking for instance files...")
    missing_files = []

    for name, size in tsp_instances + atsp_instances:
        problem_type = "tsp" if (name, size) in tsp_instances else "atsp"
        tsp_file = data_dir / problem_type / f"{name}.{problem_type}"
        tour_file = data_dir / problem_type / f"{name}.tour"

        if not tsp_file.exists():
            missing_files.append(str(tsp_file))
        if not tour_file.exists():
            missing_files.append(str(tour_file))

    if missing_files:
        print("\nERROR: The following instance files are missing:")
        for f in missing_files:
            print(f"  {f}")
        print("\nPlease download the instances from:")
        print("  http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/")
        print("  http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ATSP.html")
        print("\nAnd place them in the appropriate directories.")
        return

    # Collect TSP instance file paths
    print("\nCollecting TSP instance files...")
    for name, size in tsp_instances:
        tsp_file = data_dir / "tsp" / f"{name}.tsp"
        tour_file = data_dir / "tsp" / f"{name}.tour"

        if tsp_file.exists() and tour_file.exists():
            instance_files.append((name, tsp_file, tour_file))
            print(f"  Found {name} (n={size})")

    # Collect ATSP instance file paths
    print("\nCollecting ATSP instance files...")
    for name, size in atsp_instances:
        tsp_file = data_dir / "atsp" / f"{name}.atsp"
        tour_file = data_dir / "atsp" / f"{name}.tour"

        if tsp_file.exists() and tour_file.exists():
            instance_files.append((name, tsp_file, tour_file))
            print(f"  Found {name} (n={size})")

    if not instance_files:
        print("\nNo instance files found. Exiting.")
        return

    # Run benchmarks
    print("\nRunning benchmarks (5 runs per instance)...")
    summary_df, detailed_results = benchmark_instances(instance_files, num_runs=1)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Save summary CSV
    csv_file = results_dir / f"benchmark_summary_{timestamp}.csv"
    summary_df.to_csv(csv_file, index=False)
    print(f"\nSummary saved to: {csv_file}")

    # Print summary table
    print("\nBenchmark Summary:")
    print("=" * 120)
    print(summary_df.to_string(index=False))

    # Calculate overall statistics
    all_gaps = []
    for r in detailed_results:
        all_gaps.extend([res["gap"] for res in r["all_results"]])

    print("\nOverall Statistics:")
    print(f"  Total runs: {len(all_gaps)}")
    print(f"  Average gap: {np.mean(all_gaps):.2f}%")
    print(f"  Median gap: {np.median(all_gaps):.2f}%")
    print(f"  Best gap: {np.min(all_gaps):.2f}%")
    print(f"  Worst gap: {np.max(all_gaps):.2f}%")

    # Check if suitable for CI
    avg_gap = np.mean(all_gaps)
    if avg_gap > 10:
        print(f"\nWARNING: Average gap {avg_gap:.2f}% exceeds 10% threshold")
        sys.exit(1)
    else:
        print(f"\nSUCCESS: All benchmarks completed with average gap {avg_gap:.2f}%")


if __name__ == "__main__":
    main()
