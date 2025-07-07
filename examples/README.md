# TSP Solve Examples

This directory contains example scripts demonstrating various uses of the `tsp_solve` library.

## Examples

### 1. Basic Usage (`basic_usage.py`)
Demonstrates the simplest way to use tsp_solve with a small distance matrix.
- Shows how to solve a 5-city TSP problem
- Demonstrates using time limits
- Includes result validation

```bash
python examples/basic_usage.py
```

### 2. Random Cities (`random_cities.py`)
Shows how to work with coordinate-based problems.
- Generates random city coordinates
- Converts coordinates to distance matrix
- Visualizes the solution with matplotlib

```bash
python examples/random_cities.py
```

### 3. Benchmark Comparison (`benchmark_comparison.py`)
Compares tsp_solve performance against simple heuristics.
- Tests different problem sizes
- Compares with nearest neighbor heuristic
- Analyzes performance scaling

```bash
python examples/benchmark_comparison.py
```

### 4. Real World Example (`real_world_example.py`)
Demonstrates a realistic delivery route optimization scenario.
- Uses actual US city coordinates
- Calculates real distances using Haversine formula
- Includes cost estimation for routes

```bash
python examples/real_world_example.py
```

## Running the Examples

First, ensure you have tsp_solve installed:

```bash
# From the project root
maturin develop
```

Install additional dependencies for visualization:

```bash
pip install matplotlib numpy
```

Then run any example:

```bash
cd examples
python basic_usage.py
```

## Key Concepts

### Distance Matrix Format
All examples use integer distance matrices as required by the solver:
- Distances must be symmetric: `distance[i][j] == distance[j][i]`
- Diagonal must be zero: `distance[i][i] == 0`
- All values must be non-negative integers

### Converting Real Distances
When working with real-world coordinates:
1. Calculate actual distances (Euclidean, Haversine, etc.)
2. Multiply by a scaling factor (e.g., 100 or 1000)
3. Convert to integers

### Performance Tips
- For large problems, use time limits to get good solutions quickly
- The solver performs best on problems with up to a few hundred cities
- For very large problems (1000+ cities), consider hierarchical approaches
