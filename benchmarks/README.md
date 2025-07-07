# TSP Solver Benchmarks

This directory contains benchmarking tools for the TSP solver using standard TSPLIB instances.

## Structure

```
benchmarks/
├── data/
│   ├── tsp/         # Symmetric TSP instances
│   └── atsp/        # Asymmetric TSP instances
├── results/         # Benchmark results (CSV and JSON)
├── tsplib_parser.py # Parser for TSPLIB format files
├── run_benchmarks.py # Main benchmark runner
├── run_ci_benchmarks.py # CI-friendly wrapper
└── test_parser.py   # Parser verification script
```

## Benchmark Instances

### TSP (Symmetric)
- **si1032** (1032 cities) - Medium instance
- **u2319** (2319 cities) - Large instance
- **pr2392** (2392 cities) - Large instance
- **pcb3038** (3038 cities) - Large instance
- **fnl4461** (4461 cities) - Very large instance

### ATSP (Asymmetric)
- **ftv170** (171 cities) - Small-medium instance
- **rbg323** (323 cities) - Medium instance
- **rbg358** (358 cities) - Medium instance
- **rbg403** (403 cities) - Medium-large instance
- **rbg443** (443 cities) - Large instance

## Running Benchmarks

### Basic Usage

```bash
cd benchmarks
python run_benchmarks.py
```

This will:
1. Load all TSP and ATSP instances
2. Run each instance 5 times
3. Calculate timing and optimality gaps
4. Save results to `results/` directory
5. Display summary statistics

### CI Usage

```bash
python run_ci_benchmarks.py
```

Exit codes:
- 0: Success (average gap < 10%)
- 1: Failure (average gap ≥ 10%)
- 2: Missing instance files

### Test Parser

To verify the TSPLIB parser works correctly:

```bash
python test_parser.py
```

## Benchmark Parameters

- **Time Limit**: `max(0.5, n/1000)` seconds per instance
- **Runs**: 5 runs per instance
- **Metrics**:
  - Solution quality (optimality gap %)
  - Runtime (seconds)

## Current Performance

Recent benchmark results on medium-sized instances:

| Instance | Size | Type | Optimal | Best Found | Avg Gap | Avg Time |
|----------|------|------|---------|------------|---------|----------|
| si1032   | 1032 | TSP  | 92,650  | 93,875     | 1.32%   | 1.09s    |
| u2319    | 2319 | TSP  | 234,256 | 236,201    | 0.85%   | 2.67s    |
| ftv170   | 171  | ATSP | 2,755   | 2,646      | -3.96%  | 0.50s    |
| rbg323   | 323  | ATSP | 1,326   | 1,796      | 35.49%  | 0.51s    |

**Overall**: 8.43% average gap across all runs (well within 10% CI threshold)

## Adding New Instances

1. Download `.tsp`/`.atsp` and `.tour` files from TSPLIB
2. Place in appropriate `data/` subdirectory
3. Update instance lists in `run_benchmarks.py`

## Results Format

### CSV Summary
- Instance name, size, optimal distance
- Best/average/worst gaps
- Average runtime

### JSON Details
- Complete results for each run
- Individual tours and distances
- Full timing information
