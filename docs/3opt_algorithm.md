# 3-opt Algorithm Documentation

## Overview

The 3-opt algorithm is a local search heuristic for the Traveling Salesman Problem (TSP) that considers removing three edges from a tour and reconnecting the three resulting segments in all possible ways to find improvements.

## Algorithm Description

### Basic Concept

Given a TSP tour, the 3-opt algorithm:
1. Selects three edges to remove from the tour
2. Considers all valid ways to reconnect the three segments
3. Chooses the reconnection that provides the best improvement
4. Repeats until no improvement can be found

### Implementation Details

This implementation includes several advanced optimizations:

#### 1. Sequential Search Strategy
Instead of immediately considering all 3-opt moves, the algorithm first checks if a simpler 2-opt move would suffice:
- For each edge (c1, c2), it considers candidates c3
- First attempts 2-opt: remove (c1,c2) and (c3,c4), add (c1,c4) and (c2,c3)
- Only if 2-opt fails, extends to 3-opt by considering a fifth city c5

#### 2. Don't Look Bits (DLB)
- Maintains a boolean flag for each city
- Cities involved in recent improvements have their flag set
- Cities with unset flags are skipped during search
- Dramatically reduces redundant searches

#### 3. Alpha-nearness Candidates
- Instead of considering all possible edges, uses a candidate list
- Candidates are generated using the alpha-nearness heuristic
- Based on minimum spanning tree calculations
- Typically reduces search space by 90%+

#### 4. First Improvement Strategy
- Returns immediately when any improving move is found
- Avoids exhaustive search for the best possible move
- Trades solution quality for speed

## Code Structure

### Main Components

1. **LocalSearch struct** (`src/local_move.rs`)
   - Contains the main 3-opt implementation
   - Manages Don't Look Bits
   - Handles tour modifications

2. **Solver struct** (`src/solver.rs`)
   - Orchestrates the overall search process
   - Implements iterated local search
   - Manages diversification strategies

3. **Solution struct** (`src/solution.rs`)
   - Represents a TSP tour
   - Handles tour modifications efficiently
   - Tracks solution quality

### Key Methods

#### `execute_3opt()`
```rust
pub(crate) fn execute_3opt(&mut self, dlb: bool) -> Solution
```
Main 3-opt implementation. Searches for improving moves and returns the improved solution.

#### `apply_two_opt()`
```rust
fn apply_two_opt(&mut self, c1_pos: usize, c4_pos: usize, gain: i64)
```
Applies a 2-opt move to the current solution, reversing a segment of the tour.

## Performance Characteristics

### Time Complexity
- Worst case: O(n³) per iteration
- Average case with optimizations: O(n²k) where k is the candidate list size
- Typically k << n, providing significant speedup

### Space Complexity
- O(n) for the tour representation
- O(nk) for the candidate lists
- O(n) for Don't Look Bits

### Practical Performance
- Scales well up to several thousand cities
- Solution quality typically within 2-3% of optimal
- Execution time grows polynomially with problem size

## Usage Example

```rust
use tsp_solve::Solver;

// Create solver with distance matrix
let mut solver = Solver::new(input);

// Solve with DLB optimization enabled
let solution = solver.solve(true);

println!("Best tour distance: {}", solution.best_solution.distance);
```

## Tuning Parameters

1. **Candidate List Size**: Typically 10-20 neighbors per city
2. **DLB Reset**: Cities are reset after being involved in improvements
3. **Termination**: Stops when no improving move found

## Comparison with Other Methods

### vs 2-opt
- **Pros**: Finds better solutions, escapes more local optima
- **Cons**: Slower per iteration (O(n³) vs O(n²))

### vs Lin-Kernighan
- **Pros**: Simpler implementation, more predictable performance
- **Cons**: Generally finds slightly worse solutions

### vs Genetic Algorithms
- **Pros**: Deterministic, faster convergence
- **Cons**: More likely to get stuck in local optima

## References

1. Lin, S. (1965). "Computer solutions of the traveling salesman problem"
2. Helsgaun, K. (2000). "An effective implementation of the Lin–Kernighan traveling salesman heuristic"
3. Johnson, D. S., & McGeoch, L. A. (1997). "The traveling salesman problem: A case study in local optimization"
