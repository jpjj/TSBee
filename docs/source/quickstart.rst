Quick Start Guide
=================

This guide will help you get started with TSBee quickly.

Basic Usage
-----------

Here's a simple example to solve a 4-city TSP:

.. code-block:: python

    import tsbee

    # Define distance matrix (4 cities) - floating-point supported
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    # Solve the problem
    tour = tsbee.solve(distances)

    # Print results
    print(f"Best tour: {tour}")

    # Calculate total distance if needed
    def calculate_distance(distance_matrix, tour):
        """Calculate the total distance of a tour."""
        n =len(tour)
        return sum(distance_matrix[tour[i]][tour[(i+1)%n]] for i in range(n))

    total_distance = calculate_distance(distances, tour)
    print(f"Total distance: {total_distance}")

Output:
::

    Best tour: [0, 1, 3, 2]
    Total distance: 80

With Time Limits
----------------

For large problems, you can set a time limit:

.. code-block:: python

    import tsbee
    import time
    import random

    # Large random problem
    n = 50
    distances = [
        [random.randint(1, 1000) if i != j else 0 for j in range(n)] for i in range(n)
    ]

    # Solve with 5-second time limit
    start_time = time.time()
    tour = tsbee.solve(distances, time_limit=5.0)
    elapsed = time.time() - start_time

    print(f"Found tour: {tour}")
    print(f"Time used: {elapsed:.3f} seconds")


Best Practices
--------------

1. **Use integers for distances**: The solver works with integer distances for optimal performance.

2. **Set reasonable time limits**: For problems with 100+ cities, consider setting a time limit.

3. **Validate your input**: Ensure your distance matrix is square with zero diagonal.

4. **Consider problem size**: The solver can handle problems up to several thousand cities, but performance depends on problem structure.
