Quick Start Guide
=================

This guide will help you get started with TSBee quickly.

Basic Usage
-----------

Here's a simple example to solve a 4-city TSP:

.. code-block:: python

    import tsp_solve

    # Define distance matrix (4 cities)
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    # Solve the problem
    solution = tsp_solve.solve(distances)

    # Print results
    print(f"Best tour: {solution.tour}")
    print(f"Total distance: {solution.distance}")
    print(f"Solver iterations: {solution.iterations}")
    print(f"Solve time: {solution.time:.3f} seconds")

Output:
::

    Best tour: [0, 1, 3, 2]
    Total distance: 80
    Solver iterations: 42
    Solve time: 0.001 seconds

With Time Limits
----------------

For large problems, you can set a time limit:

.. code-block:: python

    import tsp_solve

    # Large random problem
    n = 50
    distances = [[abs(i-j)*10 if i != j else 0 for j in range(n)] for i in range(n)]

    # Solve with 5-second time limit
    solution = tsp_solve.solve(distances, time_limit=5.0)

    print(f"Found tour with distance: {solution.distance}")
    print(f"Time used: {solution.time:.3f} seconds")

Asymmetric TSP (ATSP)
--------------------

TSBee also handles asymmetric problems where distance from A to B differs from B to A:

.. code-block:: python

    import tsp_solve

    # Asymmetric distance matrix
    distances = [
        [0, 10, 15, 20],
        [12, 0, 35, 25],  # Different from [1][0]
        [18, 30, 0, 30],  # Different from [2][0] and [2][1]
        [22, 28, 35, 0]   # Different from [3][0], [3][1], [3][2]
    ]

    solution = tsp_solve.solve(distances)
    print(f"ATSP solution: {solution.tour}")

Real-World Example
------------------

Here's a more realistic example with city coordinates:

.. code-block:: python

    import tsp_solve
    import math

    # City coordinates (longitude, latitude)
    cities = [
        ("New York", 40.7128, -74.0060),
        ("Los Angeles", 34.0522, -118.2437),
        ("Chicago", 41.8781, -87.6298),
        ("Houston", 29.7604, -95.3698),
        ("Phoenix", 33.4484, -112.0740)
    ]

    # Calculate distance matrix using Euclidean distance
    def euclidean_distance(city1, city2):
        lat1, lon1 = city1[1], city1[2]
        lat2, lon2 = city2[1], city2[2]
        return int(math.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 1000)

    n = len(cities)
    distances = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                row.append(euclidean_distance(cities[i], cities[j]))
        distances.append(row)

    # Solve the TSP
    solution = tsp_solve.solve(distances)

    # Print the tour with city names
    print("Optimal tour:")
    for i, city_idx in enumerate(solution.tour):
        city_name = cities[city_idx][0]
        print(f"{i+1}. {city_name}")

    print(f"\nTotal distance: {solution.distance}")

Input Validation
---------------

TSBee validates your input and provides helpful error messages:

.. code-block:: python

    import tsp_solve

    # Invalid: non-square matrix
    try:
        invalid_matrix = [
            [0, 10, 15],
            [10, 0]  # Missing element
        ]
        tsp_solve.solve(invalid_matrix)
    except ValueError as e:
        print(f"Error: {e}")

    # Invalid: non-zero diagonal
    try:
        invalid_matrix = [
            [5, 10, 15],  # Should be 0
            [10, 0, 20],
            [15, 20, 0]
        ]
        tsp_solve.solve(invalid_matrix)
    except ValueError as e:
        print(f"Error: {e}")

Best Practices
--------------

1. **Use integers for distances**: The solver works with integer distances for optimal performance.

2. **Set reasonable time limits**: For problems with 100+ cities, consider setting a time limit.

3. **Validate your input**: Ensure your distance matrix is square with zero diagonal.

4. **Consider problem size**: The solver can handle problems up to several thousand cities, but performance depends on problem structure.

5. **Use release builds**: For production use, ensure you're using a release build for best performance.

Next Steps
----------

* Read the full :doc:`api` documentation
* Check out more :doc:`examples`
* See :doc:`benchmarks` for performance comparisons
* Learn about the algorithms in our technical documentation
