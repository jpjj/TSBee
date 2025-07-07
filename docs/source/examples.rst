Examples
========

This page contains various examples demonstrating TSBee's capabilities.

Basic Examples
--------------

Small TSP Instance
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tsp_solve

    # Simple 5-city problem
    distances = [
        [0, 215, 95, 225, 185],
        [215, 0, 305, 440, 400],
        [95, 305, 0, 140, 100],
        [225, 440, 140, 0, 40],
        [185, 400, 100, 40, 0]
    ]

    solution = tsp_solve.solve(distances)
    print(f"Tour: {' -> '.join(map(str, solution.tour))}")
    print(f"Distance: {solution.distance}")

Random City Generation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tsp_solve
    import random
    import math

    def generate_random_cities(n, max_coord=1000):
        """Generate n random cities with coordinates."""
        cities = []
        for i in range(n):
            x = random.randint(0, max_coord)
            y = random.randint(0, max_coord)
            cities.append((x, y))
        return cities

    def calculate_distance_matrix(cities):
        """Calculate Euclidean distance matrix."""
        n = len(cities)
        distances = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0)
                else:
                    x1, y1 = cities[i]
                    x2, y2 = cities[j]
                    dist = int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
                    row.append(dist)
            distances.append(row)
        return distances

    # Generate and solve random problem
    random.seed(42)  # For reproducibility
    cities = generate_random_cities(20)
    distances = calculate_distance_matrix(cities)

    solution = tsp_solve.solve(distances, time_limit=2.0)
    print(f"20-city random problem solved in {solution.time:.3f}s")
    print(f"Best distance: {solution.distance}")

Advanced Examples
-----------------

Comparing Multiple Runs
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tsp_solve
    import time

    # Large problem
    n = 100
    distances = [[abs(i-j)*10 if i != j else 0 for j in range(n)] for i in range(n)]

    # Run multiple times with different time limits
    time_limits = [0.1, 0.5, 1.0, 5.0]
    results = []

    for limit in time_limits:
        solution = tsp_solve.solve(distances, time_limit=limit)
        results.append({
            'time_limit': limit,
            'actual_time': solution.time,
            'distance': solution.distance,
            'iterations': solution.iterations
        })

    # Print comparison
    print("Time Limit | Actual Time | Distance | Iterations")
    print("-" * 50)
    for result in results:
        print(f"{result['time_limit']:9.1f} | {result['actual_time']:11.3f} | "
              f"{result['distance']:8d} | {result['iterations']:10d}")

Benchmarking Against Known Instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tsp_solve

    # Known optimal solutions for validation
    test_instances = {
        'br17': {
            'distances': [
                [0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5],
                [3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5],
                [5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24],
                # ... (truncated for brevity)
            ],
            'optimal': 39
        }
    }

    for name, instance in test_instances.items():
        solution = tsp_solve.solve(instance['distances'])
        gap = ((solution.distance - instance['optimal']) / instance['optimal']) * 100

        print(f"Instance: {name}")
        print(f"Optimal: {instance['optimal']}")
        print(f"Found: {solution.distance}")
        print(f"Gap: {gap:.2f}%")
        print(f"Time: {solution.time:.3f}s")
        print()

Integration Examples
--------------------

With Pandas DataFrames
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tsp_solve
    import pandas as pd
    import numpy as np

    # Create city data
    cities_df = pd.DataFrame({
        'city': ['A', 'B', 'C', 'D', 'E'],
        'x': [0, 10, 20, 15, 5],
        'y': [0, 15, 5, 25, 10]
    })

    # Calculate distance matrix
    def calculate_distances(df):
        n = len(df)
        distances = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt((df.iloc[i]['x'] - df.iloc[j]['x'])**2 +
                                  (df.iloc[i]['y'] - df.iloc[j]['y'])**2)
                    distances[i][j] = int(dist)
        return distances.tolist()

    distances = calculate_distances(cities_df)
    solution = tsp_solve.solve(distances)

    # Create solution DataFrame
    tour_cities = [cities_df.iloc[i]['city'] for i in solution.tour]
    solution_df = pd.DataFrame({
        'step': range(len(solution.tour)),
        'city': tour_cities,
        'city_id': solution.tour
    })

    print("Solution tour:")
    print(solution_df)

With Matplotlib Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tsp_solve
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate random cities
    np.random.seed(42)
    n_cities = 20
    cities = np.random.rand(n_cities, 2) * 100

    # Calculate distance matrix
    distances = []
    for i in range(n_cities):
        row = []
        for j in range(n_cities):
            if i == j:
                row.append(0)
            else:
                dist = np.linalg.norm(cities[i] - cities[j])
                row.append(int(dist))
        distances.append(row)

    # Solve TSP
    solution = tsp_solve.solve(distances)

    # Plot solution
    plt.figure(figsize=(10, 8))

    # Plot cities
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=5)

    # Plot tour
    tour_cities = cities[solution.tour]
    tour_cities = np.vstack([tour_cities, tour_cities[0]])  # Close the loop
    plt.plot(tour_cities[:, 0], tour_cities[:, 1], 'b-', linewidth=2, alpha=0.7)

    # Label cities
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

    plt.title(f'TSP Solution (Distance: {solution.distance})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

Performance Examples
--------------------

Scaling Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

    import tsp_solve
    import time
    import matplotlib.pyplot as plt

    # Test different problem sizes
    sizes = [10, 20, 50, 100, 200]
    times = []
    distances = []

    for n in sizes:
        print(f"Testing size {n}...")

        # Generate distance matrix
        dist_matrix = [[abs(i-j)*10 if i != j else 0 for j in range(n)] for i in range(n)]

        # Solve with time limit
        solution = tsp_solve.solve(dist_matrix, time_limit=10.0)

        times.append(solution.time)
        distances.append(solution.distance)

        print(f"  Time: {solution.time:.3f}s, Distance: {solution.distance}")

    # Plot scaling behavior
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, 'o-')
    plt.xlabel('Problem Size (cities)')
    plt.ylabel('Solve Time (seconds)')
    plt.title('Scaling: Time vs Problem Size')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(sizes, distances, 'o-')
    plt.xlabel('Problem Size (cities)')
    plt.ylabel('Tour Distance')
    plt.title('Solution Quality vs Problem Size')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

Memory Usage Monitoring
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tsp_solve
    import psutil
    import os

    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    # Monitor memory usage during solving
    sizes = [100, 500, 1000, 2000]

    for n in sizes:
        print(f"Testing size {n}...")

        # Measure memory before
        mem_before = get_memory_usage()

        # Create large distance matrix
        dist_matrix = [[abs(i-j)*10 if i != j else 0 for j in range(n)] for i in range(n)]

        # Measure memory after matrix creation
        mem_after_matrix = get_memory_usage()

        # Solve
        solution = tsp_solve.solve(dist_matrix, time_limit=5.0)

        # Measure memory after solving
        mem_after_solve = get_memory_usage()

        print(f"  Memory: {mem_before:.1f}MB -> {mem_after_matrix:.1f}MB -> {mem_after_solve:.1f}MB")
        print(f"  Matrix size: {mem_after_matrix - mem_before:.1f}MB")
        print(f"  Solver overhead: {mem_after_solve - mem_after_matrix:.1f}MB")
        print()

Running the Examples
--------------------

All examples above can be run by copying the code into a Python script or Jupyter notebook. Make sure you have TSBee installed:

.. code-block:: bash

    pip install tsp-solve

For the visualization examples, you'll also need:

.. code-block:: bash

    pip install matplotlib pandas numpy psutil

The examples demonstrate various aspects of TSBee:

* Basic usage patterns
* Input validation and error handling
* Performance characteristics
* Integration with common Python libraries
* Visualization and analysis techniques
