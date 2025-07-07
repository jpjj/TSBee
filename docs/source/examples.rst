Examples
========

This page contains various examples demonstrating TSBee's capabilities.

Basic Examples
--------------

Small TSP Instance
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tsbee

    # Simple 5-city problem
    distances = [
        [0, 215, 95, 225, 185],
        [215, 0, 305, 440, 400],
        [95, 305, 0, 140, 100],
        [225, 440, 140, 0, 40],
        [185, 400, 100, 40, 0]
    ]

    solution = tsbee.solve(distances)
    print(f"Tour: {' -> '.join(map(str, solution.tour))}")
    print(f"Distance: {solution.distance}")

Random City Generation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tsbee
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

    solution = tsbee.solve(distances, time_limit=2.0)
    print(f"20-city random problem solved in {solution.time:.3f}s")
    print(f"Best distance: {solution.distance}")
