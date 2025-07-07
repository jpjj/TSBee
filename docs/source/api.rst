API Reference
=============

This page provides detailed documentation for the TSBee Python API.

Core Functions
--------------

.. autofunction:: tsp_solve.solve

Classes
-------

.. autoclass:: tsp_solve.PySolution
   :members:
   :undoc-members:
   :show-inheritance:

Function Reference
------------------

solve
~~~~~

.. code-block:: python

    solve(distance_matrix, time_limit=None)

Solves a Traveling Salesman Problem instance.

**Parameters:**

* **distance_matrix** (*List[List[int]]*) - A square matrix where distance_matrix[i][j] represents the distance from city i to city j. The diagonal must contain zeros.
* **time_limit** (*Optional[float]*) - Maximum time in seconds to spend solving. If None, the solver will run until convergence.

**Returns:**

* **PySolution** - A solution object containing the best tour found, its total distance, and solver statistics.

**Raises:**

* **ValueError** - If the distance matrix is not square, contains non-zero diagonal elements, or is empty.

**Example:**

.. code-block:: python

    import tsp_solve

    # 4-city problem
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    solution = tsp_solve.solve(distances, time_limit=1.0)
    print(f"Tour: {solution.tour}")
    print(f"Distance: {solution.distance}")

Solution Object
---------------

PySolution
~~~~~~~~~~

The solution object returned by the solve function contains:

**Attributes:**

* **tour** (*List[int]*) - The sequence of cities in the optimal tour found
* **distance** (*int*) - The total distance of the tour
* **iterations** (*int*) - Number of iterations performed by the solver
* **time** (*float*) - Actual time spent solving in seconds

**Example:**

.. code-block:: python

    solution = tsp_solve.solve(distance_matrix)

    # Access solution properties
    cities_order = solution.tour
    total_cost = solution.distance
    solver_iterations = solution.iterations
    solve_time = solution.time

Input Requirements
------------------

Distance Matrix Format
~~~~~~~~~~~~~~~~~~~~~~

The distance matrix must satisfy the following requirements:

1. **Square Matrix**: Number of rows equals number of columns
2. **Zero Diagonal**: All diagonal elements must be zero (distance from city to itself)
3. **Non-negative Values**: All distances must be non-negative integers
4. **Symmetric/Asymmetric**: The matrix can be either symmetric (TSP) or asymmetric (ATSP)

**Valid Example:**

.. code-block:: python

    # Symmetric TSP
    symmetric_matrix = [
        [0, 10, 15],
        [10, 0, 20],
        [15, 20, 0]
    ]

    # Asymmetric TSP
    asymmetric_matrix = [
        [0, 10, 15],
        [12, 0, 20],
        [18, 25, 0]
    ]

**Invalid Examples:**

.. code-block:: python

    # Non-square matrix
    invalid_matrix = [
        [0, 10],
        [10, 0, 20]  # Different number of columns
    ]

    # Non-zero diagonal
    invalid_matrix = [
        [5, 10, 15],  # Diagonal should be 0
        [10, 0, 20],
        [15, 20, 0]
    ]

Error Handling
--------------

The solver performs input validation and raises appropriate exceptions:

* **ValueError** - For invalid distance matrix format
* **OverflowError** - For distances too large to process
* **MemoryError** - For problems too large to fit in memory

**Example Error Handling:**

.. code-block:: python

    try:
        solution = tsp_solve.solve(distance_matrix)
    except ValueError as e:
        print(f"Invalid input: {e}")
    except MemoryError:
        print("Problem too large for available memory")
