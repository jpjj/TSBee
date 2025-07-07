.. TSBee documentation master file, created by
   sphinx-quickstart on Mon Jul  7 12:11:25 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TSBee: High-Performance TSP Solver
===================================

TSBee is a high-performance Traveling Salesman Problem (TSP) solver implemented in Rust with Python bindings. It provides efficient algorithms for solving both symmetric and asymmetric TSP instances.

Features
--------

* **Fast Rust Implementation**: Core algorithms implemented in Rust for optimal performance
* **Python API**: Easy-to-use Python interface for integration with data science workflows
* **Symmetric & Asymmetric TSP**: Supports both TSP and ATSP problem types
* **Configurable Time Limits**: Control solver runtime with customizable time constraints
* **Comprehensive Examples**: Includes examples and benchmarks for various problem sizes

Quick Start
-----------

.. code-block:: python

    import tsp_solve

    # Define your distance matrix
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    # Solve the TSP
    solution = tsp_solve.solve(distance_matrix)

    print(f"Best tour: {solution.tour}")
    print(f"Total distance: {solution.distance}")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   benchmarks
