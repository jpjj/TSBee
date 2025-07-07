.. TSBee documentation master file, created by
   sphinx-quickstart on Mon Jul  7 12:11:25 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TSBee: High-Performance TSP Solver
===================================

TSBee is a high-performance Traveling Salesman Problem (TSP) solver implemented in Rust with Python bindings. It provides efficient algorithms for solving both symmetric and asymmetric TSP instances.

Features
--------

* **Fast Rust Implementation**: Advanced 3-opt local search algorithms implemented in Rust for optimal performance
* **Python API**: Simple, clean Python interface for integration with data science workflows
* **Configurable Time Limits**: Control solver runtime with customizable time constraints

Quick Start
-----------

.. code-block:: python

    import tsbee

    # Define your distance matrix (floating-point supported)
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    # Solve the TSP
    tour = tsbee.solve(distance_matrix)

    print(f"Returned tour: {tour}")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
