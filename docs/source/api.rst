API Reference
=============

This page provides detailed documentation for the TSBee Python API.

Core Functions
--------------

.. autofunction:: tsbee.solve

Input Requirements
------------------

Distance Matrix Format
~~~~~~~~~~~~~~~~~~~~~~

The distance matrix must satisfy the following requirements:

1. **Square Matrix**: Number of rows equals number of columns
2. **Zero Diagonal**: All diagonal elements must be zero (distance from city to itself)

**Valid Examples:**

.. code-block:: python

    # Integer distances
    integer_matrix = [
        [0, 10, 15],
        [10, 0, 20],
        [15, 20, 0]
    ]

    # Floating-point distances
    float_matrix = [
        [0.0, 10.5, 15.3],
        [10.5, 0.0, 20.7],
        [15.3, 20.7, 0.0]
    ]

    # Asymmetric matrix
    invalid_matrix = [
        [0, 1, 15],
        [10, 0, 2],
        [5, 25, 0]
    ]

    # negative entries
    integer_matrix = [
        [0, 10, -15],
        [10, 0, 20],
        [-15, 20, 0]
    ]

**Invalid Examples:**

.. code-block:: python

    # Non-square matrix
    invalid_matrix = [
        [0.0, 10.5],
        [10.5, 0.0, 20.7]  # Different number of columns
    ]

    # Non-zero diagonal
    invalid_matrix = [
        [5.0, 10.5, 15.3],  # Diagonal should be 0
        [10.5, 0.0, 20.7],
        [15.3, 20.7, 0.0]
    ]

Error Handling
--------------

The solver performs input validation and raises appropriate exceptions:

* **ValueError** - For invalid distance matrix format or invalid parameters
* **MemoryError** - For problems too large to fit in memory
