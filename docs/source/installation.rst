Installation
============

TSBee can be installed via pip from PyPI or built from source.

Install from PyPI
-----------------

The easiest way to install TSBee is using pip:

.. code-block:: bash

    pip install tsp-solve

This will install the pre-built wheel for your platform.

Install from Source
-------------------

To build and install from source, you'll need:

* Python 3.8+
* Rust toolchain (rustc, cargo)
* Maturin build tool

**Step 1: Install Rust**

.. code-block:: bash

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source ~/.cargo/env

**Step 2: Install build dependencies**

.. code-block:: bash

    pip install maturin

**Step 3: Clone and build**

.. code-block:: bash

    git clone https://github.com/username/tsp_solve.git
    cd tsp_solve
    maturin develop --release

Development Installation
------------------------

For development, you can install in editable mode:

.. code-block:: bash

    git clone https://github.com/username/tsp_solve.git
    cd tsp_solve

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install development dependencies
    pip install maturin pytest

    # Build and install in development mode
    maturin develop

Dependencies
------------

Runtime dependencies:
* Python 3.8+

Optional dependencies for examples and benchmarks:
* matplotlib (for plotting)
* pandas (for data handling)
* numpy (for numerical operations)

Install with optional dependencies:

.. code-block:: bash

    pip install tsp-solve[examples]

System Requirements
-------------------

**Minimum Requirements:**
* Python 3.8+
* 1GB RAM
* 100MB disk space

**Recommended:**
* Python 3.9+
* 4GB RAM for large problems (1000+ cities)
* Multi-core CPU for better performance

Verification
------------

Verify your installation by running:

.. code-block:: python

    import tsp_solve

    # Test with a simple problem
    distance_matrix = [
        [0, 10, 15],
        [10, 0, 20],
        [15, 20, 0]
    ]

    solution = tsp_solve.solve(distance_matrix)
    print(f"Installation successful! Tour: {solution.tour}")

Troubleshooting
---------------

**Common Issues:**

1. **Import Error**: Ensure you have the correct Python version and architecture
2. **Build Failures**: Make sure Rust toolchain is properly installed
3. **Performance Issues**: Use release builds for production (`maturin develop --release`)

**Getting Help:**

If you encounter issues, please:

1. Check the GitHub Issues page
2. Verify your Python and Rust versions
3. Try rebuilding with `maturin develop --release`
