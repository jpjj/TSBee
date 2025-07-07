# TSBee Documentation

This directory contains the Sphinx documentation for TSBee.

## Building Documentation

### Prerequisites

Make sure you have the documentation dependencies installed:

```bash
# From the project root
source .venv/bin/activate
uv pip install --group docs
```

### Building HTML Documentation

From the `docs/` directory:

```bash
# Build the TSBee package first (required for API docs)
make build-package

# Build the documentation
make html

# Or do both in one command
make full-build
```

The built documentation will be in `build/html/`.

### Viewing Documentation

To serve the documentation locally:

```bash
make serve
```

This will start a local server at `http://localhost:8000`.

### Live Reload During Development

For development with automatic rebuilding:

```bash
make dev-install  # Install sphinx-autobuild
make livehtml     # Start live reload server
```

## Documentation Structure

- `source/index.rst` - Main documentation page
- `source/installation.rst` - Installation instructions
- `source/quickstart.rst` - Quick start guide
- `source/api.rst` - API reference
- `source/examples.rst` - Usage examples
- `source/benchmarks.rst` - Performance benchmarks
- `source/conf.py` - Sphinx configuration

## GitHub Actions

Documentation is automatically built and deployed to GitHub Pages via the `.github/workflows/docs.yml` workflow:

- **On Pull Requests**: Documentation is built and artifacts are uploaded for review
- **On Main Branch**: Documentation is built and automatically deployed to GitHub Pages

## Configuration

The documentation uses:

- **Theme**: Read the Docs theme (`sphinx_rtd_theme`)
- **Extensions**:
  - `sphinx.ext.autodoc` - Auto-generate API docs
  - `sphinx.ext.viewcode` - Add source code links
  - `sphinx.ext.napoleon` - Support for Google/NumPy docstrings
  - `sphinx_autodoc_typehints` - Better type hint support

## Contributing

To contribute to the documentation:

1. Edit the `.rst` files in `source/`
2. Build locally to test: `make full-build`
3. View the results: `make serve`
4. Submit a pull request

The documentation will be automatically built and deployed when merged to main.
