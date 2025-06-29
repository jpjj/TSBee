![social_preview](https://github.com/user-attachments/assets/dfc3450b-09fd-4340-b238-c8812ec16a00)

# TSP Solve

High-performance Traveling Salesman Problem solver implemented in Rust with Python bindings.

## Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. The hooks include:
- **Python**: Linting and formatting with `ruff`
- **Rust**: Formatting with `cargo fmt` and linting with `cargo clippy`
- **Testing**: Automatic test execution for both Python and Rust
- **Benchmarking**: Quick benchmark compilation check

To set up pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# (Optional) Run against all files
pre-commit run --all-files
```

The hooks will automatically run on `git commit`. To skip hooks temporarily, use:
```bash
git commit --no-verify
```
