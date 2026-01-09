# Contributing

We welcome contributions to `semantic-probing`!

## Development Setup

1.  Clone the repository.
2.  Create a virtual environment: `python -m venv .venv`
3.  Activate it: `source .venv/bin/activate`
4.  Install in editable mode with dev dependencies: `pip install -e ".[dev]"`

## Running Tests

We use `pytest`.

```bash
python -m pytest tests/
```

## Code Style

-   Follow PEP 8.
-   Use type hints for all function signatures.
-   Write docstrings for all public classes and functions.

## Adding Primitives

New primitives should be added to `src/semantic_probing/encoding/semantic_dimensions.py`. Ensure they map to one of the 8 semantic dimensions.
