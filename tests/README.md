# Tests

The test suite uses Python's `unittest` framework. From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s .
```

Tests are grouped by package:

- [`evolutionary/`](evolutionary/README.md): evolutionary search.
- [`ids_tabnet/`](ids_tabnet/README.md): TabNet data and evaluation.
- [`visualization/`](visualization/README.md): history plots.

The category READMEs describe individual tests and category-specific commands.
