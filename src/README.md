# Source code

The project runner in [`main.py`](../main.py) combines these packages:

- [`evolutionary/`](evolutionary/README.md): genetic algorithm, individuals,
  crossover, evaluation interface, and run history.
- [`ids_tabnet/`](ids_tabnet/README.md): TabNet configuration, data
  preparation, and model evaluation.
- [`helpers/`](helpers/README.md): shared file-loading utilities.
- [`visualization/`](visualization/README.md): figures from recorded history.

Run the project and tests from the repository root with `PYTHONPATH=src`; see
the [project README](../README.md) for commands.
