# Evolutionary optimization

This package implements the genetic search and its reusable components:

- `algorithm.py`: population initialization, selection, evaluation, and
  generational evolution.
- `individual.py`: search-space individuals and mutation.
- `self_adaptive_sbx.py`: self-adaptive simulated binary crossover.
- `evaluation.py`: batch evaluation interface and serial objective adapter.
- `history.py`: generation records and persistent run history.

See [`tests/evolutionary/README.md`](../../tests/evolutionary/README.md) for
test coverage and execution.
