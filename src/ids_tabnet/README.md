# IDS TabNet

This package prepares and evaluates TabNet models for intrusion detection:

- `config.py`: search-space and plot-scale configuration.
- `params.py`: model hyperparameter container.
- `evaluator.py`: dataset loading, stratified train/validation/test partitions,
  training, and scoring.
- `parallel.py`: Ray-based batch evaluation.

See [`tests/ids_tabnet/README.md`](../../tests/ids_tabnet/README.md) for test
coverage and execution.
