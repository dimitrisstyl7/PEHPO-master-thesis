# IDS TabNet

[⬅ Back to main README](../../README.md#readme-files)

This package prepares and evaluates TabNet models for intrusion detection:

- `config.py`: search-space and plot-scale configuration for five tuned parameters.
- `params.py`: container for the tuned model parameters.
- `evaluator.py`: dataset loading, stratified train/validation/test partitions,
  training with TabNet defaults, and scoring.
- `parallel.py`: Ray-based batch evaluation.

See [`tests/ids_tabnet/README.md`](../../tests/ids_tabnet/README.md) for test
coverage and execution.
