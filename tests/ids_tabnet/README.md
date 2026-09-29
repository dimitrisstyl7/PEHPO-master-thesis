# IDS TabNet Tests

This directory contains standard-library `unittest` coverage for the IDS
TabNet parameter container and evaluator. The tests avoid production datasets,
real TabNet training, and GPU hardware by using an in-memory fixture and
mocks.

Run this category from the repository root with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/ids_tabnet
```

## Test modules

### `test_ids_tabnet_params.py`

Tests the public behavior of `ids_tabnet.params.ModelParams`:

| Test                                                         | Purpose                                                                                                                           |
|--------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `test_preserves_hyperparameters_and_accepts_gamma_endpoints` | Verifies supplied hyperparameters remain unchanged and inclusive gamma endpoints `1.0` and `2.0` are accepted.                    |
| `test_rejects_gamma_outside_inclusive_bounds`                | Verifies values below `1.0` and above `2.0` raise `ValueError` with the documented message.                                       |
| `test_derives_virtual_batch_size_at_ghost_batch_boundary`    | Verifies Ghost Batch Normalization sizing: batch sizes below `128` use floor-half, while batch sizes of at least `128` use `128`. |

### `test_ids_tabnet_evaluator.py`

Tests `ids_tabnet.evaluator.TabNetEvaluator` without real model training:

| Test                                                       | Purpose                                                                                                                                                 |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `test_fixture_supports_balanced_60_20_20_allocation`       | Verifies the shared fixture has balanced binary labels and produces expected 60/20/20 train, validation, and test sizes while preserving class balance. |
| `test_reproduces_fixed_seed_stratified_partitions`         | Verifies identical evaluator constructions produce identical fixed-seed stratified partitions and use the in-memory fixture instead of CSV loading.     |
| `test_evaluate_configures_cpu_model_and_scores_validation` | Mocks model training, CUDA detection, and `f1_score` to verify CPU configuration, fit arguments, validation inputs, and macro-F1 forwarding.            |
| `test_evaluate_test_scores_test_data`                      | Verifies `evaluate_test()` predicts and scores against held-out test data.                                                                              |
| `test_evaluate_selects_cuda_when_available`                | Verifies CUDA availability selects the `"cuda"` device without requiring GPU hardware.                                                                  |

## Shared fixture

`support.py` defines `make_binary_classification_frame()`. It creates numeric
features and a `Label` column with 20 samples from each of two classes by
default, providing enough observations for both stratified
`train_test_split` calls.

| Fixture or dependency                | Purpose                                                                                                          |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `make_binary_classification_frame()` | Creates numeric features and balanced labels for evaluator partition tests.                                      |
| `load_csv`                           | Supplies the in-memory DataFrame instead of reading the production CSV.                                          |
| `TabNetClassifier`                   | Records constructor, fit, and prediction arguments without real TabNet training.                                 |
| `torch.cuda.is_available`            | Controls CPU/CUDA branch coverage without GPU hardware.                                                          |
| `f1_score`                           | Returns deterministic values so evaluator forwarding is tested independently of sklearn's metric implementation. |
