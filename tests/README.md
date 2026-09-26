# IDS TabNet Tests

This directory contains standard-library `unittest` coverage for the IDS TabNet
parameter container and evaluator. The tests avoid the production dataset, real
TabNet training, and GPU hardware by using an in-memory fixture and mocks.

Run the suite from the repository root with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

## Test modules

### `test_ids_tabnet_params.py`

Tests the public behavior of `ids_tabnet.params.ModelParams`:

| Test                                                         | Purpose                                                                                                                           |
|--------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `test_preserves_hyperparameters_and_accepts_gamma_endpoints` | Verifies that supplied hyperparameters are preserved and that inclusive gamma endpoints `1.0` and `2.0` are accepted.             |
| `test_rejects_gamma_outside_inclusive_bounds`                | Verifies that values below `1.0` and above `2.0` raise `ValueError` with the documented message.                                  |
| `test_derives_virtual_batch_size_at_ghost_batch_boundary`    | Verifies Ghost Batch Normalization sizing: batch sizes below `128` use floor-half, while batch sizes of at least `128` use `128`. |

### `test_ids_tabnet_evaluator.py`

Tests `ids_tabnet.evaluator.TabNetEvaluator` without performing real model
training:

| Test                                                       | Purpose                                                                                                                                                                                                          |
|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `test_fixture_supports_balanced_60_20_20_allocation`       | Verifies that the shared fixture contains balanced binary labels and produces the expected 60/20/20 train, validation, and test sizes while preserving class balance.                                            |
| `test_reproduces_fixed_seed_stratified_partitions`         | Constructs two evaluators from identical data and verifies that their fixed-seed stratified partitions are identical. It also confirms that CSV loading is replaced by the fixture.                              |
| `test_evaluate_configures_cpu_model_and_scores_validation` | Mocks `TabNetClassifier`, CUDA detection, and `f1_score`, then verifies CPU device selection, classifier hyperparameters, fit configuration, training inputs, validation inputs, and macro-F1 result forwarding. |
| `test_evaluate_test_scores_test_data`                      | Verifies that `evaluate_test()` predicts and scores against the held-out test data rather than the validation data.                                                                                              |
| `test_evaluate_selects_cuda_when_available`                | Mocks CUDA availability and verifies that evaluator model construction selects the `"cuda"` device without requiring CUDA hardware.                                                                              |

## Shared fixture

`support.py` defines `make_binary_classification_frame()`. It creates numeric
features and a `Label` column with 20 samples from each of two classes by
default. This provides enough observations for both stratified
`train_test_split` calls used by the evaluator.

| Fixture or dependency | Purpose |
| --- | --- |
| `make_binary_classification_frame()` | Creates numeric features and a balanced `Label` column with 20 samples from each of two classes by default. The fixture supports both stratified `train_test_split` calls. |
| `load_csv` | Supplies the in-memory DataFrame instead of reading the production CSV. |
| `TabNetClassifier` | Records constructor, fit, and prediction arguments without real TabNet training. |
| `torch.cuda.is_available` | Controls CPU/CUDA branch coverage without requiring GPU hardware. |
| `f1_score` | Returns deterministic values so evaluator forwarding can be asserted independently of sklearn's metric implementation. |
