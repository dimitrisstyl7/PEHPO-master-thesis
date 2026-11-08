# IDS TabNet Tests

This directory contains standard-library `unittest` coverage for the IDS TabNet parameter container, search-space
configuration, reusable data preparation, and evaluator. Most tests avoid production datasets, real TabNet training, and
GPU hardware by using an in-memory fixture and mocks. One integration test intentionally runs tiny real CPU TabNet fits
through local Ray actors.

Run this category from the repository root with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/ids_tabnet
```

## Test modules

### `test_ids_tabnet_params.py`

Tests the public behavior of `ids_tabnet.params.ModelParams`:

| Test                                                         | Purpose                                                                                                        |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `test_preserves_hyperparameters_and_accepts_gamma_endpoints` | Verifies supplied hyperparameters remain unchanged and inclusive gamma endpoints `1.0` and `2.0` are accepted. |
| `test_rejects_gamma_outside_inclusive_bounds`                | Verifies values below `1.0` and above `2.0` raise `ValueError` with the documented message.                    |

### `test_ids_tabnet_config.py`

Tests the `ids_tabnet.config.IDS_TABNET_SEARCH_SPACE`:

| Test                                                      | Purpose                                                                                                     |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `test_search_space_entries_have_numeric_inclusive_ranges` | Verifies every entry declares numeric, ordered inclusive bounds and a supported `int` or `float` gene type. |

### `test_ids_tabnet_evaluator.py`

Tests `ids_tabnet.evaluator.TabNetEvaluator` without real model training:

| Test                                                       | Purpose                                                                                                                                                 |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `test_fixture_supports_balanced_60_20_20_allocation`       | Verifies the shared fixture has balanced binary labels and produces expected 60/20/20 train, validation, and test sizes while preserving class balance. |
| `test_reproduces_fixed_seed_stratified_partitions`         | Verifies identical evaluator constructions produce identical fixed-seed stratified partitions and use the in-memory fixture instead of CSV loading.     |
| `test_evaluate_configures_cpu_model_and_scores_validation` | Mocks model training, CUDA detection, and `f1_score` to verify CPU configuration, fit arguments, validation inputs, and macro-F1 forwarding.            |
| `test_evaluate_test_scores_test_data`                      | Verifies `evaluate_test()` predicts and scores against held-out test data.                                                                              |
| `test_evaluate_selects_cuda_when_available`                | Verifies CUDA availability selects the `"cuda"` device without requiring GPU hardware.                                                                  |

### `test_ids_tabnet_data.py`

Tests reusable `ids_tabnet.evaluator.prepare_tabnet_data()` and
`TabNetEvaluator.from_partitions()` construction:

| Test                                            | Purpose                                                                            |
|-------------------------------------------------|------------------------------------------------------------------------------------|
| `test_preparation_loads_csv_once`               | Verifies one preparation call loads one CSV and creates fixed 60/20/20 partitions. |
| `test_preparation_is_deterministic`             | Verifies repeated preparation calls preserve identical stratified partitions.      |
| `test_partition_construction_does_not_load_csv` | Verifies an evaluator reuses prepared arrays without a second CSV load.            |
| `test_serial_partitions_remain_writable`        | Verifies serial evaluator partitions preserve writable NumPy arrays.               |

### `test_ids_tabnet_parallel.py`

Tests `ids_tabnet.parallel.RayTabNetBatchEvaluator` with mocked resource and
dispatch boundaries plus local-Ray lifecycle and integration fixtures:

| Test                                                                   | Purpose                                                                                                                            |
|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `test_uninitialized_ray_is_rejected_without_lifecycle_changes`         | Verifies construction neither initializes nor shuts down caller-managed Ray.                                                       |
| `test_non_positive_worker_count_is_rejected`                           | Verifies invalid worker counts fail before dataset preparation.                                                                    |
| `test_non_positive_thread_count_is_rejected`                           | Verifies invalid thread counts fail before dataset preparation.                                                                    |
| `test_cpu_request_above_available_capacity_is_rejected`                | Verifies requested logical CPUs cannot exceed initialized Ray capacity.                                                            |
| `test_workers_share_one_prepared_dataset_and_close_keeps_ray_running`  | Verifies one preparation feeds multiple local actors, immediate recreation works, and `close()` preserves Ray initialization.      |
| `test_close_after_ray_shutdown_is_safe`                                | Verifies cleanup tolerates the caller shutting down Ray first.                                                                     |
| `test_close_with_stale_actor_handles_does_not_wait`                    | Verifies cleanup does not wait for resources that stale handles cannot release.                                                    |
| `test_empty_batch_skips_remote_evaluation`                             | Verifies empty input returns without actor submissions.                                                                            |
| `test_dispatch_preserves_input_order_and_reuses_free_actor`            | Verifies out-of-order completion preserves input order and keeps actors busy.                                                      |
| `test_fixed_seed_does_not_change_driver_random_state`                  | Verifies one evaluation seed reaches workers without consuming driver randomness.                                                  |
| `test_worker_failure_propagates_without_partial_result`                | Verifies worker errors raise instead of returning partial fitness values.                                                          |
| `test_worker_failure_cancels_remaining_submissions`                    | Verifies a failed batch cancels other candidates that are still in flight.                                                         |
| `test_worker_failure_preserves_error_if_cancellation_fails`            | Verifies cancellation cleanup errors do not mask the original worker failure.                                                      |
| `test_gene_conversion_precedes_remote_submission`                      | Verifies invalid genes fail before remote work is submitted.                                                                       |
| `test_real_tabnet_evaluation_matches_serial_and_uses_read_only_arrays` | Runs tiny real TabNet fits on two actors, checks shared arrays are read-only, and compares ordered fitness with serial evaluation. |
| `test_real_worker_failure_preserves_error_and_allows_retry`            | Verifies a real actor failure remains visible, pending work is drained, and a later batch can run.                                 |

Run parallel tests with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.ids_tabnet.test_ids_tabnet_parallel
```

## Shared fixture

`support.py` defines `make_binary_classification_frame()`. It creates numeric
features and a `Label` column with 20 samples from each of two classes by
default, providing enough observations for both stratified
`train_test_split` calls.

`test_ids_tabnet_data.py` uses the same fixture to verify one-load preparation
and partition reuse.

| Fixture or dependency                | Purpose                                                                                                                                            |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `make_binary_classification_frame()` | Creates numeric features and balanced labels for evaluator partition tests.                                                                        |
| `load_csv`                           | Supplies the in-memory DataFrame instead of reading the production CSV.                                                                            |
| `TabNetClassifier`                   | Records constructor, fit, and prediction arguments without real TabNet training.                                                                   |
| `torch.cuda.is_available`            | Controls CPU/CUDA branch coverage without GPU hardware.                                                                                            |
| `f1_score`                           | Returns deterministic values so evaluator forwarding is tested independently of sklearn's metric implementation.                                   |
| local Ray runtime                    | Starts two-CPU local runtimes for actor lifecycle, failure cleanup, and tiny real TabNet integration tests, then shuts them down after assertions. |
