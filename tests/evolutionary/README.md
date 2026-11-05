# Evolutionary Tests

This directory contains standard-library `unittest` coverage for evolutionary individuals, SBX crossover, and
genetic-algorithm orchestration. Tests use in-memory individuals, deterministic objectives, and patched random calls;
they do not require datasets, model training, or GPU hardware.

Run this category from the repository root with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/evolutionary
```

## Test modules

### `test_evolutionary_individual.py`

Tests `evolutionary.individual.Individual` initialization and bounded
GLEAM-inspired mutation. Imported `random` calls are patched so each branch is
deterministic without relying on global random seeds:

| Test                                                       | Purpose                                                                                         |
|------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `test_generates_typed_genes_with_inclusive_bounds`         | Verifies controlled endpoint samples retain declared gene types and inclusive bounds.           |
| `test_skips_mutation_when_probability_gate_fails`          | Verifies a failed mutation probability gate leaves all genes unchanged.                         |
| `test_forces_direction_at_lower_and_upper_boundaries`      | Verifies boundary genes mutate inward without random direction selection.                       |
| `test_nudges_integer_when_rounding_would_not_change_gene`  | Verifies integer mutation nudges by one when conversion would preserve the source value.        |
| `test_mutation_preserves_mixed_types_and_inclusive_bounds` | Verifies mixed integer/float mutation preserves declared types and permits inclusive endpoints. |

### `test_evolutionary_self_adaptive_sbx.py`

Tests `evolutionary.self_adaptive_sbx.SelfAdaptiveSBX` crossover and adaptive
distribution-index state:

| Test                                                             | Purpose                                                                             |
|------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `test_calculates_spread_factor_for_both_uniform_branches`        | Verifies spread-factor formulas for samples below, at, and above `u = 0.5`.         |
| `test_calculates_offspring_from_midpoint_and_step`               | Verifies midpoint-step offspring calculations.                                      |
| `test_crossover_copies_metadata_clamps_genes_and_records_memory` | Verifies copied metadata, typed bound clamping, and per-child spread-factor memory. |
| `test_discard_children_releases_memory_entries`                  | Verifies discarded children release crossover memory before adaptation.             |
| `test_equal_fitness_clears_both_memory_entries_without_adapting` | Verifies equal best fitness skips adaptation and clears both child entries.         |
| `test_adapts_for_improved_and_worsened_children`                 | Verifies bounded spread adaptation for improved and worsened children.              |
| `test_adapts_for_spread_factor_above_one`                        | Verifies logarithmic adaptation for expanded offspring spread.                      |
| `test_missing_memory_raises_for_unequal_fitness`                 | Verifies unequal-fitness adaptation raises when selected-child memory is missing.   |
| `test_clamps_eta_to_configured_bounds`                           | Verifies adaptive distribution-index proposals remain within `[0, 50]`.             |

Controlled samples cover both spread-factor branches. Direct memory fixtures
exercise adaptation and cleanup without sampling-based assertions.

### `test_evolutionary_algorithm.py`

Tests `evolutionary.algorithm.GeneticAlgorithm` orchestration with small
in-memory individuals and a deterministic objective:

| Test                                                                   | Purpose                                                                                                  |
|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `test_initializes_and_evaluates_configured_population`                 | Verifies configured population size, bounded gene initialization, and objective evaluation.              |
| `test_tournament_selection_returns_best_sampled_candidate`             | Verifies tournament selection returns the fittest sampled candidate.                                     |
| `test_get_elites_sorts_and_copies_top_individuals`                     | Verifies elites are sorted, retain fitness, and receive independent gene dictionaries.                   |
| `test_run_tracks_best_and_keeps_exact_population_size`                 | Verifies generation sizing, global best tracking, complete-pair adaptation, and per-generation logging.  |
| `test_run_without_stall_limit_runs_all_generations`                    | Verifies the disabled early-stop default preserves the full generation budget.                           |
| `test_run_stops_after_stall_limit_and_returns_global_best`             | Verifies stalled generations trigger early stopping and return the global best.                          |
| `test_run_resets_stall_counter_after_global_best_improves`             | Verifies a strict global-best improvement resets the consecutive-stall counter.                          |
| `test_rejects_non_positive_stall_limit`                                | Verifies zero and negative stall limits raise `ValueError`.                                              |
| `test_run_emits_no_generation_logs_above_info_threshold`               | Verifies generation progress is logged at INFO, not WARNING or above.                                    |
| `test_run_cleans_memory_for_discarded_odd_child_pair`                  | Verifies odd-target trimming adapts complete pairs and releases discarded-child memory.                  |
| `test_evaluates_initial_and_child_slices_in_separate_batch_calls`      | Verifies evaluation precedes selection, retains distinct elite fitness, and uses separate ordered calls. |
| `test_run_without_observer_preserves_evaluation_flow`                  | Verifies the default observer setting preserves initial and per-generation evaluation batches.           |
| `test_generation_observer_receives_initial_and_each_generation`        | Verifies snapshots include generation zero, each population, and elite identifiers.                      |
| `test_generation_observer_marks_stall_stop_and_emits_no_later_records` | Verifies the final early-stop snapshot is marked and no later records are emitted.                       |
| `test_observer_preserves_seeded_result_and_evaluation_count`           | Verifies observation preserves seeded best output and adds no fitness evaluations.                       |
| `test_generation_records_keep_gene_snapshots`                          | Verifies later individual changes cannot alter stored generation data.                                   |
| `test_history_json_round_trip`                                         | Verifies records and run metadata survive JSON persistence.                                              |

`TrackingCrossover` records child creation, adaptation pairs, and cleanup so
the odd-offspring regression remains deterministic.

### `test_evolutionary_evaluation.py`

Tests ordered batch fitness contracts and serial scalar-objective adaptation:

| Test                                                   | Purpose                                                                                             |
|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `test_serial_adapter_preserves_input_order`            | Verifies scalar objective results retain submitted gene order.                                      |
| `test_batch_evaluation_does_not_change_random_state`   | Verifies population batch evaluation consumes no genetic randomness.                                |
| `test_short_results_raise_before_partial_assignment`   | Verifies missing batch results raise without mutating fitness values.                               |
| `test_extra_results_raise_before_partial_assignment`   | Verifies extra batch results raise without mutating fitness values.                                 |
| `test_serial_adapter_preserves_fixed_seed_best_result` | Verifies serial adaptation preserves pre-change best output with mutation and multiple generations. |
| `test_run_skips_empty_child_batch_evaluator`           | Verifies all-elite generations do not trigger empty evaluator calls.                                |
