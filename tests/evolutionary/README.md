# Evolutionary Tests

This directory contains standard-library `unittest` coverage for evolutionary
search-space configuration, individuals, SBX crossover, and genetic-algorithm
orchestration. Tests use in-memory individuals, deterministic objectives, and
patched random calls; they do not require datasets, model training, or GPU
hardware.

Run this category from the repository root with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/evolutionary
```

## Test modules

### `test_evolutionary_config.py`

Tests the IDS TabNet evolutionary search-space configuration:

| Test                                                      | Purpose                                                                                                     |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `test_search_space_entries_have_numeric_inclusive_ranges` | Verifies every entry declares numeric, ordered inclusive bounds and a supported `int` or `float` gene type. |

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

| Test                                                       | Purpose                                                                                         |
|------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `test_initializes_and_evaluates_configured_population`     | Verifies configured population size, bounded gene initialization, and objective evaluation.     |
| `test_tournament_selection_returns_best_sampled_candidate` | Verifies tournament selection returns the fittest sampled candidate.                            |
| `test_get_elites_sorts_and_copies_top_individuals`         | Verifies elites are sorted, retain fitness, and receive independent gene dictionaries.          |
| `test_run_tracks_best_and_keeps_exact_population_size`     | Verifies generation sizing, global best tracking, complete-pair adaptation, and verbose output. |
| `test_run_suppresses_generation_output_when_not_verbose`   | Verifies non-verbose generations execute without printing.                                      |
| `test_run_cleans_memory_for_discarded_odd_child_pair`      | Verifies odd-target trimming adapts complete pairs and releases discarded-child memory.         |

`TrackingCrossover` records child creation, adaptation pairs, and cleanup so
the odd-offspring regression remains deterministic.
