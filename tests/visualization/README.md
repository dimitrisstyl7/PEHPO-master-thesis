# Visualization Tests

This directory contains standard-library `unittest` coverage for figures generated
from synthetic `HistoryRecorder` data. Tests do not run the genetic algorithm or
fitness evaluations.

Run this category from the repository root with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/visualization
```

## Test modules

### `test_plots.py`

Tests `visualization.plots` output and edge cases:

| Test                                                                   | Purpose                                                                           |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `test_save_all_writes_png_and_pdf_for_every_figure`                    | Verifies all four figures are written as non-empty PNG and PDF files.             |
| `test_single_generation_history_renders_without_stall_marker`          | Verifies initial-only histories render without a false stall marker.              |
| `test_scatter_deduplicates_repeated_elites_by_id`                      | Verifies Fig D deduplicates repeated elite IDs and uses a log axis for `lr`.      |
| `test_log_scales_are_configured_by_caller`                             | Verifies gene scales default to linear and do not depend on model-specific names. |
| `test_final_population_keeps_fitness_separate_from_gene_named_fitness` | Verifies a `fitness` gene cannot overwrite the objective-fitness histogram.       |

The shared fixture uses an integer `units` gene and a positive floating-point `lr`
gene, covering both ordinary and logarithmic parameter axes.
