# Execution plots

Each run stores its history and figures in a timestamped directory. The timestamp
matches the run log:

```text
plots/<YYYYMMDD_HHMMSS>/
├── history.json
├── fig_a_convergence.png
├── fig_a_convergence.pdf
├── fig_b_hyperparameter_evolution.png
├── fig_b_hyperparameter_evolution.pdf
├── fig_c_final_population.png
├── fig_c_final_population.pdf
├── fig_d_hyperparameter_vs_fitness.png
└── fig_d_hyperparameter_vs_fitness.pdf
```

Generated run directories are ignored by git. This README remains tracked.

## Figure A: Convergence dashboard

Purpose: show fitness progress, self-adaptive crossover state, and generation
runtime.

- **Axes:** generation is the shared x-axis. Top panel shows best-so-far fitness,
  population mean, and population worst fitness. Shaded band shows population
  minimum to maximum. Middle panel shows `eta_c`; bottom panel shows seconds.
- **Colors and markers:** Matplotlib default colors distinguish the fitness
  lines and band. Circular markers identify measured generations. `eta_c` uses
  purple, runtime uses green, and a dashed red vertical line marks stall stop.
- **Interpretation:** a rising best-so-far line means the run found better
  fitness. The band shows population spread. A stall marker means the configured
  stall limit ended the run; no marker means the run reached its generation
  budget.

## Figure B: Hyperparameter evolution

Purpose: show how each search-space gene changes across generations.

- **Axes:** one panel per gene, with generation on x and gene value on y.
- **Colors and markers:** dots show individual values; separate Matplotlib
  default colors show population mean and the best-fitness individual's value.
  Circular markers identify means and best-individual values.
- **Scale and interpretation:** `lr` and `lambda_sparse` use a logarithmic y-axis.
  Tightening dots indicate population convergence; movement in the best-individual
  line shows how selected hyperparameters change with fitness.

## Figure C: Final population

Purpose: summarize fitness and gene values in the last recorded generation.

- **Axes:** one histogram per fitness value and search-space gene. The x-axis is
  the value; the y-axis is the number of individuals.
- **Colors and markers:** bars use translucent blue. No point markers are used.
- **Interpretation:** narrow distributions indicate similar final candidates;
  broad distributions indicate remaining diversity.

## Figure D: Hyperparameter versus fitness

Purpose: relate each gene to fitness across all unique observed individuals.

- **Axes:** one panel per gene, with gene value on x and fitness on y.
- **Colors and markers:** point color uses the `viridis` scale for generation.
  A gold star with a black edge marks the highest-fitness individual. The colorbar
  reports generation.
- **Scale and interpretation:** `lr` and `lambda_sparse` use a logarithmic x-axis.
  Color progression shows when candidates appeared; point clusters show gene
  ranges associated with similar fitness. Repeated IDs, including carried elites,
  appear once.

## Regenerate figures

Load saved history and call `save_all`; this does not evaluate fitness:

```python
from evolutionary.history import HistoryRecorder
from visualization.plots import save_all

run_dir = "plots/<YYYYMMDD_HHMMSS>"
history = HistoryRecorder.load(f"{run_dir}/history.json")
save_all(history, run_dir)
```
