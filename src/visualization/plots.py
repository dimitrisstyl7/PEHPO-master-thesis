"""Matplotlib figures generated from saved evolutionary run histories."""

import math
from collections.abc import Collection
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from evolutionary.history import GenerationRecord, HistoryRecorder, IndividualSnapshot

__pdoc__ = {
    "_create_figure": True,
    "_require_records": True,
    "_gene_names": True,
    "_best_individual": True,
    "_save_figure": True,
}

FIGURE_FILES = {
    "convergence": "fig_a_convergence",
    "hyperparameter_evolution": "fig_b_hyperparameter_evolution",
    "final_population": "fig_c_final_population",
    "hyperparameter_vs_fitness": "fig_d_hyperparameter_vs_fitness",
}


def _create_figure(
    rows: int,
    columns: int,
    width_per_column: float = 5.0,
    height_per_row: float = 3.5,
) -> tuple[Figure, list[Axes]]:
    """Create an Agg-backed figure and its row-major axes.

    Args:
        rows: Number of subplot rows.
        columns: Number of subplot columns.
        width_per_column: Figure width contributed by each column.
        height_per_row: Figure height contributed by each row.

    Returns:
        Figure and its axes in row-major order.
    """
    figure = Figure(
        figsize=(width_per_column * columns, height_per_row * rows),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    axes = [
        figure.add_subplot(rows, columns, index + 1) for index in range(rows * columns)
    ]
    return figure, axes


def _require_records(history: HistoryRecorder) -> list[GenerationRecord]:
    """Return records after verifying that all populations are non-empty.

    Args:
        history: Recorded run data to validate.

    Returns:
        Validated generation records.

    Raises:
        ValueError: If history has no records or a record has no individuals.
    """
    if not history.records:
        raise ValueError("History must contain at least one generation.")
    for record in history.records:
        if not record.individuals:
            raise ValueError(f"Generation {record.generation} contains no individuals.")
    return history.records


def _gene_names(history: HistoryRecorder) -> list[str]:
    """Return search-space gene names in their declared order.

    Args:
        history: Run metadata containing the search space.

    Returns:
        Gene names in search-space insertion order.

    Raises:
        ValueError: If the search space contains no genes.
    """
    gene_names = list(history.search_space)
    if not gene_names:
        raise ValueError("History search space must contain at least one gene.")
    return gene_names


def _best_individual(record: GenerationRecord) -> IndividualSnapshot:
    """Return the highest-fitness individual snapshot in one generation.

    Args:
        record: Generation containing the candidate individuals.

    Returns:
        Individual snapshot with the highest fitness.

    Raises:
        ValueError: If the generation contains no individuals.
    """
    if not record.individuals:
        raise ValueError(f"Generation {record.generation} contains no individuals.")
    return max(record.individuals, key=lambda individual: individual[2])


def plot_convergence(history: HistoryRecorder) -> Figure:
    """Create fitness, crossover-index, and runtime panels by generation.

    Args:
        history: Recorded run data with at least one generation.

    Returns:
        Figure rendered by Matplotlib's non-interactive Agg canvas.
    """
    records = _require_records(history)
    figure, axes = _create_figure(3, 1, width_per_column=10.0, height_per_row=3.0)
    generations = [record.generation for record in records]
    fitness_values = [
        [individual[2] for individual in record.individuals] for record in records
    ]
    worst_fitness = [min(values) for values in fitness_values]
    best_fitness = [record.best_fitness for record in records]
    mean_fitness = [sum(values) / len(values) for values in fitness_values]
    max_fitness = [max(values) for values in fitness_values]

    axes[0].fill_between(
        generations,
        worst_fitness,
        max_fitness,
        alpha=0.2,
        label="Population min–max",
    )
    axes[0].plot(
        generations,
        best_fitness,
        marker="o",
        color="tab:green",
        label="Best so far",
    )
    axes[0].plot(
        generations,
        mean_fitness,
        marker="o",
        color="tab:orange",
        label="Population mean",
    )
    axes[0].plot(
        generations,
        worst_fitness,
        marker="o",
        color="tab:red",
        label="Population worst",
    )
    axes[0].set_ylabel("Fitness")
    axes[0].legend()

    axes[1].plot(
        generations,
        [record.eta_c for record in records],
        marker="o",
        color="tab:purple",
    )
    axes[1].set_ylabel(r"$\eta_c$")

    axes[2].plot(
        generations,
        [record.seconds for record in records],
        marker="o",
        color="tab:brown",
    )
    axes[2].set_ylabel("Seconds")
    axes[2].set_xlabel("Generation")

    for record in records:
        if record.stalled_stop:
            for axis in axes:
                axis.axvline(
                    record.generation,
                    color="tab:red",
                    linestyle="--",
                    label="Stall stop" if axis is axes[0] else None,
                )
    if any(record.stalled_stop for record in records):
        axes[0].legend()

    return figure


def plot_hyperparameter_evolution(
    history: HistoryRecorder,
    log_scale_genes: Collection[str] = (),
) -> Figure:
    """Create per-gene population and best-individual trajectories.

    Args:
        history: Recorded run data with at least one generation.
        log_scale_genes: Gene names to display on logarithmic value axes.

    Returns:
        Figure rendered by Matplotlib's non-interactive Agg canvas.
    """
    records = _require_records(history)
    gene_names = _gene_names(history)
    columns = min(3, len(gene_names))
    rows = math.ceil(len(gene_names) / columns)
    figure, axes = _create_figure(rows, columns)
    generations = [record.generation for record in records]

    for axis, gene_name in zip(axes, gene_names, strict=False):
        population_values = [
            [individual[1][gene_name] for individual in record.individuals]
            for record in records
        ]
        means = [
            sum(values) / len(values) if values else float("nan")
            for values in population_values
        ]
        best_values = [_best_individual(record)[1][gene_name] for record in records]
        axis.scatter(
            [
                generation
                for generation, values in zip(
                    generations,
                    population_values,
                    strict=True,
                )
                for _ in values
            ],
            [value for values in population_values for value in values],
            alpha=0.35,
            label="Individuals",
        )
        axis.plot(
            generations,
            means,
            marker="o",
            color="tab:orange",
            label="Population mean",
        )
        axis.plot(
            generations,
            best_values,
            marker="o",
            color="tab:green",
            label="Best individual",
        )
        if gene_name in log_scale_genes:
            axis.set_yscale("log")
        axis.set_title(gene_name)
        axis.set_xlabel("Generation")
        axis.set_ylabel("Value")
        axis.legend()

    for axis in axes[len(gene_names) :]:
        axis.set_visible(False)
    return figure


def plot_final_population(history: HistoryRecorder) -> Figure:
    """Create histograms for final-generation fitness and every gene.

    Args:
        history: Recorded run data with at least one generation.

    Returns:
        Figure rendered by Matplotlib's non-interactive Agg canvas.
    """
    records = _require_records(history)
    gene_names = _gene_names(history)
    final_individuals = records[-1].individuals
    if not final_individuals:
        raise ValueError(
            f"Generation {records[-1].generation} contains no individuals."
        )

    panel_names = ["fitness", *gene_names]
    columns = min(3, len(panel_names))
    rows = math.ceil(len(panel_names) / columns)
    figure, axes = _create_figure(rows, columns)
    values_by_panel = [
        [individual[2] for individual in final_individuals],
        *[
            [individual[1][gene_name] for individual in final_individuals]
            for gene_name in gene_names
        ],
    ]

    for axis, panel_name, values in zip(
        axes[: len(panel_names)],
        panel_names,
        values_by_panel,
        strict=True,
    ):
        axis.hist(values, bins="auto", color="tab:blue", alpha=0.8)
        axis.set_title(panel_name)
        axis.set_xlabel("Value")
        axis.set_ylabel("Individuals")
    for axis in axes[len(panel_names) :]:
        axis.set_visible(False)
    return figure


def plot_hyperparameter_vs_fitness(
    history: HistoryRecorder,
    log_scale_genes: Collection[str] = (),
) -> Figure:
    """Create gene-versus-fitness scatters for unique observed individuals.

    Args:
        history: Recorded run data with at least one generation.
        log_scale_genes: Gene names to display on logarithmic value axes.

    Returns:
        Figure rendered by Matplotlib's non-interactive Agg canvas.
    """
    records = _require_records(history)
    gene_names = _gene_names(history)
    unique_individuals: dict[str, tuple[int, IndividualSnapshot]] = {}
    for record in records:
        for individual in record.individuals:
            unique_individuals.setdefault(
                individual[0],
                (record.generation, individual),
            )
    if not unique_individuals:
        raise ValueError("History contains no individuals.")

    generations = [entry[0] for entry in unique_individuals.values()]
    individuals = [entry[1] for entry in unique_individuals.values()]
    best_individual = max(individuals, key=lambda individual: individual[2])
    max_generation = max(generations)
    norm = Normalize(
        vmin=min(generations), vmax=max(max_generation, min(generations) + 1)
    )
    columns = min(3, len(gene_names))
    rows = math.ceil(len(gene_names) / columns)
    figure, axes = _create_figure(rows, columns)
    color_mappable = ScalarMappable(norm=norm, cmap="viridis")

    for axis, gene_name in zip(axes, gene_names, strict=False):
        axis.scatter(
            [individual[1][gene_name] for individual in individuals],
            [individual[2] for individual in individuals],
            c=generations,
            cmap="viridis",
            norm=norm,
            alpha=0.75,
        )
        axis.scatter(
            [best_individual[1][gene_name]],
            [best_individual[2]],
            marker="*",
            s=180,
            color="gold",
            edgecolors="black",
            label="Best individual",
        )
        if gene_name in log_scale_genes:
            axis.set_xscale("log")
        axis.set_title(gene_name)
        axis.set_xlabel("Hyperparameter value")
        axis.set_ylabel("Fitness")
        axis.legend()

    for axis in axes[len(gene_names) :]:
        axis.set_visible(False)
    figure.colorbar(color_mappable, ax=axes[: len(gene_names)], label="Generation")
    return figure


def _save_figure(figure: Figure, run_dir: Path, file_stem: str) -> None:
    """Save a figure as PNG and PDF, then release its artists.

    Args:
        figure: Figure to save.
        run_dir: Directory receiving both output files.
        file_stem: Shared filename stem without an extension.
    """
    try:
        figure.savefig(run_dir / f"{file_stem}.png", dpi=150, format="png")
        figure.savefig(run_dir / f"{file_stem}.pdf", format="pdf")
    finally:
        figure.clear()


def save_all(
    history: HistoryRecorder,
    run_dir: str | Path,
    log_scale_genes: Collection[str] = (),
) -> None:
    """Save all execution figures in PNG and PDF, then release each figure.

    Args:
        history: Recorded run data used to create the figures.
        run_dir: Directory receiving the figure files.
        log_scale_genes: Gene names to display on logarithmic value axes.
    """
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_figure(
        plot_convergence(history),
        output_dir,
        FIGURE_FILES["convergence"],
    )
    _save_figure(
        plot_hyperparameter_evolution(history, log_scale_genes),
        output_dir,
        FIGURE_FILES["hyperparameter_evolution"],
    )
    _save_figure(
        plot_final_population(history),
        output_dir,
        FIGURE_FILES["final_population"],
    )
    _save_figure(
        plot_hyperparameter_vs_fitness(history, log_scale_genes),
        output_dir,
        FIGURE_FILES["hyperparameter_vs_fitness"],
    )
