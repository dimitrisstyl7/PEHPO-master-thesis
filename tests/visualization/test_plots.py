"""Tests for static figures generated from evolutionary histories."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evolutionary.history import GenerationRecord, HistoryRecorder
from evolutionary.individual import SearchSpaceEntry
from visualization.plots import (
    FIGURE_FILES,
    plot_convergence,
    plot_final_population,
    plot_hyperparameter_evolution,
    plot_hyperparameter_vs_fitness,
    save_all,
)

SEARCH_SPACE: dict[str, SearchSpaceEntry] = {
    "lr": {"min": 0.001, "max": 0.05, "type": float},
    "units": {"min": 4, "max": 32, "type": int},
}


def make_history(
    records: list[GenerationRecord],
) -> HistoryRecorder:
    """Build a compact synthetic run history for plot tests."""
    history = HistoryRecorder(
        population_size=2,
        generation_budget=max(0, len(records) - 1),
        stall_limit=None,
        search_space=SEARCH_SPACE,
    )
    for record in records:
        history(record)
    return history


def make_record(
    generation: int,
    individuals: tuple[tuple[str, dict[str, int | float], float], ...],
    elite_ids: tuple[str, ...] = (),
    stalled_stop: bool = False,
) -> GenerationRecord:
    """Build one generation with consistent global-best metadata."""
    return GenerationRecord(
        generation=generation,
        individuals=individuals,
        elite_ids=elite_ids,
        eta_c=2.0 + generation,
        best_fitness=max(individual[2] for individual in individuals),
        seconds=0.25 + generation,
        stalled_stop=stalled_stop,
    )


def make_single_generation_history() -> HistoryRecorder:
    """Build an initial-only history for single-generation plot coverage."""
    return make_history(
        [
            make_record(
                0,
                (
                    ("elite-a", {"lr": 0.01, "units": 8}, 0.6),
                    ("ind-b", {"lr": 0.02, "units": 12}, 0.8),
                ),
                elite_ids=("ind-b",),
            )
        ]
    )


class PlotTests(unittest.TestCase):
    """Verify generated file outputs and plot edge cases."""

    def test_save_all_writes_png_and_pdf_for_every_figure(self) -> None:
        """Each documented figure is saved in both supported formats."""
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            save_all(make_single_generation_history(), run_dir)

            for file_stem in FIGURE_FILES.values():
                for extension in ("png", "pdf"):
                    output_path = run_dir / f"{file_stem}.{extension}"
                    self.assertTrue(output_path.is_file())
                    self.assertGreater(output_path.stat().st_size, 0)

    def test_single_generation_history_renders_without_stall_marker(self) -> None:
        """An initial-only history plots without inventing a stop marker."""
        figure = plot_convergence(make_single_generation_history())
        try:
            labels = [line.get_label() for line in figure.axes[0].lines]
            self.assertNotIn("Stall stop", labels)
            self.assertEqual(len(figure.axes[0].lines), 3)
        finally:
            figure.clear()

    def test_scatter_deduplicates_repeated_elites_by_id(self) -> None:
        """An elite seen in several generations appears once in Fig D."""
        history = make_history(
            [
                make_record(
                    0,
                    (
                        ("elite-a", {"lr": 0.01, "units": 8}, 0.6),
                        ("ind-b", {"lr": 0.02, "units": 12}, 0.8),
                    ),
                    elite_ids=("ind-b",),
                ),
                make_record(
                    1,
                    (
                        ("elite-a", {"lr": 0.01, "units": 8}, 0.6),
                        ("ind-c", {"lr": 0.03, "units": 16}, 0.9),
                    ),
                    elite_ids=("ind-c",),
                ),
            ]
        )
        figure = plot_hyperparameter_vs_fitness(history, log_scale_genes={"lr"})
        try:
            observation_points = figure.axes[0].collections[0].get_offsets()
            self.assertEqual(len(observation_points), 3)
            self.assertEqual(figure.axes[0].get_xscale(), "log")
        finally:
            figure.clear()

    def test_log_scales_are_configured_by_caller(self) -> None:
        """Gene scales stay caller-controlled rather than model-specific."""
        history = make_single_generation_history()
        evolution_figure = plot_hyperparameter_evolution(history)
        scatter_figure = plot_hyperparameter_vs_fitness(history)
        try:
            self.assertEqual(evolution_figure.axes[0].get_yscale(), "linear")
            self.assertEqual(scatter_figure.axes[0].get_xscale(), "linear")
        finally:
            evolution_figure.clear()
            scatter_figure.clear()

    def test_final_population_keeps_fitness_separate_from_gene_named_fitness(
        self,
    ) -> None:
        """A search-space gene named fitness does not replace objective values."""
        search_space: dict[str, SearchSpaceEntry] = {
            "fitness": {"min": 0.0, "max": 20.0, "type": float},
        }
        history = HistoryRecorder(
            population_size=2,
            generation_budget=0,
            stall_limit=None,
            search_space=search_space,
        )
        history(
            GenerationRecord(
                generation=0,
                individuals=(
                    ("individual-a", {"fitness": 10.0}, 0.2),
                    ("individual-b", {"fitness": 20.0}, 0.8),
                ),
                elite_ids=(),
                eta_c=2.0,
                best_fitness=0.8,
                seconds=0.25,
            )
        )

        figure = plot_final_population(history)
        try:
            fitness_histogram, gene_histogram = figure.axes[:2]
            fitness_range = (
                min(patch.get_x() for patch in fitness_histogram.patches),
                max(
                    patch.get_x() + patch.get_width()
                    for patch in fitness_histogram.patches
                ),
            )
            gene_range = (
                min(patch.get_x() for patch in gene_histogram.patches),
                max(
                    patch.get_x() + patch.get_width()
                    for patch in gene_histogram.patches
                ),
            )
            self.assertLess(fitness_range[1], 1.0)
            self.assertGreater(gene_range[0], 9.0)
        finally:
            figure.clear()
