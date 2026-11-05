"""Tests for genetic-algorithm orchestration and generation lifecycle."""

import random
import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolutionary.algorithm import GeneticAlgorithm
from evolutionary.evaluation import BatchEvaluator, serial_batch_evaluator
from evolutionary.history import GenerationRecord, HistoryRecorder
from evolutionary.individual import Individual, SearchSpaceEntry

SEARCH_SPACE: dict[str, SearchSpaceEntry] = {
    "gene": {"min": 0, "max": 10, "type": int},
}


def score(genes: dict[str, int | float]) -> float:
    """Return the single gene as a deterministic maximization objective."""
    return float(genes["gene"])


def make_individual(gene: int, fitness: float = 0.0) -> Individual:
    """Construct a deterministic individual for orchestration tests."""
    return Individual(
        search_space=SEARCH_SPACE,
        mutation_rate=0.0,
        sub_areas_number=2,
        genes={"gene": gene},
        fitness=fitness,
    )


class TrackingCrossover:
    """Minimal crossover double that exposes child-memory lifecycle."""

    def __init__(
        self,
        child_values: tuple[int, int] = (10, 9),
        child_value_sequence: list[tuple[int, int]] | None = None,
    ):
        self.eta_c = 2.0
        self._child_values = child_values
        self._child_value_sequence = child_value_sequence
        self._memory: dict[str, list[float]] = {}
        self.created: list[tuple[Individual, Individual]] = []
        self.updated: list[tuple[Individual, Individual]] = []

    def crossover(
        self,
        parent_1: Individual,
        parent_2: Individual,
    ) -> tuple[Individual, Individual]:
        """Create deterministic children and register their memory entries."""
        child_values = self._child_values
        if self._child_value_sequence is not None:
            child_values = self._child_value_sequence.pop(0)
        child_1 = make_individual(child_values[0])
        child_2 = make_individual(child_values[1])
        self._memory[child_1.id] = [1.0]
        self._memory[child_2.id] = [1.0]
        self.created.append((child_1, child_2))
        return child_1, child_2

    def update_etas(
        self,
        parent_1: Individual,
        parent_2: Individual,
        child_1: Individual,
        child_2: Individual,
    ) -> None:
        """Record complete pairs and release their crossover memory."""
        del parent_1, parent_2
        self.updated.append((child_1, child_2))
        self._memory.pop(child_1.id, None)
        self._memory.pop(child_2.id, None)

    def discard_children(self, *children: Individual) -> None:
        """Release crossover memory for discarded children."""
        for child in children:
            self._memory.pop(child.id, None)


def make_algorithm(
    population_size: int,
    max_generations: int,
    elites_count: int,
    tournament_size: int = 1,
    fitness_evaluator: BatchEvaluator | None = None,
    stall_generations: int | None = None,
    generation_observer: Callable[[GenerationRecord], None] | None = None,
) -> GeneticAlgorithm:
    """Build a GA using the in-memory deterministic objective."""
    return GeneticAlgorithm(
        fitness_evaluator=(
            serial_batch_evaluator(score)
            if fitness_evaluator is None
            else fitness_evaluator
        ),
        search_space=SEARCH_SPACE,
        population_size=population_size,
        max_generations=max_generations,
        tournament_size=tournament_size,
        mutation_rate=0.0,
        sub_areas_number=2,
        elites_count=elites_count,
        stall_generations=stall_generations,
        generation_observer=generation_observer,
    )


class GeneticAlgorithmTests(unittest.TestCase):
    """Verify population orchestration without model training."""

    def test_initializes_and_evaluates_configured_population(self) -> None:
        """Initialization creates bounded genes and evaluation assigns fitness."""
        algorithm = make_algorithm(
            population_size=4,
            max_generations=0,
            elites_count=1,
        )

        population = algorithm._initialize_population()
        self.assertEqual(len(population), 4)
        for individual in population:
            self.assertIs(type(individual.genes["gene"]), int)
            self.assertTrue(0 <= individual.genes["gene"] <= 10)

        algorithm._evaluate_population(population)

        self.assertEqual(
            [individual.fitness for individual in population],
            [float(individual.genes["gene"]) for individual in population],
        )

    def test_tournament_selection_returns_best_sampled_candidate(self) -> None:
        """Selection chooses the highest fitness among the sampled candidates."""
        algorithm = make_algorithm(
            population_size=3,
            max_generations=0,
            elites_count=1,
            tournament_size=2,
        )
        population = [
            make_individual(1, fitness=1.0),
            make_individual(5, fitness=5.0),
            make_individual(3, fitness=3.0),
        ]

        with patch(
            "evolutionary.algorithm.random.sample",
            return_value=[population[0], population[2]],
        ) as sample:
            selected = algorithm._tournament_selection(population)

        sample.assert_called_once_with(population, 2)
        self.assertIs(selected, population[2])

    def test_get_elites_sorts_and_copies_top_individuals(self) -> None:
        """Elites retain fitness while isolating copied gene dictionaries."""
        algorithm = make_algorithm(
            population_size=3,
            max_generations=0,
            elites_count=2,
        )
        population = [
            make_individual(1, fitness=1.0),
            make_individual(5, fitness=5.0),
            make_individual(3, fitness=3.0),
        ]

        elites = algorithm._get_elites(population)

        self.assertEqual([elite.fitness for elite in elites], [5.0, 3.0])
        self.assertEqual([elite.genes["gene"] for elite in elites], [5, 3])
        for elite, source in zip(elites, [population[1], population[2]]):
            self.assertIsNot(elite, source)
            self.assertIsNot(elite.genes, source.genes)
            self.assertEqual(elite.id, source.id)
        elites[0].genes["gene"] = 0
        self.assertEqual(population[1].genes["gene"], 5)

    def test_run_tracks_best_and_keeps_exact_population_size(self) -> None:
        """Generations preserve size, update the best fitness, and log each generation."""
        algorithm = make_algorithm(
            population_size=4,
            max_generations=2,
            elites_count=2,
        )
        initial_population = [
            make_individual(1),
            make_individual(2),
            make_individual(3),
            make_individual(4),
        ]
        crossover = TrackingCrossover()
        algorithm._crossover_operator = crossover

        with (
            patch.object(
                algorithm,
                "_initialize_population",
                return_value=initial_population,
            ),
            patch.object(
                algorithm,
                "_tournament_selection",
                return_value=initial_population[0],
            ),
            patch("evolutionary.individual.random.random", return_value=1.0),
            patch.object(
                algorithm,
                "_get_elites",
                wraps=algorithm._get_elites,
            ) as get_elites,
            self.assertLogs("evolutionary.algorithm", level="INFO") as logs,
        ):
            best = algorithm.run()

        self.assertEqual(best.genes["gene"], 10)
        self.assertEqual(best.fitness, 10.0)
        self.assertEqual(
            [len(call.args[0]) for call in get_elites.call_args_list],
            [4, 4],
        )
        self.assertEqual(len(crossover.updated), 2)
        generation_records = [
            record
            for record in logs.records
            if record.getMessage().startswith("Generation")
        ]
        self.assertEqual(len(generation_records), 2)

    def test_run_without_stall_limit_runs_all_generations(self) -> None:
        """The disabled default preserves the full generation budget."""
        algorithm = make_algorithm(
            population_size=4,
            max_generations=3,
            elites_count=2,
        )
        initial_population = [
            make_individual(1),
            make_individual(2),
            make_individual(3),
            make_individual(4),
        ]
        algorithm._crossover_operator = TrackingCrossover(child_values=(0, 0))

        with (
            patch.object(
                algorithm,
                "_initialize_population",
                return_value=initial_population,
            ),
            patch.object(
                algorithm,
                "_tournament_selection",
                return_value=initial_population[0],
            ),
            patch("evolutionary.individual.random.random", return_value=1.0),
            patch.object(
                algorithm,
                "_get_elites",
                wraps=algorithm._get_elites,
            ) as get_elites,
        ):
            best = algorithm.run()

        self.assertEqual(best.genes["gene"], 4)
        self.assertEqual(get_elites.call_count, 3)

    def test_run_stops_after_stall_limit_and_returns_global_best(self) -> None:
        """The configured limit stops after consecutive non-improving generations."""
        algorithm = make_algorithm(
            population_size=4,
            max_generations=5,
            elites_count=2,
            stall_generations=1,
        )
        initial_population = [
            make_individual(1),
            make_individual(2),
            make_individual(3),
            make_individual(4),
        ]
        algorithm._crossover_operator = TrackingCrossover()

        with (
            patch.object(
                algorithm,
                "_initialize_population",
                return_value=initial_population,
            ),
            patch.object(
                algorithm,
                "_tournament_selection",
                return_value=initial_population[0],
            ),
            patch("evolutionary.individual.random.random", return_value=1.0),
            patch.object(
                algorithm,
                "_get_elites",
                wraps=algorithm._get_elites,
            ) as get_elites,
            self.assertLogs("evolutionary.algorithm", level="INFO") as logs,
        ):
            best = algorithm.run()

        self.assertEqual(best.genes["gene"], 10)
        self.assertEqual(get_elites.call_count, 2)
        self.assertTrue(
            any(
                record.getMessage().startswith("Stopping early at generation 2/5")
                for record in logs.records
            )
        )

    def test_run_resets_stall_counter_after_global_best_improves(self) -> None:
        """A new global best starts a fresh sequence of stalled generations."""
        algorithm = make_algorithm(
            population_size=4,
            max_generations=6,
            elites_count=2,
            stall_generations=2,
        )
        initial_population = [
            make_individual(1),
            make_individual(2),
            make_individual(3),
            make_individual(4),
        ]
        algorithm._crossover_operator = TrackingCrossover(
            child_value_sequence=[(10, 9), (8, 7), (11, 6), (5, 4), (3, 2)]
        )

        with (
            patch.object(
                algorithm,
                "_initialize_population",
                return_value=initial_population,
            ),
            patch.object(
                algorithm,
                "_tournament_selection",
                return_value=initial_population[0],
            ),
            patch("evolutionary.individual.random.random", return_value=1.0),
            patch.object(
                algorithm,
                "_get_elites",
                wraps=algorithm._get_elites,
            ) as get_elites,
        ):
            best = algorithm.run()

        self.assertEqual(best.genes["gene"], 11)
        self.assertEqual(get_elites.call_count, 5)

    def test_rejects_non_positive_stall_limit(self) -> None:
        """Stall limits must be positive integers when enabled."""
        for stall_generations in (0, -1):
            with (
                self.subTest(stall_generations=stall_generations),
                self.assertRaisesRegex(
                    ValueError,
                    "stall_generations must be a positive integer",
                ),
            ):
                make_algorithm(
                    population_size=4,
                    max_generations=3,
                    elites_count=2,
                    stall_generations=stall_generations,
                )

    def test_run_emits_no_generation_logs_above_info_threshold(self) -> None:
        """Generation progress is INFO, so a WARNING threshold suppresses it."""
        algorithm = make_algorithm(
            population_size=4,
            max_generations=1,
            elites_count=2,
        )
        initial_population = [
            make_individual(1),
            make_individual(2),
            make_individual(3),
            make_individual(4),
        ]
        algorithm._crossover_operator = TrackingCrossover()

        with (
            patch.object(
                algorithm,
                "_initialize_population",
                return_value=initial_population,
            ),
            patch.object(
                algorithm,
                "_tournament_selection",
                return_value=initial_population[0],
            ),
            patch("evolutionary.individual.random.random", return_value=1.0),
            self.assertNoLogs("evolutionary.algorithm", level="WARNING"),
        ):
            algorithm.run()

    def test_run_cleans_memory_for_discarded_odd_child_pair(self) -> None:
        """Odd target sizes adapt complete pairs and release the excess pair."""
        algorithm = make_algorithm(
            population_size=5,
            max_generations=2,
            elites_count=2,
        )
        initial_population = [
            make_individual(1),
            make_individual(2),
            make_individual(3),
            make_individual(4),
            make_individual(5),
        ]
        crossover = TrackingCrossover()
        algorithm._crossover_operator = crossover

        with (
            patch.object(
                algorithm,
                "_initialize_population",
                return_value=initial_population,
            ),
            patch.object(
                algorithm,
                "_tournament_selection",
                return_value=initial_population[0],
            ),
            patch("evolutionary.individual.random.random", return_value=1.0),
            patch.object(
                algorithm,
                "_get_elites",
                wraps=algorithm._get_elites,
            ) as get_elites,
        ):
            algorithm.run()

        self.assertEqual(
            [len(call.args[0]) for call in get_elites.call_args_list],
            [5, 5],
        )
        self.assertEqual(len(crossover.created), 4)
        self.assertEqual(len(crossover.updated), 2)
        self.assertEqual(crossover._memory, {})

    def test_evaluates_initial_and_child_slices_in_separate_batch_calls(self) -> None:
        """Evaluation forms a barrier before selection and excludes elites."""
        calls: list[list[dict[str, int | float]]] = []
        events: list[str] = []
        evaluation_number = 0

        def evaluate(
            gene_dicts: Sequence[dict[str, int | float]],
        ) -> list[float]:
            nonlocal evaluation_number
            events.append("evaluate")
            calls.append([genes.copy() for genes in gene_dicts])
            base_fitness = 100.0 if evaluation_number == 0 else 200.0
            evaluation_number += 1
            return [base_fitness + float(genes["gene"]) for genes in gene_dicts]

        algorithm = make_algorithm(
            population_size=4,
            max_generations=1,
            elites_count=2,
            fitness_evaluator=evaluate,
        )
        initial_population = [
            make_individual(1),
            make_individual(2),
            make_individual(3),
            make_individual(4),
        ]
        algorithm._crossover_operator = TrackingCrossover()
        original_get_elites = algorithm._get_elites
        captured_elites: list[list[Individual]] = []

        def capture_elites(population: list[Individual]) -> list[Individual]:
            elites = original_get_elites(population)
            captured_elites.append(elites)
            return elites

        with (
            patch.object(
                algorithm,
                "_initialize_population",
                return_value=initial_population,
            ),
            patch.object(
                algorithm,
                "_tournament_selection",
                side_effect=lambda _population: (
                    events.append("select") or initial_population[0]
                ),
            ),
            patch.object(algorithm, "_get_elites", side_effect=capture_elites),
            patch("evolutionary.individual.random.random", return_value=1.0),
        ):
            algorithm.run()

        self.assertGreater(events.index("select"), events.index("evaluate"))
        self.assertEqual(
            [elite.fitness for elite in captured_elites[0]],
            [104.0, 103.0],
        )
        self.assertEqual(
            [[genes["gene"] for genes in call] for call in calls],
            [
                [1, 2, 3, 4],
                [10, 9],
            ],
        )

    def test_run_without_observer_preserves_evaluation_flow(self) -> None:
        """The default observer setting keeps the existing evaluation batches."""
        batch_sizes: list[int] = []

        def evaluate(
            gene_dicts: Sequence[dict[str, int | float]],
        ) -> list[float]:
            batch_sizes.append(len(gene_dicts))
            return [score(genes) for genes in gene_dicts]

        algorithm = make_algorithm(
            population_size=4,
            max_generations=2,
            elites_count=2,
            fitness_evaluator=evaluate,
        )

        best = algorithm.run()

        self.assertEqual(batch_sizes, [4, 2, 2])
        self.assertEqual(best.fitness, float(best.genes["gene"]))

    def test_generation_observer_receives_initial_and_each_generation(self) -> None:
        """The observer receives ordered, complete population snapshots."""
        records: list[GenerationRecord] = []
        algorithm = make_algorithm(
            population_size=4,
            max_generations=2,
            elites_count=2,
            generation_observer=records.append,
        )
        initial_population = [make_individual(value) for value in (1, 2, 3, 4)]
        algorithm._crossover_operator = TrackingCrossover()

        with (
            patch.object(
                algorithm,
                "_initialize_population",
                return_value=initial_population,
            ),
            patch.object(
                algorithm,
                "_tournament_selection",
                return_value=initial_population[0],
            ),
            patch("evolutionary.individual.random.random", return_value=1.0),
        ):
            algorithm.run()

        self.assertEqual([record.generation for record in records], [0, 1, 2])
        self.assertTrue(all(len(record.individuals) == 4 for record in records))
        self.assertTrue(all(len(record.elite_ids) == 2 for record in records))
        self.assertEqual(records[0].best_fitness, 4.0)
        self.assertEqual(records[1].elite_ids, records[0].elite_ids)
        for record in records:
            individual_ids = {individual[0] for individual in record.individuals}
            self.assertTrue(set(record.elite_ids).issubset(individual_ids))

    def test_generation_observer_marks_stall_stop_and_emits_no_later_records(
        self,
    ) -> None:
        """The final record marks stall termination and no later generation."""
        records: list[GenerationRecord] = []
        algorithm = make_algorithm(
            population_size=4,
            max_generations=5,
            elites_count=2,
            stall_generations=1,
            generation_observer=records.append,
        )
        initial_population = [make_individual(value) for value in (1, 2, 3, 4)]
        algorithm._crossover_operator = TrackingCrossover()

        with (
            patch.object(
                algorithm,
                "_initialize_population",
                return_value=initial_population,
            ),
            patch.object(
                algorithm,
                "_tournament_selection",
                return_value=initial_population[0],
            ),
            patch("evolutionary.individual.random.random", return_value=1.0),
        ):
            algorithm.run()

        self.assertEqual([record.generation for record in records], [0, 1, 2])
        self.assertEqual(
            [record.stalled_stop for record in records],
            [False, False, True],
        )

    def test_observer_preserves_seeded_result_and_evaluation_count(self) -> None:
        """Recording consumes no randomness and adds no fitness evaluations."""
        initial_random_state = random.getstate()

        def run_with_observer(
            observer: Callable[[GenerationRecord], None] | None,
        ) -> tuple[dict[str, int | float], float, int]:
            evaluation_count = 0

            def evaluate(
                gene_dicts: Sequence[dict[str, int | float]],
            ) -> list[float]:
                nonlocal evaluation_count
                evaluation_count += len(gene_dicts)
                return [score(genes) for genes in gene_dicts]

            algorithm = make_algorithm(
                population_size=6,
                max_generations=3,
                elites_count=2,
                tournament_size=2,
                fitness_evaluator=evaluate,
                generation_observer=observer,
            )
            random.seed(2031)
            best = algorithm.run()
            return best.genes.copy(), best.fitness, evaluation_count

        try:
            baseline = run_with_observer(None)
            observed = run_with_observer(lambda _record: None)
        finally:
            random.setstate(initial_random_state)

        self.assertEqual(observed, baseline)

    def test_generation_records_keep_gene_snapshots(self) -> None:
        """Later changes to an individual cannot mutate an emitted snapshot."""
        records: list[GenerationRecord] = []
        algorithm = make_algorithm(
            population_size=4,
            max_generations=0,
            elites_count=2,
            generation_observer=records.append,
        )
        initial_population = [make_individual(value) for value in (1, 2, 3, 4)]

        with patch.object(
            algorithm,
            "_initialize_population",
            return_value=initial_population,
        ):
            algorithm.run()

        snapshot = records[0].individuals[0][1]
        initial_population[0].genes["gene"] = 10
        initial_population[0].fitness = 10.0

        self.assertEqual(snapshot, {"gene": 1})
        self.assertEqual(records[0].individuals[0][2], 1.0)

    def test_history_json_round_trip(self) -> None:
        """Saved metadata and records load to an equivalent history."""
        history = HistoryRecorder(
            population_size=4,
            generation_budget=2,
            stall_limit=1,
            search_space=SEARCH_SPACE,
        )
        history(
            GenerationRecord(
                generation=0,
                individuals=(("individual-1", {"gene": 3}, 3.0),),
                elite_ids=("individual-1",),
                eta_c=2.0,
                best_fitness=3.0,
                seconds=0.25,
            )
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            history.save(path)
            loaded = HistoryRecorder.load(path)

        self.assertEqual(loaded.records, history.records)
        self.assertEqual(loaded.population_size, history.population_size)
        self.assertEqual(loaded.generation_budget, history.generation_budget)
        self.assertEqual(loaded.stall_limit, history.stall_limit)
        self.assertEqual(loaded.search_space, history.search_space)
