"""Tests for genetic-algorithm orchestration and generation lifecycle."""

import unittest
from unittest.mock import patch

from evolutionary.algorithm import GeneticAlgorithm
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

    def __init__(self, child_values: tuple[int, int] = (10, 9)):
        self.eta_c = 2.0
        self._child_values = child_values
        self._memory: dict[str, list[float]] = {}
        self.created: list[tuple[Individual, Individual]] = []
        self.updated: list[tuple[Individual, Individual]] = []

    def crossover(
        self,
        parent_1: Individual,
        parent_2: Individual,
    ) -> tuple[Individual, Individual]:
        """Create deterministic children and register their memory entries."""
        child_1 = make_individual(self._child_values[0])
        child_2 = make_individual(self._child_values[1])
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
    verbose: bool,
    tournament_size: int = 1,
) -> GeneticAlgorithm:
    """Build a GA using the in-memory deterministic objective."""
    return GeneticAlgorithm(
        objective_fn=score,
        search_space=SEARCH_SPACE,
        population_size=population_size,
        max_generations=max_generations,
        tournament_size=tournament_size,
        mutation_rate=0.0,
        sub_areas_number=2,
        elites_count=elites_count,
        verbose=verbose,
    )


class GeneticAlgorithmTests(unittest.TestCase):
    """Verify population orchestration without model training."""

    def test_initializes_and_evaluates_configured_population(self) -> None:
        """Initialization creates bounded genes and evaluation assigns fitness."""
        algorithm = make_algorithm(
            population_size=4,
            max_generations=0,
            elites_count=1,
            verbose=False,
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
            verbose=False,
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
            verbose=False,
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
        elites[0].genes["gene"] = 0
        self.assertEqual(population[1].genes["gene"], 5)

    def test_run_tracks_best_and_keeps_exact_population_size(self) -> None:
        """Generations preserve size, update the best fitness, and print when enabled."""
        algorithm = make_algorithm(
            population_size=4,
            max_generations=2,
            elites_count=2,
            verbose=True,
        )
        initial_population = [
            make_individual(1),
            make_individual(2),
            make_individual(3),
            make_individual(4),
        ]
        crossover = TrackingCrossover()
        algorithm._crossover = crossover

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
            patch("builtins.print") as print_mock,
            patch.object(
                algorithm,
                "_get_elites",
                wraps=algorithm._get_elites,
            ) as get_elites,
        ):
            best = algorithm.run()

        self.assertEqual(best.genes["gene"], 10)
        self.assertEqual(best.fitness, 10.0)
        self.assertEqual(
            [len(call.args[0]) for call in get_elites.call_args_list],
            [4, 4],
        )
        self.assertEqual(len(crossover.updated), 2)
        self.assertEqual(print_mock.call_count, 2)

    def test_run_suppresses_generation_output_when_not_verbose(self) -> None:
        """Non-verbose runs still execute generations without printing."""
        algorithm = make_algorithm(
            population_size=4,
            max_generations=1,
            elites_count=2,
            verbose=False,
        )
        initial_population = [
            make_individual(1),
            make_individual(2),
            make_individual(3),
            make_individual(4),
        ]
        algorithm._crossover = TrackingCrossover()

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
            patch("builtins.print") as print_mock,
        ):
            algorithm.run()

        print_mock.assert_not_called()

    def test_run_cleans_memory_for_discarded_odd_child_pair(self) -> None:
        """Odd target sizes adapt complete pairs and release the excess pair."""
        algorithm = make_algorithm(
            population_size=5,
            max_generations=2,
            elites_count=2,
            verbose=False,
        )
        initial_population = [
            make_individual(1),
            make_individual(2),
            make_individual(3),
            make_individual(4),
            make_individual(5),
        ]
        crossover = TrackingCrossover()
        algorithm._crossover = crossover

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
