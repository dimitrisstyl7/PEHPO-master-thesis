"""Tests for ordered batch fitness-evaluation contracts."""

import random
import unittest
from collections.abc import Sequence

from evolutionary.algorithm import GeneticAlgorithm
from evolutionary.evaluation import BatchEvaluator, serial_batch_evaluator
from evolutionary.individual import GeneDictionary, Individual, SearchSpaceEntry

SEARCH_SPACE: dict[str, SearchSpaceEntry] = {
    "gene": {"min": 0, "max": 10, "type": int},
}


def score(genes: GeneDictionary) -> float:
    """Return the single gene as a deterministic scalar objective."""
    return float(genes["gene"])


def build_algorithm(
    evaluator: BatchEvaluator,
    population_size: int = 3,
    max_generations: int = 1,
    tournament_size: int = 1,
    mutation_rate: float = 0.0,
    sub_areas_number: int = 2,
    elites_count: int = 1,
) -> GeneticAlgorithm:
    """Build a small deterministic algorithm for contract tests."""
    return GeneticAlgorithm(
        fitness_evaluator=evaluator,
        search_space=SEARCH_SPACE,
        population_size=population_size,
        max_generations=max_generations,
        tournament_size=tournament_size,
        mutation_rate=mutation_rate,
        sub_areas_number=sub_areas_number,
        elites_count=elites_count,
    )


class BatchEvaluationTests(unittest.TestCase):
    """Verify ordered evaluation and atomic fitness assignment."""

    def test_serial_adapter_preserves_input_order(self) -> None:
        """Scalar objective results retain positional mapping."""
        evaluator = serial_batch_evaluator(score)
        genes = [{"gene": 2}, {"gene": 9}, {"gene": 4}]

        self.assertEqual(evaluator(genes), [2.0, 9.0, 4.0])

    def test_batch_evaluation_does_not_change_random_state(self) -> None:
        """Batch evaluation does not consume genetic randomness."""
        random.seed(23)
        algorithm = build_algorithm(serial_batch_evaluator(score))
        population = algorithm._initialize_population()
        state_before = random.getstate()

        algorithm._evaluate_population(population)

        self.assertEqual(random.getstate(), state_before)

    def test_short_results_raise_before_partial_assignment(self) -> None:
        """Missing fitness values leave every existing fitness unchanged."""
        algorithm = build_algorithm(
            lambda _gene_dicts: [8.0],
        )
        population = [
            Individual(
                search_space=SEARCH_SPACE,
                mutation_rate=0.0,
                sub_areas_number=2,
                genes={"gene": 1},
                fitness=1.0,
            ),
            Individual(
                search_space=SEARCH_SPACE,
                mutation_rate=0.0,
                sub_areas_number=2,
                genes={"gene": 2},
                fitness=2.0,
            ),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "returned 1 fitness values for 2 individuals",
        ):
            algorithm._evaluate_population(population)

        self.assertEqual([individual.fitness for individual in population], [1.0, 2.0])

    def test_extra_results_raise_before_partial_assignment(self) -> None:
        """Extra fitness values leave every existing fitness unchanged."""
        algorithm = build_algorithm(
            lambda _gene_dicts: [8.0, 9.0, 10.0],
        )
        population = [
            Individual(
                search_space=SEARCH_SPACE,
                mutation_rate=0.0,
                sub_areas_number=2,
                genes={"gene": 1},
                fitness=1.0,
            ),
            Individual(
                search_space=SEARCH_SPACE,
                mutation_rate=0.0,
                sub_areas_number=2,
                genes={"gene": 2},
                fitness=2.0,
            ),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "returned 3 fitness values for 2 individuals",
        ):
            algorithm._evaluate_population(population)

        self.assertEqual([individual.fitness for individual in population], [1.0, 2.0])

    def test_serial_adapter_preserves_fixed_seed_best_result(self) -> None:
        """Serial adaptation preserves the pre-change seeded GA result."""
        random.seed(17)
        algorithm = build_algorithm(
            serial_batch_evaluator(score),
            population_size=6,
            max_generations=4,
            tournament_size=2,
            mutation_rate=0.5,
            sub_areas_number=3,
            elites_count=2,
        )
        best = algorithm.run()

        self.assertEqual(best.genes, {"gene": 9})
        self.assertEqual(best.fitness, 9.0)

    def test_run_skips_empty_child_batch_evaluator(self) -> None:
        """All-elite generations do not trigger empty evaluator calls."""
        calls: list[list[GeneDictionary]] = []

        def evaluate(
            gene_dicts: Sequence[GeneDictionary],
        ) -> list[float]:
            calls.append(list(gene_dicts))
            return [score(genes) for genes in gene_dicts]

        random.seed(29)
        algorithm = build_algorithm(
            evaluate,
            population_size=3,
            max_generations=2,
            elites_count=3,
        )
        algorithm.run()

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 3)
