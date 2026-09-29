"""Tests for Self-Adaptive SBX crossover and eta adaptation."""

import math
import unittest
from unittest.mock import patch

from evolutionary.individual import Individual, SearchSpaceEntry
from evolutionary.self_adaptive_sbx import SelfAdaptiveSBX

SEARCH_SPACE: dict[str, SearchSpaceEntry] = {
    "integer": {"min": 0, "max": 10, "type": int},
    "real": {"min": 0.0, "max": 1.0, "type": float},
}


def make_individual(
    integer: int,
    real: float,
    fitness: float = 0.0,
) -> Individual:
    """Construct a small mixed-type individual for crossover tests."""
    return Individual(
        search_space=SEARCH_SPACE,
        mutation_rate=0.25,
        sub_areas_number=4,
        genes={"integer": integer, "real": real},
        fitness=fitness,
    )


class SelfAdaptiveSBXTests(unittest.TestCase):
    """Verify deterministic SBX calculations and adaptive state cleanup."""

    def test_calculates_spread_factor_for_both_uniform_branches(self) -> None:
        """Samples on either side of one-half use their declared formulas."""
        crossover = SelfAdaptiveSBX()
        exponent = 1 / (crossover.eta_c + 1)

        self.assertAlmostEqual(
            crossover._calculate_spread_factor(0.25),
            math.pow(0.5, exponent),
        )
        self.assertAlmostEqual(
            crossover._calculate_spread_factor(0.5),
            1.0,
        )
        self.assertAlmostEqual(
            crossover._calculate_spread_factor(0.75),
            math.pow(2.0, exponent),
        )

    def test_calculates_offspring_from_midpoint_and_step(self) -> None:
        """The midpoint-step decomposition produces both symmetric children."""
        child_1, child_2 = SelfAdaptiveSBX._calculate_genes(2.0, 2.0, 6.0)

        self.assertEqual(child_1, 0.0)
        self.assertEqual(child_2, 8.0)

    def test_crossover_copies_metadata_clamps_genes_and_records_memory(self) -> None:
        """Crossover preserves metadata and clamps every typed child gene."""
        parent_1 = make_individual(0, 0.0, fitness=3.0)
        parent_2 = make_individual(10, 1.0, fitness=2.0)
        crossover = SelfAdaptiveSBX()

        with patch(
            "evolutionary.self_adaptive_sbx.random.random",
            side_effect=[0.99, 0.99],
        ):
            child_1, child_2 = crossover.crossover(parent_1, parent_2)

        self.assertIs(child_1.search_space, parent_1.search_space)
        self.assertIs(child_2.search_space, parent_2.search_space)
        self.assertEqual(child_1.mutation_rate, parent_1.mutation_rate)
        self.assertEqual(child_2.mutation_rate, parent_2.mutation_rate)
        self.assertEqual(child_1.sub_areas_number, parent_1.sub_areas_number)
        self.assertEqual(child_2.sub_areas_number, parent_2.sub_areas_number)
        self.assertEqual(child_1.genes["integer"], 0)
        self.assertEqual(child_2.genes["integer"], 10)
        self.assertEqual(child_1.genes["real"], 0.0)
        self.assertEqual(child_2.genes["real"], 1.0)

        for child in (child_1, child_2):
            for name, entry in SEARCH_SPACE.items():
                value = child.genes[name]
                with self.subTest(child=child.id, name=name):
                    self.assertIs(type(value), entry["type"])
                    self.assertTrue(entry["min"] <= value <= entry["max"])

        self.assertEqual(len(crossover._memory[child_1.id]), 2)
        self.assertEqual(len(crossover._memory[child_2.id]), 2)

    def test_discard_children_releases_memory_entries(self) -> None:
        """Children removed before adaptation no longer retain crossover state."""
        parent_1 = make_individual(0, 0.0)
        parent_2 = make_individual(10, 1.0)
        crossover = SelfAdaptiveSBX()

        with patch(
            "evolutionary.self_adaptive_sbx.random.random",
            return_value=0.5,
        ):
            child_1, child_2 = crossover.crossover(parent_1, parent_2)

        crossover.discard_children(child_1, child_2)

        self.assertNotIn(child_1.id, crossover._memory)
        self.assertNotIn(child_2.id, crossover._memory)

    def test_equal_fitness_clears_both_memory_entries_without_adapting(self) -> None:
        """Equal best fitness skips adaptation and releases both child slots."""
        parent_1 = make_individual(2, 0.2, fitness=4.0)
        parent_2 = make_individual(8, 0.8, fitness=3.0)
        child_1 = make_individual(3, 0.3, fitness=4.0)
        child_2 = make_individual(7, 0.7, fitness=1.0)
        crossover = SelfAdaptiveSBX()
        crossover._memory[child_1.id] = [0.5]
        crossover._memory[child_2.id] = [0.5]

        crossover.update_etas(parent_1, parent_2, child_1, child_2)

        self.assertEqual(crossover.eta_c, 2.0)
        self.assertNotIn(child_1.id, crossover._memory)
        self.assertNotIn(child_2.id, crossover._memory)

    def test_adapts_for_improved_and_worsened_children(self) -> None:
        """Collapsed offspring use distinct updates for improvement outcomes."""
        cases = (
            (5.0, 0.0, 0.5),
            (1.0, 0.0, 5.0),
        )

        for child_fitness, other_child_fitness, expected_eta in cases:
            with self.subTest(child_fitness=child_fitness):
                parent_1 = make_individual(2, 0.2, fitness=4.0)
                parent_2 = make_individual(8, 0.8, fitness=3.0)
                child_1 = make_individual(3, 0.3, fitness=child_fitness)
                child_2 = make_individual(
                    7,
                    0.7,
                    fitness=other_child_fitness,
                )
                crossover = SelfAdaptiveSBX()
                crossover._memory[child_1.id] = [0.5]
                crossover._memory[child_2.id] = [0.5]

                crossover.update_etas(
                    parent_1,
                    parent_2,
                    child_1,
                    child_2,
                )

                self.assertAlmostEqual(crossover.eta_c, expected_eta)
                self.assertNotIn(child_1.id, crossover._memory)
                self.assertNotIn(child_2.id, crossover._memory)

    def test_adapts_for_spread_factor_above_one(self) -> None:
        """Improved children use the logarithmic update for expanded spread."""
        parent_1 = make_individual(2, 0.2, fitness=4.0)
        parent_2 = make_individual(8, 0.8, fitness=3.0)
        child_1 = make_individual(3, 0.3, fitness=5.0)
        child_2 = make_individual(7, 0.7, fitness=1.0)
        crossover = SelfAdaptiveSBX()
        crossover._memory[child_1.id] = [2.0]
        crossover._memory[child_2.id] = [2.0]

        crossover.update_etas(parent_1, parent_2, child_1, child_2)

        expected_eta = -1 + 3 * (math.log(2) / math.log(3))
        self.assertAlmostEqual(crossover.eta_c, expected_eta)

    def test_missing_memory_raises_for_unequal_fitness(self) -> None:
        """Unequal fitness cannot adapt without the selected child's memory."""
        parent_1 = make_individual(2, 0.2, fitness=4.0)
        parent_2 = make_individual(8, 0.8, fitness=3.0)
        child_1 = make_individual(3, 0.3, fitness=5.0)
        child_2 = make_individual(7, 0.7, fitness=1.0)
        crossover = SelfAdaptiveSBX()

        with self.assertRaisesRegex(ValueError, "Missing crossover memory"):
            crossover.update_etas(parent_1, parent_2, child_1, child_2)

    def test_clamps_eta_to_configured_bounds(self) -> None:
        """Adaptive proposals below zero and above fifty are clamped."""
        parent_1 = make_individual(2, 0.2, fitness=4.0)
        parent_2 = make_individual(8, 0.8, fitness=3.0)

        lower_crossover = SelfAdaptiveSBX()
        lower_crossover._eta_c = 0.0
        lower_child_1 = make_individual(3, 0.3, fitness=5.0)
        lower_child_2 = make_individual(7, 0.7, fitness=1.0)
        lower_crossover._memory[lower_child_1.id] = [0.5]
        lower_crossover._memory[lower_child_2.id] = [0.5]
        lower_crossover.update_etas(
            parent_1,
            parent_2,
            lower_child_1,
            lower_child_2,
        )

        upper_crossover = SelfAdaptiveSBX()
        upper_crossover._eta_c = 50.0
        upper_child_1 = make_individual(3, 0.3, fitness=1.0)
        upper_child_2 = make_individual(7, 0.7, fitness=0.0)
        upper_crossover._memory[upper_child_1.id] = [0.5]
        upper_crossover._memory[upper_child_2.id] = [0.5]
        upper_crossover.update_etas(
            parent_1,
            parent_2,
            upper_child_1,
            upper_child_2,
        )

        self.assertEqual(lower_crossover.eta_c, 0.0)
        self.assertEqual(upper_crossover.eta_c, 50.0)
