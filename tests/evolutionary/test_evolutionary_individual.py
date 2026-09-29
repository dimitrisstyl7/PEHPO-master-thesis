"""Tests for evolutionary individuals and bounded GLEAM mutation."""

import unittest
from unittest.mock import patch

from evolutionary.individual import Individual, SearchSpaceEntry

SEARCH_SPACE: dict[str, SearchSpaceEntry] = {
    "integer": {"min": 0, "max": 10, "type": int},
    "real": {"min": 0.0, "max": 1.0, "type": float},
}


class IndividualTests(unittest.TestCase):
    """Verify typed initialization and deterministic mutation branches."""

    def test_generates_typed_genes_with_inclusive_bounds(self) -> None:
        """Controlled endpoint samples retain declared types and bounds."""
        with (
            patch("evolutionary.individual.random.randint", side_effect=[0]),
            patch(
                "evolutionary.individual.random.uniform",
                side_effect=[1.0],
            ),
        ):
            individual = Individual(
                search_space=SEARCH_SPACE,
                mutation_rate=0.5,
                sub_areas_number=2,
            )

        for name, entry in SEARCH_SPACE.items():
            value = individual.genes[name]
            with self.subTest(name=name):
                self.assertIs(type(value), entry["type"])
                self.assertTrue(entry["min"] <= value <= entry["max"])

        self.assertEqual(individual.genes, {"integer": 0, "real": 1.0})

    def test_skips_mutation_when_probability_gate_fails(self) -> None:
        """A failed mutation gate leaves all genes untouched."""
        genes = {"integer": 5, "real": 0.5}
        individual = Individual(
            search_space=SEARCH_SPACE,
            mutation_rate=0.5,
            sub_areas_number=2,
            genes=genes.copy(),
        )

        with patch("evolutionary.individual.random.random", return_value=0.75):
            individual.mutate()

        self.assertEqual(individual.genes, genes)

    def test_forces_direction_at_lower_and_upper_boundaries(self) -> None:
        """Boundary genes move inward without consulting random direction."""
        integer_space: dict[str, SearchSpaceEntry] = {
            "integer": SEARCH_SPACE["integer"],
        }
        at_lower = Individual(
            search_space=integer_space,
            mutation_rate=1.0,
            sub_areas_number=2,
            genes={"integer": 0},
        )
        at_upper = Individual(
            search_space=integer_space,
            mutation_rate=1.0,
            sub_areas_number=2,
            genes={"integer": 10},
        )

        with (
            patch("evolutionary.individual.random.random", return_value=0.0),
            patch("evolutionary.individual.random.randint", return_value=1),
            patch(
                "evolutionary.individual.random.uniform",
                return_value=4.9,
            ),
            patch("evolutionary.individual.random.choice") as choice,
        ):
            at_lower.mutate()
            at_upper.mutate()

        self.assertGreater(at_lower.genes["integer"], 0)
        self.assertLess(at_upper.genes["integer"], 10)
        choice.assert_not_called()

    def test_nudges_integer_when_rounding_would_not_change_gene(self) -> None:
        """An integer mutation that rounds to its source value advances one step."""
        individual = Individual(
            search_space=SEARCH_SPACE,
            mutation_rate=1.0,
            sub_areas_number=2,
            genes={"integer": 5, "real": 0.5},
        )

        with (
            patch("evolutionary.individual.random.random", return_value=0.0),
            patch("evolutionary.individual.random.choice", return_value=True),
            patch("evolutionary.individual.random.randint", return_value=1),
            patch(
                "evolutionary.individual.random.uniform",
                return_value=5.9,
            ),
        ):
            individual.genes = {"integer": 5}
            individual.mutate()

        self.assertEqual(individual.genes["integer"], 6)
        self.assertIs(type(individual.genes["integer"]), int)

    def test_mutation_preserves_mixed_types_and_inclusive_bounds(self) -> None:
        """Integer and float mutations can land exactly on either bound."""
        individual = Individual(
            search_space=SEARCH_SPACE,
            mutation_rate=1.0,
            sub_areas_number=2,
            genes={"integer": 5, "real": 0.5},
        )

        with (
            patch(
                "evolutionary.individual.random.random",
                side_effect=[0.0, 0.0],
            ),
            patch(
                "evolutionary.individual.random.choice",
                side_effect=[True, False],
            ),
            patch(
                "evolutionary.individual.random.randint",
                side_effect=[2, 2],
            ),
            patch(
                "evolutionary.individual.random.uniform",
                side_effect=[10.0, 0.0],
            ),
        ):
            individual.mutate()

        self.assertEqual(individual.genes, {"integer": 10, "real": 0.0})
        for name, entry in SEARCH_SPACE.items():
            value = individual.genes[name]
            with self.subTest(name=name):
                self.assertIs(type(value), entry["type"])
                self.assertTrue(entry["min"] <= value <= entry["max"])
