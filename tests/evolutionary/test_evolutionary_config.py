"""Tests for evolutionary search-space configuration."""

import unittest

from evolutionary.config import IDS_TABNET_SEARCH_SPACE


class EvolutionaryConfigTests(unittest.TestCase):
    """Verify search-space entries expose valid inclusive constraints."""

    def test_search_space_entries_have_numeric_inclusive_ranges(self) -> None:
        """Every entry has ordered numeric bounds and a supported gene type."""
        self.assertTrue(IDS_TABNET_SEARCH_SPACE)

        for name, entry in IDS_TABNET_SEARCH_SPACE.items():
            with self.subTest(name=name):
                self.assertEqual(set(entry), {"min", "max", "type"})
                self.assertIsInstance(entry["min"], (int, float))
                self.assertIsInstance(entry["max"], (int, float))
                self.assertIn(entry["type"], (int, float))
                self.assertLessEqual(entry["min"], entry["max"])

                minimum = entry["type"](entry["min"])
                maximum = entry["type"](entry["max"])
                self.assertLessEqual(minimum, maximum)
