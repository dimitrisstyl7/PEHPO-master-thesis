"""Tests for reusable TabNet dataset preparation."""

import unittest
from unittest.mock import patch

import numpy as np

from ids_tabnet.evaluator import (
    TabNetEvaluator,
    prepare_tabnet_data,
)
from tests.ids_tabnet.support import make_binary_classification_frame


class TabNetDataPreparationTests(unittest.TestCase):
    """Verify one-load deterministic partitions and reuse."""

    dataset_path = "datasets/cicids2017_cleaned.csv"

    def setUp(self) -> None:
        """Create a balanced in-memory dataset fixture."""
        self.data = make_binary_classification_frame()

    def test_preparation_loads_csv_once(self) -> None:
        """One preparation call reads the dataset exactly once."""
        with patch(
            "ids_tabnet.evaluator.load_csv",
            return_value=self.data.copy(),
        ) as load_csv:
            partitions = prepare_tabnet_data(self.dataset_path)

        load_csv.assert_called_once_with(self.dataset_path)
        self.assertEqual(len(partitions.X_train), 24)
        self.assertEqual(len(partitions.X_val), 8)
        self.assertEqual(len(partitions.X_test), 8)
        self.assertTrue(partitions.X_train.flags.writeable)
        self.assertTrue(partitions.y_train.flags.writeable)

    def test_serial_partitions_remain_writable(self) -> None:
        """Serial TabNet evaluation keeps its existing writable-array contract."""
        with patch(
            "ids_tabnet.evaluator.load_csv",
            return_value=self.data.copy(),
        ):
            partitions = prepare_tabnet_data(self.dataset_path)

        self.assertTrue(partitions.X_train.flags.writeable)
        self.assertTrue(partitions.X_val.flags.writeable)
        self.assertTrue(partitions.X_test.flags.writeable)
        self.assertTrue(partitions.y_train.flags.writeable)
        self.assertTrue(partitions.y_val.flags.writeable)
        self.assertTrue(partitions.y_test.flags.writeable)

    def test_preparation_is_deterministic(self) -> None:
        """Repeated preparation calls produce identical fixed partitions."""
        with patch(
            "ids_tabnet.evaluator.load_csv",
            side_effect=[self.data.copy(), self.data.copy()],
        ):
            first = prepare_tabnet_data(self.dataset_path)
            second = prepare_tabnet_data(self.dataset_path)

        for first_partition, second_partition in (
            (first.X_train, second.X_train),
            (first.X_val, second.X_val),
            (first.X_test, second.X_test),
            (first.y_train, second.y_train),
            (first.y_val, second.y_val),
            (first.y_test, second.y_test),
        ):
            np.testing.assert_array_equal(first_partition, second_partition)

    def test_partition_construction_does_not_load_csv(self) -> None:
        """An evaluator can reuse prepared arrays without reading the CSV."""
        with patch(
            "ids_tabnet.evaluator.load_csv",
            return_value=self.data.copy(),
        ):
            partitions = prepare_tabnet_data(self.dataset_path)

        with patch("ids_tabnet.evaluator.load_csv") as load_csv:
            evaluator = TabNetEvaluator.from_partitions(partitions)

        load_csv.assert_not_called()
        self.assertIs(evaluator.X_train, partitions.X_train)
        self.assertIs(evaluator.X_val, partitions.X_val)
        self.assertIs(evaluator.X_test, partitions.X_test)
        self.assertIs(evaluator.y_train, partitions.y_train)
        self.assertIs(evaluator.y_val, partitions.y_val)
        self.assertIs(evaluator.y_test, partitions.y_test)
