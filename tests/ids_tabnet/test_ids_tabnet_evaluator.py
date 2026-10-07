"""Tests for TabNet evaluator data partitioning and scoring seams."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ids_tabnet.evaluator import DeviceName, TabNetEvaluator
from ids_tabnet.params import ModelParams
from tests.ids_tabnet.support import make_binary_classification_frame


class TabNetEvaluatorTests(unittest.TestCase):
    """Verify evaluator orchestration without real model training."""

    dataset_path = "datasets/cicids2017_cleaned.csv"

    def setUp(self) -> None:
        """Create a fresh balanced fixture for each test."""
        self.data = make_binary_classification_frame()
        self.params = ModelParams(
            n_da=16,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-5,
            lr=0.02,
            batch_size=64,
        )

    def make_evaluator(self) -> TabNetEvaluator:
        """Construct an evaluator while replacing CSV loading."""
        with patch(
            "ids_tabnet.evaluator.load_csv",
            return_value=self.data.copy(),
        ):
            return TabNetEvaluator(self.dataset_path)

    def test_fixture_supports_balanced_60_20_20_allocation(self) -> None:
        """The fixture has enough observations for two binary stratified splits."""
        label_counts = self.data["Label"].value_counts().to_dict()

        self.assertEqual(label_counts, {0: 20, 1: 20})
        self.assertEqual(len(self.data), 40)

        evaluator = self.make_evaluator()
        self.assertEqual(len(evaluator.X_train), 24)
        self.assertEqual(len(evaluator.X_val), 8)
        self.assertEqual(len(evaluator.X_test), 8)
        self.assertEqual(evaluator.y_train.tolist().count(0), 12)
        self.assertEqual(evaluator.y_train.tolist().count(1), 12)
        self.assertEqual(evaluator.y_val.tolist().count(0), 4)
        self.assertEqual(evaluator.y_val.tolist().count(1), 4)
        self.assertEqual(evaluator.y_test.tolist().count(0), 4)
        self.assertEqual(evaluator.y_test.tolist().count(1), 4)

    def test_reproduces_fixed_seed_stratified_partitions(self) -> None:
        """Two constructions from the same fixture yield identical partitions."""
        with patch(
            "ids_tabnet.evaluator.load_csv",
            side_effect=[self.data.copy(), self.data.copy()],
        ) as load_csv:
            first = TabNetEvaluator(self.dataset_path)
            second = TabNetEvaluator(self.dataset_path)

        load_csv.assert_any_call(self.dataset_path)
        self.assertEqual(load_csv.call_count, 2)

        for first_partition, second_partition in (
            (first.X_train, second.X_train),
            (first.X_val, second.X_val),
            (first.X_test, second.X_test),
            (first.y_train, second.y_train),
            (first.y_val, second.y_val),
            (first.y_test, second.y_test),
        ):
            np.testing.assert_array_equal(first_partition, second_partition)

    def test_dataset_path_is_required(self) -> None:
        """Construction without an explicit dataset path is rejected."""
        with self.assertRaises(TypeError):
            TabNetEvaluator()

    @patch("ids_tabnet.evaluator.f1_score", return_value=np.float64(0.625))
    @patch("ids_tabnet.evaluator.TabNetClassifier")
    def test_evaluate_configures_model_and_scores_validation(
        self,
        classifier_type: MagicMock,
        f1_score: MagicMock,
    ) -> None:
        """evaluate trains with declared settings and scores validation data."""
        evaluator = self.make_evaluator()
        classifier = classifier_type.return_value
        classifier.predict.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        result = evaluator.evaluate(self.params, seed=17)

        classifier_type.assert_called_once_with(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-5,
            optimizer_params={"lr": 0.02},
            device_name="auto",
            seed=17,
            verbose=0,
        )

        fit_kwargs = classifier.fit.call_args.kwargs
        self.assertIs(fit_kwargs["X_train"], evaluator.X_train)
        self.assertIs(fit_kwargs["y_train"], evaluator.y_train)
        self.assertEqual(fit_kwargs["eval_set"], [(evaluator.X_val, evaluator.y_val)])
        self.assertEqual(fit_kwargs["eval_metric"], ["balanced_accuracy"])
        self.assertEqual(fit_kwargs["weights"], 1)
        self.assertEqual(fit_kwargs["max_epochs"], 20)
        self.assertEqual(fit_kwargs["patience"], 5)
        self.assertEqual(fit_kwargs["batch_size"], 64)
        self.assertEqual(fit_kwargs["virtual_batch_size"], 32)

        classifier.predict.assert_called_once_with(evaluator.X_val)
        f1_args, f1_kwargs = f1_score.call_args
        self.assertIs(f1_args[0], evaluator.y_val)
        self.assertIs(f1_args[1], classifier.predict.return_value)
        self.assertEqual(f1_kwargs, {"average": "macro", "zero_division": 0})
        self.assertEqual(result, 0.625)
        self.assertIsInstance(result, float)

    @patch("ids_tabnet.evaluator.f1_score", return_value=np.float64(0.75))
    @patch("ids_tabnet.evaluator.TabNetClassifier")
    def test_evaluate_test_scores_test_data(
        self,
        classifier_type: MagicMock,
        f1_score: MagicMock,
    ) -> None:
        """evaluate_test predicts and scores the held-out test partition."""
        evaluator = self.make_evaluator()
        classifier = classifier_type.return_value
        classifier.predict.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        result = evaluator.evaluate_test(self.params, seed=23)

        classifier_type.assert_called_once_with(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-5,
            optimizer_params={"lr": 0.02},
            device_name="auto",
            seed=23,
            verbose=0,
        )
        classifier.predict.assert_called_once_with(evaluator.X_test)
        f1_args, f1_kwargs = f1_score.call_args
        self.assertIs(f1_args[0], evaluator.y_test)
        self.assertIs(f1_args[1], classifier.predict.return_value)
        self.assertEqual(f1_kwargs, {"average": "macro", "zero_division": 0})
        self.assertEqual(result, 0.75)
        self.assertIsInstance(result, float)

    @patch("ids_tabnet.evaluator.f1_score", return_value=0.5)
    @patch("ids_tabnet.evaluator.torch.cuda.is_available", return_value=True)
    @patch("ids_tabnet.evaluator.TabNetClassifier")
    def test_evaluate_forwards_explicit_device(
        self,
        classifier_type: MagicMock,
        _: MagicMock,
        f1_score: MagicMock,
    ) -> None:
        """An explicit device is forwarded to TabNet unchanged."""
        with patch("ids_tabnet.evaluator.load_csv", return_value=self.data.copy()):
            evaluator = TabNetEvaluator(self.dataset_path, DeviceName.CUDA)
        classifier_type.return_value.predict.return_value = np.array(
            [0, 1, 0, 1, 0, 1, 0, 1]
        )

        evaluator.evaluate(self.params)

        self.assertEqual(classifier_type.call_args.kwargs["device_name"], "cuda")
        f1_score.assert_called_once()

    @patch("ids_tabnet.evaluator.torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_raises_when_unavailable(self, _: MagicMock) -> None:
        """Explicit CUDA without a CUDA device raises RuntimeError."""
        with (
            self.assertRaises(RuntimeError),
            patch("ids_tabnet.evaluator.load_csv", return_value=self.data.copy()),
        ):
            TabNetEvaluator(self.dataset_path, DeviceName.CUDA)
