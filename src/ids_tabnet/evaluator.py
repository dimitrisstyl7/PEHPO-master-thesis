"""Model evaluation utilities for TabNet-based IDS experiments."""

import typing
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from numpy.typing import NDArray
from pandas.core.frame import DataFrame
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from helpers.file_loader import load_csv
from ids_tabnet.params import ModelParams

__pdoc__ = {
    "TabNetEvaluator.__init__": False,
    "TabNetEvaluator._initialize": True,
    "TabNetEvaluator._fit_and_score": True,
}


@dataclass(frozen=True)
class TabNetDataPartitions:
    """
    Fixed train, validation, and test arrays used by TabNet evaluation.

    Arrays remain writable for serial evaluation. Call `make_read_only()` before
    placing partitions in Ray's object store.
    """

    X_train: NDArray[Any]
    X_val: NDArray[Any]
    X_test: NDArray[Any]
    y_train: NDArray[Any]
    y_val: NDArray[Any]
    y_test: NDArray[Any]

    def make_read_only(self) -> None:
        """Prevent mutation before sharing partitions with Ray workers."""
        for partition in (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        ):
            partition.setflags(write=False)


def prepare_tabnet_data(dataset_path: str) -> TabNetDataPartitions:
    """
    Load one CSV and create deterministic stratified 60/20/20 partitions.

    The first split reserves 20% for testing. The second split divides the
    remaining 80% into 60% training and 20% validation data.
    """
    df = typing.cast(DataFrame, load_csv(dataset_path))
    X = df.drop(columns=["Label"]).to_numpy()
    y = df["Label"].to_numpy()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        stratify=y_temp,
        random_state=42,
    )

    return TabNetDataPartitions(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


class DeviceName(StrEnum):
    """
    Device name for TabNetClassifier.

    Supported values:
        * `AUTO`: Automatically select the device.
        * `CPU`: Run on the CPU.
        * `CUDA`: Run on a CUDA-capable GPU.
    """

    # TODO: for now, only CPU will be used in the GA evaluation, but we may want to add GPU support later.

    AUTO = "auto"
    """Automatically select the device."""

    CPU = "cpu"
    """Run on the CPU."""

    CUDA = "cuda"
    """Run on a CUDA-capable GPU."""


class TabNetEvaluator:
    """
    Evaluates a TabNet classifier on a fixed train/val/test split.

    Loads a CSV dataset, applies a **60/20/20** stratified split on construction,
    then exposes `evaluate()` to train and score a model for a given hyperparameter set.

    The dataset must contain a `Label` column used as the classification target.

    Args:
        dataset_path: Path to the dataset CSV file.
        device_name: Requested device. `AUTO` is resolved by TabNet itself (CUDA if available,
            otherwise CPU).

    Raises:
        RuntimeError: If `device_name` is `DeviceName.CUDA` and CUDA is not available.
    """

    def __init__(
        self,
        dataset_path: str,
        device_name: DeviceName = DeviceName.AUTO,
    ):
        self._initialize(device_name, prepare_tabnet_data(dataset_path))

    @classmethod
    def from_partitions(
        cls,
        partitions: TabNetDataPartitions,
        device_name: DeviceName = DeviceName.AUTO,
    ) -> "TabNetEvaluator":
        """
        Construct an evaluator from already prepared fixed partitions.

        This path avoids another CSV load and preserves the same scoring
        behavior as path-based construction.
        """
        evaluator = cls.__new__(cls)
        evaluator._initialize(device_name, partitions)
        return evaluator

    def evaluate(self, params: ModelParams, seed: int = 0) -> float:
        """
        Trains a TabNetClassifier with the given model parameters and evaluates it on the validation set.

        Args:
            params: Hyperparameters for this TabNet evaluation run.
            seed: Random seed for reproducibility.

        Returns:
            Macro F1-score of the model on the validation set.
        """
        return self._fit_and_score(params, self.X_val, self.y_val, seed)

    def evaluate_test(self, params: ModelParams, seed: int = 0) -> float:
        """
        Trains a TabNetClassifier with the given model parameters and evaluates it on the test set.

        Use this only after model selection is finished, not during GA fitness evaluation.

        Args:
            params: Hyperparameters for this TabNet evaluation run.
            seed: Random seed for reproducibility.

        Returns:
            Macro F1-score of the model on the test set.
        """
        return self._fit_and_score(params, self.X_test, self.y_test, seed)

    def _initialize(
        self,
        device_name: DeviceName,
        partitions: TabNetDataPartitions,
    ) -> None:
        """Validate device selection and attach prepared partition arrays."""
        if device_name is DeviceName.CUDA and not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA-capable GPU was requested, but CUDA is not available. Use 'CPU' or 'AUTO' instead."
            )

        self.device_name = device_name
        self.X_train = partitions.X_train
        self.X_val = partitions.X_val
        self.X_test = partitions.X_test
        self.y_train = partitions.y_train
        self.y_val = partitions.y_val
        self.y_test = partitions.y_test

    def _fit_and_score(
        self, params: ModelParams, X_score: Any, y_score: Any, seed: int
    ) -> float:
        """
        Fit TabNet once and score on the provided split.

        The training and early-stopping split stays fixed internally so GA and
        final reporting share identical model setup.
        """
        clf = TabNetClassifier(
            n_d=int(params.n_da),
            n_a=int(params.n_da),
            n_steps=int(params.n_steps),
            gamma=params.gamma,
            lambda_sparse=params.lambda_sparse,
            optimizer_params={"lr": params.lr},
            device_name=self.device_name.value,
            seed=seed,
            verbose=0,  # Suppress training logs for cleaner output during evolution
        )

        # Fit the model on the training data, using the validation set for early stopping.
        clf.fit(
            X_train=self.X_train,
            y_train=self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            eval_metric=[
                "balanced_accuracy",
            ],  # Closer to Macro F1 than accuracy; both penalize minority class neglect
            weights=1,  # weighted sampling to handle class imbalance
            max_epochs=20,  # Keep low for fast evolution
            patience=5,  # Early stopping to prevent overfitting and save time during evolution
            batch_size=int(params.batch_size),
            virtual_batch_size=int(params.virtual_batch_size),
        )

        y_pred = clf.predict(X_score)

        # Calculate and return Macro F1-Score - We use Macro because it averages the F1 of each class independently.
        # zero_division=0 prevents warnings when a model fails to predict rare classes.
        return float(f1_score(y_score, y_pred, average="macro", zero_division=0))
