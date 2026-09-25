"""Model evaluation utilities for TabNet-based IDS experiments."""

import typing
from typing import Any

import torch
from pandas.core.frame import DataFrame
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from helpers.file_loader import load_csv
from ids_tabnet.params import ModelParams

__pdoc__ = {
    "TabNetEvaluator.__init__": False,
}


class TabNetEvaluator:
    """
    Evaluates a TabNet classifier on a fixed train/val/test split.

    Loads a CSV dataset, applies a **60/20/20** stratified split on construction,
    then exposes `evaluate()` to train and score a model for a given hyperparameter set.

    The dataset must contain a `Label` column used as the classification target.

    Args:
        dataset_path: Path to the dataset CSV file.
            Default is `"datasets/cicids2017_cleaned.csv"`.
    """

    def __init__(self, dataset_path: str = "datasets/cicids2017_cleaned.csv"):
        df = typing.cast(DataFrame, load_csv(dataset_path))
        X = df.drop(columns=["Label"]).to_numpy()
        y = df["Label"].to_numpy()

        # First split: separate out the Test set (20%)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        # Second split: split the remaining 80% into Train (60%) and Val (20%)
        # test_size=0.25 here means 25% of the 80%, which is 20% of the total.
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,
            stratify=y_temp,
            random_state=42,
        )

    def evaluate(self, params: ModelParams) -> float:
        """
        Trains a TabNetClassifier with the given model parameters and evaluates it on the validation set.

        Args:
            params: Hyperparameters for this TabNet evaluation run.

        Returns:
            Macro F1-score of the model on the validation set.
        """
        return self._fit_and_score(params, self.X_val, self.y_val)

    def evaluate_test(self, params: ModelParams) -> float:
        """
        Trains a TabNetClassifier with the given model parameters and evaluates it on the test set.

        Use this only after model selection is finished, not during GA fitness evaluation.

        Args:
            params: Hyperparameters for this TabNet evaluation run.

        Returns:
            Macro F1-score of the model on the test set.
        """
        return self._fit_and_score(params, self.X_test, self.y_test)

    def _fit_and_score(
        self,
        params: ModelParams,
        X_score: Any,
        y_score: Any,
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
            device_name="cuda" if torch.cuda.is_available() else "cpu",
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
