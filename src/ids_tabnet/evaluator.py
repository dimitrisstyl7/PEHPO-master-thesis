"""Model evaluation utilities for TabNet-based IDS experiments."""

import typing

import torch
from helpers.file_loader import load_csv
from ids_tabnet.params import ModelParams
from pandas.core.frame import DataFrame
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class TabNetEvaluator:
    """
    A class to evaluate the performance of a TabNet model on a given dataset. It loads the dataset, splits it into
    training, validation, and testing sets (60/20/20), and provides a method to train and evaluate the model using
    specified hyperparameters contained in a ModelParams instance.

    The dataset is expected to contain a 'Label' column which is used as the target variable.
    """

    def __init__(self, dataset_path: str = "datasets/cicids2017_cleaned.csv"):
        """
        Initializes the TabNetEvaluator with the given dataset path. It loads the dataset, splits it
        into training/validation/testing sets, and prepares the data for evaluation.

        The input CSV file must have a 'Label' column.

        Args:
            dataset_path: Path to the dataset CSV file.
                Default is `"datasets/cicids2017_cleaned.csv"`.
        """
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
        Trains a TabNetClassifier with the given model parameters and evaluates it on the test set.

        Args:
            params: ModelParams instance containing TabNet hyperparameters.

        Returns:
            Macro F1-score of the model on the test set.
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
                "balanced_accuracy"
            ],  # Closer to Macro F1 than accuracy; both penalize minority class neglect
            weights=1,  # weighted sampling to handle class imbalance
            max_epochs=20,  # Keep low for fast evolution
            patience=5,  # Early stopping to prevent overfitting and save time during evolution
            batch_size=int(params.batch_size),
            virtual_batch_size=int(params.virtual_batch_size),
        )

        # Predict on the test set
        y_pred = clf.predict(self.X_test)

        # Calculate and return Macro F1-Score - We use Macro because it averages the F1 of each class independently.
        # zero_division=0 prevents warnings when a model fails to predict rare classes.
        return float(f1_score(self.y_test, y_pred, average="macro", zero_division=0))
