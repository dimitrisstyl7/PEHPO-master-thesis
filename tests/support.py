"""Shared fixtures for IDS TabNet tests."""

from pandas import DataFrame


def make_binary_classification_frame(samples_per_class: int = 20) -> DataFrame:
    """Build a balanced labeled frame large enough for two stratified splits."""
    rows = [
        {
            "sample_id": label * samples_per_class + sample,
            "feature_a": float(sample),
            "feature_b": float(label + sample / 10),
            "Label": label,
        }
        for label in (0, 1)
        for sample in range(samples_per_class)
    ]
    return DataFrame(rows)
