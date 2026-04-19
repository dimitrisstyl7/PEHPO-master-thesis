"""Configuration constants for evolutionary hyperparameter optimization."""

MUTATION_RATE = 0.1
"""Probability that each gene mutates during one mutation step."""

SUB_AREAS_NUMBER = 10
"""Number of mutation sub-areas used by the GLEAM-inspired mutation rule."""

IDS_TABNET_SEARCH_SPACE = {
    "n_da": {"min": 8, "max": 64, "type": int},
    "n_steps": {"min": 3, "max": 10, "type": int},
    "gamma": {"min": 1.0, "max": 2.0, "type": float},
    "lambda_sparse": {"min": 1e-6, "max": 0.1, "type": float},
    "lr": {"min": 0.001, "max": 0.05, "type": float},
    "batch_size": {"min": 1024, "max": 16384, "type": int},
}
"""Search space bounds and types for IDS TabNet hyperparameters."""
