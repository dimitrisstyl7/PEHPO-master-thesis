"""Fitness-evaluation contracts for evolutionary algorithms."""

from collections.abc import Callable, Sequence
from typing import TypeAlias

GeneDictionary: TypeAlias = dict[str, int | float]
"""Gene values for one candidate individual."""

ScalarObjective: TypeAlias = Callable[[GeneDictionary], float]
"""Objective that maps one gene dictionary to one scalar fitness."""

BatchEvaluator: TypeAlias = Callable[
    [Sequence[GeneDictionary]],
    Sequence[float],
]
"""Ordered evaluator that maps one population slice to aligned fitnesses.

The genetic algorithm does not invoke this evaluator for empty population
slices.
"""


def serial_batch_evaluator(objective_fn: ScalarObjective) -> BatchEvaluator:
    """
    Adapt a scalar objective to the ordered batch-evaluator contract.

    The adapter evaluates each gene dictionary once, in input order, and
    returns fitness values in that same order. The objective owns evaluation
    behavior; the adapter does not alter genes or add randomness.

    Args:
        objective_fn: Scalar objective used for each submitted gene dictionary.

    Returns:
        Batch evaluator compatible with `GeneticAlgorithm`.
    """

    def evaluate(gene_dicts: Sequence[GeneDictionary]) -> list[float]:
        return [objective_fn(genes) for genes in gene_dicts]

    return evaluate
