"""Self-adaptive simulated binary crossover (SBX).

Stateful crossover operator that adapts its distribution index based on offspring
performance.
"""

import math
import random

from evolutionary.individual import Individual


class SelfAdaptiveSBX:
    """
    Stateful SBX crossover operator with adaptive distribution index.

    Stores per-child spread factors during crossover and updates the distribution
    index after children are evaluated.
    """

    def __init__(self):  # type: ignore[no-untyped-def]
        """
        Initialize SBX parameters and memory.

        Sets the distribution index, adaptation factor, bounds, and prepares
        memory for per-child spread factors.
        """
        self._eta_c = 2.0
        self._alpha = 2.0
        self._eta_c_bounds = {"min": 0.0, "max": 50.0}
        self._memory: dict[str, list[float]] = {}

    def crossover(
        self, parent_1: Individual, parent_2: Individual
    ) -> tuple[Individual, Individual]:
        """
        Perform SBX crossover and record per-gene spread factors.

        **Steps:**
        - Sample a uniform value per gene and compute a spread factor.
        - Generate offspring genes, then enforce type and bounds.
        - Record per-child spread factors for later adaptation.

        This method is stochastic and uses the `random` module.
        """
        child_1 = Individual(
            search_space=parent_1.search_space,
            mutation_rate=parent_1.mutation_rate,
            sub_areas_number=parent_1.sub_areas_number,
            genes=parent_1.genes.copy(),
        )
        child_2 = Individual(
            search_space=parent_2.search_space,
            mutation_rate=parent_2.mutation_rate,
            sub_areas_number=parent_2.sub_areas_number,
            genes=parent_2.genes.copy(),
        )

        self._memory[child_1.id] = []
        self._memory[child_2.id] = []

        for key in child_1.search_space.keys():
            u = random.random()
            beta_q = self._calculate_spread_factor(u)
            child_1_gene, child_2_gene = self._calculate_genes(
                beta_q, child_1.genes[key], child_2.genes[key]
            )
            entry = child_1.search_space[key]
            gene_type = entry["type"]
            lower = entry["min"]
            upper = entry["max"]

            if gene_type is int:
                child_1_gene = int(round(child_1_gene))
                child_2_gene = int(round(child_2_gene))

                child_1_gene = min(max(child_1_gene, int(lower)), int(upper))
                child_2_gene = min(max(child_2_gene, int(lower)), int(upper))
            else:
                child_1_gene = min(max(child_1_gene, lower), upper)
                child_2_gene = min(max(child_2_gene, lower), upper)

            child_1.genes[key] = child_1_gene
            child_2.genes[key] = child_2_gene
            self._memory[child_1.id].append(beta_q)
            self._memory[child_2.id].append(beta_q)

        return child_1, child_2

    def update_etas(
        self,
        parent_1: Individual,
        parent_2: Individual,
        child_1: Individual,
        child_2: Individual,
    ) -> None:
        """
        Update the distribution index using the winning child and stored spread
        factors.

        Compares the best child against the best parent, applies the adaptive
        update for each stored spread factor, averages proposals, clamps to bounds,
        and clears memory for both children.
        """
        best_child = child_1 if child_1.fitness >= child_2.fitness else child_2
        best_parent = parent_1 if parent_1.fitness >= parent_2.fitness else parent_2

        if best_child.fitness == best_parent.fitness:
            self._memory.pop(child_1.id, None)
            self._memory.pop(child_2.id, None)
            return

        beta_values = self._memory.get(best_child.id)
        if not beta_values:
            raise ValueError(f"Missing crossover memory for child id: {best_child.id}.")

        proposed_etas: list[float] = []
        child_is_better = best_child.fitness > best_parent.fitness

        for beta_q in beta_values:
            if beta_q > 1:
                numerator = math.log(beta_q)
                if child_is_better:
                    denominator = math.log(1 + self._alpha * (beta_q - 1))
                else:
                    denominator = math.log(1 + (beta_q - 1) / self._alpha)
                eta_prime = -1 + (self._eta_c + 1) * (numerator / denominator)
            else:
                if child_is_better:
                    eta_prime = (self._eta_c + 1) / self._alpha - 1
                else:
                    eta_prime = self._alpha * (self._eta_c + 1) - 1
            proposed_etas.append(eta_prime)

        average_eta = sum(proposed_etas) / len(proposed_etas)
        lower = self._eta_c_bounds["min"]
        upper = self._eta_c_bounds["max"]
        self._eta_c = min(max(average_eta, lower), upper)

        self._memory.pop(child_1.id, None)
        self._memory.pop(child_2.id, None)

    def _calculate_spread_factor(self, u: float) -> float:
        """
        Compute the spread factor from a uniform sample and the current distribution
        index.
        """
        exponent = 1 / (self._eta_c + 1)
        base = (2 * u) if u <= 0.5 else (1 / (2 * (1 - u)))
        return math.pow(base, exponent)

    @staticmethod
    def _calculate_genes(
        beta_q: float, child_1_gene: int | float, child_2_gene: int | float
    ) -> tuple[int | float, int | float]:
        """
        Compute offspring genes from the spread factor and parent genes.
        """
        midpoint = 0.5 * (child_1_gene + child_2_gene)
        step = 0.5 * beta_q * (child_1_gene - child_2_gene)
        return midpoint + step, midpoint - step
