"""Self-Adaptive Simulated Binary Crossover (Self-Adaptive SBX).

Stateful crossover operator that adapts its distribution index based on offspring
performance.
"""

import math
import random

from evolutionary.individual import Individual

__pdoc__ = {
    "SelfAdaptiveSBX.__init__": False,
    "SelfAdaptiveSBX._calculate_spread_factor": True,
    "SelfAdaptiveSBX._calculate_genes": True,
}


class SelfAdaptiveSBX:
    """
    Stateful Self-Adaptive SBX crossover operator with adaptive distribution index.

    Stores per-child spread factors during crossover and updates the distribution
    index after children are evaluated.

    Initializes the distribution index, adaptation factor, bounds, and prepares
    memory for per-child spread factors.
    """

    def __init__(self):  # type: ignore[no-untyped-def]
        self._eta_c = 2.0
        self._alpha = 2.0
        self._eta_c_bounds = {"min": 0.0, "max": 50.0}
        self._memory: dict[str, list[float]] = {}

    @property
    def eta_c(self) -> float:
        r"""Current distribution index $\eta_c$ controlling offspring spread."""
        return self._eta_c

    def crossover(
        self,
        parent_1: Individual,
        parent_2: Individual,
    ) -> tuple[Individual, Individual]:
        """
        Perform SBX crossover and record per-gene spread factors.

        **Steps:**

        1. Sample a uniform value per gene and compute a spread factor.
        2. Generate offspring genes, then enforce type and bounds.
        3. Record per-child spread factors for later adaptation.

        This method is stochastic and uses the `random` module.
        """
        # Create children as copies of parents to inherit search space, rates, and initial genes
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

        # Initialize memory slots — will accumulate β_q values per gene
        self._memory[child_1.id] = []
        self._memory[child_2.id] = []

        for key in child_1.search_space:
            # Sample uniform random variable and compute spread factor β_q
            u = random.random()
            beta_q = self._calculate_spread_factor(u)
            child_1_gene, child_2_gene = self._calculate_genes(
                beta_q,
                child_1.genes[key],
                child_2.genes[key],
            )
            entry = child_1.search_space[key]
            gene_type = entry["type"]
            lower = entry["min"]
            upper = entry["max"]

            # Enforce type constraints and clamp to search space bounds
            if gene_type is int:
                child_1_gene = round(child_1_gene)
                child_2_gene = round(child_2_gene)

                child_1_gene = min(max(child_1_gene, int(lower)), int(upper))
                child_2_gene = min(max(child_2_gene, int(lower)), int(upper))
            else:
                child_1_gene = min(max(child_1_gene, lower), upper)
                child_2_gene = min(max(child_2_gene, lower), upper)

            child_1.genes[key] = child_1_gene
            child_2.genes[key] = child_2_gene

            # Record β_q for both children for later η_c adaptation
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

        # No fitness change — skip adaptation and clean up memory
        if best_child.fitness == best_parent.fitness:
            self._memory.pop(child_1.id, None)
            self._memory.pop(child_2.id, None)
            return

        beta_values = self._memory.get(best_child.id)
        if not beta_values:
            raise ValueError(f"Missing crossover memory for child id: {best_child.id}.")

        proposed_etas: list[float] = []
        child_is_better = best_child.fitness > best_parent.fitness

        # Compute a proposed η_c update for each stored β_q value
        for beta_q in beta_values:
            if beta_q > 1:  # Standard case: offspring spread beyond parent midpoint
                numerator = math.log(beta_q)
                if child_is_better:
                    denominator = math.log(1 + self._alpha * (beta_q - 1))
                else:
                    denominator = math.log(1 + (beta_q - 1) / self._alpha)
                eta_prime = -1 + (self._eta_c + 1) * (numerator / denominator)
            else:  # β_q ≤ 1: offspring collapsed near parent midpoint — use closed-form update
                if child_is_better:
                    eta_prime = (self._eta_c + 1) / self._alpha - 1
                else:
                    eta_prime = self._alpha * (self._eta_c + 1) - 1
            proposed_etas.append(eta_prime)

        # Average all proposed updates and clamp to valid η_c bounds
        average_eta = sum(proposed_etas) / len(proposed_etas)
        lower = self._eta_c_bounds["min"]
        upper = self._eta_c_bounds["max"]
        self._eta_c = min(max(average_eta, lower), upper)

        # Clear memory for both children regardless of which was used for the update
        self._memory.pop(child_1.id, None)
        self._memory.pop(child_2.id, None)

    def _calculate_spread_factor(self, u: float) -> float:
        r"""
        Compute the SBX spread factor $\beta_q$ from a uniform sample $u$ and $\eta_c$.

        **Formula:**
        $$\beta_q = \begin{cases} (2u)^{\frac{1}{\eta_c + 1}} & \text{for } u \le 0.5 \\ \left(\frac{1}{2(1 - u)}\right)^{\frac{1}{\eta_c + 1}} & \text{for } u > 0.5 \end{cases}$$
        Higher $\eta_c$ concentrates offspring closer to parents (smaller $\beta_q$).
        """
        exponent = 1 / (self._eta_c + 1)
        base = (2 * u) if u <= 0.5 else (1 / (2 * (1 - u)))
        return math.pow(base, exponent)

    @staticmethod
    def _calculate_genes(
        beta_q: float,
        child_1_gene: float,
        child_2_gene: float,
    ) -> tuple[int | float, int | float]:
        r"""
        Compute offspring gene values using the SBX midpoint–step decomposition.

        **Formula:**
        $$\begin{align*}
        \text{midpoint} &= \frac{p_1 + p_2}{2} \\
        \text{step} &= \beta_q \cdot \frac{p_1 - p_2}{2} \\
        \text{child}_1 &= \text{midpoint} + \text{step} \\
        \text{child}_2 &= \text{midpoint} - \text{step}
        \end{align*}$$
        """
        midpoint = 0.5 * (child_1_gene + child_2_gene)
        step = 0.5 * beta_q * (child_1_gene - child_2_gene)
        return midpoint + step, midpoint - step
