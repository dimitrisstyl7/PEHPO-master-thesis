"""Individual representation and mutation logic for evolutionary search."""

import random

from evolutionary.config import MUTATION_RATE, SUB_AREAS_NUMBER


class Individual:
    """
    Represents an individual solution in the evolutionary search.

    Each individual holds a dictionary of genes constrained by a provided search space
    and stores its fitness score.
    """

    def __init__(self, search_space: dict, genes: dict | None = None):
        """
        Initializes an Individual with either provided genes or randomly generated ones.

        Args:
            search_space: Dictionary describing valid range and type for each gene.
            genes: Optional gene dictionary. If `None`, random genes are generated.

        Raises:
            ValueError: If `search_space` is `None`.
        """
        if search_space is None:
            raise ValueError("Please provide a search space.")

        self.search_space = search_space
        self.genes = self.__generate_random_genes() if genes is None else genes
        self.fitness = 0.0

    def __generate_random_genes(self) -> dict:
        """
        Generates a random valid gene configuration based on the search space.

        Returns:
            Dictionary mapping each gene name to a randomly sampled value.
        """
        new_genes = {}
        for key, bounds in self.search_space.items():
            if bounds["type"] == int:
                new_genes[key] = random.randint(bounds["min"], bounds["max"])
            elif bounds["type"] == float:
                new_genes[key] = random.uniform(bounds["min"], bounds["max"])
        return new_genes

    def mutate(self, mutation_rate: float = MUTATION_RATE, k: int = SUB_AREAS_NUMBER):
        """
        Mutates the individual's genes using a GLEAM-inspired mutation strategy.

        Each gene has a probability of being mutated based on the mutation_rate. When a gene is selected
        for mutation, the search space between the current value and the nearest boundary is divided into
        k sub-areas. A new value is then sampled from a randomly selected sub-area, favoring smaller
        perturbations while still allowing larger jumps.

        Args:
            mutation_rate: Probability that each gene will be mutated. Default is `MUTATION_RATE`.
            k: Number of sub-areas used for mutation range partitioning. Default is `SUB_AREAS_NUMBER`.
        """
        for key, value in self.genes.items():
            if random.random() > mutation_rate:
                continue

            entry = self.search_space[key]

            # Determine mutation direction based on current position
            if value >= entry["max"]:
                should_increase = False
            elif value <= entry["min"]:
                should_increase = True
            else:
                should_increase = random.choice([True, False])

            # GLEAM logic: divide the distance to the boundary into k sub-areas
            B = entry["max"] if should_increase else entry["min"]
            d = abs(B - value) / k  # Sub-area width (δ)
            i = random.randint(1, k)  # Sub-area index

            a, b = (value, value + d * i) if should_increase else (value - d * i, value)  # fmt: skip
            new_value = entry["type"](random.uniform(a, b))

            # If it's an int and didn't change due to rounding, nudge it by 1
            if new_value == value and entry["type"] == int:
                step = 1 if should_increase else -1
                # Ensure the nudge doesn't exceed the bounds
                if entry["min"] <= value + step <= entry["max"]:
                    new_value = value + step

            self.genes[key] = new_value
