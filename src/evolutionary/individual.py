"""Individual representation and mutation logic for evolutionary search."""

import random
import uuid
from typing import TypedDict


class SearchSpaceEntry(TypedDict):
    """Constraints and type definition for a single gene in the search space."""

    min: int | float
    """The inclusive minimum value allowed for the gene."""
    max: int | float
    """The inclusive maximum value allowed for the gene."""
    type: type
    """The Python type (typically `int` or `float`) used to cast the gene."""


__pdoc__ = {
    "Individual.__init__": False,
    "Individual._generate_random_genes": True,
}


class Individual:
    """
    Represents an individual solution in the evolutionary search.

    Each individual holds a dictionary of genes constrained by a provided search space,
    stores its fitness score, and carries a unique id.

    Args:
        search_space: Dictionary describing valid range and type for each gene.
        mutation_rate: Per-gene mutation probability for this individual.
        sub_areas_number: Number of sub-areas used for mutation range partitioning.
        genes: Optional gene dictionary. If `None`, random genes are generated.

    Raises:
        ValueError: If `search_space` is `None`.
    """

    def __init__(
        self,
        search_space: dict[str, SearchSpaceEntry],
        mutation_rate: float,
        sub_areas_number: int,
        genes: dict[str, int | float] | None = None,
        fitness: float = 0.0,
    ):
        if search_space is None:
            raise ValueError("Please provide a search space.")

        self.search_space = search_space
        self.genes = self._generate_random_genes() if genes is None else genes
        self.mutation_rate = mutation_rate
        self.sub_areas_number = sub_areas_number
        self.fitness = fitness
        self.id = uuid.uuid4().hex

    def _generate_random_genes(self) -> dict[str, int | float]:
        """
        Generates a random valid gene configuration based on the search space.

        Returns:
            Gene configuration sampled uniformly within bounds, typed per the search space.
        """
        new_genes: dict[str, int | float] = {}
        for key, bounds in self.search_space.items():
            if bounds["type"] is int:
                new_genes[key] = random.randint(int(bounds["min"]), int(bounds["max"]))
            elif bounds["type"] is float:
                new_genes[key] = random.uniform(bounds["min"], bounds["max"])
        return new_genes

    def mutate(self) -> None:
        r"""
        Mutates genes using a GLEAM-inspired bounded perturbation strategy.

        Per-gene mutation probability gates entry. For each selected gene:

        - Direction is forced toward the nearer bound if at boundary, otherwise random.
        - Distance to boundary is partitioned into `sub_areas_number` sub-areas (width $\delta$).
        - A random sub-area index $i$ is drawn; new value sampled from $[a, a + \delta \cdot i]$.
        - Integer genes that round to the same value are nudged by $\pm 1$ if within bounds.
        """
        for key, value in self.genes.items():
            if random.random() > self.mutation_rate:
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
            delta = abs(B - value) / self.sub_areas_number  # Sub-area width (δ)
            i = random.randint(1, self.sub_areas_number)  # Sub-area index

            a, b = (
                (value, value + delta * i)
                if should_increase
                else (value - delta * i, value)
            )
            new_value = entry["type"](random.uniform(a, b))

            # If it's an int and didn't change due to rounding, nudge it by 1
            if new_value == value and entry["type"] is int:
                step = 1 if should_increase else -1
                # Ensure the nudge doesn't exceed the bounds
                if entry["min"] <= value + step <= entry["max"]:
                    new_value = value + step

            self.genes[key] = new_value
