"""Genetic algorithm orchestrator for evolutionary hyperparameter optimization."""

import logging
import random
import time
from collections.abc import Callable

from evolutionary.evaluation import BatchEvaluator
from evolutionary.history import GenerationRecord
from evolutionary.individual import Individual, SearchSpaceEntry
from evolutionary.self_adaptive_sbx import SelfAdaptiveSBX

logger = logging.getLogger(__name__)

__pdoc__ = {
    "GeneticAlgorithm.__init__": False,
    "GeneticAlgorithm._run_generation": True,
    "GeneticAlgorithm._create_next_population": True,
    "GeneticAlgorithm._observe_generation": True,
    "GeneticAlgorithm._initialize_population": True,
    "GeneticAlgorithm._evaluate_population": True,
    "GeneticAlgorithm._tournament_selection": True,
    "GeneticAlgorithm._get_top_individuals": True,
    "GeneticAlgorithm._get_elites": True,
}


class GeneticAlgorithm:
    r"""
    Orchestrates a generational genetic algorithm with Self-Adaptive SBX crossover and elitism.

    Wires together population initialization, tournament selection, Self-Adaptive
    SBX crossover, GLEAM mutation, and elitism into a full generational loop.
    Progress is reported through the module logger (`INFO` per generation);
    the application configures handlers and levels. An ordered batch
    evaluator is injected at construction time, keeping the GA independent of
    any specific evaluation domain. Optional stall-based stopping ends a run
    after the global best fails to improve for the configured number of
    consecutive generations.

    Args:
        fitness_evaluator: Receives one ordered population slice of gene
            dictionaries and returns aligned scalar fitness values. Higher
            fitness is better. Use `serial_batch_evaluator` to adapt a scalar
            objective.
        search_space: Bounds and type constraints for each hyperparameter gene.
        population_size: Number of individuals maintained across generations.
        max_generations: Total number of generational cycles to execute.
        tournament_size: Number of candidates drawn per tournament selection event.
        mutation_rate: Per-gene mutation probability applied to each child.
        sub_areas_number: GLEAM partition count for bounded perturbation range.
        elites_count: Top individuals carried over untouched each generation.
        stall_generations: Consecutive generations without global-best
            improvement before early stopping. `None` disables early stopping.
        generation_observer: Optional callback receiving copied snapshots of
            the evaluated initial population and each completed generation.
    """

    def __init__(
        self,
        fitness_evaluator: BatchEvaluator,
        search_space: dict[str, SearchSpaceEntry],
        population_size: int,
        max_generations: int,
        tournament_size: int,
        mutation_rate: float,
        sub_areas_number: int,
        elites_count: int,
        stall_generations: int | None = None,
        generation_observer: Callable[[GenerationRecord], None] | None = None,
    ):
        if stall_generations is not None and (
            type(stall_generations) is not int or stall_generations < 1
        ):
            raise ValueError("stall_generations must be a positive integer or None.")

        self._fitness_evaluator = fitness_evaluator
        self._search_space = search_space
        self._population_size = population_size
        self._max_generations = max_generations
        self._tournament_size = tournament_size
        self._mutation_rate = mutation_rate
        self._sub_areas_number = sub_areas_number
        self._elites_count = elites_count
        self._stall_generations = stall_generations
        self._generation_observer = generation_observer
        self._crossover_operator = SelfAdaptiveSBX()  # type: ignore[no-untyped-call]

    def run(self) -> Individual:
        r"""
        Execute the full generational evolutionary loop.

        **Algorithm per generation:**

        1. Extract elites (carried over untouched, fitness retained).
        2. Fill the remainder: tournament select → Self-Adaptive SBX crossover → mutate.
        3. Evaluate only the new children.
        4. Update $\eta_c$ for each parent–child pair via `SelfAdaptiveSBX.update_etas()`.
        5. Clear the pair buffer.
        6. Replace population with elites + trimmed children.
        7. Track the global best individual across all generations.
        8. Stop after `stall_generations` consecutive generations without
           strict global-best improvement, when configured.
        9. Notify the optional observer with snapshots of evaluated populations.

        Returns:
            Individual with the highest fitness observed before the generation
            budget or configured stall limit is reached.
        """
        logger.info(
            "Starting GA: population=%d, generations=%d, elites=%d",
            self._population_size,
            self._max_generations,
            self._elites_count,
        )
        population = self._initialize_population()
        start_time = time.perf_counter()
        self._evaluate_population(population)
        initial_seconds = time.perf_counter() - start_time
        logger.info(
            "Initial population evaluated in %.1fs",
            initial_seconds,
        )
        best = max(population, key=lambda ind: ind.fitness)

        if self._generation_observer is not None:
            initial_elite_ids = tuple(
                individual.id for individual in self._get_top_individuals(population)
            )
            self._observe_generation(
                generation=0,
                population=population,
                elite_ids=initial_elite_ids,
                best_fitness=best.fitness,
                seconds=initial_seconds,
            )

        stall_count = 0
        for generation in range(self._max_generations):
            population, best, stall_count, stalled_stop = self._run_generation(
                population=population,
                best=best,
                stall_count=stall_count,
                generation=generation,
            )
            if stalled_stop:
                break

        return best

    def _run_generation(
        self,
        population: list[Individual],
        best: Individual,
        stall_count: int,
        generation: int,
    ) -> tuple[list[Individual], Individual, int, bool]:
        """Create and record one generation, then update best and stop state."""
        generation_start = time.perf_counter()
        elites, children = self._create_next_population(population)
        next_population = elites + children

        current_best = max(next_population, key=lambda ind: ind.fitness)
        if current_best.fitness > best.fitness:
            best = current_best
            stall_count = 0
        elif self._stall_generations is not None:
            stall_count += 1

        mean_fitness = sum(ind.fitness for ind in next_population) / len(
            next_population
        )
        generation_seconds = time.perf_counter() - generation_start
        stalled_stop = (
            self._stall_generations is not None
            and stall_count >= self._stall_generations
        )
        logger.info(
            "Generation %d/%d | best=%.4f | mean=%.4f | eta_c=%.4f | %.1fs",
            generation + 1,
            self._max_generations,
            best.fitness,
            mean_fitness,
            self._crossover_operator.eta_c,
            generation_seconds,
        )

        self._observe_generation(
            generation=generation + 1,
            population=next_population,
            elite_ids=tuple(
                ind.id for ind in self._get_top_individuals(next_population)
            ),
            best_fitness=best.fitness,
            seconds=generation_seconds,
            stalled_stop=stalled_stop,
        )
        if stalled_stop:
            logger.info(
                "Stopping early at generation %d/%d: global best did not "
                "improve for %d consecutive generations (stall limit=%d)",
                generation + 1,
                self._max_generations,
                stall_count,
                self._stall_generations,
            )
        return next_population, best, stall_count, stalled_stop

    def _create_next_population(
        self,
        population: list[Individual],
    ) -> tuple[list[Individual], list[Individual]]:
        """Create and evaluate children alongside unchanged elite copies."""
        # Carry over elites untouched — their fitness from this generation remains valid
        elites = self._get_elites(population)
        children: list[Individual] = []
        pairs: list[tuple[Individual, Individual, Individual, Individual]] = []

        # Fill the remainder of the next generation via selection → crossover → mutation
        target_children = self._population_size - self._elites_count
        while len(children) < target_children:
            parent_1 = self._tournament_selection(population)
            parent_2 = self._tournament_selection(population)
            child_1, child_2 = self._crossover_operator.crossover(parent_1, parent_2)
            child_1.mutate()
            child_2.mutate()
            children.extend([child_1, child_2])
            pairs.append((parent_1, parent_2, child_1, child_2))

        # Trim to exact target in case (population_size - elites_count) is odd
        children = children[:target_children]
        # Keep only complete parent-child pairs that still have both evaluated children
        retained_pair_count = len(children) // 2
        for _, _, discarded_child_1, discarded_child_2 in pairs[retained_pair_count:]:
            self._crossover_operator.discard_children(
                discarded_child_1,
                discarded_child_2,
            )
        pairs = pairs[:retained_pair_count]

        # Only evaluate new children — elites already carry valid fitness scores
        self._evaluate_population(children)
        # Adapt eta_c based on whether each child outperformed its parents
        for parent_1, parent_2, child_1, child_2 in pairs:
            self._crossover_operator.update_etas(parent_1, parent_2, child_1, child_2)
        pairs.clear()  # Explicit cleanup — Individual refs accumulate across generations
        return elites, children

    def _observe_generation(
        self,
        generation: int,
        population: list[Individual],
        elite_ids: tuple[str, ...],
        best_fitness: float,
        seconds: float,
        stalled_stop: bool = False,
    ) -> None:
        """Send an immutable gene snapshot when an observer is configured."""
        if self._generation_observer is None:
            return
        self._generation_observer(
            GenerationRecord(
                generation=generation,
                individuals=tuple(
                    (ind.id, ind.genes.copy(), ind.fitness) for ind in population
                ),
                elite_ids=elite_ids,
                eta_c=self._crossover_operator.eta_c,
                best_fitness=best_fitness,
                seconds=seconds,
                stalled_stop=stalled_stop,
            )
        )

    def _initialize_population(self) -> list[Individual]:
        """
        Create the initial generation of randomly-initialized individuals.

        Returns:
            `population_size` individuals with genes sampled uniformly
            from the search space bounds.
        """
        return [
            Individual(
                search_space=self._search_space,
                mutation_rate=self._mutation_rate,
                sub_areas_number=self._sub_areas_number,
            )
            for _ in range(self._population_size)
        ]

    def _evaluate_population(self, population: list[Individual]) -> None:
        """
        Evaluate one population slice through one ordered batch request.

        The evaluator receives copied gene dictionaries in population order.
        Its result count is validated before any fitness is assigned, so a
        malformed result cannot partially mutate the population. Elites should
        **not** be passed here — their fitness from the prior generation remains
        valid. Empty slices are skipped without invoking the evaluator.
        """
        if not population:
            return

        gene_dicts = [individual.genes.copy() for individual in population]
        fitnesses = list(self._fitness_evaluator(gene_dicts))
        if len(fitnesses) != len(population):
            raise ValueError(
                "Batch evaluator returned "
                f"{len(fitnesses)} fitness values for "
                f"{len(population)} individuals.",
            )

        for individual, fitness in zip(population, fitnesses, strict=True):
            individual.fitness = fitness

    def _tournament_selection(self, population: list[Individual]) -> Individual:
        """
        Select one parent via $k$-tournament selection.

        Draws `tournament_size` candidates without replacement and returns the
        one with the highest fitness. Stochastic; uses `random.sample`.

        Returns:
            Highest-fitness individual among the sampled tournament candidates.
        """
        candidates = random.sample(population, self._tournament_size)
        return max(candidates, key=lambda ind: ind.fitness)

    def _get_top_individuals(self, population: list[Individual]) -> list[Individual]:
        """Return the highest-fitness source individuals in descending order."""
        return sorted(
            population,
            key=lambda ind: ind.fitness,
            reverse=True,
        )[: self._elites_count]

    def _get_elites(self, population: list[Individual]) -> list[Individual]:
        """
        Extract the top-performing individuals for guaranteed generational survival.

        Genes are copied to prevent mutation aliasing when the elite is reused
        as a parent in tournament selection during the same generation.

        Returns:
            `elites_count` highest-fitness individuals sorted descending by fitness.
        """
        return [
            Individual(
                search_space=elite.search_space,
                mutation_rate=elite.mutation_rate,
                sub_areas_number=elite.sub_areas_number,
                genes=elite.genes.copy(),
                fitness=elite.fitness,
                individual_id=elite.id,
            )
            for elite in self._get_top_individuals(population)
        ]
