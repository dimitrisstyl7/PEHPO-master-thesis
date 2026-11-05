import logging
import time
from pathlib import Path

import ray

from evolutionary.algorithm import GeneticAlgorithm
from evolutionary.history import HistoryRecorder
from ids_tabnet.config import IDS_TABNET_SEARCH_SPACE
from ids_tabnet.parallel import RayTabNetBatchEvaluator
from visualization.plots import save_all

logger = logging.getLogger(__name__)


# Evaluator constant values
DATASET_PATH = "datasets/cicids2017_cleaned.csv"
WORKER_COUNT = 12
THREADS_PER_WORKER = 1

# GeneticAlgorithm constant values
POPULATION_SIZE = 100
MAX_GENERATIONS = 500
STALL_GENERATIONS = max(1, MAX_GENERATIONS // 5)  # 20% of max generations
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1  # 10% mutation rate
SUB_AREAS_NUMBER = 3
ELITES_COUNT = 3
LOG_SCALE_GENES = ("lr", "lambda_sparse")


def main() -> None:
    # Use the same timestamp directory for logs and plots.
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    file_name = log_dir / "run.log"

    # Logging configuration to log both to console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(file_name, "w"),
        ],
    )

    # Log the start of a new run with configuration details
    logger.info("=" * 30 + " NEW RUN " + "=" * 30)
    logger.info(
        "Initializing Ray with %d workers and %d threads per worker",
        WORKER_COUNT,
        THREADS_PER_WORKER,
    )
    start_time = time.perf_counter()
    history = HistoryRecorder(
        population_size=POPULATION_SIZE,
        generation_budget=MAX_GENERATIONS,
        stall_limit=STALL_GENERATIONS,
        search_space=IDS_TABNET_SEARCH_SPACE,
    )

    # Initialize Ray with a specified object store memory limit (3 GB in this case)
    ray.init(object_store_memory=3 * 1024**3)

    # Run the genetic algorithm with the RayTabNetBatchEvaluator
    try:
        with RayTabNetBatchEvaluator(
            dataset_path=DATASET_PATH,
            worker_count=WORKER_COUNT,
            threads_per_worker=THREADS_PER_WORKER,
        ) as evaluator:
            algorithm = GeneticAlgorithm(
                fitness_evaluator=evaluator,
                search_space=IDS_TABNET_SEARCH_SPACE,
                population_size=POPULATION_SIZE,
                max_generations=MAX_GENERATIONS,
                tournament_size=TOURNAMENT_SIZE,
                mutation_rate=MUTATION_RATE,
                sub_areas_number=SUB_AREAS_NUMBER,
                elites_count=ELITES_COUNT,
                stall_generations=STALL_GENERATIONS,
                generation_observer=history,
            )
            best = algorithm.run()
            logger.info("Best individual: %s", best)
    finally:
        try:
            if history.records:
                run_dir = Path("plots") / run_id
                history.save(run_dir / "history.json")
                save_all(history, run_dir, LOG_SCALE_GENES)
                logger.info("Saved run history and plots to %s", run_dir)
        finally:
            ray.shutdown()
            logger.info("Total runtime: %.1fs\n", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
