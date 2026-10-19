"""CPU-parallel Ray evaluation for TabNet hyperparameter batches."""

import logging
import time
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import ray
import torch
from ray.exceptions import ActorDiedError, ActorHandleNotFoundError, RayError

from evolutionary.evaluation import GeneDictionary
from ids_tabnet.evaluator import (
    DeviceName,
    TabNetDataPartitions,
    TabNetEvaluator,
    prepare_tabnet_data,
)
from ids_tabnet.params import ModelParams

logger = logging.getLogger(__name__)

__pdoc__ = {
    "RayTabNetBatchEvaluator.__init__": False,
    "RayTabNetBatchEvaluator.__call__": True,
    "RayTabNetBatchEvaluator._release_actors": True,
    "RayTabNetBatchEvaluator._validate_resources": True,
    "RayTabNetBatchEvaluator._cancel_pending_evaluations": True,
    "RayTabNetBatchEvaluator._available_cpu_units": True,
    "RayTabNetBatchEvaluator._cluster_cpu_units": True,
    "_model_params_from_genes": True,
}

_CPU_RELEASE_POLL_INTERVAL_SECONDS = 0.01
_CPU_RELEASE_TIMEOUT_SECONDS = 5.0


def _model_params_from_genes(genes: GeneDictionary) -> ModelParams:
    """Convert one evolutionary gene dictionary into TabNet parameters."""
    return ModelParams(
        n_da=int(genes["n_da"]),
        n_steps=int(genes["n_steps"]),
        gamma=float(genes["gamma"]),
        lambda_sparse=float(genes["lambda_sparse"]),
        lr=float(genes["lr"]),
        batch_size=int(genes["batch_size"]),
    )


@ray.remote
class _TabNetWorker:
    """Ray actor that owns one reusable CPU TabNet evaluator."""

    def __init__(
        self,
        X_train: Any,
        X_val: Any,
        X_test: Any,
        y_train: Any,
        y_val: Any,
        y_test: Any,
        threads_per_worker: int,
    ):
        torch.set_num_threads(threads_per_worker)
        # Ray supplies outer actor parallelism; one inter-op thread avoids
        # creating another per-actor scheduling pool that can oversubscribe CPUs.
        torch.set_num_interop_threads(1)
        partitions = TabNetDataPartitions(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
        # Reassert the read-only contract after Ray deserializes the shared
        # arrays in this actor. This is idempotent if Ray already preserved it.
        partitions.make_read_only()
        self._evaluator = TabNetEvaluator.from_partitions(
            partitions,
            device_name=DeviceName.CPU,
        )

    def evaluate(self, model_params: ModelParams, seed: int) -> float:
        """Evaluate one parameter set with the actor's fixed data partitions."""
        return self._evaluator.evaluate(model_params, seed=seed)

    def partition_signature(self) -> tuple[tuple[int, ...], ...]:
        """Return partition shapes for lightweight lifecycle verification."""
        return (
            self._evaluator.X_train.shape,
            self._evaluator.X_val.shape,
            self._evaluator.X_test.shape,
            self._evaluator.y_train.shape,
            self._evaluator.y_val.shape,
            self._evaluator.y_test.shape,
        )

    def partitions_are_read_only(self) -> tuple[bool, ...]:
        """Report whether every worker-owned partition rejects writes."""
        return tuple(
            not partition.flags.writeable
            for partition in (
                self._evaluator.X_train,
                self._evaluator.X_val,
                self._evaluator.X_test,
                self._evaluator.y_train,
                self._evaluator.y_val,
                self._evaluator.y_test,
            )
        )


# `ray.remote` adds `.options()` at runtime, but mypy retains the original
# class type. This alias exposes Ray's actor API to the type checker.
_RAY_TABNET_WORKER: Any = _TabNetWorker


class RayTabNetBatchEvaluator:
    """
    Evaluate ordered TabNet gene batches on reusable CPU Ray actors.

    The caller must initialize Ray before construction and remains responsible
    for shutting it down. This evaluator owns only its actors and object-store
    references.

    The CICIDS2017 partitions require approximately **1.7 GiB** before Ray
    serialization overhead. Configure Ray's object store with additional
    capacity for that overhead.

    Args:
        dataset_path: Path to the CSV dataset used for one fixed preparation.
        worker_count: Number of reusable Ray CPU actors.
        threads_per_worker: Logical CPUs and PyTorch threads reserved per actor.
        evaluation_seed: Seed forwarded unchanged to every candidate evaluation.

    Raises:
        RuntimeError: If Ray is not initialized.
        ValueError: If resource settings are non-positive or exceed available
            Ray CPUs.
    """

    def __init__(
        self,
        dataset_path: str,
        worker_count: int,
        threads_per_worker: int,
        evaluation_seed: int = 0,
    ):
        self._validate_resources(worker_count, threads_per_worker)
        self.worker_count = worker_count
        self.threads_per_worker = threads_per_worker
        self.evaluation_seed = evaluation_seed
        self._data_refs: tuple[Any, ...] = ()
        self._actors: tuple[Any, ...] = ()
        self._closed = False

        prepared = prepare_tabnet_data(dataset_path)
        # Freeze the driver-owned arrays before placing them in Ray's object
        # store, so the shared payload cannot be mutated before actor startup.
        prepared.make_read_only()
        self._data_refs = tuple(
            ray.put(partition)
            for partition in (
                prepared.X_train,
                prepared.X_val,
                prepared.X_test,
                prepared.y_train,
                prepared.y_val,
                prepared.y_test,
            )
        )
        del prepared

        try:
            self._actors = tuple(
                _RAY_TABNET_WORKER.options(num_cpus=threads_per_worker).remote(
                    *self._data_refs,
                    threads_per_worker,
                )
                for _ in range(worker_count)
            )
            # Force actor construction to finish now, so constructor failures
            # surface during evaluator creation instead of the first batch.
            ray.get(
                [actor.partition_signature.remote() for actor in self._actors],
            )
        except Exception:
            self._release_actors()
            self._data_refs = ()
            raise

    def __call__(self, gene_dicts: Sequence[GeneDictionary]) -> list[float]:
        """
        Evaluate one ordered batch, keeping at most one task per actor in flight.

        Empty batches return immediately. Parameter conversion occurs before
        remote submission so malformed genes cannot produce partial work. If a
        worker fails, remaining in-flight references are cancelled before the
        original failure is re-raised. Already-running actor tasks are drained
        first, so failure propagation can wait for their completion.
        """
        if self._closed:
            raise RuntimeError("RayTabNetBatchEvaluator is closed.")
        if not gene_dicts:
            return []

        candidate_model_params = [
            _model_params_from_genes(genes) for genes in gene_dicts
        ]
        fitnesses = [0.0] * len(candidate_model_params)
        # Track input and actor indices so completion order cannot change
        # returned fitness order, while each actor handles one task at a time.
        pending: dict[Any, tuple[int, int]] = {}
        next_index = 0

        # Initial dispatch: submit at most one candidate per actor before
        # waiting, so every available actor starts one evaluation.
        for actor_index, actor in enumerate(self._actors):
            if next_index == len(candidate_model_params):
                break
            result_ref = actor.evaluate.remote(
                candidate_model_params[next_index],
                seed=self.evaluation_seed,
            )
            pending[result_ref] = (next_index, actor_index)
            next_index += 1

        # Drain and refill until every submitted candidate has finished.
        try:
            while pending:
                # Accept whichever actor finishes first. This avoids idle actors
                # when candidate training times differ.
                ready, _ = ray.wait(list(pending), num_returns=1)
                result_ref = ready[0]
                result_index, actor_index = pending.pop(result_ref)
                fitnesses[result_index] = float(ray.get(result_ref))
                logger.debug(
                    "Candidate %d/%d done | actor=%d | fitness=%.4f",
                    result_index + 1,
                    len(candidate_model_params),
                    actor_index,
                    fitnesses[result_index],
                )

                if next_index < len(candidate_model_params):
                    # Refill the actor that became free, keeping dispatch bounded.
                    result_ref = self._actors[actor_index].evaluate.remote(
                        candidate_model_params[next_index],
                        seed=self.evaluation_seed,
                    )
                    pending[result_ref] = (next_index, actor_index)
                    next_index += 1
        except Exception as error:
            # A failed result must not leave unrelated candidates running after
            # the batch has already failed. Queued actor tasks are cancelled,
            # running tasks are drained, and the original error is preserved.
            self._cancel_pending_evaluations(pending, error)
            raise

        return fitnesses

    def __enter__(self) -> Self:
        """Return evaluator for context-manager use."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release evaluator-owned resources without shutting down Ray."""
        self.close()

    def close(self) -> None:
        """
        Stop actors, then release driver-held object-store references.

        Ray remains initialized for the caller.
        """
        try:
            self._release_actors()
        finally:
            self._data_refs = ()
            self._closed = True

    def _release_actors(self) -> None:
        """
        Stop actors and wait until their Ray CPU reservations return.

        The wait is bounded. If an actor already died before cleanup while
        unrelated Ray work keeps CPUs busy, the expected release never fully
        appears and a `RuntimeError` is raised after the timeout.
        """
        actors = self._actors
        self._actors = ()
        if not actors or not ray.is_initialized():
            return

        # Capture the current and maximum CPU capacity before killing actors.
        # The target below is based only on actors that are successfully killed,
        # without waiting for unrelated Ray tasks to finish.
        available_before = self._available_cpu_units()
        total_cpus = self._cluster_cpu_units()
        successful_kills = 0
        for actor in actors:
            try:
                ray.kill(actor)
            except (ActorDiedError, ActorHandleNotFoundError):
                continue
            successful_kills += 1

        # Handles from a previous Ray runtime cannot release resources in the
        # current runtime, so there is nothing to wait for in that case.
        if successful_kills == 0:
            return

        target_available = min(
            total_cpus,
            available_before + successful_kills * self.threads_per_worker,
        )

        # ray.kill() is asynchronous. Poll until the actor CPU reservations
        # return, but use a deadline so cleanup cannot wait indefinitely.
        deadline = time.monotonic() + _CPU_RELEASE_TIMEOUT_SECONDS
        while ray.is_initialized():
            if self._available_cpu_units() >= target_available:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Timed out waiting for Ray actor CPU resources to be released."
                )
            time.sleep(_CPU_RELEASE_POLL_INTERVAL_SECONDS)

    @staticmethod
    def _validate_resources(
        worker_count: int,
        threads_per_worker: int,
    ) -> None:
        """Validate caller-managed Ray state and requested CPU capacity."""
        if not ray.is_initialized():
            raise RuntimeError(
                "Ray must be initialized before creating RayTabNetBatchEvaluator."
            )
        if worker_count <= 0:
            raise ValueError("worker_count must be positive.")
        if threads_per_worker <= 0:
            raise ValueError("threads_per_worker must be positive.")

        # Each actor reserves `threads_per_worker` schedulable Ray CPU units.
        # These units usually map to logical CPUs, not guaranteed physical cores.
        requested_cpus = worker_count * threads_per_worker
        available_cpus = ray.available_resources().get("CPU", 0.0)  # type: ignore[no-untyped-call]
        if requested_cpus > available_cpus:
            raise ValueError(
                f"Requested {requested_cpus} CPUs, but Ray has "
                f"{available_cpus:g} CPUs available."
            )

    @staticmethod
    def _cancel_pending_evaluations(
        pending: dict[Any, tuple[int, int]],
        original_error: Exception,
    ) -> None:
        """Cancel queued actor work and drain running tasks after a failure."""
        pending_refs = tuple(pending)
        for result_ref in pending_refs:
            try:
                # `force=False` is required for Ray actor tasks. It cancels
                # queued work; a task already running must be drained below.
                ray.cancel(result_ref)
            except (RayError, ValueError) as cancellation_error:
                original_error.add_note(
                    f"Could not cancel pending Ray task: {cancellation_error!r}"
                )

        if not pending_refs:
            return

        try:
            ready, _ = ray.wait(
                list(pending_refs),
                num_returns=len(pending_refs),
            )
        except (RayError, ValueError) as drain_error:
            original_error.add_note(
                f"Could not drain pending Ray tasks: {drain_error!r}"
            )
            return

        for result_ref in ready:
            try:
                ray.get(result_ref)
            except RayError as pending_error:
                original_error.add_note(
                    f"Pending Ray task ended during failure cleanup: {pending_error!r}"
                )

    @staticmethod
    def _available_cpu_units() -> float:
        """Read currently available Ray CPU resource units."""
        return float(
            ray.available_resources().get("CPU", 0.0)  # type: ignore[no-untyped-call]
        )

    @staticmethod
    def _cluster_cpu_units() -> float:
        """Read total Ray CPU resource units."""
        return float(
            ray.cluster_resources().get("CPU", 0.0)  # type: ignore[no-untyped-call]
        )
