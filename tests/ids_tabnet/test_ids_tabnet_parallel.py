"""Tests for CPU Ray TabNet batch evaluation."""

import random
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import numpy as np
import ray
from ray.exceptions import ActorHandleNotFoundError, RayTaskError

from ids_tabnet.evaluator import DeviceName, TabNetDataPartitions, TabNetEvaluator
from ids_tabnet.parallel import RayTabNetBatchEvaluator
from ids_tabnet.params import ModelParams

GENES = {
    "n_da": 8,
    "n_steps": 3,
    "gamma": 1.5,
    "lambda_sparse": 1e-5,
    "lr": 0.02,
    "batch_size": 128,
}


class FakeRef:
    """Hashable local stand-in for a Ray object reference."""

    def __init__(self, value: float):
        self.value = value


class FakeRemoteMethod:
    """Record actor submissions and return deterministic fake references."""

    def __init__(self, actor_index: int, submissions: list[tuple[int, int, int]]):
        self.actor_index = actor_index
        self.submissions = submissions
        self.next_ref = 0

    def remote(self, model_params: ModelParams, seed: int) -> FakeRef:
        n_da = int(model_params.n_da)
        self.submissions.append((self.actor_index, n_da, seed))
        ref = FakeRef(float(n_da))
        self.next_ref += 1
        return ref


class FakeActor:
    """Fake actor exposing the remote method used by the evaluator."""

    def __init__(self, actor_index: int, submissions: list[tuple[int, int, int]]):
        self.evaluate = FakeRemoteMethod(actor_index, submissions)


class RayTabNetResourceTests(unittest.TestCase):
    """Verify caller-managed Ray and CPU resource validation."""

    def test_uninitialized_ray_is_rejected_without_lifecycle_changes(self) -> None:
        """Construction does not initialize or shut down Ray implicitly."""
        with (
            patch("ids_tabnet.parallel.ray.is_initialized", return_value=False),
            patch("ids_tabnet.parallel.ray.init") as ray_init,
            patch("ids_tabnet.parallel.ray.shutdown") as ray_shutdown,
            self.assertRaisesRegex(
                RuntimeError,
                "Ray must be initialized before creating",
            ),
        ):
            RayTabNetBatchEvaluator(
                "dataset.csv",
                worker_count=1,
                threads_per_worker=1,
            )

        ray_init.assert_not_called()
        ray_shutdown.assert_not_called()

    def test_non_positive_worker_count_is_rejected(self) -> None:
        """Zero or negative worker counts fail before dataset preparation."""
        for worker_count in (0, -1):
            with (
                self.subTest(worker_count=worker_count),
                patch("ids_tabnet.parallel.ray.is_initialized", return_value=True),
                patch("ids_tabnet.parallel.prepare_tabnet_data") as prepare_data,
                self.assertRaisesRegex(
                    ValueError,
                    "worker_count must be positive",
                ),
            ):
                RayTabNetBatchEvaluator(
                    "dataset.csv",
                    worker_count=worker_count,
                    threads_per_worker=1,
                )

            prepare_data.assert_not_called()


class RayTabNetLifecycleTests(unittest.TestCase):
    """Verify local actor reuse and caller-owned Ray lifecycle."""

    def test_workers_share_one_prepared_dataset_and_close_keeps_ray_running(
        self,
    ) -> None:
        """Multiple actors use one preparation and close without Ray shutdown."""
        if ray.is_initialized():
            self.skipTest("Test requires ownership of a fresh local Ray runtime.")

        partitions = TabNetDataPartitions(
            X_train=np.ones((6, 2)),
            X_val=np.ones((2, 2)),
            X_test=np.ones((2, 2)),
            y_train=np.array([0, 1, 0, 1, 0, 1]),
            y_val=np.array([0, 1]),
            y_test=np.array([0, 1]),
        )
        ray.init(
            num_cpus=2,
            include_dashboard=False,
            log_to_driver=False,
        )
        evaluator: RayTabNetBatchEvaluator | None = None
        recreated: RayTabNetBatchEvaluator | None = None
        try:
            with patch(
                "ids_tabnet.parallel.prepare_tabnet_data",
                return_value=partitions,
            ) as prepare_data:
                evaluator = RayTabNetBatchEvaluator(
                    "dataset.csv",
                    worker_count=2,
                    threads_per_worker=1,
                )

            signatures = ray.get(
                [actor.partition_signature.remote() for actor in evaluator._actors],
            )
            prepare_data.assert_called_once_with("dataset.csv")
            self.assertEqual(signatures, [signatures[0], signatures[0]])

            evaluator.close()
            with patch(
                "ids_tabnet.parallel.prepare_tabnet_data",
                return_value=partitions,
            ):
                recreated = RayTabNetBatchEvaluator(
                    "dataset.csv",
                    worker_count=2,
                    threads_per_worker=1,
                )
            recreated.close()
            self.assertTrue(ray.is_initialized())
        finally:
            if evaluator is not None:
                evaluator.close()
            if recreated is not None:
                recreated.close()
            ray.shutdown()

    def test_close_after_ray_shutdown_is_safe(self) -> None:
        """Cleanup remains safe when the caller shuts down Ray first."""
        if ray.is_initialized():
            self.skipTest("Test requires ownership of a fresh local Ray runtime.")

        partitions = TabNetDataPartitions(
            X_train=np.ones((6, 2)),
            X_val=np.ones((2, 2)),
            X_test=np.ones((2, 2)),
            y_train=np.array([0, 1, 0, 1, 0, 1]),
            y_val=np.array([0, 1]),
            y_test=np.array([0, 1]),
        )
        ray.init(
            num_cpus=1,
            include_dashboard=False,
            log_to_driver=False,
        )
        with patch(
            "ids_tabnet.parallel.prepare_tabnet_data",
            return_value=partitions,
        ):
            evaluator = RayTabNetBatchEvaluator(
                "dataset.csv",
                worker_count=1,
                threads_per_worker=1,
            )

        ray.shutdown()
        evaluator.close()

        self.assertTrue(evaluator._closed)
        self.assertEqual(evaluator._actors, ())
        self.assertEqual(evaluator._data_refs, ())

    def test_close_with_stale_actor_handles_does_not_wait(self) -> None:
        """Stale handles from an old Ray runtime release no current resources."""
        evaluator = object.__new__(RayTabNetBatchEvaluator)
        evaluator._actors = (object(), object())
        evaluator.worker_count = 2
        evaluator.threads_per_worker = 1

        with (
            patch("ids_tabnet.parallel.ray.is_initialized", return_value=True),
            patch(
                "ids_tabnet.parallel.ray.available_resources",
                return_value={"CPU": 1.0},
            ) as available_resources,
            patch(
                "ids_tabnet.parallel.ray.cluster_resources",
                return_value={"CPU": 2.0},
            ),
            patch(
                "ids_tabnet.parallel.ray.kill",
                side_effect=ActorHandleNotFoundError("stale actor handle"),
            ),
        ):
            evaluator._release_actors()

        available_resources.assert_called_once_with()
        self.assertEqual(evaluator._actors, ())


class RayTabNetDispatchTests(unittest.TestCase):
    """Verify ordered, bounded, deterministic batch dispatch."""

    def make_evaluator(
        self,
        actor_count: int = 2,
    ) -> tuple[RayTabNetBatchEvaluator, list[tuple[int, int, int]]]:
        """Build an evaluator around local fake actors."""
        submissions: list[tuple[int, int, int]] = []
        evaluator = object.__new__(RayTabNetBatchEvaluator)
        evaluator._actors = tuple(
            FakeActor(actor_index, submissions) for actor_index in range(actor_count)
        )
        evaluator._data_refs = ()
        evaluator.evaluation_seed = 73
        evaluator._closed = False
        return evaluator, submissions

    def test_empty_batch_skips_remote_evaluation(self) -> None:
        """An empty batch returns without submitting any actor work."""
        evaluator, submissions = self.make_evaluator()

        self.assertEqual(evaluator([]), [])
        self.assertEqual(submissions, [])

    def test_dispatch_preserves_input_order_and_reuses_free_actor(self) -> None:
        """Out-of-order completion still returns aligned fitness values."""
        evaluator, submissions = self.make_evaluator()
        genes = [{**GENES, "n_da": n_da} for n_da in (8, 16, 24, 32)]

        def complete_latest(
            refs: list[FakeRef],
            num_returns: int,
        ) -> tuple[list[FakeRef], list[FakeRef]]:
            del num_returns
            return [refs[-1]], refs[:-1]

        with (
            patch(
                "ids_tabnet.parallel.ray.wait",
                side_effect=complete_latest,
            ),
            patch(
                "ids_tabnet.parallel.ray.get",
                side_effect=lambda ref: ref.value,
            ),
        ):
            fitnesses = evaluator(genes)

        self.assertEqual(fitnesses, [8.0, 16.0, 24.0, 32.0])
        self.assertEqual(
            submissions,
            [
                (0, 8, 73),
                (1, 16, 73),
                (1, 24, 73),
                (1, 32, 73),
            ],
        )

    def test_fixed_seed_does_not_change_driver_random_state(self) -> None:
        """Batch dispatch forwards one seed without consuming driver randomness."""
        evaluator, submissions = self.make_evaluator(actor_count=1)
        state = random.getstate()

        def complete_first(
            refs: list[FakeRef],
            num_returns: int,
        ) -> tuple[list[FakeRef], list[FakeRef]]:
            del num_returns
            return [refs[0]], refs[1:]

        with (
            patch(
                "ids_tabnet.parallel.ray.wait",
                side_effect=complete_first,
            ),
            patch(
                "ids_tabnet.parallel.ray.get",
                side_effect=lambda ref: ref.value,
            ),
        ):
            self.assertEqual(evaluator([GENES]), [8.0])

        self.assertEqual(random.getstate(), state)
        self.assertEqual(submissions, [(0, 8, 73)])

    def test_worker_failure_propagates_without_partial_result(self) -> None:
        """A remote failure raises instead of returning completed values."""
        evaluator, _ = self.make_evaluator(actor_count=1)

        def complete_first(
            refs: list[FakeRef],
            num_returns: int,
        ) -> tuple[list[FakeRef], list[FakeRef]]:
            del num_returns
            return [refs[0]], refs[1:]

        with (
            patch(
                "ids_tabnet.parallel.ray.wait",
                side_effect=complete_first,
            ),
            patch(
                "ids_tabnet.parallel.ray.get",
                side_effect=[RuntimeError("worker failed"), None],
            ),
            self.assertRaisesRegex(RuntimeError, "worker failed"),
        ):
            evaluator([GENES])

    def test_worker_failure_cancels_remaining_submissions(self) -> None:
        """A failed batch cancels candidates still running on other actors."""
        evaluator, _ = self.make_evaluator(actor_count=2)
        remaining_ref: FakeRef | None = None

        def complete_first(
            refs: list[FakeRef],
            num_returns: int,
        ) -> tuple[list[FakeRef], list[FakeRef]]:
            nonlocal remaining_ref
            del num_returns
            if len(refs) == 1:
                return [refs[0]], []
            remaining_ref = refs[1]
            return [refs[0]], refs[1:]

        with (
            patch(
                "ids_tabnet.parallel.ray.wait",
                side_effect=complete_first,
            ),
            patch(
                "ids_tabnet.parallel.ray.get",
                side_effect=RuntimeError("worker failed"),
            ),
            patch("ids_tabnet.parallel.ray.cancel") as cancel,
            self.assertRaisesRegex(RuntimeError, "worker failed"),
        ):
            evaluator([GENES, {**GENES, "n_da": 16}])

        self.assertIsNotNone(remaining_ref)
        cancel.assert_called_once_with(remaining_ref)

    def test_worker_failure_preserves_error_if_cancellation_fails(self) -> None:
        """Cleanup errors do not replace the original worker failure."""
        evaluator, _ = self.make_evaluator(actor_count=2)

        def complete_first(
            refs: list[FakeRef],
            num_returns: int,
        ) -> tuple[list[FakeRef], list[FakeRef]]:
            del num_returns
            if len(refs) == 1:
                return [refs[0]], []
            return [refs[0]], refs[1:]

        with (
            patch(
                "ids_tabnet.parallel.ray.wait",
                side_effect=complete_first,
            ),
            patch(
                "ids_tabnet.parallel.ray.get",
                side_effect=[RuntimeError("worker failed"), None],
            ),
            patch(
                "ids_tabnet.parallel.ray.cancel",
                side_effect=ValueError("cancel failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "worker failed"),
        ):
            evaluator([GENES, {**GENES, "n_da": 16}])

    def test_gene_conversion_precedes_remote_submission(self) -> None:
        """Invalid genes fail before any actor receives work."""
        evaluator, submissions = self.make_evaluator()
        invalid_genes = {key: value for key, value in GENES.items() if key != "lr"}

        with self.assertRaises(KeyError):
            evaluator([invalid_genes])

        self.assertEqual(submissions, [])

    def test_non_positive_thread_count_is_rejected(self) -> None:
        """Zero or negative thread counts fail before dataset preparation."""
        for threads_per_worker in (0, -1):
            with (
                self.subTest(threads_per_worker=threads_per_worker),
                patch("ids_tabnet.parallel.ray.is_initialized", return_value=True),
                patch("ids_tabnet.parallel.prepare_tabnet_data") as prepare_data,
                self.assertRaisesRegex(
                    ValueError,
                    "threads_per_worker must be positive",
                ),
            ):
                RayTabNetBatchEvaluator(
                    "dataset.csv",
                    worker_count=1,
                    threads_per_worker=threads_per_worker,
                )

            prepare_data.assert_not_called()

    def test_cpu_request_above_available_capacity_is_rejected(self) -> None:
        """Configurations above initialized Ray capacity fail before loading."""
        with (
            patch("ids_tabnet.parallel.ray.is_initialized", return_value=True),
            patch(
                "ids_tabnet.parallel.ray.available_resources",
                return_value={"CPU": 3.0},
            ),
            patch("ids_tabnet.parallel.prepare_tabnet_data") as prepare_data,
            self.assertRaisesRegex(ValueError, "Requested 4 CPUs"),
        ):
            RayTabNetBatchEvaluator(
                "dataset.csv",
                worker_count=2,
                threads_per_worker=2,
            )

        prepare_data.assert_not_called()


class RayTabNetIntegrationTests(unittest.TestCase):
    """Verify real CPU actors run tiny TabNet evaluations on shared arrays."""

    def test_real_tabnet_evaluation_matches_serial_and_uses_read_only_arrays(
        self,
    ) -> None:
        """Real actor training preserves order and serial fitness values."""
        if ray.is_initialized():
            self.skipTest("Test requires ownership of a fresh local Ray runtime.")

        rng = np.random.default_rng(7)
        X = rng.normal(size=(20, 3)).astype(np.float32)
        y = np.array([0, 1] * 10, dtype=np.int64)
        parallel_partitions = TabNetDataPartitions(
            X_train=X[:12],
            X_val=X[12:16],
            X_test=X[16:],
            y_train=y[:12],
            y_val=y[12:16],
            y_test=y[16:],
        )
        serial_partitions = TabNetDataPartitions(
            X_train=X[:12].copy(),
            X_val=X[12:16].copy(),
            X_test=X[16:].copy(),
            y_train=y[:12].copy(),
            y_val=y[12:16].copy(),
            y_test=y[16:].copy(),
        )
        genes = [
            {
                "n_da": 2,
                "n_steps": 1,
                "gamma": 1.5,
                "lambda_sparse": 1e-5,
                "lr": 0.02,
                "batch_size": 4,
            },
            {
                "n_da": 4,
                "n_steps": 1,
                "gamma": 1.5,
                "lambda_sparse": 1e-5,
                "lr": 0.02,
                "batch_size": 4,
            },
        ]

        ray.init(
            num_cpus=2,
            include_dashboard=False,
            log_to_driver=False,
        )
        evaluator: RayTabNetBatchEvaluator | None = None
        try:
            with patch(
                "ids_tabnet.parallel.prepare_tabnet_data",
                return_value=parallel_partitions,
            ):
                evaluator = RayTabNetBatchEvaluator(
                    "dataset.csv",
                    worker_count=2,
                    threads_per_worker=1,
                    evaluation_seed=11,
                )

            read_only = ray.get(
                [
                    actor.partitions_are_read_only.remote()
                    for actor in evaluator._actors
                ],
            )
            self.assertEqual(
                read_only,
                [(True, True, True, True, True, True)] * 2,
            )

            with redirect_stdout(StringIO()):
                parallel_fitnesses = evaluator(genes)
                serial_evaluator = TabNetEvaluator.from_partitions(
                    serial_partitions,
                    device_name=DeviceName.CPU,
                )
                serial_fitnesses = [
                    serial_evaluator.evaluate(
                        ModelParams(
                            n_da=int(genes_for_candidate["n_da"]),
                            n_steps=int(genes_for_candidate["n_steps"]),
                            gamma=float(genes_for_candidate["gamma"]),
                            lambda_sparse=float(genes_for_candidate["lambda_sparse"]),
                            lr=float(genes_for_candidate["lr"]),
                            batch_size=int(genes_for_candidate["batch_size"]),
                        ),
                        seed=11,
                    )
                    for genes_for_candidate in genes
                ]

            np.testing.assert_allclose(parallel_fitnesses, serial_fitnesses)
        finally:
            if evaluator is not None:
                evaluator.close()
            ray.shutdown()

    def test_real_worker_failure_preserves_error_and_allows_retry(self) -> None:
        """Actor failures remain visible and do not poison later batches."""
        if ray.is_initialized():
            self.skipTest("Test requires ownership of a fresh local Ray runtime.")

        rng = np.random.default_rng(11)
        X = rng.normal(size=(20, 3)).astype(np.float32)
        y = np.array([0, 1] * 10, dtype=np.int64)
        partitions = TabNetDataPartitions(
            X_train=X[:12],
            X_val=X[12:16],
            X_test=X[16:],
            y_train=y[:12],
            y_val=y[12:16],
            y_test=y[16:],
        )
        valid_genes = {
            "n_da": 2,
            "n_steps": 1,
            "gamma": 1.5,
            "lambda_sparse": 1e-5,
            "lr": 0.02,
            "batch_size": 4,
        }
        invalid_genes = {**valid_genes, "batch_size": 0}

        ray.init(
            num_cpus=2,
            include_dashboard=False,
            log_to_driver=False,
        )
        evaluator: RayTabNetBatchEvaluator | None = None
        try:
            with patch(
                "ids_tabnet.parallel.prepare_tabnet_data",
                return_value=partitions,
            ):
                evaluator = RayTabNetBatchEvaluator(
                    "dataset.csv",
                    worker_count=2,
                    threads_per_worker=1,
                    evaluation_seed=11,
                )

            with self.assertRaises(RayTaskError) as raised:
                evaluator([invalid_genes, valid_genes, valid_genes])

            self.assertIsInstance(raised.exception, RayTaskError)
            retry_fitnesses = evaluator([valid_genes])
            self.assertEqual(len(retry_fitnesses), 1)
            self.assertIsInstance(retry_fitnesses[0], float)
        finally:
            if evaluator is not None:
                evaluator.close()
            ray.shutdown()
