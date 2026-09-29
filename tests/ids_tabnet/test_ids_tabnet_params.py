"""Tests for TabNet model parameter validation and derived values."""

import unittest

from ids_tabnet.params import ModelParams


class ModelParamsTests(unittest.TestCase):
    """Verify the parameter container's public contracts."""

    def test_preserves_hyperparameters_and_accepts_gamma_endpoints(self) -> None:
        """Supplied values remain unchanged, including inclusive gamma bounds."""
        for gamma in (1.0, 2.0):
            with self.subTest(gamma=gamma):
                params = ModelParams(
                    n_da=16,
                    n_steps=5,
                    gamma=gamma,
                    lambda_sparse=1e-5,
                    lr=0.02,
                    batch_size=256,
                )

                self.assertEqual(params.n_da, 16)
                self.assertEqual(params.n_steps, 5)
                self.assertEqual(params.gamma, gamma)
                self.assertEqual(params.lambda_sparse, 1e-5)
                self.assertEqual(params.lr, 0.02)
                self.assertEqual(params.batch_size, 256)

    def test_rejects_gamma_outside_inclusive_bounds(self) -> None:
        """Values outside the declared gamma interval raise the documented error."""
        for gamma in (0.99, 2.01):
            with self.subTest(gamma=gamma), self.assertRaisesRegex(
                ValueError,
                r"^Gamma should be between 1\.0 and 2\.0\.$",
            ):
                ModelParams(
                    n_da=16,
                    n_steps=5,
                    gamma=gamma,
                    lambda_sparse=1e-5,
                    lr=0.02,
                    batch_size=256,
                )

    def test_derives_virtual_batch_size_at_ghost_batch_boundary(self) -> None:
        """Batch sizes below 128 use floor-half; larger sizes use 128."""
        expected_values = {
            64: 32,
            127: 63,
            128: 128,
            256: 128,
        }

        for batch_size, expected_virtual_batch_size in expected_values.items():
            with self.subTest(batch_size=batch_size):
                params = ModelParams(
                    n_da=16,
                    n_steps=5,
                    gamma=1.5,
                    lambda_sparse=1e-5,
                    lr=0.02,
                    batch_size=batch_size,
                )

                self.assertEqual(
                    params.virtual_batch_size,
                    expected_virtual_batch_size,
                )
