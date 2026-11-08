"""Tests for TabNet model parameter validation."""

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
                )

                self.assertEqual(params.n_da, 16)
                self.assertEqual(params.n_steps, 5)
                self.assertEqual(params.gamma, gamma)
                self.assertEqual(params.lambda_sparse, 1e-5)
                self.assertEqual(params.lr, 0.02)

    def test_rejects_gamma_outside_inclusive_bounds(self) -> None:
        """Values outside the declared gamma interval raise the documented error."""
        for gamma in (0.99, 2.01):
            with (
                self.subTest(gamma=gamma),
                self.assertRaisesRegex(
                    ValueError,
                    r"^Gamma should be between 1\.0 and 2\.0\.$",
                ),
            ):
                ModelParams(
                    n_da=16,
                    n_steps=5,
                    gamma=gamma,
                    lambda_sparse=1e-5,
                    lr=0.02,
                )
