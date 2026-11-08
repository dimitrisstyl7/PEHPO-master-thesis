"""Parameter container for TabNet model configuration."""

__pdoc__ = {
    "ModelParams.__init__": False,
}


class ModelParams:
    r"""
    Hyperparameter container for a TabNet model configuration.

    Args:
        n_da: Width of the decision prediction layer ($n_d$) and attention embedding ($n_a$).
            Bigger values give more capacity with higher overfitting risk. Typical range is 8 to 64.
            According to the paper, $n_d = n_a$ is usually a good choice.
        n_steps: Number of steps in the TabNet architecture (usually between 3 and 10).
        gamma: Coefficient for feature reusage in masks. Values close to 1 make mask selection less
            correlated between layers. Valid range is 1.0 to 2.0.
        lambda_sparse: Extra sparsity loss coefficient. Higher values enforce sparser feature selection.
        lr: Initial optimizer learning rate. A large initial value with decay is recommended in the paper.

    Raises:
        ValueError: If `gamma` is not between 1.0 and 2.0.
    """

    def __init__(
        self,
        n_da: int,
        n_steps: int,
        gamma: float,
        lambda_sparse: float,
        lr: float,
    ):
        if gamma < 1.0 or gamma > 2.0:
            raise ValueError("Gamma should be between 1.0 and 2.0.")

        self.n_da = n_da
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.lr = lr
