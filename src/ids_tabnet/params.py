"""Parameter container for TabNet model configuration."""


class ModelParams:
    """
    Hyperparameter container for a TabNet model configuration.

    `virtual_batch_size` is derived automatically from `batch_size` to satisfy
    Ghost Batch Normalization constraints (≥128 → 128, else `batch_size // 2`).
    """

    def __init__(
        self,
        n_da: int,
        n_steps: int,
        gamma: float,
        lambda_sparse: float,
        lr: float,
        batch_size: int,
    ):
        """
        Initializes the ModelParams with the given parameters for the TabNet model.

        Args:
            n_da: Width of the decision prediction layer (`n_d`) and attention embedding (`n_a`).
                Bigger values give more capacity with higher overfitting risk. Typical range is 8 to 64.
                According to the paper, `n_d = n_a` is usually a good choice.
            n_steps: Number of steps in the TabNet architecture (usually between 3 and 10).
            gamma: Coefficient for feature reusage in masks. Values close to 1 make mask selection less
                correlated between layers. Valid range is 1.0 to 2.0.
            lambda_sparse: Extra sparsity loss coefficient. Higher values enforce sparser feature selection.
            lr: Initial optimizer learning rate. A large initial value with decay is recommended in the paper.
            batch_size: Number of samples per training batch.

        Raises:
            ValueError: If `gamma` is not between 1.0 and 2.0.
        """
        if gamma < 1.0 or gamma > 2.0:
            raise ValueError("Gamma should be between 1.0 and 2.0.")

        self.n_da = n_da
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.lr = lr
        self.batch_size = batch_size
        # Safety check for Ghost Batch Normalization
        self.virtual_batch_size = 128 if batch_size >= 128 else batch_size // 2
