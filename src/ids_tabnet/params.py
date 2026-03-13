class ModelParams:
    """
    A class to hold the hyperparameters for the TabNet model.
    It automatically calculates the 'virtual_batch_size' based on the provided 'batch_size'.
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
        :param n_da: Width of the decision prediction layer (n_d) and attention embedding (n_a).
            Bigger values give more capacity to the model with the risk of overfitting.
            Values typically range from 8 to 64. According to the paper, n_d=n_a is usually a good choice.
        :param n_steps: Number of steps in the TabNet architecture (usually between 3 and 10).
        :param gamma: Coefficient for feature reusage in the masks. A value close to 1 will make mask
            selection least correlated between layers. Values range from 1.0 to 2.0.
        :param lambda_sparse: Extra sparsity loss coefficient. The bigger this coefficient is,
            the sparser the model will be in terms of feature selection. Reducing this value could help
            depending on the difficulty of the problem.
        :param lr: Initial learning rate used for the optimizer. A large initial learning rate with decay
            is recommended as per the original paper.
        :param batch_size: Number of samples per batch used during training.
        :raises ValueError: If gamma is not between 1.0 and 2.0.
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
