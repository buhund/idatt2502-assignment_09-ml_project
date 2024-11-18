class PPOHyperparameters:

    def __init__(
        self,
        # Algorithm prameters
        learning_rate = 0.0001,         # Default: 3e-4
        min_learning_rate = 0.000_001,  # Minimum agent learning rate
        clip_rate = 0.2,                # Clip rate = epsilon value. Standard choice for PPO epsilon constant is 0.2
        gamma = 0.90,                   # Discount factor. Default = 0.90
        n_steps = 128,                  # Steps per update. Default = 128
        entropy_coef = 0.01,            # Entropy coefficient to encourage exploration. Default = 0.01
        vf_coef = 0.5,                  # Value function coefficient in loss calculation. Default = 0.5
        max_grad_norm = 0.5,            # Gradient clipping. Default = 0.5
        gae_lambda = 0.95,              # Generalized Advantage Estimation discount. Default = 0.95
        number_minibatches = 5,         #
        batch_size = 16,                #
        num_epochs = 10,                #

        # Training parameters
        timesteps_per_batch = 2_000,    #
        max_timesteps_episode = 1_000,  #
        total_timesteps = 3_000_000,    # Total training steps, default 1_000_000
        eval_interval = 10_000,         # Evaluate weights every eval_interval steps

    ):
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda

        self.number_minibatches = number_minibatches
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_episode = max_timesteps_episode
        self.total_timesteps = total_timesteps
        self.eval_interval = eval_interval


