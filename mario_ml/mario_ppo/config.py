# config.py
import torch

# Environment config
WORLD = 1
STAGE = 1
ENV_VERSION = "v0"
ENV_NAME = f"SuperMarioBros-{WORLD}-{STAGE}-{ENV_VERSION}"
RENDER_MODE = True # True for visual gameplay, False for no visuals.

# Number of episodes to run
NUM_EPISODES = 10
NUM_TEST_EPISODES = 1_000

# With video recording enabled, the simulations strike out at about 100 episodes. Set to False to run longer.
# TODO Make video not crash, record only every 1000 episodes?
# TODO Or just remove recording entirely
ENABLE_VIDEO_RECORDING = False # True for video recoding (ffmpeg), False for no recording.
VIDE_RECORDING_INTERVAL_TRAINING = 2000 # Record every 2000 episode of 30_000 = 15 videos
VIDE_RECORDING_INTERVAL_TESTING = 100 # Record every 100 episode of 1_000 = 10 videos
VIDEO_OUTPUT_PATH = "logs/video"

# Use GPU/Cuda if available. Else fallback to good 'ol CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File path to Actor and Critic model weights
ACTOR_PATH = "src/weights/actor.pth" # Actor controls how our agent behaves (policy-based method).
CRITIC_PATH = "src/weights/critic.pth" # Critic measures how good the action taken is (value-based method).
WEIGHTS_PATH = "src/weights/"

# Checkpoint settings
CHECKPOINT_PATH = "src/checkpoints"
CHECKPOINT_INTERVAL = 5  # Save every 100 episodes

# Model Configs
class PPOConfig:
    # PPO Hyperparameters
    learning_rate = 0.0003          # 3e-4
    epsilon = 0.2                   # Aka Clip Rate. Standard choice for PPO epsilon constant
    gamma = 0.99                    # Discount factor
    n_steps = 128                   # Steps per update
    entropy_coef = 0.01                 # Entropy coefficient to encourage exploration
    vf_coef = 0.5                   # Value function coefficient in loss calculation
    max_grad_norm = 0.5             # Gradient clipping
    gae_lambda = 0.95               # Generalized Advantage Estimation discount

    # Training Parameters
    total_timesteps = 1000000       # Total training steps
    eval_interval = 10000           # Evaluate weights every eval_interval steps
    checkpoint_interval = 50000     # Save weights every checkpoint_interval steps

    # Environment Settings
    frame_stack = 4                 # Stack frames to include temporal context
    grayscale = True                # Use grayscale images for simplicity

