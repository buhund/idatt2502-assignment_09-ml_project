# config.py
import torch

# Model Configs
class Config:
    # PPO Hyperparameters
    learning_rate = 0.0003
    gamma = 0.99                  # Discount factor
    n_steps = 128                 # Steps per update
    ent_coef = 0.01               # Entropy coefficient to encourage exploration
    vf_coef = 0.5                 # Value function coefficient in loss calculation
    max_grad_norm = 0.5           # Gradient clipping
    gae_lambda = 0.95             # Generalized Advantage Estimation discount

    # Training Parameters
    total_timesteps = 1000000     # Total training steps
    eval_interval = 10000         # Evaluate model every eval_interval steps
    checkpoint_interval = 50000   # Save model every checkpoint_interval steps

    # Environment Settings
    frame_stack = 4               # Stack frames to include temporal context
    grayscale = True              # Use grayscale images for simplicity


# Environment config
WORLD = 1
STAGE = 1
ENV_VERSION = "v0"
ENV_NAME = f"SuperMarioBros-{WORLD}-{STAGE}-{ENV_VERSION}"
NUM_EPISODES = 10_000
RENDER_MODE = True # True for visual gameplay, False for no visuals.

# With video recording enabled, the simulations strike out at about 100 episodes. Set to False to run longer.
# TODO Make video not crash, record only every 1000? 5000?
ENABLE_VIDEO_RECORDING = False # True for video recoding (ffmpeg), False for no recording.

# Use GPU/Cuda if available. Else fallback to good 'ol CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Actor and Critic model path
ACTOR_PATH = "src/model/actor.pth"
CRITIC_PATH = "src/model/critic.pth"
""""
    An Actor that controls how our agent behaves (policy-based method).
    A Critic that measures how good the action taken is (value-based method).
"""
