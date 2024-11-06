# config.py
import torch


# Environment config
#ENV_NAME = 'SuperMarioBros-v0'
WORLD = 1
STAGE = 1
ENV_VERSION = "v0"
ENV_NAME = f"SuperMarioBros-{WORLD}-{STAGE}-{ENV_VERSION}"
NUM_EPISODES = 1_000
RENDER_MODE = True # True for visual gameplay, False for no visuals.

# With video recording enabled, the simulations strike out at about 100 episodes. Set to False to run longer.
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
