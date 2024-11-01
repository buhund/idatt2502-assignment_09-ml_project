"""
Super Mario Bros. with Machine Learning
Algorithm: Policy Gradient Method PPO
"""

import gymnasium as gym
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym_super_mario_bros import make
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import numpy as np
import cv2
import torch
import ale_py


"""
Wrapper for the environment to preprocess frames. This makes the RL algorithms perform better.
- Resize and greyscale frames to reduce computational load.
- Stack frmes to provide tempoeral information accross frames.
- Normalize observations, i.e. scalin the pixel values for stability.
"""
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        # Resize and grayscale the frame
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(obs, axis=-1)

# Initialize the environment
def make_mario_env():
    env = make("SuperMarioBros-1-1-v3")
    env = PreprocessFrame(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_mario_env()

# Initialize PPO with CnnPolicy
model = PPO("CnnPolicy", env, verbose=1, learning_rate=1e-4, n_steps=128, batch_size=64, n_epochs=10)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_super_mario_bros")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()