"""
Super Mario Bros. with Machine Learning
Algorithm: Policy Gradient Method, Proximal Policy Optimization (PPO)
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import numpy as np
import cv2
import torch
import ale_py
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation


# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v3')

# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)

# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])

# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

# This is the AI model
model = PPO('CnnPolicy', env, verbose=1, learning_rate=1e-5, n_steps=512)

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=10000)


# Start the game
state = env.reset()
# Loop through the game
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()