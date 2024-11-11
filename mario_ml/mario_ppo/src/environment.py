# src/environment.py
# TODO Probably make this into two classes: environment and wrapper

import collections
import os
import subprocess as sp

import cv2
import gym
import gym_super_mario_bros
import numpy as np
import torch
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from config import ENV_NAME, RENDER_MODE
from config import ENABLE_VIDEO_RECORDING

class ActionRepeat(gym.Wrapper):
    """Repeats the action for a specified number of frames (default = 4)."""

    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.buffer = collections.deque(maxlen=2)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.buffer.append(observation)
            if done:
                break
        max_frame = np.max(np.stack(self.buffer), axis=0)
        return max_frame, total_reward, done, info


class ResizeAndGrayscale(gym.ObservationWrapper):
    """Resizes and converts the observation to grayscale, outputting 84x84 images."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, observation):
        return ResizeAndGrayscale.convert(observation)

    @staticmethod
    def convert(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            raise ValueError("Unknown resolution.")
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        cropped_img = resized_img[18:102, :]
        processed_frame = np.reshape(cropped_img, [84, 84, 1])
        return processed_frame.astype(np.uint8)


class ConvertToTensor(gym.ObservationWrapper):
    """Converts observations to tensor format suitable for PyTorch."""

    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ObservationBuffer(gym.ObservationWrapper):
    """Maintains a sliding window of the last n observations."""

    def __init__(self, env, n_steps, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype,
        )

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class NormalizePixels(gym.ObservationWrapper):
    """Normalizes pixel values in the observation to a range of [0, 1]."""

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0




class Monitor:
    """Handles video recording for the environment using ffmpeg."""

    def __init__(self, width, height, saved_path="output"):
        if not ENABLE_VIDEO_RECORDING:
            self.pipe = None  # No video recording if disabled, i.e. config/ENABLE_VIDEO_RECORDING = False
            return

        ffmpeg_path = "/usr/bin/ffmpeg"
        if not os.path.isfile(ffmpeg_path):
            raise RuntimeError("ffmpeg not found. Please ensure ffmpeg is installed.")
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        output_file = os.path.join(saved_path, f"{ENV_NAME}.mp4")
        self.command = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            "60",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "mpeg4",
            output_file,
        ]

        try:
            self.pipe = sp.Popen(
                self.command,
                stdin=sp.PIPE,
                stderr=sp.PIPE,
                executable=ffmpeg_path,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"ffmpeg not found. Please check if ffmpeg is in the specified directory: {ffmpeg_path}."
            ) from e

    def record(self, image_array):
        if not self.pipe:
            return  # Skip recording if video recording is disabled

        try:
            self.pipe.stdin.write(image_array.tobytes())
        except BrokenPipeError as e:
            error_output = self.pipe.stderr.read().decode()
            print("ffmpeg error:", error_output)
            raise RuntimeError(
                "ffmpeg terminated unexpectedly. Check the ffmpeg command and input frames."
            ) from e

    def close(self):
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe.wait()



class CustomReward(gym.RewardWrapper):
    """A custom reward wrapper that modifies rewards based on game score and level completion."""

    def __init__(self, env=None, monitor=None):
        super().__init__(env)
        self.monitor = monitor
        self.curr_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)

        score_diff = info.get("score", 0) - self.curr_score
        reward += score_diff / 40.0
        self.curr_score = info.get("score", 0)

        if done:
            if info.get("flag_get", False):
                reward += 50  # Reward for completing the level
            else:
                reward -= 50  # Penalty for failing the level
        return state, reward, done, info

    def reset(self, **kwargs):
        self.curr_score = 0
        return self.env.reset(**kwargs)


def create_env(map=ENV_NAME, action_repeat=4, output_path=None):
    """Sets up the Super Mario Bros environment with customized wrappers."""
    env = JoypadSpace(gym_super_mario_bros.make(map), SIMPLE_MOVEMENT)
    if output_path is not None:
        monitor = Monitor(width=256, height=240, saved_path=output_path)
    else:
        monitor = None
    env = CustomReward(env, monitor=monitor)
    env = ActionRepeat(env, action_repeat)
    env = ResizeAndGrayscale(env)
    env = ConvertToTensor(env)
    env = ObservationBuffer(env, 4)
    env = NormalizePixels(env)
    return env




