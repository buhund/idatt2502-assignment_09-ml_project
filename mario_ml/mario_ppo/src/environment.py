# src/environment.py

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from config import ENV_NAME, ENABLE_VIDEO_RECORDING, VIDEO_OUTPUT_PATH
from src.utils.video_recorder import VideoRecorder
from src.wrappers import (ActionRepeat, ResizeAndGrayscale,
                          ConvertToTensor, ObservationBuffer,
                          NormalizePixels, CustomReward)


def create_env(map=ENV_NAME, action_repeat=4, output_path=None):
    """Sets up the Super Mario Bros environment with customized wrappers."""
    env = JoypadSpace(gym_super_mario_bros.make(map), SIMPLE_MOVEMENT)

    # Conditionally create the VideoRecorder based on ENABLE_VIDEO_RECORDING
    video_recorder = None
    if ENABLE_VIDEO_RECORDING:
        video_recorder = VideoRecorder(width=256, height=240, saved_path=VIDEO_OUTPUT_PATH)

    env = CustomReward(env, video_recorder=video_recorder)
    env = ActionRepeat(env, action_repeat)
    env = ResizeAndGrayscale(env)
    env = ConvertToTensor(env)
    env = ObservationBuffer(env, 4)
    env = NormalizePixels(env)
    return env




