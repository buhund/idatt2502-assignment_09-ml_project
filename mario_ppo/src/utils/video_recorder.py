# src/utils/video_recorder.py

import os
import subprocess as sp
import numpy as np
from config import ENV_NAME, ENABLE_VIDEO_RECORDING


class VideoRecorder:
    """Helper method for video recording while running the environment using ffmpeg."""

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
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", "60",
            "-i", "-",
            "-an",
            "-vcodec", "mpeg4",
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
                f"ffmpeg not found. Please check for ffmpeg installation in the specified directory: {ffmpeg_path}."
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