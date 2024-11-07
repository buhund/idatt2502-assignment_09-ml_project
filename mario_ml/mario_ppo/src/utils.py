# src/utils.py

import os


def get_unique_filename(path, filename):
    """Returns a unique filename by appending a number if the file already exists."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(path, new_filename)):
        new_filename = f"{base}({counter}){ext}"
        counter += 1
    return new_filename