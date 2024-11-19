# src/utils/unique_filename.py

import os

def create_unique_filename(path, filename):
    """Helper method for creating a unique file name, by appending a number if the file name already exists."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(path, new_filename)):
        new_filename = f"{base}({counter}){ext}"
        counter += 1
    return new_filename


