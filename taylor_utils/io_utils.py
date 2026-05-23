import os
import builtins
import pathlib
import pickle


def save_pickle(data, file_name):
    """Save data in pickle format."""
    absolute_path = pathlib.Path(file_name).absolute()
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    with builtins.open(str(absolute_path), 'wb') as f:
        pickle.dump(data, f, protocol=4)


def load_pickle(filepath):
    """Load data from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
