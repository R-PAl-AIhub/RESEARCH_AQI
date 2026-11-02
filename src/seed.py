"""One call that makes random numbers repeatable across Python, NumPy and TensorFlow."""
import os
import random

import numpy as np
import tensorflow as tf


def set_seed(seed):
    """Seed Python's `random`, NumPy's global generator and TensorFlow's global seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses started after this
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
