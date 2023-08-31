import os
import sys
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform
import numpy as np
import random as python_random

from src.config import TOTAL_STUDY_PERIODS_FILE, CURRENCY_PAIRS_FILE, SEED


def init_seeds():
    np.random.seed(SEED)
    python_random.seed(SEED)
    tf.random.set_seed(SEED)


def create_folder_tree_if_not_exists(path):
    if not os.path.exists(path):
        if '.' in os.path.basename(path):
            path = os.path.dirname(path)
        os.makedirs(path, exist_ok=True)


def save_total_study_periods(n_periods):
    create_folder_tree_if_not_exists(TOTAL_STUDY_PERIODS_FILE)
    with open(TOTAL_STUDY_PERIODS_FILE, 'w') as file:
        file.write(str(n_periods))


def read_total_study_periods():
    with open(TOTAL_STUDY_PERIODS_FILE, 'r') as file:
        n_periods = file.read()
    return int(n_periods)


def save_currency_pairs(currency_pairs):
    with open(CURRENCY_PAIRS_FILE, 'w') as file:
        file.write('\n'.join(currency_pairs))


def read_currency_pairs():
    with open(CURRENCY_PAIRS_FILE, 'r') as file:
        currency_pairs_str = file.read()
    currency_pairs = currency_pairs_str.split('\n')
    return currency_pairs


def print_system_infos():
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    # print(f"Keras Version: {tf.keras.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    print(f"SciPy {sp.__version__}")
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")
    print()
