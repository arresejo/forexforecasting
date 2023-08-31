import numpy as np

from src.config import FEATURES_CLUS_DIR, CLUS_TRAIN_FT_FILENAME_FORMAT
from src.utils import create_folder_tree_if_not_exists


def load_features(period, feature_method):
    data = np.load(f'{FEATURES_CLUS_DIR}/{CLUS_TRAIN_FT_FILENAME_FORMAT.format(period, feature_method)}')
    return data


def save_features(data, period, feature_method):
    create_folder_tree_if_not_exists(FEATURES_CLUS_DIR)
    np.save(f"{FEATURES_CLUS_DIR}/{CLUS_TRAIN_FT_FILENAME_FORMAT.format(period, feature_method)}", data)
