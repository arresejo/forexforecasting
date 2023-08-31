import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from src.config import CLUS_TEST_FILENAME_FORMAT, CLUS_TRAIN_FILENAME_FORMAT, CLUS_TRAIN_FT_FILENAME_FORMAT, \
    PREPROCESSING_CLUS_DIR, FEATURES_CLUS_DIR
from src.utils import read_total_study_periods, create_folder_tree_if_not_exists


def load_period(period):
    train_df = pd.read_csv(
        f"{PREPROCESSING_CLUS_DIR}/{CLUS_TRAIN_FILENAME_FORMAT.format(period)}",
        index_col='Date',
        parse_dates=['Date'])
    test_df = pd.read_csv(f"{PREPROCESSING_CLUS_DIR}/{CLUS_TEST_FILENAME_FORMAT.format(period)}",
                          index_col='Date',
                          parse_dates=['Date'])
    return train_df, test_df


def save_features(data, period, feature_method):
    create_folder_tree_if_not_exists(FEATURES_CLUS_DIR)
    np.save(f"{FEATURES_CLUS_DIR}/{CLUS_TRAIN_FT_FILENAME_FORMAT.format(period, feature_method)}", data)


def features_exist(period, feature_method):
    return os.path.isfile(f"{FEATURES_CLUS_DIR}/{CLUS_TRAIN_FT_FILENAME_FORMAT.format(period, feature_method)}")


def build_features(feature_method):
    print('*** Building features for clustering ***')
    total_study_periods = read_total_study_periods()

    for i in range(total_study_periods):

        if features_exist(i, feature_method):
            print(f"Features file already available for period {i}")
            continue

        # print(f"Extracting features for period {i}")
        train_df, _ = load_period(i)

        melted_df = train_df.copy()
        melted_df['date'] = melted_df.index
        melted_df = melted_df.melt(id_vars=['date'])
        melted_df.columns = ['date', 'pair', 'return']

        features_df = extract_features(melted_df,
                                       column_id='pair',
                                       column_sort='date',
                                       column_value='return',
                                       impute_function=impute)

        n_unique = features_df.nunique()
        cols_to_drop = n_unique[n_unique == 1].index
        features_df = features_df.drop(cols_to_drop, axis=1)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)

        save_features(scaled_features, i, feature_method)


if __name__ == '__main__':
    build_features()
