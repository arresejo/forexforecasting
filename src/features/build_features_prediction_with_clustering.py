import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import N_LAGS, TARGET_CURRENCY_PAIRS, CLUSTERS_DIR, PRED_CLUS_TRAIN_FILENAME_FORMAT, \
    PRED_CLUS_TEST_FILENAME_FORMAT, PRED_CLUS_X_TRAIN_FT_FILENAME_FORMAT, PRED_CLUS_y_TRAIN_FT_FILENAME_FORMAT, \
    PRED_CLUS_X_TEST_FT_FILENAME_FORMAT, PRED_CLUS_y_TEST_FT_FILENAME_FORMAT, FEATURES_PRED_CLUS_DIR, \
    PREPROCESSING_PRED_CLUS_DIR, HDBS_CLUSTERS_FILENAME
from src.utils import read_total_study_periods, create_folder_tree_if_not_exists


def create_sequences(X, y):
    Xs, ys = [], []
    for i in range(len(X) - N_LAGS):
        Xs.append(X[i:(i + N_LAGS)].flatten())
        ys.append(y[i + N_LAGS])
    return np.array(Xs), np.array(ys)


def load_period(period, feature_method, ma_windows_size):
    train_df = pd.read_csv(
        f"{PREPROCESSING_PRED_CLUS_DIR}/{PRED_CLUS_TRAIN_FILENAME_FORMAT.format(period, ma_windows_size, feature_method)}")
    test_df = pd.read_csv(f"{PREPROCESSING_PRED_CLUS_DIR}/{PRED_CLUS_TEST_FILENAME_FORMAT.format(period, ma_windows_size, feature_method)}")
    return train_df, test_df


def save_features(X_train_seq, y_train_seq, X_test_seq, y_test_seq, currency_pair, period, feature_extraction_method, ma_windows_size):
    create_folder_tree_if_not_exists(FEATURES_PRED_CLUS_DIR)
    np.save(f'{FEATURES_PRED_CLUS_DIR}/{PRED_CLUS_X_TRAIN_FT_FILENAME_FORMAT.format(currency_pair, period, ma_windows_size, feature_extraction_method)}',
            X_train_seq)
    np.save(f'{FEATURES_PRED_CLUS_DIR}/{PRED_CLUS_y_TRAIN_FT_FILENAME_FORMAT.format(currency_pair, period, ma_windows_size, feature_extraction_method)}',
            y_train_seq)
    np.save(f'{FEATURES_PRED_CLUS_DIR}/{PRED_CLUS_X_TEST_FT_FILENAME_FORMAT.format(currency_pair, period, ma_windows_size, feature_extraction_method)}', X_test_seq)
    np.save(f'{FEATURES_PRED_CLUS_DIR}/{PRED_CLUS_y_TEST_FT_FILENAME_FORMAT.format(currency_pair, period, ma_windows_size, feature_extraction_method)}', y_test_seq)


def read_clusters(feature_method):
    with open(f"{CLUSTERS_DIR}/{HDBS_CLUSTERS_FILENAME.format(feature_method)}", 'rb') as file:
        # Load the pickled object
        obj = pickle.load(file)
    return obj


def find_target_cluster(cluster, target_pair):
    for key, values in cluster.items():
        if target_pair in values:
            target_cluster = key
    return target_cluster


def build_cluster_features(df, cluster, target_pair):
    target_cluster = find_target_cluster(cluster, target_pair)

    if target_cluster == -1:
        print(f"Target pair {target_pair} assigned to noise cluster")

    assert target_cluster != -1  # Target pair cannot be assigned to noise cluster

    df_list = []

    for pair in cluster[target_cluster]:
        if pair == target_pair:
            df_list.append(df[[f"{pair}_Returns"]])
        else:
            df_list.append(df[[f"{pair}_Returns_MA"]])

    features_df = pd.concat(df_list, axis=1)

    features_df['Cluster_Returns'] = features_df.median(axis=1)
    features_df['Target'] = np.where(features_df[f"{target_pair}_Returns"] >= features_df['Cluster_Returns'], 1, 0)
    features_df = features_df[[f"{target_pair}_Returns", 'Target']]

    return features_df


def build_features(feature_method, ma_windows_size):
    print('*** Building features for prediction with clustering ***')

    total_study_periods = read_total_study_periods()
    period_clusters = read_clusters(feature_method)

    for currency_pair in TARGET_CURRENCY_PAIRS:
        for i in range(total_study_periods):
            train_df, test_df = load_period(i, feature_method, ma_windows_size)

            period_cluster = period_clusters[i]

            train_df = build_cluster_features(train_df, period_cluster, currency_pair)
            test_df = build_cluster_features(test_df, period_cluster, currency_pair)

            X_train = train_df[[f"{currency_pair}_Returns"]]
            y_train = train_df['Target']

            X_test = test_df[[f"{currency_pair}_Returns"]]
            y_test = test_df['Target']

            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm = scaler.transform(X_test)

            X_train_seq, y_train_seq = create_sequences(X_train_norm, y_train)
            X_test_seq, y_test_seq = create_sequences(X_test_norm, y_test)

            save_features(X_train_seq, y_train_seq, X_test_seq, y_test_seq, currency_pair, i, feature_method, ma_windows_size)

