import os
import pickle

import hdbscan
import numpy as np

from src.config import CLUS_TRAIN_FT_FILENAME_FORMAT, TARGET_CURRENCY_PAIRS, CLUSTERS_DIR, FEATURES_CLUS_DIR, \
    HDBS_CLUSTERS_FILENAME, HDBSCAN_CONFIG, HDBSCAN_REPORT_DIR, HDBSCAN_REPORT_FILENAME
from src.evaluation import get_metrics_df, evaluate_clustering
from src.plotting import plot_features
from src.utils import read_total_study_periods, read_currency_pairs, create_folder_tree_if_not_exists


def load_features(period, feature_method):
    data = np.load(f'{FEATURES_CLUS_DIR}/{CLUS_TRAIN_FT_FILENAME_FORMAT.format(period, feature_method)}')
    return data


def extract_clusters(currency_pairs, labels):
    cluster_dict = {}

    for label, pair in zip(labels, currency_pairs):
        if label == -1 and pair in TARGET_CURRENCY_PAIRS:
            print(f"Warning: {pair} assigned to noise cluster")
        # assert not (label == -1 and pair in TARGET_CURRENCY_PAIRS)

        if label in cluster_dict:
            cluster_dict[label].append(pair)
        else:
            cluster_dict[label] = [pair]

    return cluster_dict


def save_clusters(clusters, feature_method):
    create_folder_tree_if_not_exists(CLUSTERS_DIR)
    with open(f"{CLUSTERS_DIR}/{HDBS_CLUSTERS_FILENAME.format(feature_method)}", 'wb') as file:
        pickle.dump(clusters, file)
        file.close()


def run_model(feature_method, config=HDBSCAN_CONFIG, precomputed=False, plotting=True, verbose=True):
    metrics_list = []
    study_periods_clusters = []

    currency_pairs = read_currency_pairs()
    total_study_periods = read_total_study_periods()

    for i in range(total_study_periods):
        if verbose:
            print(f"*** Running model on period {i} ***")

        data = load_features(i, feature_method)

        if precomputed:
            config['metric'] = 'precomputed'

        clusterer = hdbscan.HDBSCAN(**config)
        clusterer.fit(data)

        labels = clusterer.labels_
        # probabilities = clusterer.probabilities_

        cluster_dict = extract_clusters(currency_pairs, labels)

        metrics_df = evaluate_clustering(data, labels, precomputed=precomputed)

        if plotting:
            plot_features(data, currency_pairs, f"Clusters for period {i}:", labels)

        if verbose:
            print(f"\nClusters for period {i}:")
            print(cluster_dict, '\n')
            print(f"\nMetrics for period {i}:")
            print(metrics_df, '\n')

        study_periods_clusters.append(cluster_dict)
        metrics_list.append(metrics_df)

    # print(get_metrics_df(metrics_list))

    metrics_df = get_metrics_df(metrics_list)

    if not os.path.exists(HDBSCAN_REPORT_DIR):
        os.makedirs(HDBSCAN_REPORT_DIR)

    metrics_df.to_csv(f"{HDBSCAN_REPORT_DIR}/{HDBSCAN_REPORT_FILENAME}")

    save_clusters(study_periods_clusters, feature_method)

    # return get_metrics_df(metrics_list)

    return metrics_df
