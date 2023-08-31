import numpy as np
from sklearn.decomposition import PCA

from src.config import PCA_CONFIG
from src.features.reduction.common import load_features, save_features
from src.utils import read_total_study_periods


def reduce_features(feature_method):
    total_study_periods = read_total_study_periods()

    for i in range(total_study_periods):
        features_data = load_features(i, feature_method)
        pca = PCA(**PCA_CONFIG)
        pca_scaled = pca.fit_transform(features_data)
        save_features(pca_scaled, i, f"{feature_method}_pca")

        total_variance = np.sum(pca.explained_variance_ratio_)
        print(f"Total variance explained by the selected components for period {i}: {total_variance * 100:.2f}%")


if __name__ == '__main__':
    reduce_features('tsfresh')
