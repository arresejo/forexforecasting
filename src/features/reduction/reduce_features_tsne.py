from sklearn.manifold import TSNE

from src.config import TSNE_CONFIG
from src.features.reduction.common import load_features, save_features
from src.utils import read_total_study_periods


def reduce_features(feature_method):
    total_study_periods = read_total_study_periods()

    for i in range(total_study_periods):
        features_data = load_features(i, feature_method)
        tsne_scaled = TSNE(**TSNE_CONFIG).fit_transform(features_data)
        save_features(tsne_scaled, i, f"{feature_method}_tsne")


if __name__ == '__main__':
    reduce_features('tsfresh')
