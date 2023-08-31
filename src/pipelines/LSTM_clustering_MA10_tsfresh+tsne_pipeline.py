from src.config import TSFRESH, TSFRESH_TSNE
from src.features import build_features_prediction_with_clustering, build_features_clustering_tsfresh
from src.models.clustering import hdbs
from src.models.prediction import LSTM_clustering_model
from src.preprocessing import preprocess_prediction_with_clustering, preprocess_clustering, preprocess_prediction
from src.features.reduction import reduce_features_tsne

MA_WINDOW_SIZE = 10

if __name__ == '__main__':
    preprocess_prediction.preprocess()
    preprocess_clustering.preprocess()
    preprocess_prediction_with_clustering.preprocess(ma_windows_size=MA_WINDOW_SIZE)

    build_features_clustering_tsfresh.build_features(feature_method=TSFRESH)
    reduce_features_tsne.reduce_features(feature_method=TSFRESH)

    hdbs.run_model(TSFRESH_TSNE, plotting=False)

    build_features_prediction_with_clustering.build_features(feature_method=TSFRESH_TSNE, ma_windows_size=MA_WINDOW_SIZE)

    metrics_df = LSTM_clustering_model.run_model(feature_method=TSFRESH_TSNE, ma_windows_size=MA_WINDOW_SIZE, plotting=False)
