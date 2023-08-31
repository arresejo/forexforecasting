from src.config import TSFRESH, TSFRESH_TSNE
from src.features import build_features_prediction_with_clustering, build_features_clustering_tsfresh
from src.models.clustering import hdbs
from src.models.prediction import RNN_clustering_model
from src.preprocessing import preprocess_prediction_with_clustering, preprocess_clustering, preprocess_prediction
from src.features.reduction import reduce_features_tsne

import tensorflow as tf

# Disable all GPUS for simple RNNs
# it seems there are some issues with the GPU support on M1 chips
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

MA_WINDOW_SIZE = 5

if __name__ == '__main__':
    preprocess_prediction.preprocess()
    preprocess_clustering.preprocess()
    preprocess_prediction_with_clustering.preprocess(ma_windows_size=MA_WINDOW_SIZE)

    build_features_clustering_tsfresh.build_features(feature_method=TSFRESH)
    reduce_features_tsne.reduce_features(feature_method=TSFRESH)

    hdbs.run_model(TSFRESH_TSNE, plotting=False)

    build_features_prediction_with_clustering.build_features(feature_method=TSFRESH_TSNE, ma_windows_size=MA_WINDOW_SIZE)

    metrics_df = RNN_clustering_model.run_model(feature_method=TSFRESH_TSNE, ma_windows_size=MA_WINDOW_SIZE, plotting=False)




