from src.features import build_features_prediction
from src.models.prediction import RNN_model
from src.preprocessing import preprocess_prediction

import tensorflow as tf

# Disable all GPUS for simple RNNs
# it seems there are some issues with the GPU support on M1 chips
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

if __name__ == '__main__':

    preprocess_prediction.preprocess()
    build_features_prediction.build_features()

    metrics_df = RNN_model.run_model(plotting=False)
