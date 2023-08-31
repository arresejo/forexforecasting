import tensorflow as tf

from src.config import FEATURES_PRED_DIR, N_LAGS, GRU_REPORT_DIR
from src.models.prediction import common
from src.models.prediction.ANN_model import predict, train_model
from src.models.prediction.RNN_model import training_params


def create_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(N_LAGS, 1)))
    model.add(tf.keras.layers.GRU(
        50,
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid'
    ))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.GRU(
        50,
        return_sequences=True,
        activation='tanh',
        recurrent_activation='sigmoid'
    ))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.GRU(
        50,
        return_sequences=False,
        activation='tanh',
        recurrent_activation='sigmoid'
    ))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss=tf.keras.metrics.binary_crossentropy,
                  optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model


def run_model(plotting=True):
    metrics_df = common.run_model(create_model,
                                  training_params,
                                  train_model,
                                  predict,
                                  FEATURES_PRED_DIR,
                                  GRU_REPORT_DIR,
                                  plotting=plotting)
    return metrics_df


if __name__ == '__main__':
    run_model()
