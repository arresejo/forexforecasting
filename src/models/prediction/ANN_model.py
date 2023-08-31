import tensorflow as tf

from src.config import FEATURES_PRED_DIR, ANN_REPORT_DIR, N_LAGS, ANN_PATIENCE, ANN_EPOCHS, ANN_BATCH_SIZE
from src.models.prediction import common
from src.models.prediction.LR_model import train_model, predict


def create_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(N_LAGS, 1)))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss=tf.keras.metrics.binary_crossentropy,
                  optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model


def training_params():
    return {
        'patience': ANN_PATIENCE,
        'epochs': ANN_EPOCHS,
        'batch_size': ANN_BATCH_SIZE
    }


def train_model(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, training_params):
    params = training_params()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='min',
                                                      patience=params['patience'],
                                                      restore_best_weights=True,
                                                      verbose=True)
    history = model.fit(X_train_seq, y_train_seq,
                        epochs=params['epochs'],
                        batch_size=params['batch_size'],
                        validation_data=(X_test_seq, y_test_seq),
                        callbacks=[early_stopping],
                        verbose=True)

    return history


def predict(model, X_test_seq):
    y_pred_proba = model.predict(X_test_seq)
    y_pred_labels = (y_pred_proba >= 0.5).astype(int)
    return y_pred_labels, y_pred_proba


def run_model(plotting=True):
    metrics_df = common.run_model(create_model,
                                  training_params,
                                  train_model,
                                  predict,
                                  FEATURES_PRED_DIR,
                                  ANN_REPORT_DIR,
                                  plotting=plotting)
    return metrics_df


if __name__ == '__main__':
    run_model()
