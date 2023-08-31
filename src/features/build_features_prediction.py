import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import N_LAGS, TARGET_CURRENCY_PAIRS, PRED_TRAIN_FILENAME_FORMAT, PRED_TEST_FILENAME_FORMAT, \
    PRED_X_TRAIN_FT_FILENAME_FORMAT, PRED_y_TRAIN_FT_FILENAME_FORMAT, PRED_X_TEST_FT_FILENAME_FORMAT, \
    PRED_y_TEST_FT_FILENAME_FORMAT, PREPROCESSING_PRED_DIR, FEATURES_PRED_DIR
from src.utils import read_total_study_periods, create_folder_tree_if_not_exists


def create_sequences(X, y):
    Xs, ys = [], []
    for i in range(len(X) - N_LAGS):
        Xs.append(X[i:(i + N_LAGS)].flatten())
        ys.append(y[i + N_LAGS])
    return np.array(Xs), np.array(ys)


def load_period_for_currency_pair(currency_pair, period):
    train_df = pd.read_csv(f"{PREPROCESSING_PRED_DIR}/{PRED_TRAIN_FILENAME_FORMAT.format(currency_pair, period)}")
    test_df = pd.read_csv(f"{PREPROCESSING_PRED_DIR}/{PRED_TEST_FILENAME_FORMAT.format(currency_pair, period)}")
    return train_df, test_df


def save_features(X_train_seq, y_train_seq, X_test_seq, y_test_seq, currency_pair, period):
    create_folder_tree_if_not_exists(FEATURES_PRED_DIR)
    np.save(f'{FEATURES_PRED_DIR}/{PRED_X_TRAIN_FT_FILENAME_FORMAT.format(currency_pair, period)}', X_train_seq)
    np.save(f'{FEATURES_PRED_DIR}/{PRED_y_TRAIN_FT_FILENAME_FORMAT.format(currency_pair, period)}', y_train_seq)
    np.save(f'{FEATURES_PRED_DIR}/{PRED_X_TEST_FT_FILENAME_FORMAT.format(currency_pair, period)}', X_test_seq)
    np.save(f'{FEATURES_PRED_DIR}/{PRED_y_TEST_FT_FILENAME_FORMAT.format(currency_pair, period)}', y_test_seq)


def build_features():
    print('*** Building features for prediction ***')
    total_study_periods = read_total_study_periods()

    for currency_pair in TARGET_CURRENCY_PAIRS:
        for i in range(total_study_periods):
            train_df, test_df = load_period_for_currency_pair(currency_pair, i)

            X_train, y_train = train_df[['Returns']], train_df['Target']
            X_test, y_test = test_df[['Returns']], test_df['Target']

            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm = scaler.transform(X_test)

            X_train_seq, y_train_seq = create_sequences(X_train_norm, y_train)
            X_test_seq, y_test_seq = create_sequences(X_test_norm, y_test)

            save_features(X_train_seq, y_train_seq, X_test_seq, y_test_seq, currency_pair, i)


if __name__ == '__main__':
    build_features()
