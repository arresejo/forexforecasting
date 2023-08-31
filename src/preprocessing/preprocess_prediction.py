import numpy as np
import pandas as pd

from src.config import FILENAME_FORMAT, PREPROCESSING_PRED_DIR, TARGET_CURRENCY_PAIRS, STUDY_PERIOD_DAYS, TESTING_DAYS, \
    TRAINING_DAYS, N_LAGS, PRED_TRAIN_FILENAME_FORMAT, PRED_TEST_FILENAME_FORMAT, DATASETS_DIR
from src.utils import save_total_study_periods, create_folder_tree_if_not_exists


def load_data(files_dir, filename):
    df = pd.read_csv(f"{files_dir}/{filename}", index_col='Date', parse_dates=['Date'])
    df['Returns'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Returns'] >= 0, 1, 0)
    df = df[['Returns', 'Target']].dropna()
    return df


def preprocess():
    print('*** Preprocessing for prediction ***')
    for currency_pair in TARGET_CURRENCY_PAIRS:
        df = load_data(DATASETS_DIR, FILENAME_FORMAT.format(currency_pair))
        total_study_periods = (len(df) - STUDY_PERIOD_DAYS) // TESTING_DAYS
        save_total_study_periods(total_study_periods)
        for i in range(total_study_periods):
            start_index = i * TESTING_DAYS
            train_df = df[start_index:start_index + TRAINING_DAYS]
            test_df = df[start_index + TRAINING_DAYS - N_LAGS:start_index + TRAINING_DAYS + TESTING_DAYS]

            create_folder_tree_if_not_exists(PREPROCESSING_PRED_DIR)

            train_df.to_csv(f"{PREPROCESSING_PRED_DIR}/{PRED_TRAIN_FILENAME_FORMAT.format(currency_pair, i)}")
            test_df.to_csv(f"{PREPROCESSING_PRED_DIR}/{PRED_TEST_FILENAME_FORMAT.format(currency_pair, i)}")


if __name__ == '__main__':
    preprocess()
