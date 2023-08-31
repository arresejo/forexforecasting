from pathlib import Path

import pandas as pd

from src.config import FILENAME_FORMAT, PREPROCESSING_PRED_CLUS_DIR, STUDY_PERIOD_DAYS, TESTING_DAYS, \
    TRAINING_DAYS, N_LAGS, PRED_CLUS_TEST_FILENAME_FORMAT, PRED_CLUS_TRAIN_FILENAME_FORMAT, DATASETS_DIR

from src.utils import save_total_study_periods, create_folder_tree_if_not_exists


def load_data(files_dir, filename, ma_windows_size):
    df_list = []

    data_files = sorted(list(Path(files_dir).glob(filename)))
    assert data_files

    for data_file in data_files:
        pair = data_file.stem.split('_')[2]
        df = pd.read_csv(data_file, index_col='Date', parse_dates=['Date'])
        df['Returns'] = df['Close'].pct_change()
        df['Returns_MA'] = df['Close'].rolling(ma_windows_size).mean().pct_change()
        df = df[['Close', 'Returns', 'Returns_MA']].dropna()
        df = df.add_prefix(f"{pair}_")

        df_list.append(df)

    df = pd.concat(df_list, axis=1)
    df = df.sort_index(axis=1)
    df = df.ffill().dropna()

    return df


def preprocess(ma_windows_size):
    print('*** Preprocessing for prediction with clustering ***')
    df = load_data(DATASETS_DIR, FILENAME_FORMAT.replace('{}', '*'), ma_windows_size)
    total_study_periods = (len(df) - STUDY_PERIOD_DAYS) // TESTING_DAYS
    save_total_study_periods(total_study_periods)
    for i in range(total_study_periods):
        start_index = i * TESTING_DAYS
        train_df = df[start_index:start_index + TRAINING_DAYS]
        test_df = df[start_index + TRAINING_DAYS - N_LAGS:start_index + TRAINING_DAYS + TESTING_DAYS]
        create_folder_tree_if_not_exists(PREPROCESSING_PRED_CLUS_DIR)
        train_df.to_csv(f"{PREPROCESSING_PRED_CLUS_DIR}/{PRED_CLUS_TRAIN_FILENAME_FORMAT.format(i, ma_windows_size)}")
        test_df.to_csv(f"{PREPROCESSING_PRED_CLUS_DIR}/{PRED_CLUS_TEST_FILENAME_FORMAT.format(i, ma_windows_size)}")


if __name__ == '__main__':
    preprocess()
