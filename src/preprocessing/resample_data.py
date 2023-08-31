import os
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR, START_DATE, END_DATE, DATASETS_DIR


def load_and_merge_m1_data(pair_path):
    m1_data_list = []

    for year_folder in pair_path.iterdir():
        if year_folder.is_dir():
            for csv_file in year_folder.glob('*.csv'):
                m1_data = pd.read_csv(
                    csv_file,
                    names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
                    parse_dates=['Date'],
                    sep=';').dropna()
                m1_data_list.append(m1_data)

    m1_data = pd.concat(m1_data_list)
    m1_data = m1_data.drop(columns=['Volume'])
    m1_data.sort_values(by='Date', inplace=True)

    return m1_data


def resample(m1_data):
    m1_data.set_index('Date', inplace=True)
    daily_data = m1_data.resample('B').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    }).ffill()

    return daily_data


def exists(currency_pair):
    return os.path.isfile(f"{DATASETS_DIR}/DAT_ASCII_{currency_pair}_DAILY.csv")


def save(daily_data, currency_pair):
    os.makedirs(DATASETS_DIR, exist_ok=True)
    daily_data.to_csv(f"{DATASETS_DIR}/DAT_ASCII_{currency_pair}_DAILY.csv")


def preprocess_single_pair(currency_pair_folder):
    currency_pair = currency_pair_folder.name.upper()

    print(f"Processing {currency_pair}...")

    if exists(currency_pair):
        print(f"{currency_pair.upper()} file already exists, skipped")
        return

    m1_data = load_and_merge_m1_data(currency_pair_folder)
    daily_data = resample(m1_data)
    daily_data = daily_data[(daily_data.index >= START_DATE) & (daily_data.index <= END_DATE)]

    if daily_data.index.max() < pd.Timestamp(END_DATE) - pd.Timedelta(days=5):
        print(f"{currency_pair.upper()} missing data, skipped")
        return

    save(daily_data, currency_pair)


def resample_data():
    for currency_pair_folder in Path(RAW_DATA_DIR).iterdir():
        if currency_pair_folder.is_dir():
            preprocess_single_pair(currency_pair_folder)
    print("All done!")


if __name__ == '__main__':
    resample_data()
