import pandas as pd

from src.config import REPORTS_DIR, TARGET_CURRENCY_PAIRS, FILENAME_FORMAT, DATASETS_DIR


def calculate_stats_for_currency_pair(filepath):
    data = pd.read_csv(filepath)

    data['Return'] = data['Close'].pct_change()

    stat_props = data['Return'].describe()
    skew_kurt = pd.Series({
        'skewness': data['Return'].skew(),
        'kurtosis': data['Return'].kurtosis()
    }, name='')

    stat_props = pd.concat([stat_props, skew_kurt])

    return stat_props


def calculate_stats(verbose=True):
    results = pd.DataFrame()

    for currency_pair in TARGET_CURRENCY_PAIRS:
        stats = calculate_stats_for_currency_pair(f"{DATASETS_DIR}/{FILENAME_FORMAT.format(currency_pair)}")
        results[currency_pair] = stats

    results.to_csv(f"{REPORTS_DIR}/stats.csv")

    if verbose:
        print(results)

    return results


if __name__ == '__main__':
    calculate_stats()
