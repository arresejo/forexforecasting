import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from src.config import TARGET_CURRENCY_PAIRS, DATASETS_DIR, FILENAME_FORMAT, NAIVE_REPORT_DIR
from src.evaluation import annualised_sharpe_ratio, annualised_std
from src.preprocessing.preprocess_prediction import load_data
from src.utils import create_folder_tree_if_not_exists


def evaluate_prediction_model(df, y, y_pred):
    test_returns = df['Returns']

    test_accuracy = accuracy_score(y, y_pred)
    test_auc = roc_auc_score(y, y_pred)
    strategy_returns = test_returns * y_pred

    total_simple_pred_returns = (1 + strategy_returns).prod() - 1

    sharpe_ratio = annualised_sharpe_ratio(strategy_returns)
    std_dev = annualised_std(strategy_returns)

    metrics_dict = {
        "Accuracy": [test_accuracy],
        "AUC": [test_auc],
        "Returns": [total_simple_pred_returns],
        "SD": [std_dev],
        "SR": [sharpe_ratio]
    }

    metrics_df = pd.DataFrame(metrics_dict)

    return metrics_df


def run_model():
    metrics_list = []

    for currency_pair in TARGET_CURRENCY_PAIRS:
        df = load_data(DATASETS_DIR, FILENAME_FORMAT.format(currency_pair))
        y = df['Target']
        y_pred = np.roll(y, shift=1)

        metrics_df = evaluate_prediction_model(df, y, y_pred)
        metrics_list.append(metrics_df)

        create_folder_tree_if_not_exists(NAIVE_REPORT_DIR)
        metrics_df.to_csv(f"{NAIVE_REPORT_DIR}/{currency_pair}.csv")

    final_metrics_df = pd.concat(metrics_list, axis=0, ignore_index=True)
    final_metrics_df.index = TARGET_CURRENCY_PAIRS
    mean_df = pd.DataFrame(final_metrics_df.mean(axis=0), columns=['Mean']).T
    final_metrics_df = pd.concat([final_metrics_df, mean_df], ignore_index=False)

    final_metrics_df.to_csv(f"{NAIVE_REPORT_DIR}/summary.csv")

    return final_metrics_df


if __name__ == '__main__':
    print(run_model())
