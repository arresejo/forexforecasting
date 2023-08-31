import os
import time

import numpy as np
import pandas as pd

from src.config import PRED_X_TRAIN_FT_FILENAME_FORMAT, PRED_y_TRAIN_FT_FILENAME_FORMAT, PRED_X_TEST_FT_FILENAME_FORMAT, \
    PRED_y_TEST_FT_FILENAME_FORMAT, TARGET_CURRENCY_PAIRS, PRED_CLUS_X_TRAIN_FT_FILENAME_FORMAT, \
    PRED_CLUS_y_TRAIN_FT_FILENAME_FORMAT, PRED_CLUS_X_TEST_FT_FILENAME_FORMAT, PRED_CLUS_y_TEST_FT_FILENAME_FORMAT, \
    PRED_OVERALL_REPORT_FILENAME_FORMAT, PRED_CLUS_OVERALL_REPORT_FILENAME_FORMAT, PRED_REPORT_FILENAME_FORMAT, \
    PRED_CLUS_REPORT_FILENAME_FORMAT
from src.evaluation import evaluate_prediction_model, get_metrics_df
from src.features.build_features_prediction import load_period_for_currency_pair
from src.utils import read_total_study_periods, init_seeds, print_system_infos
from src.plotting import plot_cum_returns, plot_history


def load_features(features_dir, currency_pair, period, feature_method, ma_windows_size, with_clustering):
    if with_clustering:
        X_train_seq_filename = PRED_CLUS_X_TRAIN_FT_FILENAME_FORMAT
        y_train_seq_filename = PRED_CLUS_y_TRAIN_FT_FILENAME_FORMAT
        X_test_seq_filename = PRED_CLUS_X_TEST_FT_FILENAME_FORMAT
        y_test_seq_filename = PRED_CLUS_y_TEST_FT_FILENAME_FORMAT
        format_params = (currency_pair, period, ma_windows_size, feature_method)
    else:
        X_train_seq_filename = PRED_X_TRAIN_FT_FILENAME_FORMAT
        y_train_seq_filename = PRED_y_TRAIN_FT_FILENAME_FORMAT
        X_test_seq_filename = PRED_X_TEST_FT_FILENAME_FORMAT
        y_test_seq_filename = PRED_y_TEST_FT_FILENAME_FORMAT
        format_params = (currency_pair, period)

    X_train_seq = np.load(f'{features_dir}/{X_train_seq_filename.format(*format_params)}').astype(
        'float32')
    y_train_seq = np.load(f'{features_dir}/{y_train_seq_filename.format(*format_params)}').astype(
        'float32')
    X_test_seq = np.load(f'{features_dir}/{X_test_seq_filename.format(*format_params)}').astype(
        'float32')
    y_test_seq = np.load(f'{features_dir}/{y_test_seq_filename.format(*format_params)}').astype(
        'float32')

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq


def run_model_for_currency_pair(create_model_fn, training_params_fn, train_model_fn, predict_fn, features_dir,
                                currency_pair, feature_method, ma_windows_size, with_clustering):
    metrics_list = []
    cum_returns_list = []
    history_list = []

    total_study_periods = read_total_study_periods()

    for i in range(total_study_periods):
        X_train_seq, y_train_seq, X_test_seq, y_test_seq = load_features(features_dir, currency_pair, i, feature_method, ma_windows_size, with_clustering)

        model = create_model_fn()
        history = train_model_fn(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, training_params_fn)

        _, test_df = load_period_for_currency_pair(currency_pair, i)
        metrics_df, cum_actual_returns, cum_pred_returns = evaluate_prediction_model(
            model, predict_fn, X_test_seq, y_test_seq, test_df)

        metrics_list.append(metrics_df)
        cum_returns_list.append({
            'actual': cum_actual_returns,
            'pred': cum_pred_returns
        })
        history_list.append(history)

    return get_metrics_df(metrics_list), cum_returns_list, history_list


def run_model(create_model_fn,
              training_params_fn,
              train_model_fn,
              predict_fn,
              features_dir,
              report_dir,
              ma_windows_size=None,
              feature_method=None,
              with_clustering=False,
              plotting=True,
              verbose=True,
              random_seed=False):
    start_time = time.time()

    if not random_seed:
        init_seeds()

    if verbose:
        if ma_windows_size:
            print(f"\nMA Window Size: {ma_windows_size}\n")
        print_system_infos()

    metrics_df_list = []

    for currency_pair in TARGET_CURRENCY_PAIRS:
        if verbose:
            print(f"*** Running model on {currency_pair} ***")

        metrics_df, cum_returns, history = run_model_for_currency_pair(
            create_model_fn,
            training_params_fn,
            train_model_fn,
            predict_fn,
            features_dir,
            currency_pair,
            feature_method,
            ma_windows_size,
            with_clustering
        )

        if plotting:
            plot_cum_returns(cum_returns, f"\nCumulative Returns for {currency_pair}")
            if np.any(history):
                plot_history(history)

        if verbose:
            print(f"\nMetrics for {currency_pair}:")
            print(metrics_df, '\n')

        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        if ma_windows_size:
            metrics_df.to_csv(f"{report_dir}/{PRED_CLUS_REPORT_FILENAME_FORMAT.format(currency_pair, ma_windows_size)}")
        else:
            metrics_df.to_csv(f"{report_dir}/{PRED_REPORT_FILENAME_FORMAT.format(currency_pair)}")

        metrics_df_list.append(metrics_df)

    overall_metrics_df = pd.DataFrame(pd.concat([df.loc[df.index == 'Mean'] for df in metrics_df_list]))
    overall_metrics_df = pd.concat([overall_metrics_df, pd.DataFrame(overall_metrics_df.mean()).T])
    overall_metrics_df.index = TARGET_CURRENCY_PAIRS + ['Mean']

    if ma_windows_size:
        overall_metrics_df.to_csv(f"{report_dir}/{PRED_CLUS_OVERALL_REPORT_FILENAME_FORMAT.format(ma_windows_size)}")
    else:
        overall_metrics_df.to_csv(f"{report_dir}/{PRED_OVERALL_REPORT_FILENAME_FORMAT}")

    if verbose:
        print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))

    return overall_metrics_df
