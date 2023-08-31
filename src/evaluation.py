import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from src.config import N_LAGS


def annualised_std(strategy_returns, trading_days=252):
    daily_std = np.std(strategy_returns)
    return daily_std * np.sqrt(trading_days)


def annualised_sharpe_ratio(strategy_returns, risk_free_rate=0.0, trading_days=252):
    avg_return = np.mean(strategy_returns - risk_free_rate)
    std_return = np.std(strategy_returns)

    daily_sharpe_ratio = avg_return / std_return

    return np.sqrt(trading_days) * daily_sharpe_ratio


def evaluate_prediction_model(model, predict_fn, X_test_seq, y_test_seq, test_df):
    test_returns = test_df['Returns'][N_LAGS:]

    y_pred_labels, y_pred_proba = predict_fn(model, X_test_seq)
    test_loss = log_loss(y_test_seq, y_pred_proba)
    test_accuracy = accuracy_score(y_test_seq, y_pred_labels)
    test_auc = roc_auc_score(y_test_seq, y_pred_proba)
    strategy_returns = test_returns * np.where(y_pred_proba >= 0.5, 1, -1).squeeze()

    # total_simple_actual_returns = (1 + test_returns).prod() - 1
    total_simple_pred_returns = (1 + strategy_returns).prod() - 1

    # annualized_sharpe_ratio = (252**0.5) * (strategy_returns.mean() / strategy_returns.std())
    #sharpe_ratio = strategy_returns.mean() / strategy_returns.std()

    sharpe_ratio = annualised_sharpe_ratio(strategy_returns)
    std_dev = annualised_std(strategy_returns)

    """
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print("Test AUC:", test_auc)

    print(f"Actual Returns: {total_simple_actual_returns:.4f}")
    print(f"Predicted Returns: {total_simple_pred_returns:.4f}")
    print(f"Annualized Sharpe Ratio: {annualized_sharpe_ratio:.4f}")
    print()
    """

    # Calculate the cumulative returns
    cum_actual_simple_returns = (1 + test_returns).cumprod()
    cum_pred_simple_returns = (1 + strategy_returns).cumprod()

    """
    plt.plot(test_returns.index, cum_actual_simple_returns, label='Actual')
    plt.plot(test_returns.index, cum_pred_simple_returns, label='Predicted')
    plt.grid(True)
    plt.legend()
    plt.show()   
    """

    metrics_dict = {
        "Log Loss": [test_loss],
        "Accuracy": [test_accuracy],
        "AUC": [test_auc],
        "Returns": [total_simple_pred_returns],
        "SD": [std_dev],
        "SR": [sharpe_ratio]
    }

    metrics_df = pd.DataFrame(metrics_dict)

    return metrics_df, cum_actual_simple_returns, cum_pred_simple_returns


def evaluate_clustering(data, labels, ignore_noise=True, precomputed=False):
    new_data = data
    new_labels = labels

    if ignore_noise and not precomputed:
        new_data = np.delete(data, labels == -1, axis=0)
        new_labels = np.delete(labels, labels == -1, axis=0)

    score_metric = 'precomputed' if precomputed else 'euclidean'

    silhouette_score = metrics.silhouette_score(new_data, new_labels, metric=score_metric)
    calinski_harabasz_score = metrics.calinski_harabasz_score(new_data, new_labels)
    davies_bouldin_score = metrics.davies_bouldin_score(new_data, new_labels)

    metrics_dict = {
        'Silhouette': [silhouette_score],
        'Calinski Harabasz': [calinski_harabasz_score],
        'Davies Bouldin': [davies_bouldin_score]
    }
    metrics_df = pd.DataFrame(metrics_dict)

    return metrics_df


def get_metrics_df(metrics_list):
    final_metrics_df = pd.concat(metrics_list, axis=0, ignore_index=True)
    final_metrics_df.index = final_metrics_df.index.map(lambda x: f"Period {str(x + 1)}")
    mean_df = pd.DataFrame(final_metrics_df.mean(axis=0), columns=['Mean']).T
    final_metrics_df = pd.concat(
        [final_metrics_df, mean_df],
        ignore_index=False)
    return final_metrics_df
