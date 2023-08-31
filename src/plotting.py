import plotly.express as px

from matplotlib import pyplot as plt
from sklearn.manifold import MDS

from src.config import SEED


def plot(data):
    num_rows = len(data) // 3 if len(data) % 3 == 0 else len(data) // 3 + 1

    fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows * 3))
    axs = axs.flatten()

    for i, d in enumerate(data):
        axs[i].plot(d['actual'])
        axs[i].plot(d['pred'])
        axs[i].set_title(f'Cumulative Returns for period {i + 1}')
        axs[i].set_xlabel('Actual')
        axs[i].set_ylabel('Pred')
        axs[i].grid(True)

    if len(data) < num_rows * 3:
        for i in range(len(data), num_rows * 3):
            fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


def plot_cum_returns(cum_returns_list, title):
    num_rows = len(cum_returns_list) // 3 if len(cum_returns_list) % 3 == 0 else len(cum_returns_list) // 3 + 1

    fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows * 3))
    axs = axs.flatten()

    for i, d in enumerate(cum_returns_list):
        axs[i].plot(d['actual'])
        axs[i].plot(d['pred'])
        axs[i].set_title(f'Cumulative Returns for period {i + 1}')
        axs[i].set_xlabel('Actual')
        axs[i].set_ylabel('Pred')
        axs[i].grid(True)

    if len(cum_returns_list) < num_rows * 3:
        for i in range(len(cum_returns_list), num_rows * 3):
            fig.delaxes(axs[i])

    fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    plt.show()


def plot_history(history):
    num_rows = len(history) // 3 if len(history) % 3 == 0 else len(history) // 3 + 1

    fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows * 3))
    axs = axs.flatten()

    for i, d in enumerate(history):
        axs[i].plot(d.history['loss'])
        axs[i].plot(d.history['val_loss'])
        axs[i].set_title(f'Losses period {i + 1}')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Loss')
        axs[i].grid(True)

    if len(history) < num_rows * 3:
        for i in range(len(history), num_rows * 3):
            fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


def plot_features(data, currency_pairs, title, cluster_labels=None, apply_mds=False, plot_3d=False):

    if apply_mds:
        embedding = MDS(n_components=3, dissimilarity='precomputed', random_state=SEED)
        data = embedding.fit_transform(data)

    fig = px.scatter(data, x=0, y=1, text=currency_pairs, color=cluster_labels, title=title)
    fig.update_traces(textposition='top center')
    fig.show()

    if plot_3d:
        fig = px.scatter_3d(data, x=0, y=1, z=2, text=currency_pairs, color=cluster_labels, title=title)
        fig.show()
