#!/bin/bash

notebooks_to_execute=(
    "LR_clustering_MA10_tsfresh+tsne_pipeline.ipynb"
    "RF_clustering_MA10_tsfresh+tsne_pipeline.ipynb"
    "ANN_clustering_MA10_tsfresh+tsne_pipeline.ipynb"
    "RNN_clustering_MA10_tsfresh+tsne_pipeline.ipynb"
    "LSTM_clustering_MA10_tsfresh+tsne_pipeline.ipynb"
    "GRU_clustering_MA10_tsfresh+tsne_pipeline.ipynb"
)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

"$DIR"/notebook_executor.sh "${notebooks_to_execute[@]}"
