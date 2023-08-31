#!/bin/bash

pipelines_to_execute=(
    "LR_clustering_MA10_tsfresh+tsne_pipeline.py"
    "RF_clustering_MA10_tsfresh+tsne_pipeline.py"
    "ANN_clustering_MA10_tsfresh+tsne_pipeline.py"
    "RNN_clustering_MA10_tsfresh+tsne_pipeline.py"
    "LSTM_clustering_MA10_tsfresh+tsne_pipeline.py"
    "GRU_clustering_MA10_tsfresh+tsne_pipeline.py"
)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

"$DIR"/pipeline_executor.sh "${pipelines_to_execute[@]}"
