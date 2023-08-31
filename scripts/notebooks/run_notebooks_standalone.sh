#!/bin/bash

notebooks_to_execute=(
    "Naive_predictor.ipynb"
    "LR_pipeline.ipynb"
    "RF_pipeline.ipynb"
    "ANN_pipeline.ipynb"
    "RNN_pipeline.ipynb"
    "LSTM_pipeline.ipynb"
    "GRU_pipeline.ipynb"
)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

"$DIR"/notebook_executor.sh "${notebooks_to_execute[@]}"
