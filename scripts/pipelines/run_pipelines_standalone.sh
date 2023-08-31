#!/bin/bash

pipelines_to_execute=(
    #"LR_pipeline.py"
    #"RF_pipeline.py"
    #"ANN_pipeline.py"
    "RNN_pipeline.py"
    #"LSTM_pipeline.py"
    #"GRU_pipeline.py"
)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

"$DIR"/pipeline_executor.sh "${pipelines_to_execute[@]}"
