#!/bin/bash

VENV="venv-forex-forecasting"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

notebooks_folder="$DIR/../../notebooks"

if [ "$#" -eq 0 ]; then
    echo "Please provide a list of notebook names to execute."
    exit 1
fi

for nb in "$@"; do
    full_path="$notebooks_folder/$nb"
    if [[ -f "$full_path" ]]; then
        echo "Executing $nb with papermill..."
        papermill --log-output "$full_path" "$full_path" -k "$VENV"
        echo "-----------------------------------"
    else
        echo "Notebook $nb does not exist in the directory $notebooks_folder."
    fi
done
