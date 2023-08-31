#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pipelines_folder="$DIR/../../src/pipelines"
PROJECT_ROOT="$DIR"/../..

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

if [ "$#" -eq 0 ]; then
    echo "Please provide a list of pipeline names to execute."
    exit 1
fi

for pl in "$@"; do
    full_path="$pipelines_folder/$pl"
    if [[ -f "$full_path" ]]; then
        echo "Executing $pl..."
        python $full_path
        echo "-----------------------------------"
    else
        echo "Pipeline $pl does not exist in the directory $pipelines_folder."
    fi
done
