#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

python "$PROJECT_ROOT/src/preprocessing/resample_data.py"
