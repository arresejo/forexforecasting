#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

start_total_time=$(date +%s)

echo "Executing notebooks standalone..."
start_time=$(date +%s)
"$DIR"/run_notebooks_standalone.sh
end_time=$(date +%s)
echo "Execution time for notebooks standalone: $((end_time - start_time)) seconds"
echo "-----------------------------------"

echo "Executing notebooks hybrid MA5..."
start_time=$(date +%s)
"$DIR"/run_notebooks_hybrid_MA5.sh
end_time=$(date +%s)
echo "Execution time for notebooks hybrid MA5: $((end_time - start_time)) seconds"
echo "-----------------------------------"

echo "Executing notebooks hybrid MA10..."
start_time=$(date +%s)
"$DIR"/run_notebooks_hybrid_MA10.sh
end_time=$(date +%s)
echo "Execution time for notebooks hybrid MA10: $((end_time - start_time)) seconds"
echo "-----------------------------------"

end_total_time=$(date +%s)

echo "Total execution time: $((end_total_time - start_total_time)) seconds"
