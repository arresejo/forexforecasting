#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

start_total_time=$(date +%s)

echo "Executing pipelines standalone..."
start_time=$(date +%s)
"$DIR"/run_pipelines_standalone.sh
end_time=$(date +%s)
echo "Execution time for pipelines standalone: $((end_time - start_time)) seconds"
echo "-----------------------------------"

echo "Executing pipelines hybrid MA5..."
start_time=$(date +%s)
"$DIR"/run_pipelines_hybrid_MA5.sh
end_time=$(date +%s)
echo "Execution time for pipelines hybrid MA5: $((end_time - start_time)) seconds"
echo "-----------------------------------"

echo "Executing pipelines hybrid MA10..."
start_time=$(date +%s)
"$DIR"/run_pipelines_hybrid_MA10.sh
end_time=$(date +%s)
echo "Execution time for pipelines hybrid MA10: $((end_time - start_time)) seconds"
echo "-----------------------------------"

end_total_time=$(date +%s)

echo "Total execution time: $((end_total_time - start_total_time)) seconds"
