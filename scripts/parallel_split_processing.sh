#!/bin/bash

# Example script for parallel processing using data splits
# This demonstrates how to process large datasets in parallel
# by running multiple instances of vllm-generator with different splits

# Configuration
CONFIG_FILE="configs/qwen3.yaml"
NUM_SPLITS=4
BASE_PORT=8000

# Function to run a split
run_split() {
    local split_id=$1
    local port=$((BASE_PORT + split_id - 1))
    
    echo "Starting split $split_id/$NUM_SPLITS on port $port..."
    
    vllm-generator generate \
        --config "$CONFIG_FILE" \
        --port "$port" \
        --split-id "$split_id" \
        --num-splits "$NUM_SPLITS"  &
    
    echo "Split $split_id started with PID $!"
}

# Start all splits
echo "Starting parallel processing with $NUM_SPLITS splits..."
for i in $(seq 1 $NUM_SPLITS); do
    run_split $i
done

echo "All splits started. Waiting for completion..."
wait

echo "All splits completed!"
