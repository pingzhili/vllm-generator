#!/bin/bash

# Start vLLM servers on 8 GPUs with ports 8000-8007
MODEL_NAME="Qwen/Qwen3-8B"
VLLM_ARGS="--enable-reasoning --reasoning-parser deepseek_r1"
#VLLM_ARGS=""

echo "Starting vLLM servers on 8 GPUs..."

# Start servers on each GPU
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME $VLLM_ARGS --port 8000 &
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME $VLLM_ARGS --port 8001 &
CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_NAME $VLLM_ARGS --port 8002 &
CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_NAME $VLLM_ARGS --port 8003 &
CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_NAME $VLLM_ARGS --port 8004 &
CUDA_VISIBLE_DEVICES=5 vllm serve $MODEL_NAME $VLLM_ARGS --port 8005 &
CUDA_VISIBLE_DEVICES=6 vllm serve $MODEL_NAME $VLLM_ARGS --port 8006 &
CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_NAME $VLLM_ARGS --port 8007 &

echo "All servers started. Ports: 8000-8007"
echo "Press Ctrl+C to stop all servers"

# Wait for all background processes
wait