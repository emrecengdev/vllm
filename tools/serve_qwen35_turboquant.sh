#!/usr/bin/env bash
set -euo pipefail

MODE="${TURBOQUANT_MODE:-turbo4}"
MODEL="${TURBOQUANT_MODEL:-Qwen/Qwen3.5-27B}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
SPARSE_V="${TURBOQUANT_SPARSE_V:-0}"
LAYER_ADAPTIVE="${TURBOQUANT_LAYER_ADAPTIVE:-0}"

export VLLM_ENABLE_V1_MULTIPROCESSING="${V1_MULTIPROCESSING}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --attention-config "{\"turboquant_enabled\": true, \"turboquant_mode\": \"${MODE}\", \"turboquant_sparse_v\": ${SPARSE_V}, \"turboquant_layer_adaptive\": ${LAYER_ADAPTIVE}}" \
  "$@"
