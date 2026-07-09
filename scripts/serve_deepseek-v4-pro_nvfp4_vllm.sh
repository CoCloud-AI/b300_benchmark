#!/bin/bash
# serve_deepseek-v4-pro_nvfp4_vllm.sh
# Status: PENDING (queued as post-queue coda — user request 2026-07-09:
#   re-run DeepSeek on vLLM after everything else finishes, for the full
#   8/8 vLLM matrix + SGLang-vs-vLLM A/B on driver 610)
# V4-Pro (851 GB, deepseek_v4, mixed precision). Quant auto-detect. fp8 KV explicit. SGLang baseline: 6107/7035/2986. NOTE ~5h of sweeps — run LAST.
set -u
NAME=dsv4-pro-vllm
IMAGE=vllm/vllm-openai:v0.24.0
MODEL=nvidia/DeepSeek-V4-Pro-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: container(s) running: $RUNNING (docker rm -f first)" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/vllm/deepseek-v4-pro/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 (vLLM, port 8000)"
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v /mnt/nvme/hf-cache:/root/.cache/huggingface \
  -v ~/benchmark/results_610:/results \
  -p 8000:8000 \
  --name "$NAME" \
  --entrypoint vllm \
  "$IMAGE" \
  serve "$MODEL" \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 16384 \
  --host 0.0.0.0 --port 8000
