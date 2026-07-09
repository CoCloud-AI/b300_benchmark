#!/bin/bash
# serve_deepseek-v4-flash_nvfp4_vllm.sh
# Status: PENDING (queued as post-queue coda — user request 2026-07-09:
#   re-run DeepSeek on vLLM after everything else finishes, for the full
#   8/8 vLLM matrix + SGLang-vs-vLLM A/B on driver 610)
# Flash (168 GB, deepseek_v4, mixed NVFP4-experts+FP8-attn). Quant auto-detect (NO --quantization — same as the SGLang lesson). fp8 KV explicit (MLA pairing). SGLang baseline: 14304/17049/9073.
set -u
NAME=dsv4-flash-vllm
IMAGE=vllm/vllm-openai:v0.24.0
MODEL=nvidia/DeepSeek-V4-Flash-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: container(s) running: $RUNNING (docker rm -f first)" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/vllm/deepseek-v4-flash/{json,logs}

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
