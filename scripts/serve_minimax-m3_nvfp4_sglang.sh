#!/bin/bash
# serve_minimax-m3_nvfp4_sglang.sh
# Status: PENDING VERIFICATION (driver 610 / SGLang 0.5.14 / minimax_m3_vl arch)
#
# MiniMax-M3 NVFP4 on B300 TP=8. nvidia/MiniMax-M3-NVFP4 (233 GB).
#   model_type=minimax_m3_vl / MiniMaxM3SparseForConditionalGeneration (multimodal
#   VL sparse MoE). Benchmarked text-only. trust-remote-code required.
# Following the driver-610 lesson: MINIMAL flags, let SGLang auto-select
# quant/attention/moe backends. Add flags only if bring-up fails.
set -u
NAME=minimax-m3
IMAGE=sglang-b300:v0.5.14
MODEL=nvidia/MiniMax-M3-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: container(s) running: $RUNNING (docker rm -f first)" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/sglang/minimax-m3/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 EP=1"
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v /mnt/nvme/hf-cache:/root/.cache/huggingface \
  -v ~/benchmark/results_610:/results \
  -p 30000:30000 \
  -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  --name "$NAME" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 8 \
  --trust-remote-code \
  --context-length 16384 \
  --host 0.0.0.0 --port 30000
