#!/bin/bash
# serve_step-3.7-flash_nvfp4_sglang.sh
# Status: PENDING — SGLang attempt after vLLM exhausted (5 tries).
# Step-3.7-Flash NVFP4 has moe_intermediate_size=1280 -> per-rank 160 at TP=8,
# which fails EVERY vLLM v0.24/nightly NVFP4 MoE kernel: trtllm asserts M%128,
# flashinfer_cutlass lacks SWIGLUSTEP activation, vllm-cutlass padding path is
# NotImplemented. SGLang may route differently; if its trtllm path asserts the
# same, the model is unservable at TP=8 NVFP4 today (document + move on).
set -u
NAME=step37-sglang
IMAGE=sglang-b300:v0.5.14
MODEL=stepfun-ai/Step-3.7-Flash-NVFP4
RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: running: $RUNNING" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/sglang/step-3.7-flash/{json,logs}
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v /mnt/nvme/hf-cache:/root/.cache/huggingface \
  -v ~/benchmark/results_610:/results \
  -p 30000:30000 \
  -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  --name "$NAME" "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" --tp 8 --trust-remote-code \
  --context-length 16384 --host 0.0.0.0 --port 30000
