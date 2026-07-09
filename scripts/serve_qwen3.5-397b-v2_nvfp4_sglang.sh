#!/bin/bash
# serve_qwen3.5-397b-v2_nvfp4_sglang.sh
# Status: PENDING VERIFICATION — SGLang FALLBACK (vLLM-first policy exception)
#
# nvidia/Qwen3.5-397B-A17B-NVFP4-V2 GARBLES ("!!!!...") on vLLM on 8x B300:
#   - v0.24.0 stable: garbled; VLLM_USE_V2_MODEL_RUNNER=0 env confirmed set,
#     no effect (symptom matches issues #47239/#47367)
#   - nightly-2afa3f7e9: garbled identically
# Two-image cap reached -> SGLang 0.5.14 fallback. Precedent: the V1 397B ran
# perfectly on SGLang on the 595 node (peak 10652 t/s), same qwen3_5_moe arch.
# Minimal flags per driver-610 lesson. No parsers.
set -u
NAME=qwen35v2-sglang
IMAGE=sglang-b300:v0.5.14
MODEL=nvidia/Qwen3.5-397B-A17B-NVFP4-V2

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: container(s) running: $RUNNING" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/sglang/qwen3.5-397b-a17b-nvfp4-v2/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 EP=1 (SGLang, port 30000)"
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
