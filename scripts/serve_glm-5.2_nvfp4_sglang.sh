#!/bin/bash
# serve_glm-5.2_nvfp4_sglang.sh
# Status: RETRY QUEUED on sglang-b300:v0.5.14-tfdsa (transformers ALLOWED_LAYER_TYPES
#   backport — see Dockerfile.sglang-b300-tfdsa). First attempt on the unpatched
#   image failed at config load: layer_types deepseek_sparse_attention rejected
#   by transformers 5.8.1. Config-load verified OK with patch (CPU test).
#
# GLM-5.2 NVFP4 on B300 TP=8. nvidia/GLM-5.2-NVFP4 (433 GB).
#   model_type=glm_moe_dsa / GlmMoeDsaForCausalLM (78L, 256e/8act, DSA attention,
#   1M ctx). Registered natively in SGLang 0.5.14 (glm4_moe family). NVFP4.
# Driver-610 lesson: MINIMAL flags, let SGLang auto-select backends.
set -u
NAME=glm-5.2
IMAGE=sglang-b300:v0.5.14-tfdsa
MODEL=nvidia/GLM-5.2-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: container(s) running: $RUNNING (docker rm -f first)" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/sglang/glm-5.2/{json,logs}

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
  --kv-cache-dtype fp8_e4m3 \
  --context-length 16384 \
  --host 0.0.0.0 --port 30000
