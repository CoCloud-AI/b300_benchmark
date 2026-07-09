#!/bin/bash
# serve_mimo-v2.5_nvfp4_vllm.sh
# Status: PENDING VERIFICATION (driver 610 / vLLM v0.24.0 / vLLM-first policy 2026-07-09)
# MiMo-V2.5 (184 GB, mimo_v2 / MiMoV2ForCausalLM). Community quant (lukealonso).
# Minimal flags: quant auto-detects from hf_quant_config.json (M3/V4 lesson:
# forcing --quantization on mixed-precision checkpoints breaks; pure-NVFP4 also
# auto-detects fine). Add flags only on empirical failure. No parsers.
set -u
NAME=mimo25-vllm
IMAGE=vllm/vllm-openai:v0.24.0
MODEL=lukealonso/MiMo-V2.5-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: container(s) running: $RUNNING (docker rm -f first)" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/vllm/mimo-v2.5/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 (vLLM, port 8000)"
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v /mnt/nvme/hf-cache:/root/.cache/huggingface \
  -v ~/benchmark/results_610:/results \
  -p 8000:8000 \
  --name "$NAME" \
  --entrypoint vllm \
  "$IMAGE" \
  serve "$MODEL" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --max-model-len 16384 \
  --host 0.0.0.0 --port 8000
