#!/bin/bash
# serve_step-3.7-flash_nvfp4_vllm.sh
# Status: PENDING VERIFICATION (driver 610 / vLLM v0.24.0 / vLLM-first policy 2026-07-09)
# Step-3.7-Flash (129 GB, step3p7 / Step3p7ForConditionalGeneration). StepFun official NVFP4.
# Minimal flags: quant auto-detects from hf_quant_config.json (M3/V4 lesson:
# forcing --quantization on mixed-precision checkpoints breaks; pure-NVFP4 also
# auto-detects fine). Add flags only on empirical failure. No parsers.
set -u
NAME=step37-vllm
IMAGE=vllm/vllm-openai:v0.24.0
MODEL=stepfun-ai/Step-3.7-Flash-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: container(s) running: $RUNNING (docker rm -f first)" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/vllm/step-3.7-flash/{json,logs}

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
  --max-model-len 16384 \
  --host 0.0.0.0 --port 8000
