#!/bin/bash
# serve_glm-5.2_nvfp4_vllm.sh
# Status: PENDING VERIFICATION (driver 610 / vLLM v0.24.0 stable / glm_moe_dsa)
#
# GLM-5.2 NVFP4 on B300 TP=8 via vLLM — user decision 2026-07-09: run GLM-5.2 on
# vLLM right after V4-Pro, followed by MiniMax-M3 (also vLLM).
#
# Why vLLM here: SGLang v0.5.14 rejects this checkpoint's layer_types
# (transformers 5.8.1 pin; fixed upstream post-release — see
# results_610/metadata/blocked_on_sglang_0.5.14.md). vLLM v0.24.0 stable
# registers GlmMoeDsaForCausalLM and NVIDIA tested this checkpoint on B300 with
# vllm-openai (v0.23.0-era recipe; 0.24.0 is current stable with fixes).
# The SGLang tfdsa-patched retry stays available as an optional later A/B
# (scripts/serve_glm-5.2_nvfp4_sglang.sh).
#
# Flags:
# - --quantization modelopt_fp4 per the NVIDIA vLLM recipe (checkpoint is pure
#   NVFP4, quant_method=modelopt — exact match, unlike V4's mixed precision).
# - --kv-cache-dtype fp8_e4m3: NVFP4 KV is rejected with MLA/DSA; fp8 KV is the
#   documented pairing.
# - max-model-len 16384, TP=8, no parsers (project benchmark rules).
# - Known open issue vllm#47239 (8xB300 Model Runner V2 accuracy + TPOT
#   fluctuation): if output is garbled or TPOT oscillates, relaunch with
#   -e VLLM_USE_V2_MODEL_RUNNER=0 and document the change here.
set -u
NAME=glm52-vllm
IMAGE=vllm/vllm-openai:v0.24.0
MODEL=nvidia/GLM-5.2-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: container(s) running: $RUNNING (docker rm -f first)" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/vllm/glm-5.2/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 (vLLM, port 8000)"
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v /mnt/nvme/hf-cache:/root/.cache/huggingface \
  -v ~/benchmark/results_610:/results \
  -p 8000:8000 \
  --name "$NAME" \
  --entrypoint vllm \
  "$IMAGE" \
  serve "$MODEL" \
  --quantization modelopt_fp4 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 16384 \
  --host 0.0.0.0 --port 8000
