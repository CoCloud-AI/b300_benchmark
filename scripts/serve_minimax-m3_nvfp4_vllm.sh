#!/bin/bash
# serve_minimax-m3_nvfp4_vllm.sh
# Status: PENDING VERIFICATION (driver 610 / vLLM NIGHTLY / minimax_m3_vl)
#
# MiniMax-M3 NVFP4 on B300 TP=8 via vLLM nightly — the ONLY framework+version
# that serves this checkpoint (verified 2026-07-09):
# - SGLang: model files still in OPEN PR #28715; even that dev branch has no
#   NVFP4 path (MoE kernel drops clamped-swiglu params -> garbage).
# - vLLM v0.24.0 stable: registry has the arch, but NVFP4 support (PR #46380,
#   merged 06-25) MISSED the branch cut. NVIDIA card: "you currently need the
#   nightly docker image".
# IMAGE below must be pinned to a digest-tagged nightly at pull time — update
# the tag after `docker pull` and record the digest in results_610/metadata.
#
# Flags (from NVIDIA card + vLLM recipe, parsers stripped per project rule):
# - --block-size 128 is MANDATORY for M3.
# - --language-model-only: text-only serving; skips the vision tower, which
#   otherwise pre-allocates ~192k encoder tokens of KV (HF discussion #1).
#   Correct for our text-only throughput benchmark.
# - No --quantization flag: mixed-precision NVFP4 auto-detects via
#   hf_quant_config.json (same lesson as DeepSeek-V4 on SGLang).
set -u
NAME=minimax-m3-vllm
IMAGE=vllm/vllm-openai:nightly   # PIN DIGEST AT PULL: nightly-<commit> tag
MODEL=nvidia/MiniMax-M3-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: container(s) running: $RUNNING (docker rm -f first)" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/vllm/minimax-m3/{json,logs}

echo "Launching $NAME  model=$MODEL  image=$IMAGE  TP=8 (vLLM nightly, port 8000)"
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v /mnt/nvme/hf-cache:/root/.cache/huggingface \
  -v ~/benchmark/results_610:/results \
  -p 8000:8000 \
  --name "$NAME" \
  --entrypoint vllm \
  "$IMAGE" \
  serve "$MODEL" \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --language-model-only \
  --max-model-len 16384 \
  --host 0.0.0.0 --port 8000
