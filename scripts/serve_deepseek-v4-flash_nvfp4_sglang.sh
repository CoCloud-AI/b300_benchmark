#!/bin/bash
# serve_deepseek-v4-flash_nvfp4_sglang.sh
# Status: WORKING (verified 2026-07-08, driver 610.43.02 / SGLang 0.5.14 / deepseek_v4)
#   Application startup complete, warmup 200, 1k1k sweep running.
#   Lesson: MINIMAL flags — SGLang 0.5.14 natively supports deepseek_v4 and
#   auto-selects attention=dsv4, quant=ModelOptNvFp4FusedMoEMethod, moe=auto.
#   Forcing --quantization modelopt_fp4 / --moe-runner-backend flashinfer_trtllm /
#   --attention-backend trtllm_mla all FAIL (see removed flags below).
#
# DeepSeek-V4-Flash NVFP4 on B300 TP=8, driver 610.43.02, SGLang 0.5.14.
#
# Model: nvidia/DeepSeek-V4-Flash-NVFP4 (168 GB)
#   - model_type: deepseek_v4 / DeepseekV4ForCausalLM (MLA lineage of R1/V3)
#   - 43 layers, hidden=4096, 64 heads, 1 KV head, q_lora_rank=1024
#   - 256 routed experts, 6 experts/token, moe_intermediate=2048
#   - max_position_embeddings=1048576 (1M ctx; we cap at 16384 for the bench)
#   - Quant: MIXED_PRECISION — experts NVFP4 (group 16), attn/shared FP8.
#     SGLang reads hf_quant_config.json; --quantization modelopt_fp4 is passed
#     to match the R1 precedent. If 0.5.14 rejects it on the mixed-precision
#     checkpoint, drop the flag and let SGLang auto-detect.
#
# Flag basis: serve_deepseek-r1_nvfp4_sglang.sh (same DeepSeek MLA class), plus
# TORCHINDUCTOR_COMPILE_THREADS=1 carried forward (inductor compile-worker
# CUDA-init bug on the DeepSeek-V3/V4 vocab_parallel_embedding path). No parsers
# (benchmark rule). EP=1 (TP-only).
#
# NOTE (this node): HF cache lives on the NVMe pool (/mnt/nvme/hf-cache) and
# results go to results_610/. Plain `docker` (ken is in the docker group — use a
# fresh shell/tmux, or `sg docker -c '...'`).

set -u

NAME=dsv4-flash
IMAGE=sglang-b300:v0.5.14
MODEL=nvidia/DeepSeek-V4-Flash-NVFP4

RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then
  echo "ERROR: container(s) already running: $RUNNING" >&2
  echo "Kill first: docker rm -f <name>" >&2
  exit 2
fi

mkdir -p ~/benchmark/results_610/sglang/deepseek-v4-flash/{json,logs}

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
# Backend flags intentionally NOT forced. SGLang 0.5.14 natively supports
# model_type=deepseek_v4 and auto-selects attention_backend='dsv4'. Forcing
# --moe-runner-backend flashinfer_trtllm (R1 era) CRASHES here: the TRTLLM FP4
# MoE kernel rejects V4-Flash's grouped routing with
#   "For DeepSeek routing method, must have topkGroup <= 4"
# (trtllm_fused_moe_runner.cu:148). Let auto pick the V4-compatible MoE path.
