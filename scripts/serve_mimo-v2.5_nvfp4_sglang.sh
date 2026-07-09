#!/bin/bash
# serve_mimo-v2.5_nvfp4_sglang.sh
# Status: ITERATION — --dp-size 2 --enable-dp-attention added per the model's own
#   error: "MiMoV2ForCausalLM requires effective attention TP size 4 because its
#   fused qkv_proj weights are TP=4-interleaved". tp=8/dp=2 -> attn TP=4, all 8
#   GPUs used (MoE TP=8, attention DP=2xTP=4). Earlier: SGLang attempt. vLLM v0.24 chain: (1) transformers lacks
# mimo_v2 + quant stripped custom code -> grafted configuration_mimo_v2.py +
# auto_map + HF_HUB_OFFLINE=1 (config fix WORKED); (2) then hard assert
# "TP size must evenly split the number of KV heads" (num_kv_heads=4, TP=8).
# SGLang replicates KV heads at tp>kv for GQA models — testing that here.
# HF_HUB_OFFLINE=1 required for the grafted config to be used.
set -u
NAME=mimo25-sglang
IMAGE=sglang-b300:v0.5.14
MODEL=/root/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4/snapshots/efadb57636ca42844aa96079b323863fad83a847  # local path -> dynamic module loads from dir, no hub lookup
SNAPHASH_NOTE="grafted configuration_mimo_v2.py lives here"
RUNNING=$(docker ps --format '{{.Names}}')
if [ -n "$RUNNING" ]; then echo "ERROR: running: $RUNNING" >&2; exit 2; fi
mkdir -p ~/benchmark/results_610/sglang/mimo-v2.5/{json,logs}
docker run --gpus all --shm-size 32g --ipc=host --ulimit memlock=-1 \
  -v /mnt/nvme/hf-cache:/root/.cache/huggingface \
  -v ~/benchmark/results_610:/results \
  -p 30000:30000 \
  -e HF_HUB_OFFLINE=1 \
  -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  --name "$NAME" "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name lukealonso/MiMo-V2.5-NVFP4 --tp 8 --dp-size 2 --enable-dp-attention --language-only --trust-remote-code \
  --context-length 16384 --host 0.0.0.0 --port 30000
