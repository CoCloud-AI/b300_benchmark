#!/usr/bin/env bash
# download_extras_610.sh — second wave of NVFP4 weights for the driver-610 bench.
# Added after the initial 5-model queue (see download_models_610.sh), per user
# request 2026-07-08 to also cover StepFun + Xiaomi(MiMo) and re-run Qwen on the
# new driver. Sequential, one at a time (shared network). Resumable.
set -uo pipefail
export HF_HOME=/mnt/nvme/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"
LOGDIR=/home/ken/dl_logs; mkdir -p "$LOGDIR"

MODELS=(
  nvidia/Qwen3.5-397B-A17B-NVFP4-V2   # 244 GB — newer V2 requant of the 595-era Qwen; qwen3_5_moe
  stepfun-ai/Step-3.7-Flash-NVFP4     # 129 GB — StepFun, step3p7 MoE (official)
  lukealonso/MiMo-V2.5-NVFP4          # 184 GB — Xiaomi MiMo-V2.5, mimo_v2 (48L/256e/8act), reputable quant
)

for m in "${MODELS[@]}"; do
  tag=$(echo "$m" | tr '/' '_')
  echo "=== $(date -u '+%F %T') START  $m ===" | tee -a "$LOGDIR/download.log"
  hf download "$m" >>"$LOGDIR/$tag.log" 2>&1
  echo "=== $(date -u '+%F %T') FINISH $m rc=$? ===" | tee -a "$LOGDIR/download.log"
done
echo "=== $(date -u '+%F %T') EXTRAS DOWNLOAD COMPLETE ===" | tee -a "$LOGDIR/download.log"
