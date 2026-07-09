#!/usr/bin/env bash
# download_models_610.sh — pull NVFP4 weights for the driver-610 B300 benchmark node.
# Runs in the `download` tmux session (CLAUDE.md: one download at a time, shared network).
# Smallest-first so the first model can be benchmarked while the big ones still pull.
# Cache lands under $HF_HOME/hub (NVMe pool). NVFP4-only: each repo is a dedicated -NVFP4 checkpoint.
set -uo pipefail
export HF_HOME=/mnt/nvme/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"

LOGDIR=/home/ken/dl_logs          # logs NOT under results/ (results dir holds benchmark data only)
mkdir -p "$LOGDIR"

MODELS=(
  nvidia/DeepSeek-V4-Flash-NVFP4    # 168 GB
  nvidia/MiniMax-M3-NVFP4           # 250 GB
  nvidia/GLM-5.2-NVFP4              # 465 GB
  nvidia/Kimi-K2.7-Code-NVFP4      # 595 GB
  nvidia/DeepSeek-V4-Pro-NVFP4     # 913 GB
)

for m in "${MODELS[@]}"; do
  tag=$(echo "$m" | tr '/' '_')
  echo "=== $(date -u '+%F %T') START  $m ===" | tee -a "$LOGDIR/download.log"
  hf download "$m" >>"$LOGDIR/$tag.log" 2>&1
  rc=$?
  echo "=== $(date -u '+%F %T') FINISH $m rc=$rc ===" | tee -a "$LOGDIR/download.log"
done
echo "=== $(date -u '+%F %T') ALL DOWNLOADS COMPLETE ===" | tee -a "$LOGDIR/download.log"
