#!/usr/bin/env bash
# run_3profiles_610.sh <hf_model_id> [framework] — run the 1k1k, 1k4k, 4k1k
# concurrency sweeps sequentially against the ALREADY-WARM server container.
#
# Exists because chaining bench_sweep_610.sh invocations through
# `tmux send-keys 'sg docker -c "for P in ...; do ... $P ...; done"'` is a
# quoting trap: the pane shell expands $P (unset -> empty) before sg's child
# shell runs, so every iteration loses the profile argument and dies with
# "Unknown seq profile". (Bit us on V4-Pro 2026-07-09, cost ~30 idle GPU min.)
# A script has no such problem — always drive multi-profile runs through this.
#
# Usage (from bench tmux):
#   sg docker -c "bash ~/benchmark/scripts/run_3profiles_610.sh nvidia/DeepSeek-V4-Pro-NVFP4"
set -u
MODEL="${1:?hf model id required}"
FRAMEWORK="${2:-sglang}"
SHORT=$(basename "$MODEL"); SHORT=${SHORT%-NVFP4}; SHORT=${SHORT%-nvfp4}; SHORT=${SHORT,,}
LOGDIR=/home/ken/dl_logs
# vLLM serves on 8000 (SGLang on 30000). bench_sweep_610.sh honors $PORT.
if [ "$FRAMEWORK" = "vllm" ]; then export PORT="${PORT:-8000}"; fi

for PROF in 1k1k 1k4k 4k1k; do
  echo "=== $(date '+%F %T') run_3profiles: starting $PROF for $MODEL ($FRAMEWORK) ==="
  bash ~/benchmark/scripts/bench_sweep_610.sh "$MODEL" nvfp4 "$PROF" "$FRAMEWORK" 2>&1 \
    | tee "$LOGDIR/sweep_${SHORT}_${PROF}.log"
done
echo "=== $(date '+%F %T') run_3profiles: ALL 3 PROFILES DONE for $MODEL ==="
