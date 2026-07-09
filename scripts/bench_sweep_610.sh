#!/bin/bash
# bench_sweep_610.sh — Concurrency sweep, driver 610 cycle (writes to results_610/)
#
# Usage:
#   ./bench_sweep.sh <model_id> <precision> <seq_profile> [framework]
#
# Positional args:
#   model_id    HuggingFace ID of the served model (must match server --model)
#   precision   nvfp4 | fp8
#   seq_profile 1k1k | 1k4k | 4k1k
#   framework   sglang | vllm   (default: sglang)
#
# Env overrides:
#   PORT   Server port (default: 30000 = SGLang. Use PORT=8000 for vLLM.)
#   HOST   Server host (default: 127.0.0.1)
#
# Sequence profiles:
#   1k1k  ISL=1024  OSL=1024   balanced
#   1k4k  ISL=1024  OSL=4096   decode-heavy
#   4k1k  ISL=4096  OSL=1024   prefill-heavy
# max-model-len=16384 covers all three plus headroom.
#
# Bench tool per framework (chosen to match the server's own image, so we
# always docker exec into a running container and don't spawn extras):
#   sglang -> python3 -m sglang.bench_serving   (backend: sglang-oai-chat)
#   vllm   -> vllm bench serve                  (backend: openai-chat)
#
# Results layout (per-model, per-framework):
#   ~/benchmark/results_610/${framework}/${model_short}/json/${TAG}.json
#   ~/benchmark/results_610/${framework}/${model_short}/logs/${TAG}.log
# with TAG = "<model>_<precision>_<framework>_tp8_conc<C>_<SEQ>"
#
# model_short is auto-derived from model_id by taking the basename and
# stripping the precision suffix. Examples:
#   lukealonso/MiniMax-M2.7-NVFP4 → minimax-m2.7
#   nvidia/Kimi-K2.5-NVFP4        → kimi-k2.5
#   nvidia/Qwen3.5-397B-A17B-NVFP4 → qwen3.5-397b-a17b
#   lukealonso/GLM-5.1-NVFP4      → glm-5.1
# Override with: MODEL_SHORT=my-name ./bench_sweep.sh ...

set -u

MODEL="${1:?model id required}"
PRECISION="${2:?precision required (nvfp4|fp8)}"
SEQ="${3:?seq profile required (1k1k|1k4k|4k1k)}"
FRAMEWORK="${4:-sglang}"

PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"

case "$SEQ" in
  1k1k) ISL=1024; OSL=1024 ;;
  1k4k) ISL=1024; OSL=4096 ;;
  4k1k) ISL=4096; OSL=1024 ;;
  *) echo "Unknown seq profile: $SEQ (expected 1k1k|1k4k|4k1k)"; exit 1 ;;
esac

case "$FRAMEWORK" in
  sglang|vllm) : ;;
  *) echo "Unknown framework: $FRAMEWORK (expected sglang|vllm)"; exit 1 ;;
esac

CID=$(docker ps -q | head -1)
if [ -z "$CID" ]; then
  echo "ERROR: no running container. Launch the server first." >&2
  exit 2
fi

# Auto-derive a short, lowercase model identifier from the HF model_id.
# Strip leading org/, then strip trailing -NVFP4 / -FP8 precision suffix,
# then lowercase. Allow override via MODEL_SHORT env var.
_model_base="${MODEL##*/}"                    # MiniMax-M2.7-NVFP4
_model_base="${_model_base%-NVFP4}"           # MiniMax-M2.7
_model_base="${_model_base%-nvfp4}"
_model_base="${_model_base%-FP8}"
_model_base="${_model_base%-fp8}"
MODEL_SHORT="${MODEL_SHORT:-${_model_base,,}}"   # lowercase

# Per-(framework, model_short) result directories.
RESULTS_DIR=~/benchmark/results_610/${FRAMEWORK}/${MODEL_SHORT}
JSON_DIR=${RESULTS_DIR}/json
LOG_DIR=${RESULTS_DIR}/logs
mkdir -p "$JSON_DIR" "$LOG_DIR"

echo "Sweep: framework=$FRAMEWORK model=$MODEL precision=$PRECISION seq=$SEQ model_short=$MODEL_SHORT"
echo "Server at $BASE_URL (container $CID)"
echo "  json -> $JSON_DIR"
echo "  logs -> $LOG_DIR"

# Plateau threshold: if output_throughput gain vs. previous level drops below
# this fraction, stop the sweep. 0.10 = 10% — matches the observed Kimi 1k1k
# knee (conc=64→128 was +0.6%, clearly plateaued; conc=32→64 was +4.3%, also
# effectively plateaued). Tweak by exporting PLATEAU_THRESHOLD before calling.
PLATEAU_THRESHOLD="${PLATEAU_THRESHOLD:-0.10}"

PREV_TPS=""   # output_throughput from previous concurrency level (blank on first)

for CONC in 1 2 4 8 16 32 64 128 256 512; do
  PROMPTS=$((CONC > 4 ? CONC * 10 : 40))
  TAG="${MODEL##*/}_${PRECISION}_${FRAMEWORK}_tp8_conc${CONC}_${SEQ}"
  LOG=${LOG_DIR}/${TAG}.log
  JSON=${JSON_DIR}/${TAG}.json
  echo "=== $(date) | Conc=$CONC | Prompts=$PROMPTS | $SEQ | $FRAMEWORK ==="

  if [ "$FRAMEWORK" = "sglang" ]; then
    # sglang.bench_serving ships inside the sglang container. We docker exec
    # so the client lives next to the server (CPU-only load, no new container).
    # --ready-check-timeout-sec 0 skips the built-in readiness probe since we
    # curl-warmup once at container launch per SOP.
    #
    # --random-range-ratio 1.0 is CRITICAL. The default is 0.0 which does NOT
    # mean "no variation" — it means "sample output_len uniformly from [1,
    # full_len]". With range_ratio=0.0 every prompt gets a random output_len
    # in [1, 1024] (for 1k1k), so total_output_tokens ends up ~half the
    # requested 40*1024=40960 (observed 20398 on 2026-04-13, then 141 tokens
    # on a single-prompt probe). Setting range_ratio=1.0 makes the sampler
    # return exactly full_len every time, which is what we want for a clean
    # concurrency sweep. See compute_random_lens() in sglang's
    # benchmark/datasets/common.py — the np.random.randint(lower, upper)
    # call with lower=max(int(full_len*0.0), 1)=1 produces random lengths.
    #
    # ignore_eos is already set to True by default inside sglang.bench_serving
    # (disable_ignore_eos=False → payload["ignore_eos"] = not False = True),
    # so no --extra-request-body is needed here.
    docker exec -i "$CID" \
      python3 -m sglang.bench_serving \
      --backend sglang-oai-chat \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --dataset-name random \
      --random-input-len $ISL \
      --random-output-len $OSL \
      --random-range-ratio 1.0 \
      --num-prompts $PROMPTS \
      --max-concurrency $CONC \
      --ready-check-timeout-sec 0 \
      --output-file /results/${FRAMEWORK}/${MODEL_SHORT}/json/${TAG}.json \
      2>&1 | tee "$LOG"
  else
    # vLLM bench tool — lives in the vLLM container.
    # v0.24 verified 2026-07-09 (--help=all):
    # - --random-range-ratio default 0.0 = EXACT fixed lengths on vLLM (samples
    #   [len*(1-r), len*(1+r)] — OPPOSITE convention from sglang's 0.0-trap), so
    #   we deliberately pass nothing.
    # - --ignore-eos added: sglang's chat backend sets ignore_eos implicitly,
    #   vLLM's default is False; without it early EOS would silently shorten
    #   outputs and break apples-to-apples token counts.
    docker exec -i "$CID" \
      vllm bench serve \
      --backend openai-chat \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --dataset-name random \
      --random-input-len $ISL \
      --random-output-len $OSL \
      --num-prompts $PROMPTS \
      --max-concurrency $CONC \
      --trust-remote-code \
      --ignore-eos \
      --ready-check-timeout-sec 0 \
      --endpoint /v1/chat/completions \
      --save-result \
      --result-filename /results/${FRAMEWORK}/${MODEL_SHORT}/json/${TAG}.json \
      2>&1 | tee "$LOG"
  fi

  # Guard 1: bench tool reports a non-zero failure count.
  if grep -q "Failed requests:" "$LOG"; then
    FAILS=$(grep "Failed requests:" "$LOG" | awk '{print $NF}')
    if [ "$FAILS" != "0" ] 2>/dev/null; then
      echo "!!! $FAILS failures at conc=$CONC — stopping sweep"
      break
    fi
  fi

  # Guard 2: Python exception / HTTP 400 / context-window overflow / OCI runtime
  # errors (e.g. missing executable in the container PATH).
  if grep -qE "ValueError: Initial test run failed|Error: Bad Request|^Traceback|OCI runtime exec failed" "$LOG"; then
    echo "!!! bench tool error at conc=$CONC (see $LOG) — stopping sweep"
    break
  fi

  # Guard 3: expected result file must exist and be non-empty.
  if [ ! -s "$JSON" ]; then
    echo "!!! no JSON result produced at conc=$CONC — stopping sweep"
    break
  fi

  # Guard 4: plateau detection. Parse output_throughput (top-level float, same
  # key for sglang.bench_serving and vllm bench serve) and compare to the
  # previous level. Stop if the relative gain is below PLATEAU_THRESHOLD.
  # This is the mechanism that avoids wasting time at conc=256/512 when the
  # throughput curve has already flattened (see Kimi K2.5 1k1k — plateaued by
  # conc=64, stopped at conc=128 manually in the 2026-04-14 session).
  CUR_TPS=$(python3 -c "
import json, sys
try:
    d = json.load(open('$JSON'))
    v = d.get('output_throughput')
    print(v if v is not None else '')
except Exception as e:
    sys.stderr.write(f'parse failed: {e}\n')
    print('')
" 2>/dev/null)

  if [ -z "$CUR_TPS" ]; then
    echo "!!! could not parse output_throughput from $JSON — stopping sweep"
    break
  fi

  if [ -n "$PREV_TPS" ]; then
    # Use python for the float math — bash doesn't handle floats natively.
    read -r GAIN_PCT STOP < <(python3 -c "
prev = float('$PREV_TPS')
cur  = float('$CUR_TPS')
thr  = float('$PLATEAU_THRESHOLD')
gain = (cur - prev) / prev if prev > 0 else 0.0
stop = 1 if gain < thr else 0
print(f'{gain*100:.2f} {stop}')
")
    echo "+++ conc=$CONC output_throughput=${CUR_TPS} tok/s (gain vs prev ${GAIN_PCT}%)"
    if [ "$STOP" = "1" ]; then
      echo "!!! plateau detected at conc=$CONC (gain ${GAIN_PCT}% < $(python3 -c "print(float('$PLATEAU_THRESHOLD')*100)")%) — stopping sweep"
      break
    fi
  else
    echo "+++ conc=$CONC output_throughput=${CUR_TPS} tok/s (baseline)"
  fi
  PREV_TPS="$CUR_TPS"
done
