# B300 Single-Node Benchmark: Standard Operating Procedure

## Aligned with InferenceX methodology

This document is **process-focused**: how to run a benchmark, step by step. The permanent rules, hardware facts, framework/config constants, known issues, and model queue all live in the project rulebook:

> **Read `~/benchmark/CLAUDE.md` first.** It is the single source of truth for:
> - Hard rules (1-container, no Ctrl+C on server, warmup-once, tmux session whitelist, results layout)
> - Node hardware and software versions (links to `results/metadata/node_info.yaml` for full detail)
> - Framework choices (SGLang primary, vLLM fallback) and their critical flags
> - Benchmark constants (TP=8, EP, max-model-len=16384, sequence profiles, concurrency sweep levels, num_prompts formula, stop conditions)
> - Known issues (vLLM EP+NVFP4 broken, SGLang TMA fallback, sglang bench `--random-range-ratio` gotcha, etc.)
> - Model queue with HF IDs and download status
>
> If a rule contradicts something below, **CLAUDE.md wins** — fix the SOP. Live state ("what's running right now, what's blocked, recent change log") lives in `~/benchmark/STATUS.md`.

The remainder of this file is the canonical step-by-step process and the historical Lessons Learned.

---

## Step -1: Cycle-Start Checklist (once per measurement cycle, BEFORE planning sweeps)

Added 2026-07-17 after the EP blind spot (April's "EP broken" verdict silently governed the July 610 cycle while upstream had already shipped working EP paths — and the fix was posted in our own issue thread 9 days before the cycle started).

1. **Re-verify stale Known Issues** (CLAUDE.md entries marked STALE or older than one cycle): run a ~1 h capability smoke matrix — at minimum EP × each framework × one model — before locking the sweep plan. Version jumps (SGLang 0.5.10→0.5.14, vLLM 0.13→0.24 class) invalidate old verdicts.
2. **Read our upstream threads** (sgl-project/sglang#24502 #31503 #31556, vllm-project/vllm#48921, and threads we've commented on): maintainer replies often contain working flag combinations. Check for new NVFP4 checkpoint revisions (V2 requants) of queue models while at it.
3. **Tag every new finding** with "as of <framework versions, date>". Untagged or expired findings must not gate configuration choices.

## Step 0: Pre-launch Check

In the `bench` tmux session:

```bash
docker ps                      # Nothing running? Good. (`howell` is in docker group, no sudo.)
docker rm -f <name>             # If something is running, kill it first.
```

---

## Step 1: Server Launch (in `server` tmux session)

**Use the wrapper.** Every model-specific serve script is a thin wrapper around `launch_server.sh`, which encapsulates the hard rules (single-container check, standard mounts, TP=8, max-model-len 16384, GPU mem 0.9, NGC 26.01 image). This is the only sanctioned way to launch a container — hand-crafted `docker run` lines keep drifting and forgetting mounts.

Either run the per-model wrapper directly:

```bash
bash ~/benchmark/scripts/serve_minimax-m2.7_nvfp4.sh   # or serve_kimi-k2.5_nvfp4.sh, etc.
```

Or invoke the launcher by hand for one-offs:

```bash
# launch_server.sh <container_name> <model_id> [extra vllm serve args...]
bash ~/benchmark/scripts/launch_server.sh \
  <container_name> \
  <model_id> \
  --quantization modelopt_fp4 \
  [--tool-call-parser ...] [--reasoning-parser ...]
```

The wrapper bind-mounts:
- `~/hf_hub_cache` → `/root/.cache/huggingface` (HF model cache — older FP8 path)
- `~/.cache/huggingface/hub` → `/root/.cache/huggingface/hub` (HF model cache — NVFP4 path)
- **`~/benchmark/results` → `/results`** (where benchmark JSONs and logs land, side-by-side)

Per-model wrapper scripts under `~/benchmark/scripts/serve_*.sh`:
- `serve_minimax-m2.7_nvfp4.sh` — vLLM path, WORKING (legacy / historical baseline, M2.7 NVFP4 vLLM 1k1k complete)
- `serve_minimax-m2.7_nvfp4_sglang.sh` — SGLang path, **WORKING** (canonical M2.7 SGLang 1k1k complete)
- `serve_minimax-m2.7_nvfp4_sglang_ep8.sh` — SGLang path with `--expert-parallel-size 8`. **DOES NOT WORK** — crashes during ModelOpt weight post-processing (modelopt_quant.py:1754 shape mismatch). Retained for the next time we test EP after an upstream fix.
- `serve_kimi-k2.5_nvfp4.sh` — vLLM path, never benchmarked
- `serve_kimi-k2.5_nvfp4_sglang.sh` — SGLang path, **WORKING** (Kimi K2.5 1k1k complete-to-knee)
- `serve_glm-5.1_nvfp4_sglang.sh` — SGLang path, **WORKING** (GLM-5.1 1k1k parser-off complete)
- `serve_deepseek-r1_nvfp4_sglang.sh` — SGLang path, **WORKING** (DeepSeek R1 1k1k complete; requires `TORCHINDUCTOR_COMPILE_THREADS=1`)
- `serve_qwen3.5-397b_nvfp4_sglang.sh` — SGLang path, **WORKING** (Qwen 3.5 397B-A17B 1k1k complete, peak 10652 t/s)
- `serve_minimax-m2.7_fp8_vllm_inst0.sh` / `inst1.sh` — dual-instance TP=4 FP8 (block-alignment workaround, see `docs/quantization_block_alignment.md`)
- (add more here as models come online)

**Framework convention in script names:** `serve_<model>_<precision>.sh` is vLLM; `serve_<model>_<precision>_sglang.sh` is SGLang. Matches the `results/vllm/` vs `results/sglang/` directory split.

Always pass `--quantization modelopt_fp4`. Never pass `--enable-expert-parallel` / `--expert-parallel-size N` — EP+NVFP4 is broken on **both** vLLM 0.13.0 (kernel dispatch rejects `expert_map`) and SGLang 0.5.10.post1 (modelopt loader shape mismatch at `modelopt_quant.py:1754`). Verified runtime on 2026-04-14. See CLAUDE.md Known Issues.

**Never pass `--reasoning-parser` or `--tool-call-parser`** for benchmark runs. SGLang reasoning parsers buffer `<think>...</think>` content before streaming, which inflates client-measured TTFT by the entire reasoning-phase duration (GLM-5.1 A/B: 3895 ms parser-on → 197 ms parser-off at conc=1).

Wait for `Application startup complete.` in logs. **Do not Ctrl+C** — stop the container from the `bench` pane via `docker rm -f <name>`.

---

## Step 2: Warmup (from `bench` session) — ONCE per container

SGLang primary — use port 30000 (vLLM still listens on 8000; swap the port for the fallback path):

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
```

Wait for a 200 response. This is the **only** warmup needed. Do not repeat it between concurrency levels or between sequence-length profiles — the per-container curl flushes CUDA graphs / JIT for all subsequent sweep levels against the same container. Pass `--ready-check-timeout-sec 0` to the bench tool so it doesn't run its own 10-min per-level readiness probe (baked into `bench_sweep.sh`).

---

## Step 3: Sequence Length Matrix (InferenceX standard)

| Profile | ISL | OSL | What it measures |
|---------|-----|-----|------------------|
| **1k1k** | 1024 | 1024 | Balanced (standard comparison point) |
| **1k4k** | 1024 | 4096 | Decode-heavy (reasoning, long generation) |
| **4k1k** | 4096 | 1024 | Prefill-heavy (RAG, long context input) |

All three profiles fit comfortably under `max-model-len=16384` (max need is 5120 tokens with buffer).

**Previous profiles `1k8k` and `8k1k` have been retired** — they required max-model-len ≥ 9216 and added decode time that didn't improve the quality of the comparison versus the shorter decode-heavy/prefill-heavy ones.

---

## Step 4: Concurrency Sweep (per sequence length)

```
Concurrency levels: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
```

Start with 1k1k. `bench_sweep.sh` applies four layered stop guards — any one triggers `break`:
1. **Non-zero `Failed requests:`** in the log
2. **Python exception / HTTP 400 / OCI runtime error** — grep `ValueError: Initial test run failed|Error: Bad Request|^Traceback|OCI runtime exec failed`
3. **Missing / empty JSON** result file
4. **Plateau** — output_throughput gain vs. previous level < `PLATEAU_THRESHOLD` (default 10%). Motivating case: Kimi K2.5 plateaued by conc=64. Realized case on this node: Qwen 3.5 397B hit conc=256=10652 t/s then conc=512=10258 t/s (−3.7%), guard fired cleanly.

### Num prompts per concurrency level

```
num_prompts = max(concurrency × 10, 40)
```

### Usage

Always invoke the canonical sweep script — it encodes all guards, the port/host override, the framework split, and the result layout:

```bash
# From the `bench` tmux session, never `server`:
# Usage: bench_sweep.sh <model_id> <precision> <seq_profile> [framework]
#   precision     nvfp4 | fp8
#   seq_profile   1k1k | 1k4k | 4k1k
#   framework     vllm | sglang  (default: vllm)
#
# Env overrides:
#   PORT          (default: 30000 — SGLang. Use PORT=8000 for vLLM.)
#   HOST          (default: 127.0.0.1)

# Example — SGLang M2.7 NVFP4, balanced profile:
bash ~/benchmark/scripts/bench_sweep.sh \
  lukealonso/MiniMax-M2.7-NVFP4 nvfp4 1k1k sglang

# Example — vLLM M2.7 NVFP4, decode-heavy profile:
PORT=8000 bash ~/benchmark/scripts/bench_sweep.sh \
  lukealonso/MiniMax-M2.7-NVFP4 nvfp4 1k4k vllm
```

The script handles: `--ready-check-timeout-sec 0`, `--base-url http://$HOST:$PORT`, `--save-result --result-filename /results/${framework}/${TAG}.json`, tee'd logs next to the JSON, and the three-layer stop conditions (failures / ValueError / missing JSON).

---

## Step 5: Repeat for other sequence lengths

After 1k1k completes, repeat Step 4 with:
- **1k4k**: ISL=1024, OSL=4096 (expect lower max concurrency due to larger KV per request)
- **4k1k**: ISL=4096, OSL=1024

---

## Step 6: Key Metrics to Report

For each (concurrency, sequence length) pair:

| Metric | JSON key | Unit |
|--------|----------|------|
| Output throughput | `output_throughput` | tok/s |
| Total throughput | `total_token_throughput` | tok/s |
| **Throughput per GPU** | total / 8 | tok/s/GPU |
| Mean TTFT | `mean_ttft_ms` | ms |
| P99 TTFT | `p99_ttft_ms` | ms |
| Mean TPOT | `mean_tpot_ms` | ms |
| P99 TPOT | `p99_tpot_ms` | ms |
| Mean ITL | `mean_itl_ms` | ms |
| Failed requests | `failed_requests` | count |

**Throughput per GPU** is the normalization metric for cross-model / cross-hardware comparison.

---

## Step 7: Precision

**NVFP4 only.** FP8 is skipped entirely for B300 — NVFP4 tensor cores are the whole point of the platform, and an FP8 baseline on the same node would just re-measure a previous-gen code path. If we need an FP8 comparison later, pull it from the H200/H100 InferenceX data instead of burning B300 time on it.

| Run | Weights | Flag |
|-----|---------|------|
| NVFP4 | NVIDIA/community NVFP4 | `--quantization modelopt_fp4` |

---

## Step 8: Result Layout & Naming Convention

Per-framework, per-model, json/logs split directories:
```
~/benchmark/results/
  <framework>/
    <model_short>/
      json/   ← all result JSONs for this (framework, model) pair
      logs/   ← matching stdout tee logs
```

`bench_sweep.sh` derives `model_short` from the HF model_id by lowercasing the basename and stripping the precision suffix (`-NVFP4` or `-FP8`). Override at runtime with `MODEL_SHORT=...`.

Filename pattern (JSON and matching log share the same TAG):
```
{model}_{precision}_{framework}_tp{TP}_conc{CONC}_{SEQ}.json
{model}_{precision}_{framework}_tp{TP}_conc{CONC}_{SEQ}.log
```

Examples:
```
~/benchmark/results/sglang/minimax-m2.7/json/MiniMax-M2.7-NVFP4_nvfp4_sglang_tp8_conc32_1k1k.json
~/benchmark/results/sglang/kimi-k2.5/json/Kimi-K2.5-NVFP4_nvfp4_sglang_tp8_conc64_1k4k.json
~/benchmark/results/vllm/minimax-m2.7/json/MiniMax-M2.7-NVFP4_nvfp4_vllm_tp8_conc16_4k1k.json
```

EP is no longer part of the tag because we always run TP-only on B300 NVFP4.

---

## Step 9: Summary Report

After all sweeps, produce a per-model summary:

```
Model: MiniMax M2.7 | Precision: NVFP4 | TP=8 | B300 NVL8

| Seq  | Conc | Out tok/s | tok/s/GPU | TTFT P99 | TPOT mean | Failed |
|------|------|-----------|-----------|----------|-----------|--------|
| 1k1k | 4    | 402       | 50.3      | 61 ms    | 9.89 ms   | 0      |
| 1k1k | 8    | ...       | ...       | ...      | ...       | 0      |
| ...  | ...  | ...       | ...       | ...      | ...       | ...    |
```

Aggregate across all models in `~/benchmark/results/summary.csv`.

---

## Model-Specific Configs

### MiniMax M2.7 (NVFP4)
```bash
MODEL=lukealonso/MiniMax-M2.7-NVFP4
# Add: --quantization modelopt_fp4
# Do NOT add --enable-expert-parallel (unsupported for NVFP4 MoE in 0.13.0)
```

### GLM-5.1 (NVFP4)
```bash
MODEL=lukealonso/GLM-5.1-NVFP4
# --quantization modelopt_fp4
# NO parsers (SGLang glm45 reasoning parser buffers <think>...</think>,
# inflates TTFT 20× at conc=1). Full details in configs/glm-5.1.yaml.
```

### Kimi K2.5 (NVFP4)
```bash
MODEL=nvidia/Kimi-K2.5-NVFP4
# --quantization modelopt_fp4
# NO parsers (project rule; also avoids kimi_k2 parser reasoning buffering).
# Known caveat: SGLang chat preprocessor takes the slow tokenizer path
# (tokenization_kimi.py:178). TTFT inflated 10–30%. See configs/kimi-k2.5.yaml.
```

### Qwen 3.5 397B-A17B (NVFP4)
```bash
MODEL=nvidia/Qwen3.5-397B-A17B-NVFP4
# --quantization modelopt_fp4 --trust-remote-code --moe-runner-backend flashinfer_trtllm
# NO parsers. TORCHINDUCTOR_COMPILE_THREADS=1 in docker -e (preventative).
```

### DeepSeek R1 (NVFP4)
```bash
MODEL=nvidia/DeepSeek-R1-NVFP4
# --quantization modelopt_fp4 --trust-remote-code
# --attention-backend trtllm_mla --moe-runner-backend flashinfer_trtllm
# --kv-cache-dtype fp8_e4m3 (auto-promotes on DeepSeek-V3 MLA anyway; explicit for auditability)
# REQUIRED env: TORCHINDUCTOR_COMPILE_THREADS=1 to bypass the inductor
# compile-worker CUDA-init bug hit in vocab_parallel_embedding. Full
# trace in results/metadata/DeepSeek-R1-NVFP4_sglang_startup.yaml.
# NVIDIA HF card lists TRT-LLM only but SGLang and vLLM both work.
```

### DeepSeek V3.2 (NVFP4)
```bash
MODEL=nvidia/DeepSeek-V3.2-NVFP4
# Not downloaded; HF card marks TRT-LLM only. Deferred.
# Same DeepSeek-V3 class as R1, so same flag set should work once downloaded.
```

---

## Pre-launch sanity check

Before launching any container, verify `max_model_len ≥ max(ISL+OSL)` across **every** profile you plan to benchmark against it. If a request asks for more tokens than the context window allows, the server returns HTTP 400 and the bench tool raises `ValueError: Initial test run failed ... Error: Bad Request`. For the standard three InferenceX profiles (1k1k, 1k4k, 4k1k) `max-model-len=16384` is sufficient — this is the default in `launch_server.sh`.

`bench_sweep.sh` has layered stop conditions that will now catch this (added 2026-04-13 after the fact): **(1)** non-zero `"Failed requests:"` count, **(2)** `ValueError: Initial test run failed`, `Error: Bad Request`, or `^Traceback` in the log, **(3)** missing `${TAG}.json` result file. Any one trips `break` on the for loop, so the harness stops at the first failing concurrency level instead of marching through the rest producing empty logs.

---

## Lessons Learned (2026-04-13 session)

Recording mistakes made during the first real bench day so they don't repeat.

1. **Ctrl+C on the server tmux pane killed a vLLM process mid-compile**, losing ~7 min of torch.compile work. Even if Ctrl+C seems to "just queue as stdin", it really reaches the foreground process. **Rule:** never signal the `server` pane. Stop containers from `bench` via `sudo docker rm -f`.
2. **Saved benchmark JSONs into `~/hf_hub_cache/` (the HF model cache) just because that bind mount already existed.** Mixed results with model blobs. **Rule:** results belong under `~/benchmark/results/raw/`. Every container must bind-mount `~/benchmark/results:/results` and pass `--result-filename /results/raw/...`. Baked into `launch_server.sh` so this can't happen again.
3. **Ran `vllm bench sweep --help` in the `download` tmux session because `bench` was busy.** That cross-polluted a session meant for one thing. **Rule:** one session, one purpose. If `bench` is busy and you need to probe something, use `scratch`, never `download` or `server`.
4. **Launched a duplicate `sudo docker run` by accident** — my first attempt hit a sudo password prompt and I reissued from a different tmux session, so when the password was finally entered both queued invocations fired. **Rule:** always `sudo docker ps -a` first; kill any `Created` phantoms before relaunching. The wrapper script now refuses to run if anything is already live.
5. **Did not notice `vllm bench serve` has `--ready-check-timeout-sec`** — spent time considering monkey-patching `benchmark_serving.py` before finding a supported flag that already does exactly what we wanted. **Rule:** `grep add_argument` in the source file before reaching for a patch.
6. **`hf_transfer` download hung silently in `CLOSE-WAIT`** for 1h40m — process alive, zero bytes, no error. **Rule:** monitor with file mtime on the `.incomplete` shards, not just `ps`. If mtime is stale, kill `huggingface-cli` with SIGTERM and restart (HF skips completed blobs; hf_transfer re-fetches any partial `.incomplete` file from scratch).
7. **Tee'd the Kimi download log into `~/benchmark/results/raw/`** — results dir should hold results only, not incidental command stdout. **Rule:** if a command is producing logs that aren't benchmark data, send them to `/tmp` or rely on tmux scrollback, not `results/raw`.
8. **Launched the container with `max-model-len=8192` and did not check it against the 1k8k profile.** 1k8k asks for ISL+OSL = 1024+8192 = 9216 tokens > 8192, so every request returned HTTP 400. The bench tool raised `ValueError: Initial test run failed ... Error: Bad Request`, which does not match `bench_sweep.sh`'s "Failed requests" guard string, so the loop kept marching through all 9 levels producing empty logs and no JSONs for **hours** before I noticed. **Rules:** (a) always verify `max_model_len ≥ max(ISL+OSL)` across every planned profile before launching; (b) also scan logs for `ValueError`, `Bad Request`, and `Traceback` in addition to "Failed requests" when deciding whether to stop a sweep.
9. **Misread `--random-range-ratio 0.0` as "no variation".** sglang's `bench_serving` random dataset uses `np.random.randint(max(int(full_len * range_ratio), 1), full_len + 1)` to sample output lengths per prompt. With `range_ratio=0.0` (the default), lower bound becomes 1 and every prompt gets a uniformly random `output_len ∈ [1, full_len]`. Observed on 2026-04-13: a 1k1k sweep produced `total_output_tokens=20398` instead of the expected 40960 (40 × 1024), breaking apples-to-apples against vLLM's fixed-length output. I initially misdiagnosed this as `ignore_eos` not being honored, spent ~30 min chasing source-level patches in the chat completions function, and only found the real bug after dumping the actual payload to see `max_completion_tokens: 141` (server was faithfully generating exactly what was asked). **Rules:** (a) for fixed-length generation under sglang.bench_serving use `--random-range-ratio 1.0` (this makes the randint range `[full_len, full_len+1)`, i.e. exactly `full_len`); (b) when the observed `total_output_tokens` diverges from `num_prompts × random_output_len`, suspect the client-side dataset sampler before blaming server-side EOS handling; (c) inject a payload debug print in the bench tool rather than debugging via server behavior alone — it turns a multi-hour guessing game into a 2-minute confirmation. Both `bench_sweep.sh` and the SOP now bake in `--random-range-ratio 1.0` for the sglang branch.

10. **EP+NVFP4 is broken on SGLang too — verified by trying it (2026-04-14 00:52).** I'd been stating throughout the day that "SGLang's flashinfer cutlass MoE allgather FP4 path is enabled by default, so EP probably works" based on the presence of `--expert-parallel-size` and `--disable-flashinfer-cutlass-moe-fp4-allgather` flags in `sglang.launch_server --help`. When we actually launched `--expert-parallel-size 8` on M2.7 NVFP4, it crashed during model load:
    ```
    File "sglang/srt/layers/quantization/modelopt_quant.py", line 1754, in process_weights_after_loading
        (w13_input_scale * w13_weight_scale_2).to(torch.float32),
    RuntimeError: The size of tensor a (256) must match the size of tensor b (32) at non-singleton dimension 0
    ```
    The ModelOpt loader replicates `w13_input_scale` across all 256 experts but EP-shards `w13_weight_scale_2` to the per-rank 32, then multiplies them element-wise. So both vLLM and SGLang are currently broken for EP+NVFP4 — vLLM at kernel dispatch, SGLang at weight post-processing. **Rules:** (a) treat "flag exists in `--help`" as a *necessary but not sufficient* signal — runtime verification is required before claiming EP support; (b) when documenting upstream feature support, separate the column "flag accepted" from the column "runtime verified at this version on this hardware"; (c) the cheap test is a small-num-prompts, low-conc launch — under 5 minutes for M2.7 — so always do this before a long sweep, and use it specifically to validate any flag whose behavior you're not 100% sure of.
