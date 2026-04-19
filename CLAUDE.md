# B300 Benchmark Project — Claude Code Instructions

## Rules (always follow)
- **No `--reasoning-parser` or `--tool-call-parser` in benchmark runs.** These post-processors are for production / agentic deployments, not throughput testing. SGLang's reasoning parsers (e.g. `glm45`) buffer `<think>...</think>` content before streaming, so client-measured TTFT becomes "time to first *post-reasoning* token" — inflated by the entire reasoning-phase duration. On GLM-5.1 1k1k conc=1 the parser-on TTFT was 3895 ms; parser-off TTFT is expected to be ~200–500 ms. All sweeps must be apples-to-apples across models, so **no parsers anywhere**. Original observation 2026-04-15: see `configs/glm-5.1.yaml` parser_notes. Parser-on results are never canonical; archive them to `json_withparsers/` if useful for comparison.
- **No sudo for docker.** `howell` is in the `docker` group (added 2026-04-15). All docker commands (`docker run`, `docker ps`, `docker exec`, `docker rm`) run as the user — never prefix with sudo. The tty_ticket sudo-cache pain is gone.
- Only 1 Docker container at a time. Kill before starting new.
- Never Ctrl+C a running server. Use `docker rm -f` from another session to stop it. Use `kill <pid>` (not Ctrl+C) for bench loops and other non-server processes.
- Warmup once per container launch, not between concurrency levels. The per-container curl flushes CUDA graphs / JIT for all subsequent sweep levels against the same container.
- Use tmux sessions: `server`, `bench`, `download`, `claudecode` only. No extra sessions or panes. Do not cross-pollute (e.g. no bench probes in `download`).
- Save results under `~/benchmark/results/<framework>/<model_short>/{json,logs}/`. `bench_sweep.sh` derives `model_short` automatically (lowercased basename with `-NVFP4`/`-FP8` stripped — e.g. `lukealonso/MiniMax-M2.7-NVFP4` → `minimax-m2.7`). Override with `MODEL_SHORT=...` env var.
- Do not touch the `download` session unless explicitly told to — active downloads live there.
- Queue downloads one at a time (network is shared with the bench node).
- When making important changes, update CLAUDE.md, SOP.md, STATUS.md, and `configs/*.yaml` immediately. Persistent memory in `~/.claude/projects/-home-howell/memory/` holds cross-session gotchas that are too deep for a rulebook line.
- **Auto-commit after each model's full sweep.** After completing all 3 profiles (1k1k/1k4k/4k1k) for a model, automatically run:
  ```bash
  cd ~/benchmark
  git add -A
  git commit -m "results: <model> <precision> <driver> - <profiles_completed>"
  git push
  ```
  Example message: `results: qwen3.5-397b nvfp4 driver595 - 1k1k+1k4k+4k1k`. Git repo is at `~/benchmark` (GitHub: kenzhangwangshu/b300_benchmark). Don't wait for user prompt — commit+push before killing the container so work is safe.

## Node
- Hardware: 8× NVIDIA B300 SXM6 AC, 288 GB HBM3e each, SM 103a (detected as B200-class by some tooling)
- Driver: 590.48.01, CUDA runtime 13.1
- Host: Intel Xeon 6776P, 2× 64-core × 2 threads, 2.0 TiB RAM
- OS: Ubuntu 22.04.5 LTS, kernel 5.15.0-119-generic, Python 3.10.12, user `howell`
- Full detail: `~/benchmark/results/metadata/node_info.yaml` (canonical source — this section is a summary only)
- Driver 590.48 limits: **NGC 26.03 crashes at high concurrency** (needs 595+). Use NGC 26.01 for vLLM until driver upgrade.

## Framework
- **Primary: SGLang** — `lmsysorg/sglang:latest-cu130-runtime`, version 0.5.10.post1, port 30000
- Secondary: vLLM NGC 26.01 — `nvcr.io/nvidia/vllm:26.01-py3`, version 0.13.0, port 8000 (only for historical/fallback runs)
- SGLang bench tool: `python3 -m sglang.bench_serving --backend sglang-oai-chat`
- vLLM bench tool: `vllm bench serve --backend openai-chat`
- **Critical sglang bench flag: `--random-range-ratio 1.0`** — the default 0.0 silently samples per-prompt output lengths uniformly from `[1, random_output_len]`, producing ~half the expected total tokens. 1.0 pins every prompt to exactly `random_output_len`. `ignore_eos=true` is already set by default in the chat-backend payload, so no `--extra-request-body` is needed. Both already baked into `bench_sweep.sh`.
- Always pass `--ready-check-timeout-sec 0` to skip the hardcoded per-invocation readiness probe (the bench tool's own JIT flush that decodes `random_output_len` tokens). We curl-warmup once at container launch per SOP.

## Benchmark Config
- TP: 8 for all models (8 GPUs)
- EP: **broken on both frameworks** for NVFP4 at this snapshot. vLLM 0.13.0 rejects EP at kernel dispatch (`cutlass_moe_fp4` won't accept `expert_map`). SGLang 0.5.10.post1 accepts the flag but **crashes during model load** in `sglang/srt/layers/quantization/modelopt_quant.py:1754` with a shape mismatch — `w13_input_scale` is loaded full-size [256, ...] but `w13_weight_scale_2` is EP-sharded to [32, ...] per rank, then their element-wise multiply explodes. Verified on M2.7 NVFP4 2026-04-14 00:52. Use TP-only sharding everywhere until the loader is fixed upstream. Possible workarounds (untried): `--moe-runner-backend flashinfer_trtllm`, or re-quantize the checkpoint with EP-aware ModelOpt scales.
- Sequence profiles: **1k1k (1024/1024), 1k4k (1024/4096), 4k1k (4096/1024)**. Previous 1k8k/8k1k retired (required max-model-len ≥ 9216).
- Concurrency sweep: 1, 2, 4, 8, 16, 32, 64, 128, 256 (extend to 512 if throughput is still climbing at 256)
- max-model-len: **16384** (fits all three profiles with headroom; avoids per-profile restart cost)
- num_prompts = max(concurrency × 10, 40) — 40 is the P99 stability floor
- Stop sweep on any of: non-zero `Failed requests`; `ValueError: Initial test run failed`, `Error: Bad Request`, `Traceback`, `OCI runtime exec failed`; missing JSON output file; **plateau** (output_throughput gain vs previous concurrency level < 10%, threshold overridable via `PLATEAU_THRESHOLD` env var). All layered into `bench_sweep.sh`. The plateau guard parses `output_throughput` from each JSON (same key on sglang and vllm) and compares to the prior level — this avoids wasting time at conc=256/512 when the curve has already flattened (motivating case: Kimi K2.5 1k1k, plateaued by conc=64).

## Known Issues
- **EP+NVFP4 broken on BOTH vLLM 0.13.0 AND SGLang 0.5.10.post1.** vLLM rejects at the kernel dispatcher; SGLang crashes during ModelOpt weight post-processing with a w13_input_scale [256] vs w13_weight_scale_2 [32] shape mismatch. Use TP-only on B300 NVFP4 until upstream fixes land. (Verified on M2.7 2026-04-14 00:52.)
- **Block-FP8 TP alignment** — block-FP8 (DeepSeek / vLLM / SGLang `fp8.py`) uses 128×128 blocks; per-rank output dim must be divisible by 128. **M2.7's `moe_intermediate_size=1536` caps FP8 at TP=4** (1536/8 = 192 breaks the block; 1536/4 = 384 is fine). NVFP4 uses 16-element groups and has no such problem at TP=8 for any queue model. On an 8-GPU node, M2.7 FP8 **must** be deployed as 2× TP=4 replicas. Full details and per-model TP ceilings: `docs/quantization_block_alignment.md`.
- **M2.7 FP8 TP=8 on vLLM 0.13.0 deadlocks silently** — vLLM accepts `--tensor-parallel-size 8 --enable-expert-parallel` at argparse time, logs "Expert parallelism is enabled" for all 8 ranks, then hangs during weight load / KV cache profiling with 0% GPU utilization for 15+ minutes. Processes are in `S` state (blocked), not running. Root cause is the same block-alignment issue as SGLang; vLLM just fails later and more quietly. Use dual-instance TP=4 instead.
- **torch.inductor compile-worker subprocess CUDA-init bug (DeepSeek V3 class models on SGLang 0.5.10.post1).** When SGLang captures CUDA graphs, the `get_masked_input_and_mask` function in `vocab_parallel_embedding.py:474` is dynamo-decorated and triggers on-demand Triton kernel compilation — **regardless of `--enable-torch-compile=False`**. Inductor spawns compile-worker subprocesses with `--kind=fork --workers=32`; fork'd workers don't inherit CUDA context cleanly, so `triton_helpers.set_driver_to_gpu()` raises `RuntimeError: Could not find an active GPU backend` and the capture dies. **Flaky** — Kimi K2.5 (same DeepseekV3ForCausalLM class) and GLM-5.1 didn't hit it due to non-deterministic kernel compile ordering; DeepSeek R1 hit it reproducibly 2026-04-15. **Fix: `-e TORCHINDUCTOR_COMPILE_THREADS=1`** in the docker run, which forces in-process single-thread compilation and bypasses the subprocess pool. **Bake this env var into every new DeepSeek V3 class serve script** (R1, V3.2, future Kimi variants) until upstream fixes compile-worker CUDA init. Full trace + timeline: `results/metadata/DeepSeek-R1-NVFP4_sglang_startup.yaml` `bringup_problem` block.
- **SGLang moe_runner_backend="auto" resolves to flashinfer_trtllm on B300**, not some unnamed fallback. CLAUDE.md earlier said "auto" after reading a CUTLASS-disabled warning — the warning is real but the resolved backend is `flashinfer_trtllm`, confirmed via `/get_server_info` on Kimi K2.5, GLM-5.1, and R1. Not an issue, just a documentation correction.
- **kv_cache_dtype auto-promotes to fp8_e4m3 on DeepSeek V3 class MLA models** (Kimi K2.5, GLM-5.1, DeepSeek R1). Even if you don't pass `--kv-cache-dtype fp8_e4m3`, SGLang picks it automatically on B300 for these architectures. Passing it explicitly is safe and makes the config auditable.
- **DeepGEMM not available on NGC 26.01** (26.03 had it, but 26.03 crashes on this driver).
- **Missing B300 MoE config file in vLLM:** `E=64,N=1536,device_name=NVIDIA_B300_SXM6_AC,dtype=fp8_w8a8,block_shape=[128,128].json` — affects FP8 MoE tuning; NVFP4 path is unaffected.
- **SGLang auto-disables CUTLASS MoE on B300** with warning: *"CUTLASS backend is disabled when piecewise cuda graph is enabled due to TMA descriptor initialization issues on B200. Using auto backend instead for stability."* The `auto` backend then resolves to `flashinfer_trtllm` on B300 (confirmed via `/get_server_info` on Kimi, GLM-5.1, R1) — which is actually the fast path we'd want, not a degraded fallback. Safe to keep the warning; nothing to tune.
- **SGLang bench `--random-range-ratio 0.0` is a trap:** it samples per-prompt output lengths uniformly from `[1, random_output_len]` instead of pinning to `random_output_len`. Always pass `--random-range-ratio 1.0`. (Baked into `bench_sweep.sh` — only relevant if you ever call `sglang.bench_serving` directly.)
- **Cold start:** vLLM ~7 min (FlashInfer JIT + torch.compile + CUDA graphs). SGLang ~2–5 min (weight load ~45 s + CUDA graph capture ~1–2 min; no torch.compile phase).

## Key Files
- `~/benchmark/SOP.md` — how to benchmark (process, step-by-step, lessons learned)
- `~/benchmark/STATUS.md` — what's happening now (live state, change log)
- `~/benchmark/configs/*.yaml` — per-model configs
- `~/benchmark/scripts/launch_server.sh` — unified vLLM launch wrapper (single-container guard, standard mounts)
- `~/benchmark/scripts/bench_sweep.sh` — concurrency sweep script (framework-aware, layered stop guards)
- `~/benchmark/scripts/serve_*.sh` — per-model server launch scripts
- `~/benchmark/scripts/check_kimi_download.sh` — read-only download progress check
- `~/benchmark/scripts/patch_bench*.py` — (one-off, reverted) bench_serving.py diagnostic patches from the 2026-04-13 debug session
- `~/benchmark/results/<framework>/<model>/json/` — benchmark result JSONs (e.g. `sglang/minimax-m2.7/json/`, `vllm/kimi-k2.5/json/`)
- `~/benchmark/results/<framework>/<model>/logs/` — per-run stdout tees
- `~/benchmark/results/metadata/` — node_info.yaml, per-container startup yamls, GPU stats

## Model Queue (NVFP4)
Status as of 2026-04-16 00:00 UTC — **node wrap-up snapshot for driver upgrade**. Always cross-check `STATUS.md` for the latest sweep state before relying on this.

**This node is being retired.** All 1k1k SGLang sweeps below are tagged `driver 590.48.01, sglang 0.5.10.post1, B300 SXM6 AC`. Do not merge new-node results into the existing tables without explicit re-tagging.

1. **MiniMax M2.7** — `lukealonso/MiniMax-M2.7-NVFP4` (126 GB, 36 shards, downloaded). 1k1k **complete** on both vLLM 0.13.0 and SGLang 0.5.10.post1 TP=8 (conc 1–512). Peak SGLang 10284 t/s @ conc=512. 1k4k / 4k1k deferred to new node. EP=8 confirmed broken on both frameworks. Serve: `scripts/serve_minimax-m2.7_nvfp4_sglang.sh`.
2. **Kimi K2.5** — `nvidia/Kimi-K2.5-NVFP4` (551 GB, 119 shards, downloaded). 1k1k SGLang **complete to knee** (conc 1–128, plateaued at ~2600 t/s by conc=64). 1k4k / 4k1k deferred. DeepSeek-V3-class architecture. Tokenizer slow-path caveat documented in `configs/kimi-k2.5.yaml` — TTFT inflated 10–30%. Serve: `scripts/serve_kimi-k2.5_nvfp4_sglang.sh`.
3. **GLM-5.1** — `lukealonso/GLM-5.1-NVFP4` (434 GB, 85 shards, downloaded). 1k1k SGLang **complete parser-off** (conc 1–512). Peak 8913 t/s @ conc=512. Parser-on diagnostic preserved at `json_withparsers/`. 1k4k / 4k1k deferred. Serve: `scripts/serve_glm-5.1_nvfp4_sglang.sh`.
4. **DeepSeek R1** — `nvidia/DeepSeek-R1-NVFP4` (~395 GB, 80 shards, downloaded 2026-04-15). 1k1k SGLang **complete** (conc 1–512). Peak 9891 t/s @ conc=512. Required `TORCHINDUCTOR_COMPILE_THREADS=1` env var (see Known Issues). Serve: `scripts/serve_deepseek-r1_nvfp4_sglang.sh`. NVIDIA HF card lists TRT-LLM only; community (verda, vLLM blog) runs on SGLang/vLLM fine.
5. **Qwen 3.5 397B-A17B** — `nvidia/Qwen3.5-397B-A17B-NVFP4` (234 GB, 11 shards, downloaded). 1k1k SGLang **complete** (conc 1–512). **Peak 10652 t/s @ conc=256** — highest in the queue. Only model that peaks before conc=512 (conc=512 regressed −3.7%, plateau guard stopped sweep). 1k4k / 4k1k deferred. Serve: `scripts/serve_qwen3.5-397b_nvfp4_sglang.sh`. New `qwen3_5_moe` model type in SGLang; TORCHINDUCTOR_COMPILE_THREADS=1 carried forward preventatively.
6. **DeepSeek V3.2** — `nvidia/DeepSeek-V3.2-NVFP4` (~415 GB, **NOT downloaded**, HF card marks TRT-LLM-only, deferred).
