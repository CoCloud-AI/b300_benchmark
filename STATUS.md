# B300 Benchmark — Live Status

Last updated: 2026-07-08 (driver-610 node bring-up)

---

## ⭐ NEW NODE (driver 610.43.02 / SGLang 0.5.14) — ACTIVE 2026-07-08

> **Provenance for ALL results under `results_610/`**: `host 28-197-33-3.maas, 8x B300 SXM6 AC, driver 610.43.02 (open kernel modules), NVSwitch fabricmanager+nvlsm, SGLang 0.5.14 (derived image sglang-b300:v0.5.14), TencentOS Server 4 / kernel 6.6.119, TP=8 EP=1 NVFP4`. Do NOT merge into results_590/ or results_595/. Full node record: `results_610/metadata/node_info.yaml`.

**This is a fresh bare-metal node** — provisioned from scratch this session (nothing preinstalled). Provisioning recorded in `scripts/provision_node_610.sh`. The old `results/` folder was relabeled `results_590/` (its node_info says driver 590.48.01).

### Provisioning done (all verified)
- NVIDIA **610.43.02** open modules via `.run` (CUDA rhel9 repomd was broken server-side). Pinned `kernel-devel-6.6.119-49.22` to match running kernel. nouveau blacklisted + reboot.
- **28 TB NVMe** RAID0 (`/mnt/nvme`) holds HF cache + docker data-root (70 GB / root too small).
- Docker 29.6.1 + nvidia-container-toolkit 1.19.1; docker data-root on NVMe.
- **NVSwitch fabricmanager 610.43.02 + nvlsm** — MANDATORY (NVL5+): without it CUDA-in-container = "error 802 system not yet initialized". See `results_610/metadata/node_info.yaml` and memory.
- SGLang image fix: stock v0.5.14 missing `distro` pkg → derived `sglang-b300:v0.5.14` (`scripts/Dockerfile.sglang-b300`).

### Model queue (user-selected, NVFP4 only, on NVMe). All initial 5 downloaded ✅
1. **DeepSeek-V4-Flash** `nvidia/DeepSeek-V4-Flash-NVFP4` (168 GB, deepseek_v4) — ✅ **1k1k COMPLETE**: peak 14,074 tok/s @ conc=256 (knee), plateau guard fired at 512 (+1.6%), 0 failures. 1k4k/4k1k pending. Serve: `scripts/serve_deepseek-v4-flash_nvfp4_sglang.sh` (WORKING).
2. **MiniMax-M3** `nvidia/MiniMax-M3-NVFP4` (233 GB, minimax_m3_vl, multimodal) — ✅ downloaded, not yet served.
3. **GLM-5.2** `nvidia/GLM-5.2-NVFP4` (433 GB, glm_moe_dsa) — ✅ downloaded, not yet served.
4. **Kimi-K2.7-Code** `nvidia/Kimi-K2.7-Code-NVFP4` (555 GB, kimi_k25) — ✅ downloaded (plain K2.7 NVFP4 not published).
5. **DeepSeek-V4-Pro** `nvidia/DeepSeek-V4-Pro-NVFP4` (851 GB, deepseek_v4) — ✅ downloaded, not yet served.

### Extras — second wave (user-added 2026-07-08, `scripts/download_extras_610.sh`)
6. **Qwen3.5-397B V2** `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` (244 GB, qwen3_5_moe) — ⏳ downloading. Newer V2 requant of the 595-era Qwen (same arch), for an updated driver-610 number. No *new* large Qwen model exists (Qwen3.6 only small variants; no Qwen4 NVFP4).
7. **Step-3.7-Flash** `stepfun-ai/Step-3.7-Flash-NVFP4` (129 GB, step3p7) — ⏸ queued. StepFun's only official NVFP4.
8. **MiMo-V2.5** `lukealonso/MiMo-V2.5-NVFP4` (184 GB, mimo_v2, 48L/256e/8act) — ⏸ queued. Xiaomi MiMo; reputable community quant (no official XiaomiMiMo NVFP4 org repo). Skipped MiMo-V2.5-Pro (597 GB) — obscure author, gated config, non-standard MXFP8-attn mix, not clean NVFP4.


### Revised sequence (user decision 2026-07-09)
V4-Pro (SGLang, running) → **maintenance window** (containerd→NVMe migration —
Docker-29 snapshotter puts image layers on 70GB root; script
`scripts/maintenance_containerd_to_nvme_610.sh` — then pull vLLM stable+nightly)
→ **GLM-5.2 on vLLM v0.24.0** (`serve_glm-5.2_nvfp4_vllm.sh`; SGLang blocked by
transformers 5.8.1 layer_types pin, tfdsa-patch retry kept as optional A/B)
→ **MiniMax-M3 on vLLM nightly** (`serve_minimax-m3_nvfp4_vllm.sh`; only
framework+version serving this NVFP4 checkpoint — vLLM PR #46380 missed the
0.24.0 cut; --block-size 128 + --language-model-only, digest pinned)
→ back to SGLang 0.5.14: Kimi-K2.7-Code → Qwen3.5-397B-V2 → Step-3.7-Flash → MiMo-V2.5.
vLLM results land under results_610/vllm/<model>/ (framework split per SOP).

### Key lesson (driver-610 / SGLang 0.5.14)
**Pass MINIMAL serve flags.** SGLang 0.5.14 natively supports the new archs (deepseek_v4, etc.) and auto-selects the right quant/attention/MoE backends. Every forced R1-era flag broke DeepSeek-V4-Flash: `--quantization modelopt_fp4` (mixed-precision mismatch), `--moe-runner-backend flashinfer_trtllm` (topkGroup<=4 crash), `--attention-backend trtllm_mla` (unneeded; auto → `dsv4`). Working set: `--tp 8 --trust-remote-code --kv-cache-dtype fp8_e4m3 --context-length 16384`.

---

## (RETIRED) driver 590.48.01 node — historical, below

## Node tag for ALL results in this file

> **Node / driver / framework provenance**: `b300-node-ark-h212, driver 590.48.01, sglang 0.5.10.post1, B300 SXM6 AC`. Every JSON under `~/benchmark/results/sglang/<model>/json/` and `~/benchmark/results/vllm/<model>/json/` was produced against this pinned combination. **This node is being retired** — it is moving to a newer driver, and subsequent runs will live in a dedicated "New node" section below once the move happens. Do not mix the two without explicit re-tagging.

## Currently Running

Nothing. Qwen 3.5 397B-A17B 1k1k sweep completed 2026-04-15 23:55 UTC. Container `qwen35-397b` still warm on the node as of this writeup — can be used to run 1k4k/4k1k profiles before shutdown if time permits, otherwise it's fine to tear down. All other benchmark containers were cleaned up in prior sessions.

## Environment Changes (this session)

- **Docker group fix** (2026-04-15): `howell` added to `docker` group. All scripts (`bench_sweep.sh`, `launch_server.sh`, `serve_*.sh`) and CLAUDE.md updated to drop `sudo` prefix. `grep -r sudo ~/benchmark/scripts/` returns nothing. SOP.md references have been updated to match.
- **TORCHINDUCTOR_COMPILE_THREADS=1** baked into every DeepSeek-V3-class SGLang serve script to work around an inductor compile-worker CUDA-init bug (full trace in `results/metadata/DeepSeek-R1-NVFP4_sglang_startup.yaml`, CLAUDE.md Known Issues).
- **No reasoning / tool-call parsers in benchmark runs.** Added as a hard rule in CLAUDE.md after the GLM-5.1 TTFT A/B (3895 ms parser-on → 197 ms parser-off at conc=1). SGLang's reasoning parser buffers `<think>...</think>` content, inflating client-measured TTFT.

## Completed Results (SGLang 0.5.10.post1, NVFP4, TP=8, 1k1k)

### MiniMax M2.7 (`lukealonso/MiniMax-M2.7-NVFP4`)

| Profile | Concurrencies | Peak | JSON dir |
|---|---|---:|---|
| 1k1k | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 | 10284 t/s @ conc=512 | `results/sglang/minimax-m2.7/json/` |

Plus a parallel vLLM 0.13.0 (NGC 26.01) 1k1k sweep at the same concurrencies for A/B comparison: `results/vllm/minimax-m2.7/json/`. Two historical FP8 TP=4 JSONs at `results/vllm/minimax-m2.7/json/m27_fp8_tp4_conc{4,16}_1k1k.json` retained for reference (pre-SOP NGC 26.03 run, not part of the canonical NVFP4 matrix).

### Kimi K2.5 (`nvidia/Kimi-K2.5-NVFP4`)

| Profile | Concurrencies | Peak | JSON dir |
|---|---|---:|---|
| 1k1k | 1, 2, 4, 8, 16, 32, 64, 128 | 2595 t/s @ conc=128 | `results/sglang/kimi-k2.5/json/` |

**conc=256 and conc=512 intentionally not collected.** Throughput plateaued at ~2600 t/s by conc=64 (conc=32→64 = +4.3%, conc=64→128 = +0.6%). Extending the sweep would only add latency without adding throughput. The curve IS complete to its knee — document as such in the writeup.

### GLM-5.1 (`lukealonso/GLM-5.1-NVFP4`)

| Profile | Concurrencies | Peak | JSON dir |
|---|---|---:|---|
| 1k1k | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 | 8913 t/s @ conc=512 | `results/sglang/glm-5.1/json/` |

Parser-off canonical. Parser-on diagnostic run (`--reasoning-parser glm45 --tool-call-parser glm47`) preserved for comparison at `results/sglang/glm-5.1/json_withparsers/` and `logs_withparsers/`.

### DeepSeek R1 (`nvidia/DeepSeek-R1-NVFP4`)

| Profile | Concurrencies | Peak | JSON dir |
|---|---|---:|---|
| 1k1k | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 | 9891 t/s @ conc=512 | `results/sglang/deepseek-r1/json/` |

Required `TORCHINDUCTOR_COMPILE_THREADS=1` env var to bypass the inductor compile-worker bug (hit on R1 2026-04-15 21:47, documented in CLAUDE.md Known Issues and in `results/metadata/DeepSeek-R1-NVFP4_sglang_startup.yaml`).

### Qwen 3.5 397B-A17B (`nvidia/Qwen3.5-397B-A17B-NVFP4`) — NEW

| Profile | Concurrencies | Peak | JSON dir |
|---|---|---:|---|
| 1k1k | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 | **10652 t/s @ conc=256** | `results/sglang/qwen3.5-397b-a17b/json/` |

Qwen is the **only** model in the queue that peaks before conc=512. conc=512 regressed −3.7% vs conc=256 and the plateau guard fired, stopping the sweep cleanly. Sweep was interrupted at conc=256 by a server reboot 2026-04-15 23:26 UTC (graceful SIGTERM during decode, no JSON written) and resumed 2026-04-15 23:41 UTC against the same container image and flags — conc=256 re-ran from scratch, then conc=512 fired normally.

## Headline Comparison (1k1k peak throughput)

| Rank | Model | Peak t/s | @ conc | tps/GPU |
|---|---|---:|---:|---:|
| 1 | **Qwen 3.5 397B-A17B** | **10652** | **256** | **1332** |
| 2 | MiniMax M2.7 | 10284 | 512 | 1285 |
| 3 | DeepSeek R1 | 9891 | 512 | 1236 |
| 4 | GLM-5.1 | 8913 | 512 | 1114 |
| 5 | Kimi K2.5 | 2595 | 128 | 324 |

Full side-by-side tables (throughput, per-GPU, TTFT, TPOT at conc 1/16/64/128/256/512) live in **`results/summary_1k1k.md`** — generated from the JSONs on 2026-04-16.

## Headline Findings

1. **Qwen 3.5 397B-A17B wins both peak throughput and peak-per-GPU** on 1k1k despite being the second-smallest model by total params. Only 17B active params + aggressive GQA (32:2 query:KV ratio) + hybrid 45-linear/15-full-attention layers combine to make it both the fastest at conc=1 (194 t/s, 5.0 ms TPOT) and the highest-throughput at saturation (10652 t/s). It's also the only model to peak before conc=512 — the hybrid attention scheme likely saturates memory bandwidth earlier than peers.

2. **SGLang vs vLLM on M2.7 1k1k (TP-only, NVFP4):**
   - At conc=1: SGLang +42.7% throughput (104 vs 73 t/s), −63% TTFT (45 vs 121 ms), −30% TPOT (9.5 vs 13.5 ms).
   - At conc=512 the throughput gap vanishes (+2.4%), and SGLang's TTFT is 4.4× **worse** (3484 ms vs 791 ms). SGLang TTFT scales linearly with concurrency on B300; vLLM scales sublinearly.
   - **Sweet spots:** SGLang dominates conc≤32 (interactive), vLLM wins conc≥128 (high-load batch). Open question whether `--moe-runner-backend flashinfer_trtllm` on the M2.7 SGLang container closes the high-conc TTFT gap. Untested — deferred to the new node.

3. **EP+NVFP4 is broken on BOTH frameworks** (verified by trying):
   - vLLM 0.13.0 — `cutlass_moe_fp4` kernel rejects EP at dispatch.
   - SGLang 0.5.10.post1 — `modelopt_quant.py:1754` shape mismatch during weight post-processing: `w13_input_scale[256] × w13_weight_scale_2[32]`. Loader replicates input_scale across all experts but EP-shards weight_scale_2 to per-rank 32 — the multiply explodes. Both fail at runtime regardless of `--moe-runner-backend`.

4. **Kimi vs R1 vs Qwen (all MoE, all NVFP4 TP=8):**
   - Kimi plateaus at 2595 t/s by conc=128 while R1 (same DeepSeek-V3 class) climbs to 9891 t/s at conc=512. Likely drivers: R1 is smaller per-rank (53 GB vs 79 GB weight), larger KV pool (5.65M vs 4.24M tokens), and was launched with explicit `flashinfer_trtllm` MoE backend + `fp8_e4m3` kv cache from the start rather than relying on auto-selection.
   - Qwen beats both R1 and Kimi at every concurrency. Active-params-per-token is the dominant factor: Qwen 17B < R1 37B ≈ Kimi 32B, and Qwen has hybrid linear attention whereas R1/Kimi are full MLA.

5. **`--random-range-ratio 0.0` is a trap** in `sglang.bench_serving`. Default samples per-prompt output_len uniformly from `[1, full_len]`, not "no variation". Must pass `--random-range-ratio 1.0` for fixed-length outputs. Cost ~1 hour of misdiagnosis on 2026-04-13. Now baked into `bench_sweep.sh`.

6. **`max_model_len=8192` was too small** for the original 1k8k/8k1k profiles (1024+8192 = 9216 > 8192). Sweep silently failed for hours because `bench_sweep.sh`'s only stop guard was `"Failed requests"`, which does NOT match `ValueError: Initial test run failed ... Error: Bad Request`. Fixed: max-model-len now 16384, and `bench_sweep.sh` now layers in `ValueError|Bad Request|Traceback|OCI runtime|missing-JSON` stop guards. Retired 1k8k/8k1k in favor of 1k1k / 1k4k / 4k1k which fit with headroom.

## Pending (deferred to new node)

### High priority — retried first on the new node after driver upgrade

- **1k4k and 4k1k profiles for every model** (5 models × 2 profiles = 10 sweeps). These are the decode-heavy and prefill-heavy complements to the already-collected 1k1k curves. All model containers are warm-start templates — `~/benchmark/scripts/serve_*_sglang.sh` encodes the exact flags per model. Keep the same TP=8, EP=1, max-model-len=16384, no-parser rules.
- **DeepSeek R1 on NGC / newer driver** — if the driver upgrade allows NGC 26.03, retest both vLLM and SGLang against R1 as the baseline and compare to the 590.48 SGLang numbers.

### Medium priority

- **`--moe-runner-backend flashinfer_trtllm` on M2.7 SGLang 1k1k** — A/B test to see if it fixes the high-concurrency TTFT regression (3.5 s at conc=512). Cheap — 15 min including container restart. If it works, re-run as canonical M2.7 numbers and note the flag change.
- **Retest EP+NVFP4 on both frameworks** once SGLang ships a fix to `modelopt_quant.process_weights_after_loading` or vLLM adds EP to `cutlass_moe_fp4`. Cheap verification launch (5 min on M2.7) is enough.

### Lower priority

- **DeepSeek V3.2** (`nvidia/DeepSeek-V3.2-NVFP4`, ~415 GB) — marked TRT-LLM-only on the HF card but community (verda, vLLM blog) runs R1/V3.2 on SGLang and vLLM. If 1k1k is clean on R1, V3.2 should work too.
- **MiniMax M2.7 FP8 dual-instance TP=4** (`configs/minimax-m2.7-fp8-dual.yaml`, `serve_minimax-m2.7_fp8_vllm_inst{0,1}.sh`) — block-FP8 forces TP=4 for M2.7 (see `docs/quantization_block_alignment.md`). Not part of the canonical NVFP4 matrix but the dual-instance setup is ready to run if an FP8 baseline is ever requested.

## Blockers & Open Issues

- **EP+NVFP4 broken on both frameworks** (see above). `--moe-runner-backend` swap will NOT fix it (failure is in the weight loader, not the MoE kernel). Real fix requires either an upstream patch to `modelopt_quant.process_weights_after_loading` or re-quantizing the checkpoint with EP-aware ModelOpt.
- **SGLang TTFT scales linearly with concurrency on B300** — likely tied to the CUTLASS MoE auto-disable (`auto` → `flashinfer_trtllm` fallback) plus prefill scheduling. Untested follow-up: pin `flashinfer_trtllm` explicitly from the start for M2.7 (the other four models already pass it explicitly).
- **Kimi tokenizer slow-path warning** — every `encode()` call from SGLang's chat preprocessor takes the slow `super().encode()` path because SGLang passes `add_special_tokens=False` and Kimi's fast path bails on any kwargs. Inflates TTFT 10–30%. Documented in `configs/kimi-k2.5.yaml`; real fix is on the SGLang side. Not a blocker but should be flagged in any writeup that compares Kimi TTFT to other models.
- **Driver 590.48 blocks NGC 26.03** — driver upgrade scheduled (the new node is the upgrade path).
- **torch.inductor compile-worker CUDA-init bug on DeepSeek V3 class SGLang** — worked around with `TORCHINDUCTOR_COMPILE_THREADS=1`. Kept as a defensive env var in every new serve script until upstream fixes compile-worker CUDA init.
- ~~**Sudo cache is per-tty**~~ — resolved 2026-04-15 (docker group).

## Downloads State (on disk as of 2026-04-16)

Verified zero `.incomplete` files and valid `model.safetensors.index.json` for each:

- ✅ `lukealonso/MiniMax-M2.7-NVFP4` — 126 GB, 36 shards
- ✅ `nvidia/Kimi-K2.5-NVFP4` — 551 GB, 119 shards
- ✅ `nvidia/Qwen3.5-397B-A17B-NVFP4` — 234 GB, 11 shards
- ✅ `lukealonso/GLM-5.1-NVFP4` — 434 GB, 85 shards
- ✅ `nvidia/DeepSeek-R1-NVFP4` — ~395 GB, 80 shards (resumed and completed 2026-04-15)
- ⏸ `nvidia/DeepSeek-V3.2-NVFP4` (~415 GB) — not downloaded; marked TRT-LLM-only on HF card, deferred

**NVFP4 cache total on this node: ~1.74 TB across 5 models.**

Also in HF cache: a stale FP8 `MiniMaxAI/MiniMax-M2.7` (~215 GB) from before the NVFP4 pivot, at the old `~/hf_hub_cache/` path. Deprioritized. Can be deleted to reclaim 215 GB if disk pressure on the new node.

## New Node Section (placeholder)

> This section will be populated once the node moves to newer driver / newer SGLang / newer NGC. Everything above this line is tagged `driver 590.48.01, sglang 0.5.10.post1, B300 SXM6 AC`. Runs from the new node MUST land in a new subsection below with their own provenance line — do NOT append them under the Completed Results tables above. Suggested format:
>
> ```
> ## Completed Results (NEW NODE — driver <X>, sglang <Y>)
> ### MiniMax M2.7 (re-run on new driver)
> ...
> ```

## Change Log

- 2026-04-13 15:45 — Started M2.7 FP8 on vLLM 26.01 TP=4 (historical, retained)
- 2026-04-13 17:01 — Started Kimi K2.5 NVFP4 download
- 2026-04-13 18:30 — Launched M2.7 NVFP4 on vLLM 26.01 TP=8 (EP broken with FP4)
- 2026-04-13 19:00 — Completed M2.7 vLLM 1k1k sweep (conc 1–512)
- 2026-04-13 19:30 — Discovered 1k8k/8k1k context overflow; wasted ~2 h on silently-failing runs
- 2026-04-13 20:00 — Switched profiles to 1k1k / 1k4k / 4k1k, max-model-len 16384
- 2026-04-13 20:45 — Switched framework to SGLang (lmsysorg/sglang:latest-cu130-runtime), pulled image
- 2026-04-13 23:04 — M2.7 SGLang container ready; first sweep attempt
- 2026-04-13 23:18 — First SGLang 1k1k sweep finished but produced wrong total_output_tokens (range_ratio bug)
- 2026-04-13 23:50 — Root-caused to `--random-range-ratio 0.0` (samples uniformly from [1, full_len])
- 2026-04-14 00:00 — Re-launched M2.7 SGLang 1k1k sweep with fix; correct workload confirmed
- 2026-04-14 00:30 — M2.7 SGLang 1k1k sweep complete (conc 1–256)
- 2026-04-14 00:42 — Reorganized `results/` into framework/json+logs split
- 2026-04-14 00:48 — One-off conc=512 against same warm container; throughput tied with vLLM, TTFT 4.4× worse
- 2026-04-14 00:52 — **EP=8 attempt FAILED** — modelopt loader shape mismatch during weight post-processing
- 2026-04-14 01:14 — Reorganized `results/` into per-model subfolders (`<framework>/<model>/json,logs`)
- 2026-04-14 01:15 — Kimi K2.5 SGLang container up (~2 min cold start), warmup successful
- 2026-04-14 01:26 — Kimi K2.5 1k1k sweep started
- 2026-04-14 01:35 — Identified Kimi tokenizer slow-path warning (`Calling super().encode` from `tokenization_kimi.py:178`) — documented as caveat, not patched
- 2026-04-14 02:15 — Stopped Kimi sweep at conc=128 (throughput plateaued at ~2600 t/s, no point pushing further)
- 2026-04-14 02:30 — Killed Kimi container, session-end snapshot written
- 2026-04-15 19:00 — Docker group fix: `howell` added to `docker` group, removed `sudo` from all benchmark scripts and CLAUDE.md
- 2026-04-15 19:05 — Restarted stalled DeepSeek-R1-NVFP4 download (127 GB on disk, 8 incomplete blobs, no running process) in `download` tmux
- 2026-04-15 19:08 — Kimi marked complete; launched GLM-5.1 NVFP4 SGLang container `glm51` (TP=8, parser-on first attempt)
- 2026-04-15 19:17 — GLM-5.1 server ready (~9 min cold start); warmup curl OK
- 2026-04-15 19:18 — GLM-5.1 1k1k sweep started
- 2026-04-15 20:05 — GLM-5.1 1k1k (parser-on) complete; peak 8828 t/s conc=512, TTFT 22 s (inflated)
- 2026-04-15 20:28 — GLM-5.1 relaunched parser-off; TTFT A/B confirmed (3895 ms → 197 ms at conc=1)
- 2026-04-15 21:25 — GLM-5.1 1k1k parser-off complete; peak 8913 t/s conc=512, TTFT 6306 ms at conc=512
- 2026-04-15 21:34 — Attempted conc=1024 probe (aborted manually)
- 2026-04-15 21:47 — Launched DeepSeek R1 NVFP4 TP=8 (first attempt, crashed during CUDA graph capture)
- 2026-04-15 21:52 — Debugged: torch.inductor compile-worker subprocess CUDA-init bug in `vocab_parallel_embedding.get_masked_input_and_mask`
- 2026-04-15 21:55 — R1 relaunched with `TORCHINDUCTOR_COMPILE_THREADS=1`; clean bring-up
- 2026-04-15 22:02 — R1 ready (~7 min cold start, fastest weight load of any model: 55 s/rank, 53 GB/rank)
- 2026-04-15 22:05 — R1 1k1k sweep started
- 2026-04-15 22:52 — R1 1k1k sweep complete; 10 levels, peak 9891 t/s conc=512
- 2026-04-15 22:53 — New serve script `scripts/serve_deepseek-r1_nvfp4_sglang.sh`, config `configs/deepseek-r1.yaml`, metadata `results/metadata/DeepSeek-R1-NVFP4_sglang_startup.yaml` created
- 2026-04-15 22:53 — CLAUDE.md Known Issues updated: inductor compile-worker bug, flashinfer_trtllm as true default, kv_cache_dtype auto-promotion on DeepSeek V3 MLA
- 2026-04-15 22:55 — Qwen 3.5 397B-A17B SGLang TP=8 launched (first container; 1k1k sweep started)
- 2026-04-15 23:08 — Qwen 1k1k conc=1 complete (194 t/s)
- 2026-04-15 23:21 — Qwen 1k1k conc=128 complete (7867 t/s, still climbing strong)
- 2026-04-15 23:26 — **Server reboot.** Qwen container SIGTERM'd mid-conc=256 (exit code 137 at 23:26:20). 8 conc levels already saved; conc=256 JSON not written.
- 2026-04-15 23:35 — Session resumed. Dead container removed, tmux sessions re-created (`server`, `bench`, `claudecode`), serve script relaunched.
- 2026-04-15 23:41 — Qwen server ready again (`Application startup complete.`, ~6 min cold start)
- 2026-04-15 23:42 — Warmup curl OK, resume sweep script launched (conc=256, conc=512 only)
- 2026-04-15 23:49 — Qwen 1k1k conc=256 complete (**10652 t/s — peak**, +35% vs conc=128)
- 2026-04-15 23:55 — Qwen 1k1k conc=512 complete (10258 t/s, −3.7% vs conc=256, plateau guard fired)
- 2026-04-16 00:00 — **Node wrap-up**: STATUS.md / CLAUDE.md / SOP.md / per-model configs audited, `summary_1k1k.md` generated from JSONs, node provenance tagged. Node being retired for driver upgrade.
