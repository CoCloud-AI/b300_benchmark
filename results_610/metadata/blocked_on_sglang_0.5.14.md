# Models blocked on SGLang 0.5.14 (driver-610 node, 2026-07-08)

- **MiniMax-M3** (nvidia/MiniMax-M3-NVFP4, MiniMaxM3SparseForConditionalGeneration):
  arch NOT registered in SGLang 0.5.14 (only MiniMaxM2ForCausalLM). Upstream `main`
  still only has minimax_m2.py as of 2026-07-08 nightly — NOT fixable by nightly.
- **GLM-5.2** (nvidia/GLM-5.2-NVFP4, GlmMoeDsaForCausalLM): class IS registered, but
  config layer_types are all `deepseek_sparse_attention` (DSA x78), which the
  transformers bundled in 0.5.14 rejects (ValueError: layer_types must be in {...}).
  MAY be fixed by a newer nightly (newer transformers + GLM DSA support) — unverified.

Newest SGLang: stable v0.5.14 (2026-06-26, what we run); nightlies exist to
nightly-dev-cu13-20260708. Decision pending on whether to use a nightly for GLM-5.2.

## Update 2026-07-09 (Fable review)

**GLM-5.2 — root cause CONFIRMED + fix built.** transformers 5.8.1 in the image
already ships `GlmMoeDsaConfig` AND SGLang 0.5.14 registers `GlmMoeDsaForCausalLM`
(glm4_moe.py, wired into deepseek_common/deepseek_weight_loader.py). The ONLY
blocker is the generic `ALLOWED_LAYER_TYPES` allowlist in
transformers/configuration_utils.py, which predates the `deepseek_sparse_attention`
entry that upstream main now carries (verbatim: `"deepseek_sparse_attention",  #
for models with DSA indexer (GLM MoE DSA, DeepSeek V32)`). Built
`sglang-b300:v0.5.14-tfdsa` backporting that one string; GLM-5.2 config load
verified OK (CPU-side). GPU end-to-end retry queued after the V4-Pro sweep.

**MiniMax-M3 — genuinely unservable on SGLang (any build).** Checkpoint ships NO
custom modeling code (auto_map has AutoConfig only; snapshot has configuration/
processor .py files but no modeling_*.py). SGLang has no M3 class (main still only
minimax_m2.py as of 2026-07-08 nightly), and the transformers fallback can't
construct it either. Upstream transformers main DOES now know the
`minimax_m3_sparse` layer type, suggesting M3 enablement is in flight upstream,
but nothing servable today. Web research on vLLM/TRT-LLM alternatives pending.

## vLLM findings + disk gotcha (2026-07-09)

**vLLM main registers EVERYTHING we're blocked on** (verified from its registry):
MiniMaxM3SparseForCausalLM + MiniMaxM3SparseForConditionalGeneration + MiniMaxM3MTP,
GlmMoeDsaForCausalLM, and all queued archs (KimiK25, DeepseekV4, Qwen3_5Moe,
Step3p7, MiMoV2). Plan: serve MiniMax-M3 on vLLM (stable v0.24.0 if it has M3,
else nightly pinned by digest) at queue end, results under results_610/vllm/.
Stable-image registry check pending — blocked by the disk issue below.

**Docker-29 containerd-snapshotter disk gotcha:** daemon.json `data-root` on NVMe
is NOT sufficient — image layers go to /var/lib/containerd on the 70 GB root fs
(37 GB of SGLang layers live there). The vLLM v0.24.0 pull failed with ENOSPC and
auto-rolled-back. **Maintenance window fix (between model containers, docker
restart required):** stop docker+containerd, move /var/lib/containerd to
/mnt/nvme/containerd, set `root = "/mnt/nvme/containerd"` in
/etc/containerd/config.toml, restart, verify images, then pull vLLM.

## Upstream research findings (web agent, 2026-07-09, source-verified)

**GLM-5.2 — root cause matches our forensics, upstream already fixed it post-release:**
- transformers pin 5.8.1 in sglang v0.5.14 vs `deepseek_sparse_attention` added in
  transformers **5.11.0** (huggingface/transformers#41251). Empirically: config fails
  on 5.4.0–5.10.4, loads on >=5.11.0.
- ONLY the nvidia NVFP4 checkpoint hardcodes `layer_types` (written w/ 5.11.0);
  zai-org/GLM-5.2 + FP8 configs omit the key entirely — why v0.5.14 release notes
  claim GLM-5.2 support yet the NVFP4 checkpoint crashes.
- SGLang main fixed it twice after the v0.5.14 cut: PR #29454 (bypass layer_types
  validation, 06-26) then PR #29393 (bump transformers to 5.12.1, 06-30, "makes
  nvidia/GLM-5.2-NVFP4 usable out of the box"). NO tagged release has either fix.
- Official escape hatches: `lmsysorg/sglang:dev-glm52-nvfp4` image (NVIDIA card +
  SGLang cookbook; GB300 CI test test/registered/gb300/test_glm52_nvfp4.py exists),
  or pip transformers==5.12.1, or strip layer_types (maintainer-validated #29337).
- Our `sglang-b300:v0.5.14-tfdsa` = same effect as #29454 with a smaller blast
  radius (one allowlist string; transformers stays 5.8.1). Fallback if the GPU run
  fails: official dev-glm52-nvfp4 image.
- `--json-model-override-args` FALSIFIED as a workaround (override applied after
  AutoConfig raises).

**MiniMax-M3 — SGLang verdict hardened: not possible for NVFP4 anywhere in SGLang:**
- Model files still in OPEN PR sgl-project/sglang#28715 (split 4/4; kernels/HiCache/
  disagg splits 1-3 merged). Tracking issue #27536. Cookbook serves M3 only from
  PR-branch image `dev-cu13-minimax-m3` — and even there NO NVFP4 path (cutlass
  NVFP4 MoE doesn't forward M3's clamped-swiglu params -> garbage output).
- **vLLM: M3 BF16/MXFP8 shipped in stable v0.24.0, but NVFP4 (PR #46380, merged
  06-25) MISSED the 0.24.0 branch cut** — NVIDIA card: "you currently need the
  nightly docker image". So: `vllm/vllm-openai:nightly` (pin digest) is the path.
- vLLM M3 serve requirements: `--block-size 128` (mandatory), TP=8;
  `--language-model-only` skips the vision tower (frees ~192k encoder KV tokens) —
  right choice for our text-only throughput bench. No parsers (project rule).
- Caveats to note in writeup: vLLM M3 roadmap #45668 lists NVFP4-indexer work
  pending; open B300 vLLM issue #47239 (Model Runner V2 accuracy/TPOT — workaround
  VLLM_USE_V2_MODEL_RUNNER=0) if we ever cross-check GLM on vLLM.

## Kimi-K2.7-Code vLLM crash at 1k4k conc=512 (2026-07-09)

EngineCore fatal: `TimeoutError: RPC call to sample_tokens timed out` at ~69%
through the 1k4k conc=512 level (5120 prompts). Progress log shows periodic
~2min whole-batch stalls at every 512-request boundary, growing to a final
~4.8min stall before the RPC watchdog killed the engine. Consistent with open
vLLM issue #47239 (8xB300 Model Runner V2 TPOT fluctuation; workaround
VLLM_USE_V2_MODEL_RUNNER=0 — NOT applied, to keep runner consistency with the
other vLLM models which ran clean). 1k4k curve stands at 9 levels (peak 8757
@ conc=256, +45.6% still climbing). Guard 3 (missing JSON) caught it; retry
of the single level attempted post-4k1k.
UPDATE: single-level retry on a freshly relaunched engine SUCCEEDED cleanly
(13537 tok/s, ttft_p99 8.9s, tpot 37.5ms, 0 failures) — the RPC timeout was
TRANSIENT, likely accumulated engine state after 3 chained profiles, not a
reproducible saturation limit. 1k4k curve complete at 10 levels.

## Qwen3.5-397B-V2: GARBLED on vLLM (both images) — SGLang fallback (2026-07-09)

nvidia/Qwen3.5-397B-A17B-NVFP4-V2 outputs "!!!!..." on 8x B300 under vLLM
v0.24.0 stable AND nightly-2afa3f7e9. VLLM_USE_V2_MODEL_RUNNER=0 confirmed set
in-container, no effect. Symptom matches open issues #47239 / #47367. GLM-5.2,
MiniMax-M3, Kimi-K2.7 are all CLEAN on the same images — bug is specific to the
qwen3_5_moe NVFP4 path. Fallback: SGLang 0.5.14 (V1 397B precedent on 595 node).
Partial garbage results purged before any JSON landed in the tree.

## Qwen 397B single-stream regression vs 595 node (2026-07-09)
1k1k conc=1: 97 tok/s / 10.14ms TPOT on (driver610, sglang 0.5.14, V2 quant) vs
194 tok/s / 5.0ms TPOT on (driver595, sglang 0.5.10, V1 quant). Peak throughput
near-parity (10329@512 climbing vs 10652@256 kneed). Factor not isolated
(sglang version / driver / V2 requant). Flag in writeup; candidate A/B later.

## Step-3.7-Flash: UNSERVABLE at TP=8 NVFP4 on all frameworks (2026-07-09)

stepfun-ai/Step-3.7-Flash-NVFP4: moe_intermediate_size=1280 -> per-rank 160 at
TP=8 (320 at TP=4 — also broken). Every kernel path rejects it:
- vLLM v0.24 + nightly, trtllm backend: flashinfer assert M % 128 == 0
- vLLM flashinfer_cutlass: kernel lacks MoEActivation.SWIGLUSTEP
- vLLM cutlass: "Intermediate size padding ... not currently supported"
- SGLang 0.5.14: "intermediate size required padding ... gated activations" assert
Only TP=2 aligns (640) — breaks full-node TP=8 methodology; a 4x TP=2 replica
deployment is future work, not comparable. Same alignment family as the 590-node
M2.7-FP8 TP ceiling, now on the NVFP4 path. SKIPPED; 129 GB weights retained.
