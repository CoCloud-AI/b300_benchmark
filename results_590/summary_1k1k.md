# B300 NVFP4 — 1k1k SGLang Summary (all models)

**Node**: 8× NVIDIA B300 SXM6 AC, 288 GB HBM3e each. Driver 590.48.01, CUDA 13.1.
**Framework**: SGLang 0.5.10.post1 (`lmsysorg/sglang:latest-cu130-runtime`, image sha `715c461258...`).
**Config**: TP=8, EP=1, max-model-len=16384, `--disable-radix-cache`, `--quantization modelopt_fp4`,
`--moe-runner-backend flashinfer_trtllm` (explicit or auto-resolved), **no reasoning/tool-call parsers**.
**Workload**: `sglang.bench_serving --dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0`, `num_prompts = max(conc*10, 40)`.
**Profile**: 1k1k (ISL=1024, OSL=1024). All values are client-measured from JSON result files.

Generated 2026-04-16 from `~/benchmark/results/sglang/*/json/*_1k1k.json`.

---

## Output throughput (tokens/sec)

| Model | Size (total/active) | conc=1 | conc=16 | conc=64 | conc=128 | conc=256 | conc=512 | **Peak** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MiniMax M2.7       | 230B / 10B | 105 | 1242 | 3134 | 5567  | 8582  | 10284 | **10284** @ conc=512 |
| Kimi K2.5          | ~1T / ~32B | 170 | 1556 | 2579 | 2595  | —     | —     | **2595**  @ conc=128 |
| GLM-5.1            | 744B / 40B |  82 |  960 | 2610 | 4520  | 6787  | 8913  | **8913**  @ conc=512 |
| DeepSeek R1        | 671B / 37B | 166 | 1596 | 3947 | 6163  | 8644  | 9891  | **9891**  @ conc=512 |
| Qwen 3.5 397B-A17B | 397B / 17B | 194 | 2119 | 5190 | 7867  | 10652 | 10258 | **10652** @ conc=256 |

## Output throughput per GPU (tokens/sec/GPU, /8)

| Model | conc=1 | conc=16 | conc=64 | conc=128 | conc=256 | conc=512 | **Peak tps/GPU** |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniMax M2.7       | 13 | 155 | 392 | 696 | 1073 | 1285 | **1285** @ conc=512 |
| Kimi K2.5          | 21 | 194 | 322 | 324 | —    | —    | **324**  @ conc=128 |
| GLM-5.1            | 10 | 120 | 326 | 565 |  848 | 1114 | **1114** @ conc=512 |
| DeepSeek R1        | 21 | 199 | 493 | 770 | 1081 | 1236 | **1236** @ conc=512 |
| Qwen 3.5 397B-A17B | 24 | 265 | 649 | 983 | 1332 | 1282 | **1332** @ conc=256 |

## Mean TTFT (ms)

| Model | conc=1 | conc=16 | conc=64 | conc=128 | conc=256 | conc=512 |
|---|---:|---:|---:|---:|---:|---:|
| MiniMax M2.7       |  45 | 150 |  533 |  959 | 1814 | 3484 |
| Kimi K2.5          | 260 | 577 | 1274 | 2462 | —    | —    |
| GLM-5.1            | 197 | 471 |  960 | 1365 | 2364 | 6306 |
| DeepSeek R1        | 157 | 436 |  968 | 1545 | 2644 | 4990 |
| Qwen 3.5 397B-A17B | 197 | 536 | 1045 | 1451 | 3032 | 4840 |

## Mean TPOT (ms)

| Model | conc=1 | conc=16 | conc=64 | conc=128 | conc=256 | conc=512 |
|---|---:|---:|---:|---:|---:|---:|
| MiniMax M2.7       |  9.5 | 12.7 | 19.9 | 22.1 | 28.0 | 46.2 |
| Kimi K2.5          |  5.6 |  9.7 | 23.6 | 46.9 | —    | —    |
| GLM-5.1            | 12.1 | 16.2 | 23.6 | 27.0 | 35.4 | 51.1 |
| DeepSeek R1        |  5.9 |  9.6 | 15.3 | 19.3 | 27.0 | 46.8 |
| Qwen 3.5 397B-A17B |  5.0 |  7.0 | 11.3 | 14.8 | 21.0 | 45.1 |

---

## Peak-throughput ranking (aggregate)

1. **Qwen 3.5 397B-A17B — 10652 t/s @ conc=256** (first-and-only model to peak before conc=512)
2. MiniMax M2.7 — 10284 t/s @ conc=512
3. DeepSeek R1 — 9891 t/s @ conc=512
4. GLM-5.1 — 8913 t/s @ conc=512
5. Kimi K2.5 — 2595 t/s @ conc=128 (plateaued; conc=256/512 skipped)

## Peak-throughput ranking (per GPU)

1. **Qwen 3.5 397B-A17B — 1332 t/s/GPU @ conc=256**
2. MiniMax M2.7 — 1285 t/s/GPU @ conc=512
3. DeepSeek R1 — 1236 t/s/GPU @ conc=512
4. GLM-5.1 — 1114 t/s/GPU @ conc=512
5. Kimi K2.5 — 324 t/s/GPU @ conc=128

## Interactive-regime ranking (conc=1 throughput × TPOT)

Low concurrency favors models with compact KV and small active-param count.
**Qwen 3.5 397B** wins on both fronts — 194 t/s at conc=1 (highest) and 5.0 ms TPOT (lowest), benefiting from aggressive GQA (32:2) and only 17B active params out of a 397B total.

| Rank | Model | conc=1 t/s | conc=1 TPOT ms |
|---|---|---:|---:|
| 1 | Qwen 3.5 397B-A17B | 194 | 5.0 |
| 2 | Kimi K2.5 | 170 | 5.6 |
| 3 | DeepSeek R1 | 166 | 5.9 |
| 4 | MiniMax M2.7 | 105 | 9.5 |
| 5 | GLM-5.1 | 82 | 12.1 |

---

## Per-model notes

- **MiniMax M2.7** — 230B/10B, most compact active-param count in the queue. Scales cleanly to conc=512. Also has a historical vLLM 1k1k sweep at `~/benchmark/results/vllm/minimax-m2.7/json/` — SGLang wins at low conc, ties at high conc with ~4× worse TTFT.
- **Kimi K2.5** — DeepSeek-V3 class (MLA compact KV). Saturates early at ~2600 t/s by conc=64 (conc=32→64 = +4.3%, conc=64→128 = +0.6%). Sweep stopped at conc=128 by design. TTFT is inflated 10–30% by a slow tokenizer path in `tokenization_kimi.py:178` (SGLang's chat preprocessor passes `add_special_tokens=False`, Kimi's fast path bails). Documented in `configs/kimi-k2.5.yaml`.
- **GLM-5.1** — Parser-off canonical. Earlier parser-on run (glm45 reasoning, glm47 tool-call) showed 3895 ms TTFT at conc=1 because the reasoning parser buffers `<think>...</think>` content before streaming. Parser-off TTFT is 197 ms (A/B confirmed). Parser-on results archived at `json_withparsers/`, `logs_withparsers/`. **Project rule**: no parsers in benchmark runs.
- **DeepSeek R1** — Required `TORCHINDUCTOR_COMPILE_THREADS=1` env var to bypass a torch.inductor compile-worker subprocess CUDA-init bug in `vocab_parallel_embedding.get_masked_input_and_mask`. Same DeepSeek-V3 class as Kimi but scales cleanly all the way to conc=512 — beats Kimi at every concurrency.
- **Qwen 3.5 397B-A17B** — New `qwen3_5_moe` model type in SGLang, hybrid 45 linear + 15 full attention (every 4th layer full), 512 experts with 10 active, aggressive GQA (32:2). Peaked at conc=256; conc=512 regressed −3.7% (plateau guard stopped the sweep). Only model in the queue that peaks before conc=512, which is consistent with its small active-param count and the fact that linear attention saturates memory bandwidth earlier than full attention. Sweep was interrupted at conc=256 by a server reboot 2026-04-15 23:26 and resumed 2026-04-15 23:41 against the same flags on the same container image.

## Data locations

```
~/benchmark/results/sglang/minimax-m2.7/json/       (10 files, 1k1k)
~/benchmark/results/sglang/kimi-k2.5/json/          ( 8 files, 1k1k — plateau stop)
~/benchmark/results/sglang/glm-5.1/json/            (10 files, 1k1k parser-off)
~/benchmark/results/sglang/glm-5.1/json_withparsers/(10 files, 1k1k parser-on, diagnostic)
~/benchmark/results/sglang/deepseek-r1/json/        (10 files, 1k1k)
~/benchmark/results/sglang/qwen3.5-397b-a17b/json/  (10 files, 1k1k)
```

All file names follow the `<model>_<precision>_<framework>_tp<TP>_conc<C>_<SEQ>.json` pattern
baked into `bench_sweep.sh`. Logs for every run live next to the JSON in `logs/`.

**Provenance tag for every row in this table**: `b300-node-ark-h212, driver 590.48.01, sglang 0.5.10.post1, 2026-04-13 → 2026-04-16 session`. This node is being retired; future runs on a newer-driver node must be tracked in a separate summary file.
