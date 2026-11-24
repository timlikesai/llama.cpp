# Brief U — Phase 5: prove the lossless file is *actually* lossless with perplexity and KL, not just anecdotal prompts

## Why this matters

Everything so far rests on a 4-prompt anecdote (`68 B: 64 …`). "Bit-exact bytes" is a strong claim; the honest version is: **the expert tensors are byte-identical to the checkpoint, so any measurable difference between our file and a dequantized build must come from the runtime path, not from quantization loss.** A PPL/KL comparison is how you demonstrate that — and how you'd catch a silent convention bug that produces plausible-but-wrong text. It also produces the number a PR needs.

**No builds.** Another agent owns `build-cuda` (compile, and it may be editing `mmvq.cu`/`vecdotq.cuh`). Use only the existing binaries in `build-cuda/bin` — `llama-perplexity`, `llama-completion`, `llama-quantize`, `llama-file` — and python. If you conclude a rebuild is needed, write the change into your doc and stop.

## Artifacts already on disk (do not regenerate)

| file | size | what it is |
|---|---|---|
| `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` | 18,974,614,688 B | **ours**: 40 NVFP4 + 8 MXFP8 experts, bit-exact, BF16/F32 rest, 744 tensors, arch `inkling2_loop`, file_type 39 |
| `~/models/timlikesai/Inkling-DEQ-Q80.gguf` | 27,820,099,456 B | our converter's **dequantized** build (`--outtype q8_0`, no `--native-experts`). This is the reference — it already generates `68 B: 64 …` correctly (`logs/deq-q80-run.log`) |
| `~/models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf` | 17.63 GB | public **lossy** reference |
| `~/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf` | — | has 40 native NVFP4 expert tensors (byte-identical to ours) + Q8_0 for L20–23 |
| `~/models/timlikesai/Inkling-2-Mini-NATIVE-scale1.gguf` | 18,974,614,688 B | diagnostic variant with all 40 `.scale` = 1.0 — expected to be broken; useful as a "sensitivity" control |
| HF checkpoint | 20.19 GB | `/home/tim/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337` |

## Task

1. **Write `docs/mxfp8/20-ppl-kl-validation.md` in your first 4 tool calls**: the plan, then fill it as numbers land. Same rule that bit seven siblings — no invisible success.
2. **Perplexity**, same corpus/context/seed for every file, `llama-perplexity` on GPU (auto-fit; **never pin a nonzero `-ngl`** — it disables memory fitting and OOMs on these cards):
   - `Inkling-2-Mini-NATIVE.gguf` (ours, bit-exact)
   - `Inkling-DEQ-Q80.gguf` (dequantized reference)
   - `Inkling-2-Mini-IQ4_NL.gguf` (public lossy)
   Use a real text corpus (`--help` for the file option; `logs/g0-prompt-long.txt` is only 2.8 KB — download or synthesize something substantially larger, and record what you used). Report mean PPL ± error and the token count. **A lossless repack should be statistically indistinguishable from the dequantized build** — if it isn't, that's the finding, and it invalidates my byte-exactness claim, so chase it and say so loudly.
3. **KL divergence vs the dequantized reference** — the sharper test, because it compares distributions rather than one likelihood. Get per-token logits/probs from both files on the same prompt set (`llama-completion --probabilities` or the perplexity tool's output; check what this build supports) and compute mean KL + max KL + perplexity-of-reference-under-ours. If no binary exposes full logits, write a small python harness over `llama-cpp-python`-style calls **only if** it's already installed — otherwise state the limitation and use the `--probabilities` top-k route, documenting what it can and cannot detect.
4. **Sensitivity control**: run the same measurement on `Inkling-2-Mini-NATIVE-scale1.gguf` (broken `.scale`). It must be dramatically worse. If it isn't, your harness can't see the difference and the KL result is worthless — fix the harness before reporting anything.
5. **Long-context check while you're at it** (Phase 6 groundwork, GPU only, no code changes): at `-c 8192` and beyond, record memory per card from the load log and note where fp8 KV vs f16 KV matters. Child E's earlier math (in `docs/mxfp8/`) says the SWA cap of 1024 tokens on the 20 sliding-window layers is load-bearing: fp8 KV = 296/552/1064/2088 MiB at 32k/64k/128k/262k, and f16 KV uncapped at 262k = 42,590 MiB (over budget by ~9.8 GB). Verify that against an actual run if you can, and flag any contradiction.

## Rules

- **Read-only for source**: you may not edit `conversion/**`, `src/**`, `ggml/**`, `tools/**`, `tests/**`, or any model file. Create only your doc, scripts under `scripts/inkling_native/`, and logs under `logs/`.
- **Never delete a file or log. Never `rm` anything in `~/models/`.** Disk: ~250 GB free; do not write another multi-GB GGUF — measure the ones that exist.
- No docker/systemctl. GPUs are free but a build lane may be running benchmarks on them — if timings look contaminated, note it and re-run rather than reporting a number you don't trust.
- Report negative and ambiguous results plainly. My byte-exactness claim is load-bearing and I'd rather have it broken in public than have you smooth it over.
- 40-minute budget; finalize at 34.
