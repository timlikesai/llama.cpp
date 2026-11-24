# 47. Setting sweep on the shipped server configuration

Measured 2026-11-23 on the deployment itself: `Inkling-2-Mini-FULL.gguf` (19.61 GB, 798 tensors =
744 trunk + 54 drafter), 2x RTX 5060 Ti 16 GB, ctx **262144**, `-sm layer`, `-fa on`, `-b 2048`,
`-ub 256`, `q4_1` target KV **and** `q4_1` draft KV, `--spec-type draft-dspark,ngram-mod,ngram-map-k`,
`--spec-draft-n-max 2`, `--spec-draft-n-ubatch 64`, vision on CPU (`--no-mmproj-offload`).

Harness: `scripts/inkling_native/profile_shipped.sh sweep`, which reads the flag vector straight out of
`/home/tim/code/services/docker-compose.yml` so it cannot drift from what is deployed. Prompt:
`logs/prompt-code.txt` (2,527 tokens), 192 tokens generated, `temperature 0`, `seed 42`,
`cache_prompt: false`, `timings: true`.

## Round 1 - CUDA/backend environment knobs

| config | prefill t/s | decode t/s | draft n | accepted | verdict |
|---|---:|---:|---:|---:|---|
| baseline | 1,036.2 | 52.2 | 247 | 0.482 | reference |
| `GGML_CUDA_MMQ_PREC=q8` | 878.1 | 49.2 | 201 | 0.557 | **-15% prefill - rejected** |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` | - | - | - | - | **OOM, cannot start - rejected** |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` | 690.6 | 41.6 | 247 | 0.482 | **-33% prefill, -20% decode - rejected** |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 1,044.6 | 52.1 | 247 | 0.482 | +0.8% = wash |

Two of these are informative beyond "no gain":

- **`MMQ_PREC=q8` is a dial on the 20 NVFP4 layers, not on MXFP8.** It selects how src1 (the
  activation) is quantized for the block-scaled MMA (`ggml/src/ggml-cuda/mmq.cu:100-122`). The default
  already gives FP4 weight types native 4-bit activations (W4A4) and gives MXFP8 `e4m3` activations
  (W8A8) - see `44-intrinsics-audit.md`. Paying 15% of prefill to widen FP4 activations confirms the
  fast path is the one in use. The output also *changes* (different md5 on the generated text, same
  prompt/seed): coarser-to-finer activation rounding flips near-ties at temperature 0, which is
  expected and is not a bug.
- **Unified memory is actively harmful here.** `token_embd.weight` (883 MB) sits in host RAM by
  buffer-type policy, and earlier I had hypothesized that making `CUDA_Host` real (unified memory)
  would recover the per-token host crossing. It costs a third of prefill instead. Combined with the
  `-ot token_embd.weight=CUDA0` test (flat: 1,022 vs 1,025 prefill, +2.4% decode, acceptance
  bit-identical), the host-resident embedding is **not** a meaningful ceiling and should be left alone.

## Round 2 - the server-only flags no CLI measurement ever included

| config | prefill t/s | decode t/s | draft n | accepted | text md5 | verdict |
|---|---:|---:|---:|---:|---|---|
| baseline (r2) | 1,038.6 | 52.1 | 247 | 0.482 | `00dec267` | reproduces r1 (0.2%) |
| drop `--ctx-checkpoints --checkpoint-min-step --cache-idle-slots --cache-reuse` | 1,039.2 | 52.1 | 247 | 0.482 | same | wash |
| **`--spec-type draft-dspark`** (ngram chain removed) | 1,036.1 | **44.6** | 181 | 0.552 | same | **-14% decode - the chain is a win** |
| `--threads 16` | 1,034.4 | 51.7 | 247 | 0.482 | same | slightly worse |
| `--threads 24` | 1,035.9 | 51.4 | 247 | 0.482 | same | slightly worse |

**Reversal worth stating plainly.** The ngram chain had looked free-to-harmful in CLI measurements:
it exactly doubles host readback traffic (217 -> 434 tensor reads, `46-...` Finding) and was a wash at
small context (1,014.8/42.8 vs 1,021.1/42.5). On the shipped server at 256K with a code prompt,
**removing it costs 14% of decode.** The mechanism is visible in the counters: with the chain, 247
draft tokens are proposed at 0.482 acceptance; without it, only 181 at 0.552 - the ngram speculators
propose more tokens per step, and even at a lower per-token rate the extra accepted tokens win. So the
earlier "drop the chain" prior recorded in `46-...` is **wrong at long context and retracted here**.
Per-domain behaviour still differs (`41-kv-cache-type-study.md`, `39-...`), so this is a statement
about code prompts at 256K, not a universal.

Everything else in the deployed flag set is load-bearing or neutral; no flag change found this session
improves on the shipped configuration. The remaining levers are code-level, tracked as TODO #4 (bulk
movement for the ~1.8 s of 5-20 us copy/concat/scale kernels), #6 (MoE decode batching), #19
(`-sm tensor`), #20 (ncu audit of `mul_mat_q<44>`).

## Harness defects found and fixed (these produced false results, so they are part of the record)

1. **Fixed harness port.** All configs polled `127.0.0.1:8099`; a leftover server from an interrupted
   run answered `/health` there, and a config whose own process had exited was reported "healthy after
   3 polls". Now each config takes a random port and the health probe also requires the process to be
   alive.
2. **`set -e` + a failing `grep` in the health loop** aborted stages silently. The sweep block now runs
   `set +e` and records the last HTTP code explicitly.
3. **Results came from the HTTP response, not log scraping.** `timings` in the chat-completion response
   gives `prompt_per_second` / `predicted_per_second` / `draft_n` / `draft_n_accepted` per config, and
   the generated text is hashed so an output change is caught rather than rationalized. Responses are
   saved as `logs/sweep-<round>-<tag>.log.json`.
4. **Never delete previous probe output.** The first rewrite did `rm -f` on the prior round's logs;
   removed. Rounds are namespaced by `ROUND` instead (`r1`, `r2`).
5. **`-v` changes what this build prints.** Without `-v`, `llama-cli` writes only warnings; with `-v`,
   the `[ Prompt | Generation ]` summary line disappears in favour of `print_timing`. Both modes now
   have distinct parsers (see the traps list in `README.md`).

## 4. Speculation knobs on the *deployed* flag vector (2026-11-23, late)

Same window, same file (`Inkling-2-Mini-FULL.gguf`), 262,144 ctx, `q4_1` caches, the ngram chain
(`draft-dspark,ngram-mod,ngram-map-k`) included because that is what compose ships. `mode_shiptune`.

| `--spec-draft-n-max` | `--spec-draft-n-ubatch` | decode t/s | fitter retries |
|---:|---:|---:|---:|
| **2** | **64** (deployed) | **52.01** | 0 |
| 2 | 32 | 51.83 | 0 |
| 3 | 64 | 50.65 | 0 |
| 3 | 32 | 50.50 | 0 |

The shipped pair is the best of the grid, so nothing to change. `n_max 3` costs 2.6% — the block gets
wider faster than acceptance holds up. `--spec-draft-n-ubatch` is a wash at this context (0.3%), which
is worth knowing because it looked like a VRAM lever.

Independent corroboration of the chain: the same file at the same context measured **48.2 t/s** with
`draft-dspark` alone (§1-3 and doc 41) and **52.0 t/s** here with the chain — +7.9%.

Note the acceptance column is empty: with the chain the server prints its per-slot report in a different
shape than `llama-cli`'s `draft acceptance = X (N/M)`, and I am not going to guess a regex. Decode t/s
is the metric that decides anyway, and zero `failed to allocate` lines means these numbers are clean.
