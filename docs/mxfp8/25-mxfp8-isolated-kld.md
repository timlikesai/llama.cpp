# 25 — MXFP8 isolated KLD/PPL on Qwen3.5-0.8B (dense, no Inkling confounds)

Status: **IN PROGRESS** (brief AA resumed; §1 verified below, packer + measurements land after).
Briefs: `docs/mxfp8/briefs/Y-mxfp8-isolated-kld.md` (original), `docs/mxfp8/briefs/AA-mxfp8-kld-finish.md` (resume, narrow scope)

## 0. Claim under test

MXFP8 (8.25 bpw, block-scaled e4m3 + ue8m0) should land **between** `q8_0` (8.5 bpw) and
`mxfp4` (4.25 bpw) on KLD vs the bf16 ground truth, and much closer to `q8_0`. If MXFP8 is
materially worse than q8_0 at similar size, something in the type or its runtime use is
wrong — report loudly. Second axis: CPU vs CUDA KLD for the *same* mxfp8 file must agree;
disagreement is a compute-path bug (undetectable on Inkling because of its arch defect).

## 1. Environment facts (verified, not assumed)

- [x] **source GGUF present + arch/type** —
      `/home/tim/.cache/huggingface/hub/models--ggml-org--Qwen3.5-0.8B-GGUF/snapshots/004fd086eccea9f7cf867aa07ba328773beef81d/Qwen3.5-0.8B-BF16.gguf`,
      1,557,662,496 B, `general.architecture = qwen35`, `general.file_type = 32` (BF16), 335 tensors
      = 195 BF16 (2-D weights, `ne[0] % 32 == 0`) + 140 F32 (norms / 1-D). It is a hybrid model
      (attn + `ssm_*` + `attn_gate`) with one `blk.24.nextn.*` MTP head.
- [x] **`llama-quantize` cannot emit MXFP8** — the allowed-type list printed by
      `./build-cuda/bin/llama-quantize` (no args) contains `38 MXFP4_MOE` but **no mxfp8 entry**.
      The type *is* reachable through `--tensor-type <name>=mxfp8` (see `logs/mxfp8-y/c-mxfp8.log`:
      every 2-D tensor gets "applying manual override: q8_0 -> mxfp8") and the quantizer does run,
      but `ggml_validate_row_data` has no `GGML_TYPE_MXFP8` case, so the post-quant validation
      aborts on tensor 2/335 with `ggml_validate_row_data: invalid type 44` ->
      `llama_model_quantize: failed to quantize`. That abort is what left
      `~/models/timlikesai/Qwen3.5-0.8B-MXFP8-C.gguf` truncated at 10,966,560 B with an unreadable
      header (`GGUFReader: ValueError: GGUF magic invalid`). Left in place, never deleted.
      => the MXFP8 file for this study must come from the python packer.
- [x] **binaries** — `build-cuda/bin` is a full tree dated Nov 21 12:33-12:35 (libggml-cuda.so
      12:35, `llama-perplexity`, `test-backend-ops`, `llama-quantize` all present). Not rebuilt
      by this lane (no builds allowed; the tree was already free of other users at 13:45,
      `nvidia-smi` showed 0% util / 13 MiB used on both GPUs).
- [x] **regression gate** — `test-backend-ops -b CPU -p mxfp8`: **306/306 tests passed, Backend CPU:
      OK, 3/3 backends passed** (`logs/y-gate-cpu-mxfp8.log`, 13:47). CUDA0 run in flight ->
      `logs/y-gate-cuda0-mxfp8.log`.
- [ ] corpus: exact file, bytes, token count, chunk count (filled when the first run reports it).

Candidate files already on disk and structurally valid (all 335 tensors, 140 F32 + 195 quantized):

| file | size | file_type | quantized type |
|---|---|---|---|
| `Qwen3.5-0.8B-BF16.gguf` (source, = KL base) | 1,557,662,496 B | 32 | BF16 x195 |
| `~/models/timlikesai/Qwen3.5-0.8B-Q8_0.gguf` | 833,592,096 B | 7 | Q8_0 x195 |
| `~/models/timlikesai/Qwen3.5-0.8B-MXFP4.gguf` | 423,285,536 B | 7 | MXFP4 x195 |
| `~/models/timlikesai/Qwen3.5-0.8B-MXFP8-C.gguf` | 10,966,560 B | unreadable | BROKEN (aborted `llama-quantize` write) |

## 2. Python MXFP8 packer — convention

(to be filled: scale rule, e4m3 rounding, in-range assertion)

## 3. Self-check before any measurement

(to be filled: python decode vs C `from_mxfp8`, per-block relative error, requantize idempotence)

## 4. Perplexity + KLD table (base = bf16 source)

| candidate | bpw | size | mean PPL ± err | mean KLD ± err | same top p |
|---|---|---|---|---|---|
| bf16 (base, self) | 16 | | | | |
| q8_0 | 8.5 | | | | |
| mxfp8 (python) | 8.25 | | | | |
| mxfp4 | 4.25 | | | | |

## 5. CPU vs CUDA on the same mxfp8 file

## 6. Verdict — is MXFP8 fit for a PR?

## 7. Corrections to my brief

## 8. Artifacts
