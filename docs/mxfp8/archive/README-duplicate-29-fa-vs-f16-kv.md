> [!NOTE]
> Archived 2026-11-21: this duplicated the topic of `../24-fa-kv-memory.md`, which is canonical (it carries the
> corrected two-axis reading: both iswa `llama_kv_cache: size =` lines and the final `sched_reserve` pass only,
> plus the finding that FA does NOT cost more memory than f16 KV here). Kept verbatim for provenance; not deleted.

# 29 — FA vs f16 KV: the memory matrix (item #22, now measurable)

Status: **running.** Task from `docs/mxfp8/00-status.md` (14:40 entry): "#22 (FA using more
memory than f16 KV) is now *measurable* — it was blocked precisely because `resolve_fused_ops`
force-disabled FA on the alias-straddling split, which `e046294a7` fixes."

Matrix to measure: {FA off, on} x {f16, fp8, q8_0 KV} x {2k, 8k, 32k} against theoretical size.
Three specific sub-questions from the brief:

1. Is KV allocated **per applied slot (48)** instead of **per unique layer (24)**?
2. Is `sliding_window = 1024` honoured in the FA path?
3. Do the shared loop slots get separate KV state in the FA allocator?

## Model facts this is computed against

`/home/tim/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` (`logs/inkling-gguf-meta-iq4nl.log`
for the same hparams in the public IQ4_NL file):

| hparam | value |
|---|---|
| physical layers (`block_count`) | 24 |
| loop core | `[6, 18)`, `count = 3` -> `n_slot = 24 + 2*12 = 48` |
| `layer_phys(il)` | `il < 24 ? il : 6 + (il - 24) % 12` (`src/models/inkling2-loop.cpp:3`) |
| `n_head_kv` / `key_length` / `value_length` | 8 / 128 / 128 -> `n_embd_k_gqa = n_embd_v_gqa = 1024` |
| `sliding_window` | 1024, pattern 5-of-6 True (full-attn at physical 5, 11, 17, 23) |
| arch memory class | `llama_memory_hybrid_iswa` (`src/llama-model.cpp:3370`, `:3409`) |

## Theoretical KV bytes (no measurement, arithmetic only)

Per slot per token: K = `1024 * tsz`, V = `1024 * tsz` -> f16 4096 B, q8_0 2176 B, f8_e4m3 2048 B.

Slots by kind (48 total, from `layer_phys` + the SWA pattern): **8 full** (5, 11, 17, 23, 29, 35,
41, 47) and **40 SWA**.

Cache sizes come from `src/llama-kv-cache-iswa.cpp:75`: `size_base = n_ctx_seq`,
`size_swa = GGML_PAD(min(size_base, n_swa*n_seq_max + n_ubatch), 256)`.

| n_ctx | size_base | size_swa (ub=2048) | f16, 48 slots | f16, unique 24 | f8_e4m3, 48 slots |
|---|---|---|---|---|---|
| 2048 | 2048 | 2048 | 384.0 MiB | 192.0 | 192.0 |
| 8192 | 8192 | 3072 | 736.0 MiB | 368.0 | 368.0 |
| 32768 | 32768 | 3072 | 1504.0 MiB | 752.0 | 752.0 |

## Measured

(TODO — `logs/ad-fa-kv-matrix.log`, driver `scripts/inkling_native/ad_fa_kv_matrix.py`)

## Verdicts on the three sub-questions

(TODO)

## Corrections to my brief

(TODO)
