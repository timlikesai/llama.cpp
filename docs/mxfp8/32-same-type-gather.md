# 32 — Same-type gather: −11% compute at 128K, +27% auto-fit context

Two commits: `06a5a7cd8 ggml : add ggml_get_rows_same_type` and `b18b212c6 model : gather the Inkling relative bias in the mask type`. Follows #30 (`77bd382db`, `docs/mxfp8/30-bias-window-compression.md`) — that fold removed the broadcast add; this removes the remaining f32 intermediate.

## The blocker was one line, and it had a second symptom

```c
// ggml/src/ggml.c, ggml_get_rows()
// TODO: implement non F32 return
enum ggml_type type = GGML_TYPE_F32;
if (a->type == GGML_TYPE_I32) { type = a->type; }
```

This explains the session's most confusing null result (§1 of doc 30): casting the bias table to f16 *before* the gather changed the ctx slope by exactly **0.0%**, because the gather returned f32 no matter what went in. A dtype optimization upstream of a gather is simply inert while that line stands.

## Scoping: why the blast radius is small, and fail-safe

Verified by reading, not assumed:

| backend | state |
|---|---|
| **CUDA** | **no change needed** — `get_rows_cuda(..., dst->type, ...)` at `getrows.cu:463` already takes the destination type independently of src0, asserted with `GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type))`; `supports_op` (`ggml-cuda.cu:6474`) switches on src0 only, so it accepts the new node |
| **CPU** | the only forward that hard-coded the assumption (`ggml_compute_forward_get_rows_f16/_bf16` passed `(float *) dst->data`); now copies the row verbatim when `dst->type == src0->type` |
| Metal / Vulkan / SYCL / OpenCL / OpenVINO | one guard line each in `supports_op` |
| spacemit `ime.cpp` | already checks `op->src[0]->type == op->type` — no change |

Design choice: **`ggml_get_rows` is untouched**; a new `ggml_get_rows_same_type()` opts in, and exactly one call site uses it. So no existing caller on any backend changes behavior by construction.

The guard condition at each backend is `dst->type == F32 || src0->type == I32` — **a statement true of every graph buildable today**, so it can decline nothing that used to work. And a decline is *fail-safe*: the scheduler moves the node to CPU (correct, just unaccelerated) rather than writing an unexpected element type. That directionality is what makes the untested guards acceptable — they can only ever make a backend claim less.

**Honest limitation:** the Metal/Vulkan/SYCL/OpenCL/OpenVINO guards are reviewed but **not compile-verified** — no toolchains for them on this host. Filed as #36.

## Results (lossless 19 GB checkpoint, 2× RTX 5060 Ti, FA on, f8_e4m3 KV, `-sm layer`)

dense-layer chain in units of `n_kv × n_head × n_ubatch`:

```
original:  F32 gather 1.0 + F32 permute-copy 1.0 + F32 broadcast add 1.0 + narrowing cast 0.5 = 3.5
after #30: F32 gather 1.0 + cast 0.5 + F16 permute-copy 0.5                                    = 2.0
after #32: F16 gather 0.5 + F16 permute-copy 0.5                                                = 1.0
```

| metric | before | after | Δ |
|---|---|---|---|
| compute @128K, `-ub 256` | 9,892.88 MiB | **8,781.78** | **−11.2%** |
| auto-fit context, `-ub 128` | 202,496 | **258,048** | **+27.4%** |
| auto-fit context, `-ub 256` | 107,776 | **143,616** | **+33.3%** |
| prefill / decode @8k | 906.09 / 40.40 | 907.83 / 40.41 | none |

Cumulative ctx-proportional term across #30+#32: **3.5 → 1.0 units**. By unit arithmetic that implies slope 0.1698 → ~0.048 MiB/token (−72%) — but note this is **derived, not measured**: while the fitter is rebalancing weights between the two cards, a two-point slope from `sched_reserve` totals is not trustworthy (see the anomaly below). The measured facts are the 128K total (−11.2%) and the auto-fit ceilings (+27.4% / +33.3%). A clean slope needs `-sm none` or post-#34 measurements.

## The anomaly I did not smooth over

At ctx 32768 / `-ub 512` the *summed* compute buffer got **worse**: 6,894.69 → 7,360.39 MiB. Per device: CUDA0 **−530**, CUDA1 **+995**. The fitter rebalances weights once buffers shrink, so the sum of per-device peaks moves independently of the per-token cost — which is why I report the auto-fit ceiling and the 128K total as the meaningful numbers, not a mid-range point. Folded into #34 (attributing the flat batch-proportional term), since that term is what the fitter is trading against.

## Verification

New op coverage (`same_type` flag on `test_get_rows`): F16/BF16 × {plain, view src0} plus the real `{24, 1026}` gathered 512 rows — **265/265 CPU, 241/241 CUDA0**. Gates: `test-backend-ops -p mxfp8` OK on CPU/CUDA0/CUDA1, `test-llama-archs` 587/587, `test-quantize-fns` exit 0, greedy probe unchanged on CUDA and CPU (`A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:`). Logs: `logs/rt-build-{1,2}.log`, `logs/rt-mem-{8192,32768}.log`, `logs/rt-128k.log`, `logs/rt-auto-{128,256}.log`.

## Next levers, in the ranked order from the plan

1. **#34** attribute the flat ~4.5 GiB batch-proportional term — now the largest single component, and the thing the fitter trades against.
2. **#35** bias operand in `FLASH_ATTN_EXT` — would take the remaining 1.0 unit to ~0 and make compute genuinely flat in context.
3. **#27** the capacity matrix (unblocked): with 258,048 auto-selected at `-ub 128`, a *pinned* 262,144 is now within reach — worth testing `-ub 64` and q4 KV variants.
