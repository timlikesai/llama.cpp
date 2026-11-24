# Phase 4c (CUDA Tier B) - MXFP8 decode route in `mmvq`

Brief: `docs/mxfp8/briefs/T-decode-mxfp8.md`. Scope is the single-token / small-batch
`mmvq` route only. The `mmq` tile route for prefill landed in `c39ab0a00`.

## Baseline (before this change)

Command (from the brief, unchanged for the A/B):

```
llama-completion -m ~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf -c 8192 \
  -f logs/g0-prompt-long.txt -n 32 --temp 0 -no-cnv --seed 42
```

| metric | value |
|---|---|
| prefill | 506.95 t/s |
| decode  | 21.95 t/s |
| lossy IQ4_NL reference decode (parity line) | 23.62 t/s |

Source: `logs/r-perf-auto.log`.

## Dispatch finding

Confirmed in `ggml-cuda.cu` (`ggml_cuda_mul_mat_id`, ~L2655): for `ne2 <= MMVQ_MAX_BATCH_SIZE` the kernel
is picked by `ggml_cuda_mmvq_supported(src0->type)`. MXFP8 was explicitly excluded there (that function
returned `false` for it), so decode fell to the next branch, `(F8_E4M3 || MXFP8) &&
ggml_cuda_should_use_mmvf(...)` -> `ggml_cuda_mul_mat_vec_f`, i.e. the SIMT dequantize-to-float route. Same
shape in `ggml_cuda_mul_mat_id_needs_sync` (`[TAG_MUL_MAT_ID_CUDA_GRAPHS]`).

## Hypothesis

`mmvq` is organised around a `vec_dot_q_cuda_t` function pointer plus a `vdr` (values per thread per
iteration) table; the kernel body is type-agnostic apart from that dot, so MXFP8 needs no new kernel - one
dot plus three table entries. It then inherits the fused gate/up/GLU mmvq path for free, because
`ggml_cuda_should_fuse_mul_mat_vec_q` is driven by `ggml_cuda_should_use_mmvq`.

## Blocker found while implementing

`QI_MXFP8` was wrong for the `mmvq` indexing convention. In `mul_mat_vec_q` the type trait `qi` means "32-bit
payload words per block" (`blocks_per_iter = vdr*nwarps*warp_size/qi`, `kqs = vdr*(tid % (qi/vdr))`), which is
what QI is for every type already on the route: Q4_0 16 B -> 4, MXFP4 16 B -> 4, NVFP4 32 B -> 8, Q8_0 32 B ->
8. `ggml-common.h` had `QR_MXFP8 4` -> `QI_MXFP8 2`, read as "4 e4m3 codes per 32-bit word". With qi 2 the
kernel covers 8 of the block's 32 bytes and silently drops three quarters of every row. Fixed to `QR_MXFP8 1`
so `QI_MXFP8` is 8, matching `QR8_0`/`QI8_0` - MXFP8 is the same 1 byte per value shape as Q8_0. Nothing else
in the tree reads `QR_MXFP8`/`QI_MXFP8` (grep: only `ggml_cuda_type_traits<GGML_TYPE_MXFP8>`, whose `qr`
member has no readers in `ggml-cuda`), and `ggml.c`'s CPU traits do not use them, so the fix is contained.
This is outside the brief's file fence, and is flagged rather than worked around: a local override in
`mmvq.cu` would leave the trap in place for the next caller.

## Design as implemented

- `vecdotq.cuh`: `vec_dot_mxfp8_q8_1` + `VDR_MXFP8_Q8_1_MMVQ 2` - same per-thread shape as Q8_0 (2 int words
  = 8 values, `qi/vdr = 4` threads per block). Payload is read with `get_int_b1` because `block_mxfp8` is 33
  bytes, so its `qs` is never 4-byte aligned (same reason MXFP4 uses `get_int_b1`). Codes go through
  `ggml_cuda_cast<float2>` on `__nv_fp8x2_e4m3`, 2 per cvt, accumulated in f32 against the sign-extended
  `q8_1` int8s. No dp4a: an e4m3 payload has no integer form.
- Scale is `ggml_cuda_e8m0_to_fp32(e) * __low2float(bq8_1->ds)`, no `* 0.5`, and no `bq8_1->s` correction -
  the e4m3 codes are the values themselves, like MXFP4's table rather than the offset int4 types.
- `mmvq.cu`: MXFP8 added to `get_vec_dot_q_cuda` and `get_vdr_mmvq`; `ggml_cuda_mmvq_supported` is back to
  plain `ggml_is_quantized`.
- `ggml-cuda.cu`: dropped the now-dead `|| type == GGML_TYPE_MXFP8` terms in the two MUL_MAT_ID dispatch
  chains. MMVF still serves MXFP8 for `ne2 > MMVQ_MAX_BATCH_SIZE` and for plain MUL_MAT.
- No per-arch batch table entries: unlisted types get `MMVQ_MAX_BATCH_SIZE`, which is what consumer
  Blackwell already returns for every type (`get_mmvq_mmid_max_batch` short-circuits at `cc >= ADA_LOVELACE`).

## FINAL: route REJECTED (measured), trait fix kept as `b16720b83`

The lane timed out with the implementation building green but the gates unrun (blocked on GPU contention from the parallel PPL lane — `logs/t-baseline-run1..3.log` record the OOMs honestly rather than reporting contaminated numbers). The parent ran the gates.

**With the mmvq route enabled:**

| check | result |
|---|---|
| `test-backend-ops -p mxfp8` CUDA0 / CUDA1 / CPU | 237/237 / 237/237 / 306/306 |
| correctness, GPU and CPU | `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` unchanged |
| prefill @2817, 3 repeats | 507.38 / 508.79 / 508.34 t/s (baseline 506.95-509.03, flat in noise) |
| **decode @2817, 3 repeats** | **21.98 / 21.90 / 21.96 t/s** |
| decode short-prompt, 3 repeats | 28.34 / 28.25 / 28.20 t/s (baseline 28.11) |

**After reverting the route (`git checkout HEAD -- vecdotq.cuh mmvq.cu ggml-cuda.cu`):**

| check | result |
|---|---|
| CPU / CUDA0 ops | 306/306 / 237/237 |
| correctness | unchanged |
| prefill / **decode** @2817 | 509.03 / **22.24 t/s** |

**Decision:** the mmvq route is a small regression, not a neutral change — 21.90-21.98 t/s versus 22.24 t/s for the existing mmvf dequantize-to-float route, with prefill flat. It would also have traded f32 activations for q8_1 ones (strictly less accurate) to go slower, so it is reverted and only the `QI_MXFP8`/`QR_MXFP8` correction is committed, as `b16720b83`.

**Why this is still the most valuable part of the task:** `QI_MXFP8` was 2 where the header's own convention demands 8 (Q8_0 8, MXFP4 4, NVFP4 8), and nothing would have complained — the kernel indexes by `qi`, so with `qi == 2` it covers 8 of each block's 32 payload bytes and **silently drops three quarters of every row**, with no assert and plausible-looking output. The lane flagged the fix as outside its file fence instead of papering over it with a local override in `mmvq.cu`, which is what kept the trap from being inherited by the next caller.

**Decode is latency/occupancy-bound, not bandwidth-bound — measured by arithmetic.** Per token, Inkling reads only the routed experts: 8 of 128, so per NVFP4 layer `8 x (2,985,984 + 1,492,992) B = 35.8 MB`, across 20 NVFP4 layers = 716 MB; the 4 MXFP8 layers move `8 x (5.48 + 2.74) MB = 65.7 MB` each = 263 MB; attention/other adds a couple hundred MB. The loop model applies layers 2-13 two extra times, so MoE traffic is roughly 1.5-2x a plain 24-layer read: **~1.3-2.0 GB per token.** At 22 t/s that is only **~29-40 GB/s** against the ~450 GB/s per card available on two 5060 Tis — i.e. single-digit percent of peak bandwidth.

**So the ceiling is not bytes moved.** It is the shape of the work: ~384 tiny expert GEMVs per token (8 experts x 48 MoE applications x gate_up/down), dispatched across two cards, plus the ids gather and per-layer sync. A different src1 kernel was never going to fix that, which is consistent with `mmvq` coming out slower than `mmvf`. Real levers, in order of expected effect: **speculative decoding from the shipped MTP head** (multiplies tokens per wall-clock second directly, and the 636 MB BF16 draft is nearly free against ~12 GB of slack), then reducing per-layer launch/sync overhead for the MoE path. Note the earlier "~10.8 GB per token" figure in this document's first draft was wrong — that is the total NVFP4 expert bytes in the file, not per-token traffic.

## Progress

| Step | State |
|---|---|
| Write doc + capture baseline | done (this file) |
| Confirm today's decode kernel | done: MMVF via `ggml_cuda_mul_mat_id` |
| Implement `vec_dot_mxfp8_q8_1` + `mmvq` wiring | done, building (`logs/t-build-2.log`) |
| Gate 1: `test-backend-ops` CUDA0/CUDA1/CPU | done by parent: 237/237, 237/237, 306/306 |
| Gate 2: real-file correctness | done by parent: unchanged on GPU and CPU |
| Gate 3: decode A/B | done by parent once the GPU freed: 21.90-21.98 vs 22.24 baseline => **worse** |
| Decide: selected or documented negative | **documented negative; route reverted, trait fix committed** |
