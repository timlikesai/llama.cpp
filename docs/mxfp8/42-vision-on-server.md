# 42 — Vision on the server: GPU projector OOM-crashes, CPU offload fixes it

**Date:** 2026-11-23 · **Machine:** spare-parts-2, 2× RTX 5060 Ti 16 GB
**Status:** shipped and verified — multimodal works on the lossless model at 262144 ctx with the
DSpark drafter engaged.

## Why

`~/.pi/agent/models.json` advertises `"input": ["text","image"]` for `inkling/inkling-2-mini`, but
the compose service had `--mmproj` commented out. So the declaration was a lie: an agent sending an
image got

```
HTTP 500 {"error":{"message":"image input is not supported - hint: if this is unexpected,
          you may need to provide the mmproj"}}
```

Enabling vision makes the claim true. Projector: `/home/tim/models/timlikesai/Inkling-2-Mini-vision-mmproj.gguf`
(641,806,144 B, 422 tensors, `clip.vision.projector_type = inkling2v`, 23 layers, hidden 1024,
`image_size 640`, `patch_size 20`, `spatial_merge_size 2`, `projection_dim 2304`) — the converter
output validated earlier in `docs/mxfp8/17`.

## What happened on the GPU (the failure, kept because it is the interesting part)

With `--mmproj` and default offload settings the server **loaded fine** and printed

```
I srv load_model: loaded multimodal model, '/models/timlikesai/Inkling-2-Mini-vision-mmproj.gguf'
W srv load_model: cache_reuse is not supported by multimodal, it will be disabled
```

…then the **first image request killed the process**:

```
/app/ggml/src/ggml-cuda/ggml-cuda.cu:119: CUDA error
E CUDA error: out of memory
  server_context_impl::iterate(...)
  server_context_impl::pre_decode()
  server_context_impl::update_slots()
```

`docker inspect` showed `RestartCount=1`, `OOMKilled=false` — an in-process CUDA OOM, not the kernel
evicting us. Client-side it looked like `http.client.RemoteDisconnected: Remote end closed connection
without response`, which is worth knowing: **a dropped connection on the first image is a server crash,
not a bad request.**

Placement is the reason. The projector goes to the *first* device (CUDA0), which is the tighter card
here — at that moment CUDA0 had ~442 MiB free against CUDA1's ~1.1 GiB. The ViT's own compute buffers
(1024 patches before the 2×2 merge, 23 blocks, f16) had nowhere to go.

## The flag surface (answers "does llama.cpp have Strata's CPU-vision trick?")

Yes — all of it lives in `common/arg.cpp`, examples-gated to `mmproj_examples`:

| flag | effect | source |
|---|---|---|
| `--mmproj-offload` / `--no-mmproj-offload` | GPU or CPU for the projector; **default is enabled (GPU)** | `arg.cpp:2576-2582` → `params.mmproj_use_gpu` |
| `-mmdev`, `--mmproj-device DEVICE` | explicit device; `none` = don't offload. Env: `MTMD_BACKEND_DEVICE` | `arg.cpp:2585-2602` |
| `--mmproj` / `-mm`, `--mmproj-url`, `--mmproj-auto` / `--no-mmproj` | which projector, and whether to auto-fetch one with `-hf` | `arg.cpp:2553-2575` |
| `--image-min-tokens` / `--image-max-tokens` | clamp the vision token budget (dynamic-resolution models) | `arg.cpp:2618,2625` |
| `-ctim`, `--cache-type-images` | image **embedding** cache type (caches the encoded image, not a KV cache) | see `docs/multimodal/README.md` |

Under the hood `mtmd_select_device(ctx_params.use_gpu, ctx_params.device)` picks the backend and falls
back to the CPU buffer type (`tools/mtmd/mtmd.cpp:936-947`), and `mtmd_context_params` carries
`n_threads`, so CPU encoding uses `--threads`.

**Shipped:** `--no-mmproj-offload`.

## Measured cost of CPU encoding

Isolated by submitting a request with `max_tokens: 1` (so the round trip is essentially encode +
prefill) against a text-only control — script `scripts/inkling_native/probe_server_vision.py`:

| | round trip |
|---|---|
| image + 1 token | 205 ms |
| text + 1 token | 49 ms |
| **≈ CPU ViT encode** | **~156 ms per image** |

(My first estimate in the compose comment was "~a second"; the measurement is 6× better, and the
comment now carries the number.) Full request round trips including a short answer: 890 ms (red) and
1374 ms (blue).

VRAM: freeing the projector took CUDA0 from 15,869 → 15,175 MiB (~690 MiB back) and it stays back
during image requests, which is the point — no ViT buffers on either card.

## Verification (all green)

```
$ python3 scripts/inkling_native/probe_server_vision.py
PASS  health: {'status': 'ok'}
PASS  alias: alias=inkling/inkling-2-mini ftype=NVFP4 modalities={'vision': True, 'video': True, 'audio': False}
PASS  vision_advertised: modalities=... (vision=True => mmproj loaded)
PASS  text_oracle: answer='68'
PASS  vision_red: answer='Red'
PASS  vision_blue: answer='Blue'
PASS  alive_after_vision: {'status': 'ok'}
RESULT: ALL PASS     # RestartCount stayed 0 across the whole probe
```

Two details that make this a real test rather than a formality:

1. **Two colors, same prompt.** A model guessing from a language prior gets one of red/blue wrong;
   reading pixels gets both. Single-image tests cannot distinguish those.
2. **The text control runs after the images** — proving the encode didn't corrupt the slot or leak
   into subsequent decodes.
3. `--no-mmproj-offload` did **not** degrade text throughput: decode and acceptance are unchanged
   from the text-only baseline, and the lossless answer to `17 times 4` is still `68`.

## Trade-offs now in effect

| gained | lost |
|---|---|
| image input is real (declaration honest), agents can use this endpoint multimodally | `--cache-reuse 4096` is **disabled by multimodal** (server-enforced, see the warning above). Within-slot prompt caching still works (`cached_tokens=40` observed on a repeat); idle/evicted-slot reuse is gone. |
| ~690 MiB of VRAM stays free, no ViT buffers, no OOM | +156 ms per image on CPU. Higher resolution / multiple images scale up from there; `--image-max-tokens` and `-ctim` are the levers. |

If agent latency across many turns matters more than vision, the rollback is one line: delete
`--no-mmproj-offload` **and** the two `-mm`/path lines (or just the `-mm` pair, which removes
multimodal entirely and brings `--cache-reuse` back).

## Not claimed

- **No measurement of image *quality* beyond solid colors** — a color check proves the pixels reach
  the decoder, not that the tower describes a photograph well. Lane U's earlier `llama-mtmd-cli` test
  used a color-swap control too. A real eval (e.g. MMMU-style or even a screenshot description) is
  unrun.
- **Multi-image, video and high-resolution paths are untested.** `modalities` says `video: True`.
- **Long-context retrieval with the q4_1 cache is still unverified** (open item in #27) — unrelated to
  vision, but it is the same "quality under the shipped config" gap.
