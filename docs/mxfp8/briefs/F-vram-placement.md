# Brief F — VRAM/placement plan for Inkling-2-Mini on 2×16 GB (and why the docker container is crash-looping)

**Deliverable:** write `/home/tim/code/services/repos/llama.cpp/docs/mxfp8/07-vram-placement.md`.
**Do not edit any file outside that path. Do NOT start/stop/restart any docker container or service** (the parent is handling that decision with Tim). Read-only investigation + analysis. Print a 12-line summary to stdout. Cite file:line / log lines.

## Observed problem (ground truth, do not re-verify from scratch — but do read the live state)

`/home/tim/code/services/docker-compose.yml` was edited today ~17:47 and the `llama-cpp` service now points at `/models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf`. The container has **RestartCount=57** and dies with:

```
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 15861.32 MiB on device 0: cudaMalloc failed: out of memory
E ggml_backend_buft_alloc_buffer_n_default: failed to allocate CUDA0 buffer of size 16631803392
E llama_model_load: error loading model: unable to allocate CUDA0 buffer
```
Hardware: 2× GeForce RTX 5060 Ti 16 GB (16311 MiB each), driver 615.71.09, both cards hard-capped at **150 W** by `gpu-power-limit.service`. There is a separate `llama-cpp-worker` (Thunderbolt RPC) that is profile-gated and currently not running.

## Tasks

1. Read the current `llama-cpp` service block in `/home/tim/code/services/docker-compose.yml` **in full** (command/flags, `-ngl`, `-sm`, `--split-mode`, `-ot`, `--cache-type-k/v`, `-c/--ctx-size`, `-fa`, `--spec-*`, `--override-tensor`, env). Quote the flag list verbatim in the doc. Identify precisely why it asks for 15.86 GB on device 0 and OOMs: 17.63 GB model + KV on a 16.31 GB card, `-sm layer` vs `tensor` behaviour, `unified KV`, and any `--n-gpu-layers` value.
2. Read `/home/tim/code/services/docs/kv-cache-investigation.md` and any `/home/tim/code/services/docs/*.md` + `llama-cpp-vram-investigation.md` for the measured VRAM/KV numbers on this box (262k ctx, q4_1 KV, DFlash2 drafter, `-sm layer` required because `tensor` GGML_ASSERTs with the drafter). Extract what's reusable and what is Qwen-specific.
3. Compute an **arithmetic VRAM budget table** for Inkling-2-Mini (arch facts: 24 physical layers, hidden 2304, 8 KV heads, head_dim 128, `sliding_window_size 1024` on 20 of 24 layers — so 4 layers keep full KV; vocab 201024; MTP head exists separately). Bytes/token for KV at f16 / q8_0 / f8_e4m3 / MXFP8-equivalent; then per-card totals for these three model variants: (a) ggml-org IQ4_NL 17.63 GB, (b) unsloth UD-Q4_K_XL 17.63 GB (40 native NVFP4 expert tensors + Q8_0 for blk.20-23), (c) our target **lossless native** ≈ 20.4 GB (40 NVFP4 + 8 MXFP8 experts). For each: max ctx at 2 cards with `-sm layer`, and which tensors must stay on CPU (via `-ot`) to make it fit, with the expected t/s penalty direction (MoE expert layers on CPU hurt much less than attention, and blk.20-23 are only 4 of 24 layers).
4. State the **recommended placement recipe** for the lossless file: `-sm layer` vs `tensor` (note master's Inkling + draft interaction), `--tensor-split`, `-ot` carve-out string to try first, KV type choice, and a fallback ladder (what to relax if it still OOMs: ctx → KV → one expert group to CPU). Note the 150 W cap is a *throughput* constraint, not a capacity one, and that the box's `llama-cpp` image is sm_120/CUDA 13.2 (see `/home/tim/code/services/AGENTS.md` gotchas; also note we are building **locally outside docker** for this project, so the recipe must be expressible as plain `llama-server` flags too).
5. Cross-check with the fleet context in `/home/tim/code/services/SWARM.md` (the Thunderbolt RPC pair: spare-parts-1 is the head, this box can serve as an RPC worker exposing the second card) — mention whether RPC offload is a legitimate alternative to `-ot` CPU offload here and its measured bandwidth (21–24 Gbps TCP, ~0.4 ms RTT).

Output must be decision-ready: Tim should be able to pick flags from this doc without re-deriving anything.
