# Inkling-2-Mini MXFP8 in llama.cpp — Roadmap & Architecture

Status: **v1 (plan)** · Branch: `mxfp8-weights` off `master` `29b52d398` · Box: spare-parts-2 (2× RTX 5060 Ti 16 GB, sm_120, CUDA 13.2) · Mode: **local build, outside docker**

| Doc | Contents | Owner |
|---|---|---|
| `00-status.md` | live status log, what's done/broken/next | parent |
| `01-roadmap.md` | **this file** — phases, gates, decision records | parent |
| `02-prior-art-branches.md` | index of `mxfp4-w4a8`, `w4a8-beast`, MXFP-KV branches, patch archives | child A |
| `03-touchpoint-checklist.md` | every file:line a new weight type must touch, ordered build sequence | child B |
| `04-inkling2-master.md` | Inkling-2 as implemented in master (converter + graph + ops + sizes) | child C |
| `05-hw-and-ecosystem.md` | sm_120a `mxf8f6f4` PTX facts, SFA/SFB layouts, vLLM/CUTLASS refs, upstream PRs | child D |
| `06-lossless-repack-design.md` | byte-level repack contract + size budget + verifier results | child E |
| `briefs/*.md` | the delegation briefs handed to the children | parent |

---

## 0. STATUS — 2026-11-21, phases 0-5 complete, phase 6 exceeded, two new workstreams opened

Written later than everything below it; where this section and a phase description disagree, **this section is current**.

**Delivered.** The unadulterated checkpoint runs locally on 2x RTX 5060 Ti as a lossless GGUF: 19.0 GB / 744 tensors, 40 NVFP4 + 8 MXFP8 expert tensors byte-identical to the safetensors (4 layers of MXFP8 required a new `GGML_TYPE_MXFP8 = 44`), CUDA + CPU in agreement, vision working end to end.

| phase | gate | outcome |
|---|---|---|
| 0 baseline | G0 | done — and its root cause for the docker crash loop: a pinned `-ngl` disables auto-fit |
| 1 MXFP8 core + CPU | G1 | done — 306/306 CPU ops, byte-exact fixed-point requantization test |
| 2 lossless repack | G2 | done — **NVFP4 needed no new code at all** (master already decodes it); MXFP8 converter + verifier, 320 rows byte-exact |
| 3 CUDA correctness | G3 | done — 237/237 (later 241) on both cards |
| 4 CUDA speed | G4 | done — native `mxf8f6f4` W8A8 `e4m3.e4m3` MMQ, prefill +4.6%; decode route tried and **rejected** (regression, kept the `QI_MXFP8` landmine fix) |
| 5 validation | G5 | done — PPL 5.5757 lossless vs 5.5636 lossy, KL resolves loss at 173x noise, sensitivity control proves the harness |
| 6 fit on 2 cards + fp8 KV | G6 | **superseded by the memory workstream below** — fp8 KV confirmed to halve KV, but KV was never the wall |

**Two things the plan got wrong, corrected by measurement rather than argument:** (1) the decode/prefill bottleneck was not bandwidth but **host-sync-bound device placement** — looped layers aliased onto physical layers whose weights sat on the other GPU (doc 27, +78% prefill / +82% decode); (2) the long-context wall was **not KV cache bytes** but a materialized per-head relative-bias chain in the compute buffers (doc 29-32). Assumption R-series and DR-6's sizing math are obsolete where they conflict with those docs.

### Workstream A — memory (docs 24, 26-32)

| result | numbers |
|---|---|
| alias-aware placement `e046294a7` | prefill 507 -> 901, decode 22 -> 40 t/s; peer copies 5,460 -> 1,106 |
| causal-mask fold `77bd382db` | ctx compute slope 0.1698 -> 0.0728 MiB/token; **128K unlocked** |
| fitter zero-span guards `9dae3d4c8` | SIGFPE -> legible error; latent upstream bug that the fold made reachable |
| same-type gather `4c394b02f` + `8395793da` | chain 3.5 -> 2.0 -> **1.0 units**; compute @128K -11.2%; auto-fit +27.4% |
| **indexed-bias FA, upstream #32808 adopted `84d9bc0dc`** | **prefill 850.9 -> 1,091.0 t/s (+28.3%)**, compute @8K 4,077 -> 3,568 MiB, PPL 5.5787 -> 5.5656; the tensor is gone instead of the nodes, which also retires our fold chain (doc 40) |
| **now** | **256K runs WITH the drafter, all weights resident, reproducibly: 1,027 t/s prefill / 52.1 decode, acceptance 0.733, zero fitter retries. Required `-ctk/-ctv q4_0` on the target — its cache had been f16, which is what was really blocking 256K (doc 41). Session arc on the complete model: 507/22 -> 1,027/52** |

Open: #58 (auto-fit + draft over-commits — the fitter cannot find a feasible plan on 2x16 GB, reproduces on the PR worktree build, so pre-existing; the shared root cause is allocator address-span, doc 40), #56 (re-baseline acceptance / retrieval / CPU-vs-CUDA on the merged tree), #21 (Inkling CPU batched-eval nan — gates any host-offload config), #57 (CLI placement-arg limitation), #27 (cache matrix: types done at 200K/256K, context sweep and long-context retrieval quality remain), #36 compile-verify the cross-backend guards, #28 host-spill (measured: **worse for speed**, 13.5 t/s all-host — capacity escape hatch only). ~~#34~~ attributed in doc 40, ~~#35~~ delivered by adopting #32808, ~~#55~~ closed by the target cache type (doc 41 §6).

### Workstream B — speculative decoding (docs 23, 33, 34)

Decode is launch-bound (~384 tiny expert GEMVs/token), so a drafter is the highest-leverage remaining win — and the measurement that proved it: the same 27B model with its DFlash2 draft decodes at **70.7 t/s vs 23.1 without (3.06x)**. **My original verdict that Inkling's head was "unretrievable / a new architecture" was wrong and is retracted**: it is DFlash + DFlare layer-wise fusion + a `rnn_hidden` Markov head, and the reference implementation is in vLLM (`vllm/models/inkling2/nvidia/dspark.py`, `ops/dspark.py`). All math is transcribed with its bf16 rounding points in doc 33.

**Delivered**: the 54 draft tensors embed in the main file as nextn layers (`e0292ff64`) *and* load as a separate `dflash` draft; per-stage DFlare tap mixing with a mutation-proven oracle (`55bf894f8`); the `rnn_hidden` readout and state channel; visit-indexed taps (the fix that took acceptance from 0.09 to 0.67 — found in upstream PR #32218's comment, not by reasoning); a padded-vocab draft-logit bug that affected every dflash draft. **Running today: 128K with the drafter = 1,024 t/s prefill / 50.1 decode, acceptance 0.625-0.673 depending on prompt.** Still short of shipping-quality drafting: acceptance is domain-dependent (code 0.67, prose ~0.2), and 256K + drafter does not yet fit (#55). Memory: 8.9 GiB free at 8K, but the draft's context at 262144 is what breaks co-residency. **Method correction that must survive**: speculation is only *algorithmically* output-invariant - four `--spec-type` values produced four different greedy strings here, so acceptance rate, not output text, is the gate.

Also outstanding, non-memory: #21 (Inkling CPU batched-eval nan, all quantizations), #18 (my DEQ reference emits non-finite logits), #23 (`llama-quantize` cannot emit MXFP8 - a gap I introduced when adding the file type).

### Housekeeping for the PR

Branch `mxfp8-weights` is **28 commits, rebased clean onto `upstream/master` at `5bacb7c86`** (2026-11-22, zero conflicts, twice), plus the squash-merge of `#32808`. Verified post-merge: ops mxfp8 green on CPU/CUDA0/CUDA1, `test-flash-attn-indexed-bias` green on both cards (incl. `rows=1`), archs 591/591, dflare oracle + mutation, `test-jinja`, and Qwen3.8-27B + DFlash2 draft statistics identical to the PR worktree build. Before submitting, split per **#41** (re-scoped after the adoption — our fold commits are gone, and `ggml_get_rows_same_type` no longer has a consumer in-tree, so justify or drop it).



## 1. What we are building

Run the **unadulterated** `thinkingmachines/Inkling-2-Mini` weights (~20 GB) in llama.cpp by *repackaging* them — never dequantizing/requantizing — into GGUF, and executing them natively on consumer Blackwell, which has the matching tensor-core instruction.

### The checkpoint (measured, on disk)

`~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/`
`model-00001-of-00001.safetensors` = 19,526,019,168 B (1244 tensors: 994 BF16, 154 F32, **48 F8_E4M3**, **48 U8**) + `mtp.safetensors` 635,801,416 B (54 BF16). `hf_quant_config.json` (`quant_method: inkling2`) splits the routed-expert weights into **two different native formats**:

| group | tensors | what | block | scale dtype | where |
|---|---|---|---|---|---|
| `nvfp4_nearest` | 40 | `layers.{0..19}.mlp.experts.{w13,w2}_weight` | 1×16 along K | `float8_e4m3` (stored) + F32 `…_amax`/`…_outer_scale_denominator` global | ~11.5 GB |
| `mxfp8_nearest` | 8 | `layers.{20..23}.mlp.experts.{w13,w2}_weight` | 1×32 along K | **`ue8m0`** | ~3.95 GB |
| `bfloat16` | 993 | attention, embeddings, sconv, controllers, vision | – | – | ~4.4 GB |
| `float32` | 74 | `attn_res.*` loop queries | – | – | small |

Payload shapes (E=128 experts): NVFP4 `w13` U8 `(128,2304,1152)` (packed ⇒ 2304×2304 fp4) + scale `(128,2304,144)`; `w2` U8 `(128,2304,576)` (⇒ 2304×1152) + scale `(128,2304,72)`. MXFP8 `w13` F8_E4M3 `(128,2304,2304)` + scale `(128,2304,72)`; `w2` F8_E4M3 `(128,2304,1152)` + scale `(128,2304,36)`. Scales are 1×N along K, per row → **classic MX**, despite the config's declared `weight_block_shape [32,32]` (master already resolves this conflict in favour of the stored shapes, `conversion/inkling2.py:157-160`).

Both groups declare a *required activation quantization* — NVFP4 layers want `nvfp4_e2m1`/e4m3/block-16 activations, MXFP8 layers want `mxfp8_e4m3`/**`ue8m0`**/block-32 activations. So the reference implementation is W4A4 and **W8A8** respectively. That is precisely the instruction family this hardware has.

Architecture: `inkling2_loop` — 24 physical layers replayed into ~49 logical steps (`loop_config {core_start 6, core_end 18, num_passes 3}`), hidden 2304, 128 routed experts × top-8, intermediate 1152, 24 q-heads / 8 kv-heads, head_dim 128, sliding window 1024 on 20 of 24 layers, sigmoid router + `route_scale 8` + gate bias, log-scaling RoPE-ish relative attention (`rel_logits_proj`, `d_rel 16`), softcaps, `final_logit_softcapping 30`, plus a 23-layer ViT (bf16) and a 3-layer MTP head (bf16).

### What master already has (verified)

- **The whole Inkling-2 architecture** landed in `29b52d398` (1 day old): `src/models/inkling2.cpp`, `src/models/inkling2-loop.cpp`, `conversion/inkling2.py`, `tools/mtmd/models/inkling2v.cpp`, `common/parsers/inkling.cpp`, arch kwargs in `llama-arch.cpp`.
- `GGML_TYPE_MXFP4 = 39` (`block_mxfp4 {uint8_t e /*e8m0*/; uint8_t qs[16];}`, 17 B / 32 elems) and `GGML_TYPE_NVFP4 = 40` (`block_nvfp4 {uint8_t d[4]; uint8_t qs[32];}`, 36 B / 64 elems), plus `GGML_TYPE_F8_E4M3 = 43` (raw 1-byte element, **no block scale**).
- Native FP4 tensor cores on Blackwell: `mma.cuh:1288` `kind::mxf4.block_scale.scale_vec::2X.m16n8k64…ue8m0`, `:1295` `kind::mxf4nvf4…ue4m3`, gated by `native_fp4_available` (`mmq.cuh:1040`) and selected by `prec_src1 == GGML_PREC_Q4` (`mmq.cu:75-83`). Build system already forces `120a-real` (`ggml/src/ggml-cuda/CMakeLists.txt:63-95`).
- Lossless repack machinery in the converter: `base.py:repack_mxfp4_blocks` (nibble de/re-interleave, writes `raw_dtype=MXFP4`), `_nvfp4_pack` (preserves e4m3 scale bits as ue4m3, `[4 scales][32 qs]` super-blocks), `_generate_nvfp4_tensors`/`_flush_nvfp4_experts` (stacks per-expert NVFP4 into `[E, rows, cols]` and writes **per-expert F32 `.scale` + `.input_scale` sidecars**), `MOSTLY_NVFP4`/`MOSTLY_MXFP4_MOE` ftypes; runtime loads those sidecars (`llama-model.cpp:1917-1990`) and applies them with `MUL`.
- `conversion/deepseek*.py` copy **MXFP4 experts bit-exact** ⇒ a proven in-tree template for "pass the bytes through, just repack".

### What master does NOT have

1. **No MXFP8 weight type at all.** `GGML_TYPE_F8_E4M3` is an unscaled 1-byte element type; `_prepare_fp8_e4m3_tensors` (`base.py:692`) only preserves **per-tensor**-scaled fp8 (`prod(scale.shape) != 1 ⇒ continue`). Inkling's 1×32 `ue8m0` block scales have no home. Consequently master's Inkling converter **dequantizes both expert groups to F32** (`inkling2.py:176 LazyChunkedTensor(..., np.float32)`) and requantizes.
2. Ground truth for the gap: the public `unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf` (17.63 GB) holds **40 native NVFP4 tensors for blk.0–19 + F32 `[128]` `.scale` sidecars** (so a lossless-NVFP4 repack is *possible* and somebody shipped one — out-of-tree) while **blk.20–23 experts fell back to Q8_0** (requantized, ~same bytes, wrong values, no fp8 tensor cores).
   → **Our delta vs. what exists: exactly the MXFP8 layers, plus upstreaming the NVFP4 export so one file is fully native.**

---

## 2. Decision records

### DR-1 — Add `GGML_TYPE_MXFP8 = 44` (block-scaled fp8), bump `GGML_TYPE_COUNT` to 45
`block_mxfp8 { uint8_t e /* ue8m0, value 2^(e-127) */; uint8_t qs[32]; }` = **33 B per 32 elements** (8.25 bpw), `QK_MXFP8 = 32`. Deliberate 1:1 analogue of `block_mxfp4`.

- *Why not* `F8_E4M3` payload + separate scales tensor: no fused matmul would ever see them together; doubles allocation; breaks every "row bytes = ne·type_size/block_size" assumption; loses the native MMA.
- *Why not* fold the e8m0 scale into the payload (i.e. requantize): that is exactly the "adulteration" we are eliminating.
- *Rejected*: reusing the old KV-cache `MXFP8` type from the `mxfp-*-fa` branches — those are **KV-cache** types with different layout/semantics (Hadamard-rotated, SoA) and were never weight types; do not conflate (see `02-prior-art-branches.md`).
- Byte 255 (`NaN` scale) must be rejected at load time (loudly, not silently).

### DR-2 — Two CUDA tiers, in this order: **A16-dequant mmq first, native `mxf8f6f4` W8A8 second**
- **Tier A (correct, ~fast):** implement `vec_dot_type = F16` for MXFP8 and let the existing f16 mmq tile path run: `e4m3 → f16` (`cvt.rn.f16x2.e4m3x2`) then `× 2^(e-127)`. Reuses all of mmq's existing f16 machinery, needs no activation quantizer, is trivially checkable against CPU. Decode/MoE gets `mmvq`/`dmmv` with a new dequant functor. This tier alone gets us a *runnable, lossless* 20 GB model.
- **Tier B (fast):** `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32…e4m3.e4m3.f32.ue8m0` with **fp8 activations** — the recipe the checkpoint itself specifies. This is the `mxfp4-w4a8` branch's exact contribution (it used the same instruction in its *mixed* form for A8×W4); the SFA/SFB scale-register packing is already written there (`02-prior-art-branches.md` §2, §5). Selected via a new `GGML_PREC_FP8` prec tag set by the converter per tensor.
- Do **not** plan around `tcgen05`/TMEM or `wgmma`: consumer Blackwell (sm_120) has neither — `mma.sync` with block scaling is the one (confirm in `05-hw-and-ecosystem.md`).

### DR-3 — Keep the two expert groups' *own* scales; only the NVFP4 global factor goes to a sidecar
NVFP4's `amax / outer_scale_denominator` per-expert fp32 factor cannot live in `block_nvfp4`. Use master's existing convention verbatim: `blk.N.ffn_gate_up_exps.scale` / `.input_scale` as F32 `[E]` tensors, applied at runtime by the existing `MUL`. MXFP8 needs **no** sidecar (its scales are in-block). If `06-lossless-repack-design.md` finds that factor is always a power of two we may revisit; assume it is not.

### DR-4 — Lossless means "bytes are a permutation", and we will *prove* it
Gate for Phase 2 is a standalone verifier that dequantizes the produced GGUF and compares to an independent HF reference with `max|Δ| == 0` (and asserts payload/scale byte multiset equality). Pure numpy on the real checkpoint (child E) before any C++ is trusted.

### DR-5 — One branch, upstream-shaped commits
`mxfp8-weights` holds everything; the ggml type + CUDA paths are generic (PR-able), the Inkling-specific converter export is arch-scoped. Never merge the old KV-cache MXFP branches; cherry-pick only `quantize.cu`/`mma.cuh` hunks identified as reusable in `02-prior-art-branches.md`.

### DR-6 — Memory plan on 2×16 GB *(corrected 22:40 by child E's measurement)*
Lossless weights = **18,014.3 MiB (~18.0 GB)**, not the 20.4 GB originally estimated ⇒ ~9 GB per card on 2×16 GB, leaving **≥12.4 GiB slack** for KV + compute buffers. With 8 KV heads × head_dim 128 × 24 layers and 20 of 24 layers windowed at 1024 tokens: **fp8 KV costs 296 / 552 / 1064 / 2088 MiB at 32k / 64k / 128k / 262k**. f16 KV uncapped at 262k would be 42,590 MiB — **over by 9,822 MiB**, so the sliding-window cap is load-bearing rather than an optimisation. This is why MXFP8 (weights) + fp8 KV (cache) are one project. Arithmetic: `06-lossless-repack-design.md`.

---

## 3. Phases, each with a hard test gate

> Ordering rule: nothing CUDA-optimised before a byte-level contract is proven (Phase 2's verifier) and nothing perf-work before the numbers are right (Phase 3).

### Phase 0 — Baseline outside docker *(running now)*
1. `cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON` → local master build (`logs/build-master.log`, tmux `mxfp8build`).
2. Run `blk`-native existing file: `unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf` on both GPUs (both cards are free of docker only if `llama-cpp` container is stopped — **check before starting**, the compose stack currently serves Qwen3.8-27B on 9931).
3. Record: t/s decode/prefill, `--no-cnv` prompt log, `llama-perplexity` on a small set, memory per card, and which kernels were chosen (NVFP4 mmq vs mmvq, `GGML_CUDA_FORCE_...` off).
- **Gate G0:** model produces coherent text locally; baseline numbers in `00-status.md`. This validates the (1-day-old) arch support *before* we layer a new type on top, so any later breakage is attributable.

### Phase 1 — `GGML_TYPE_MXFP8` in ggml (CPU + reference) → *testable in seconds*
Deliverable per `03-touchpoint-checklist.md` groups A+B+E: enum, `block_mxfp8`, traits (`to_float`, `from_float_ref`), CPU `vec_dot`/dequant, `gguf-py` enum + `QUANT_SIZE_MAP[MXFP8] = (32, 1+32)` + `MOSTLY_MXFP8`, `llama-arch` type-name string, `test-quantize-fns` round-trip + `test-backend-ops` `MUL_MAT`/`MUL_MAT_ID` MXFP8 cases on CPU.
- **Gate G1:** `ctest` green; round-trip test asserts **byte-identity** of payload+scale for already-MXFP8 input; `test-backend-ops` MXFP8 vs F32 reference within tolerance on CPU.

### Phase 2 — Lossless repack (python only, no C++ risk)
`conversion/inkling2.py` (+ small `base.py` helper) gains a **native export** path: MXFP8 experts → `GGML_TYPE_MXFP8` bit-exact (with the gate/up row permutation applied to payload *and* scale rows identically); NVFP4 experts → existing `_nvfp4_pack` + `.scale`/`.input_scale` sidecars; everything else untouched in dtype-faithful form (BF16 stays BF16, F32 stays F32). Ship it behind an explicit flag/ftype so the default path is unchanged.
- **Gate G2a:** `scripts/inkling_native/check_layout.py --full` exits 0 (child E's invariants, incl. cross-check vs the shipped unsloth NVFP4 bytes).
- **Gate G2b:** produced GGUF: `llama-print`/`gguf-dump` shows exactly 40 NVFP4 + 8 MXFP8 + 48 `.scale` sidecars, size within ±2 % of the predicted ~20.4 GB, and it **loads and runs** on CPU via the Phase-1 reference path (slow is fine) → first end-to-end lossless artifact.

### Phase 3 — CUDA correctness (Tier A)
MXFP8 through the existing f16-tile mmq + `mmvq`/`dmmv` decode path + `mul_mat_id`; `convert.cu`/`getrows.cu`/`ggml-cuda.cu` row-size + op-support entries; explicit "unsupported" for Metal/Vulkan/SyCL/Hexagon rather than silent fallback.
- **Gate G3:** identical logits (within fp tolerance) CPU vs CUDA on the small synthetic Inkling from T3-below, and on the real file for a fixed prompt set; `test-backend-ops --backend cuda` green for `MUL_MAT{MXFP8}` + `MUL_MAT_ID{MXFP8}` across M ∈ {1, 5, 17, 64, 256} (tile-boundary cases).

### Phase 4 — CUDA speed (Tier B: native `mxf8f6f4` W8A8)
Fused **activation MXFP8 quantizer** (from `mxfp4-w4a8` `quantize.cu`, block 32, `ue8m0`), scale-register (SFA/SFB) packing per `05-hw-and-ecosystem.md`, mmq k-tile/`sram_stride` config for 33-byte blocks, dispatch via `GGML_PREC_FP8` per-tensor tag, and a heuristic for when A8 beats A16 (large-M prefill/dense; small-M stays on the memory-bound path).
- **Gate G4:** ≥1.5× prefill t/s vs Tier A on the 4 MXFP8 layers (isolated microbench: `scripts/w4a-harness/` revival), no regression on the NVFP4 layers, KL(logits‖TierA) ≈ 0 within fp8-activation tolerance, and `test-backend-ops` green with the native path force-enabled.

### Phase 5 — Accuracy & perf validation
- **Gate G5:** perplexity + KL-divergence of the lossless file vs the *dequantized-F32* file (should be ~0 — proving the repack is faithful end-to-end through the kernels) and vs the unsloth Q4_K_XL/Q8_0 file (should be **better**, and by a quantifiable margin on the last 4 layers); t/s and GB-per-card table across ctx sizes; all numbers appended to `00-status.md` with commands and logs kept (never deleted).

### Phase 6 — Fit it on the two cards + fp8 KV
- **Gate G6:** `llama-server` with the lossless file, `-sm layer`, both cards ≤ ~14 GB, `--cache-type-k/v q8_0` (then MXFP8/fp8-e4m3 KV if the FA path supports it), 64k+ ctx, long-context retrieval smoke test, no OOM under 150 W caps. Optional stretch: `--model-draft` with unsloth's `dspark-Inkling-2-Mini-Q8_0.gguf` / the MTP head; vision via `mtmd`.

### Backlog (explicitly not now)
MXFP8 *KV* cache + fp8 flash-attention (old `mxfp-*-fa` branches) · MTP head in-graph · CPU AVX-512/NEON MXFP8 repack · Metal/Vulkan · upstreaming a PR (worth doing after G4; keep commits clean for it).

---

## 4. Risk register

| # | Risk | Impact | Mitigation / who |
|---|---|---|---|
| R1 | 33 B/block is not power-of-two → smem tile/swizzle friction in mmq | Phase 4 perf | Precedent exists: MXFP4 = 17 B/block works today; `03` §C lists exact tile code; fall back to Tier A |
| R2 | `MUL_MAT_ID` (MoE) may not support block-scaled types or the `.scale` sidecar path | G3 blocker — Inkling is *all* MoE | child C Q3/Q4 verifies; if missing, add a scale-apply step in the graph (small, well-scoped) |
| R3 | `GGML_TYPE_COUNT`-indexed arrays / python tables silently mis-handled | silent numeric corruption | child B "Risk list"; add an assertion + a type-count test |
| R4 | sm_120 `mxf8f6f4` operand/format constraints differ from the A4W8 mixed case the branch implemented | G4 unachievable, Tier B dropped | child D Q1/Q2 from PTX ISA; Tier A still ships the lossless model |
| R5 | Inkling arch support is 24 h old (bugs, missing swa/loop details) | noisy baselines | Phase 0 baseline with the *published* file; keep upstream pinned for rebase (`29b52d398`) |
| R6 | Both GPUs are normally held by the docker `llama-cpp` container | can't test locally | stop/start `llama-cpp` deliberately, coordinate with fleet (compose `up -d llama-cpp` afterwards); never pack VRAM to the edge (see services/AGENTS.md gotcha) |
| R7 | 19.5 GB checkpoint → conversion needs chunked/streaming to fit 61 GB RAM | slow/flaky Phase 2 | master already uses `LazyChunkedTensor`; keep lazy, verify peak RSS in the harness |
| R8 | NVFP4 `outer_scale_denominator` folding could silently change bytes | breaks DR-4 | child E Q-c measures whether the factor is a power of two; sidecar otherwise (DR-3) |

---

## 5. Immediate next actions (parent)

1. Watch Phase 0 build (tmux `mxfp8build`) → then G0 run.
2. Collect child outputs A–E, merge corrections into this file (esp. Q4/R2: MoE type support; Q2/R8: NVFP4 export; D-Q2: SFA layout).
3. Start Phase 1 (ggml type) as its own worker task once `03-touchpoint-checklist.md` lands — it is the critical path.
4. Add a `Makefile`/`scripts/inkling_native/make_native_gguf.sh` one-shot so Phase 2 is reproducible.
5. Keep `00-status.md` updated with every measured number and every command that produced it.
