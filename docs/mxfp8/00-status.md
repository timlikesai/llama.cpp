# Inkling MXFP8 — live status log

Update this file as work lands. Every number gets the command that produced it. Never delete an entry — append.

## 2026-11-20 (session start)

**Branch created:** `mxfp8-weights` off `master` = `29b52d398` ("model : Inkling 2 Mini (#32694)"), which equals `upstream/master` (0 commits behind — verified `git rev-list --count master..upstream/master` → `0`).

**Local (non-docker) CUDA build:** ✅ `JOB_DONE exit=0`
```
cmake -B build-cuda -S . -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
cmake --build build-cuda -j 22          # logs/configure-master.log, logs/build-master.log  (tmux: mxfp8build)
```
Binaries in `build-cuda/bin/`. CUDA 13.2 (nvcc V13.2.86), driver 615.71.09, both 5060 Ti idle at test time.

**Checkpoint recon (facts, all measured this session):**
- Raw HF repo cached: `model-00001-of-00001.safetensors` 19,526,019,168 B (1244 tensors: 994 BF16 / 154 F32 / **48 F8_E4M3** / **48 U8**) + `mtp.safetensors` 635,801,416 B (54 BF16).
- `hf_quant_config.json`: experts split **NVFP4 = layers 0–19**, **MXFP8 = layers 20–23** (ue8m0 scales, 1×32 along K; `weight_block_shape [32,32]` is inconsistent with stored shapes — master already prefers the stored shapes).
- Existing public GGUFs on disk read with `gguf-py`:
  - `ggml-org/.../Inkling-2-Mini-IQ4_NL.gguf` 17.63 GB → types {Q8_0: 186, F32: 478, IQ4_NL: 40} (everything requantized; no native formats).
  - `unsloth/.../Inkling-2-Mini-UD-Q4_K_XL.gguf` 17.63 GB → types {F32: 508, Q8_0: 154, BF16: 32, **NVFP4: 40**}; NVFP4 = `blk.0–19.ffn_gate_up_exps/ffn_down_exps.weight` (+ F32 `[128]` `.scale` sidecars); **`blk.20–23` experts are Q8_0** ← the exact layer group our MXFP8 work replaces.
- Master gap confirmed by reading code: `conversion/inkling2.py:176` dequantizes *both* expert groups to F32 `LazyChunkedTensor`; `base.py:692 _prepare_fp8_e4m3_tensors` only preserves **per-tensor** fp8 (`prod(scale.shape) != 1 ⇒ continue`). No `GGML_TYPE_MXFP8` exists (max id 43 `F8_E4M3`, `GGML_TYPE_COUNT = 44`).

**Docs created:** `docs/mxfp8/01-roadmap.md` (plan + decision records DR-1…DR-6 + risk register), `docs/mxfp8/briefs/{A..H}-*.md` (delegation briefs).

**Children launched (model `qwen/qwen3.8-flash-next`):**
| id-ish | agent | brief | deliverable |
|---|---|---|---|
| task-1 | scout | A prior art | `02-prior-art-branches.md` |
| task-2 | scout | B touchpoints | `03-touchpoint-checklist.md` |
| task-3 | scout | C inkling/master | `04-inkling2-master.md` |
| task-4 | researcher | D hw + ecosystem | `05-hw-and-ecosystem.md` |
| task-5 | worker | E layout harness | `scripts/inkling_native/{layout,check_layout}.py`, `06-lossless-repack-design.md` |
| + | scout | F vram placement | `07-vram-placement.md` |
| + | reviewer | G plan review | `08-plan-review.md` |
| + | scout | H other engines | `09-other-engines-mxfp8.md` |

## 2026-11-20 ~19:47 — host reboot (Tim brought inference capacity online)

- `/tmp` was cleared → **all subagent runs lost**. Briefs A–E were killed by the 30-minute subagent timeout and F/G/H by the reboot, **all before writing a single file**: 8 research sessions, zero artifacts. Fix applied to the briefs: `briefs/00-HOW-TO-WORK.md` now forces skeleton-within-6-tool-calls + incremental rewrite, `timeoutMs=2400000`, and explicit per-child model pin (`qwen/qwen3.8-flash-next`).
- Relaunched all 8 children ~20:30 (A scout, B scout, C scout, D researcher, E worker, F scout, G reviewer, H scout) — ids in the session log.
- `build-cuda/` **survived** (lives under `/home`), branch `mxfp8-weights` intact. Verified post-reboot:
  - `./build-cuda/bin/llama-cli --list-devices` → `CUDA0/CUDA1: NVIDIA GeForce RTX 5060 Ti (15941 MiB, 15792 MiB free)`
  - `strings build-cuda/bin/libllama.so | grep -i inkling` → `llama_model_inkling2`, `llama_model_inkling2_loop` symbols present ⇒ arch compiled in.
  - `strings build-cuda/bin/libggml-base.so` type names → `q8_0 iq4_nl mxfp4 nvfp4 f8_e4m3` (**no `mxfp8`** — the gap, as planned).
- `llama-cpp` resumed crash-looping on boot: `RestartCount=48` by 20:26, same OOM.

## 2026-11-21 14:40 — **e046294a7**: alias-aware device placement. prefill 507 → 901 t/s, decode 22 → 40 t/s

Lane AB implemented the fix for the finding above and I verified it independently before committing (12 commits on the branch, tree clean).

| | before | after (auto, FA on) | after (`-fa off`, matched comparison) |
|---|---|---|---|
| prefill @2817 | 507.61 t/s | **901.28** (`logs/ac-verify-perf.log`) | 847.40 |
| decode | 22.18 t/s | **40.36** | 39.59 |
| `cudaMemcpyPeerAsync` | 5,460 | **1,106** | — |
| peer traffic | 57,353 MiB | **8,867 MiB (−85%)** | — |
| `cudaStreamSynchronize` | 6,991 / 4.457 s | **972 / 0.083 s** | — |
| Flash Attention | force-disabled | **enabled** | off |
| weight split | 13,152 / 4,048 MiB | **7,855 / 9,344 MiB** | — |

**The change is 24 lines and principled**: `layer_phys(int il)` becomes a virtual on `llama_model` returning `il`, `load_tensors` computes `n_layer_unit = max(layer_phys(il))+1` and runs the split arithmetic in *unique-layer* space, and each slot's device comes from `get_layer_buft_list(layer_phys(il))` — so an aliased slot inherits the device of the layer whose tensors it reads, and slots that add no weights stop consuming device budget. Inkling's existing `layer_phys` just gains `override`.

**Why this is safe rather than Inkling-specific hackery:** for any model without aliasing `layer_phys` is the identity, so `n_layer_unit == n_layer_all` and the arithmetic is bit-for-bit unchanged. Verified directly: the dense `Qwen3.5-0.8B-Q8_0` control reproduces its pre-change per-chunk perplexities **exactly** (`7.4483, 11.9865, 14.1357, …` — identical to `logs/ab2-q80-gpu.log` measured before the patch). Plus Inkling unchanged on both backends (`68 B: 64 C: 72 D: 70 E: 66 F: 69 G:`, `The capital of France is Paris.`), ops 306/306 CPU and 237/237 CUDA0.

**Two things I want on the record.** (1) The residual peer traffic is activation handoff, not weight fetch — the lane proved it by showing that at `-b 256` the total is *identical* while the dominant transfer shrinks from 4,718,592 B (the `attn_k`/`attn_v` BF16 weight fingerprint) to 2,359,296 B. That is how you demonstrate a mechanism instead of asserting one. (2) The lane rejected the tempting shortcut in writing: non-blocking copies would remove 5,400 host blocks, leave 57 GiB on the critical path, and break copy liveness. My original hypothesis ranking was wrong about the majority of the syncs (`MUL_MAT_ID` was ~21%, not the cause) — corrected in the doc rather than quietly.

**Knock-on effects for the plan:** #22 (FA using more memory than f16 KV) is now *measurable* — it was blocked precisely because `resolve_fused_ops` force-disabled FA on the alias-straddling split, which this commit fixes. Same for #19 Phase 6 fp8 KV at long context. And the perf question resets: at 901/40 t/s the earlier “5–10% of peak” framing is obsolete, so the new hot path needs re-profiling before more optimization is attempted.



### Prefill: it is looped-layer cross-device traffic, not kernels

Lane Z's report (`docs/mxfp8/26-sync-attribution.md`, 15 KB; `scripts/inkling_native/z_sync_attrib.py`; 13 `logs/z-*` artifacts) attributes the sync storm precisely, and **my hypothesis ranking was wrong about the majority**:

- **~77% of the 6,991 syncs (5,411 / 3.12 s) sit around CUDA0<->CUDA1 peer copies** — the pattern `cudaMemcpyPeerAsync, cudaEventRecord, cudaStreamWaitEvent | SYNC | cudaMemcpyPeerAsync` occurs 2,700 times (1,997 ms); 5,460 peer copies per run.
- **Cause:** `-sm layer` puts layer 24+ on CUDA1, but `src/models/inkling2-loop.cpp:68` aliases the replayed layers onto the **physical tensors of layers 0..23 on CUDA0**. So every replay reaches across the PCIe link for weights. Size attribution matches weight/control tensors, not per-token activations: 4,718,592 B (`attn_k/attn_v` BF16 [2304,1024]) x1,785 each way, `attn_res_query` 27,648 B x336, `attn_q_clip` 12,288 B x336. **~33 GB of peer traffic over a 5.5 s prefill = 5.4-10.7 GB/s, with the host inside a sync 90-100 ms of every 100 ms window.**
- **`MUL_MAT_ID` is only ~21%** (its fallback emits 2 copies + 2 syncs per node, bounding it at 746 nodes). It still matters secondarily because one flagged node sets `use_cuda_graph = false` for the whole cgraph (`ggml-cuda.cu:3328-3338`).
- Refuted: hypotheses 4 and 5 (`cudaMemGetInfo` x250 and `cudaGraphInstantiate` x429 are load/teardown, not per-layer polling or per-ubatch rebuilds; the 382/700/926 MB memcpies are model-load weight uploads in a window with zero kernel activity).
- **Isolating measurement:** `-sm none` removes the ping-pong and gives **prefill 506 -> 642 t/s (+27%)**, but decode drops 20.6 -> 11.5 t/s because CUDA1's weight capacity is lost and the fitter starts overriding tensors to CPU (`W llama_model_loader: tensor overrides to CPU are used with mmap enabled`). So it is proof of cause, **not a shippable config**. `-sm row` cannot load at all: `device CUDA0 does not support split buffers`.
- Z also rejected the tempting-but-wrong fix in its own fence: making the cross-device copies non-blocking would delete ~5,400 host blocks but leave 33 GB on the critical path and break the liveness guarantee the scheduler relies on (a copy source may be handed to the next node). Good judgment — the problem is *traffic*, not *waiting*.

**Fix direction (mine to land):** device assignment for a replayed layer must follow the device of its **physical** tensors — after `post_arch_tensors` aliases `layers[il] = layers[layer_phys(il)]`, the fitter / `resolve_fused_ops` / `llama-context` must not place `il >= 24` nodes on a different card. Filed as a task; upstream-report skeleton is in doc 26.

### MXFP8: clean-room validated on a dense model

`AA` produced `~/models/timlikesai/Qwen3.5-0.8B-MXFP8-PY.gguf` (809,456,416 B, **195 MXFP8 + 140 F32** — the same tensor selection as the Q8_0 build) and the parent measured it on `logs/corpora/wiki.test.slice64k.raw`, `-c 1024 -b 1024 -ub 512 -s 42` (`logs/ab2-mxfp8-{gpu,cpu}.log`, `logs/ab2-q80-{gpu,cpu}.log`):

| candidate | chunk 2 | chunk 9 | chunk 12 | CPU vs CUDA |
|---|---|---|---|---|
| MXFP8 (8.25 bpw) | 12.3981 | 19.6536 | 17.2813 | **identical to every printed digit, all 15 chunks** |
| Q8_0 (8.5 bpw) | 11.9865 | 18.9678 | 16.6361 | identical |

**Verdicts:** (1) the MXFP8 type and its CPU+CUDA paths are **correct and bit-consistent across backends** on a model with no architectural confounds; (2) this independently confirms task #21 — the Inkling CPU nan is *not* a quantization defect, since the same CPU path here is perfectly well-behaved; (3) MXFP8 costs roughly **3-4% PPL versus q8_0 at slightly fewer bits-per-weight**, which is what power-of-two-only ue8m0 scales should cost — a sensible result to publish, not a bug.

Also learned the hard way twice more: the perplexity tool needs **>= 2x ctx tokens** (a 1,910-token corpus silently errors at `-c 1024`), and `AA`'s first attempt wrote an 11 MB file that wasn't even a valid GGUF — size sanity (450-500 MB expected for 0.8B at 8.25 bpw) is the one-second check that catches it, same lesson as the 48 GB Inkling build.

## 2026-11-21 12:45 — V's report needed a control it didn't run: the CPU nan is model-wide, not ours

Lane V (`docs/mxfp8/21-dequant-nan.md`) did good work — an offline scan of all 704 tensors / 27.81 GB of `Inkling-DEQ-Q80.gguf` found **zero** non-finite block scales or values (largest expert value bound 0.85), and a value comparison against the healthy public build showed the 8 ex-MXFP8 tensors are **bit-identical** while the 40 ex-NVFP4 ones differ by exactly 4-bit rounding (rms 0.002 vs |x| 0.1). That kills my dtype-policy/overflow hypothesis and lane U's “prime suspect: MXFP8 → Q8_0 scale overflow”: **the dequantized data is fine.**

It then reported that `Inkling-2-Mini-NATIVE.gguf` is nan on the CPU path and concluded the native expert types have an independent CPU defect. **I ran the control it omitted, and the conclusion does not survive** (`logs/E1.log`, `E2.log`, `E3.log`, corpus `logs/corpora/deq-probe.raw`, `-c 1024 -b 1024 -ub 512 -s 42`):

| file | CPU (`-ngl 0`) | GPU |
|---|---|---|
| `IQ4_NL` — public reference, **no native types at all** | **nan** `[1]nan,[2]nan` | finite |
| `NATIVE` (ours) | **nan** | finite `[1]3.5669,[2]5.3751` |
| `DEQ-Q80` | **nan** | nan |
| `NATIVE` at `-b 256 -ub 256` (different CPU MoE kernel path) | **nan** | — |

**So CPU batched evaluation of Inkling-2-Mini produces non-finite logits for every quantization, including the upstream lossy build.** It is an Inkling CPU-path bug, not a consequence of the MXFP8/NVFP8 work — and lane U's “NATIVE is numerically healthy” PPL remains a **CUDA-scoped** claim, which I have now corrected in `20-*` §7's caveat too. Greedy single-token decode on CPU is fine for both files, so the defect is in batched eval, not the type decode.

Leads recorded in task #21: `ggml` has **no `GGML_OP_SITU` anywhere** (grep empty in both backends) — Inkling's clamped activation is composed at `src/llama-graph.cpp:2666-2679` as `beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)` from generic `TANH`/`SIGMOID`/`MUL`, so CPU-vs-CUDA differences in those primitives (or `0*inf` from an upstream overflow) are the first place to look; also the loop/SWA/controller (`attn_res`) math, which has the same CPU/GPU asymmetry profile.

Also filed as **#22** on Tim's report that enabling Flash Attention can use **more** memory than an f16 KV cache, with the looped layers as prime suspect (48 applications from 24 weight layers): the matrix to measure is {FA off,on} × {f16,fp8 KV} × {2k,8k,32k} against theoretical size, checking specifically whether KV is allocated per applied layer instead of per unique layer, whether `sliding_window 1024` is honoured in the FA path, and whether the shared loop slots from `690b87d88` still get separate KV state in the FA allocator. This ties directly to the already-observed fact that the 2-card split disables FA outright.

**Lanes now:** `W` — profile why prefill sits at ~5–10% of peak (steered to include #22). `X` — MTP draft packaging + mapping table, no builds.

## 2026-11-21 12:05 — Phase 5 done: the lossless file is validated, and my own control artifact was broken

`docs/mxfp8/20-ppl-kl-validation.md` (313 lines) + reusable drivers `scripts/inkling_native/ppl_kl_u{,2}.sh`, WikiText-2 via the `Salesforce/wikitext` HF parquet (the older zip mirrors in my notes are dead), 16,705 tokens, 8 chunks.

| file | mean PPL | KLD vs NATIVE base | same top token |
|---|---|---|---|
| **NATIVE (ours, bit-exact)** | **5.5757 ± 0.1442** | 0 (max 7e-5 — the u16 logit downcast) | 100.000% |
| IQ4_NL (public, lossy) | 5.5636 ± 0.1437 | 0.012061 ± 0.000318 | 95.687% |
| DEQ-Q80 (my dequantized control) | **nan** | nan | 0.012% |
| SCALE1 (broken-`.scale` control) | 5,260,013.9 ± 318,084.7 | 13.787 ± 0.0499 | 0.672% |

**What is now actually demonstrated:** our bit-exact repack is numerically healthy on real text and statistically indistinguishable from the public lossy build on PPL (0.22% apart against ±2.6% bars), while the KL axis does resolve loss (IQ4_NL at 173× the instrument's noise floor), and a wrong scale convention is catastrophic and unmissable (5.26e6). The lane's own summary: *“no plausible silent-convention-bug survives this test.”* That is the standard I wanted before claiming this thing works.

**Correction I owe the record: `Inkling-DEQ-Q80.gguf` emits non-finite logits** — nan on all 8 chunks, reproduced on two independent code paths, and nan as a candidate against a known-good base. That is the file I used as the **positive control** in the elimination chain that localized the bug to native bytes (“our converter's values are right because the q8_0 build answers `68`”). Greedy generation giving the right answer is evidently a weaker guarantee than I treated it as. The final diagnosis does *not* depend on it — upstream `690b87d88` plus this PPL/KL table stand on their own — but any conclusion drawn from that file is now suspect and is being chased in its own lane (brief `V`). Notably the direct `NATIVE == dequantized` KL was therefore **never measured**; byte-exactness still rests on the offline tensor comparison against the `.safetensors` checkpoint, and the lane said so plainly instead of claiming corroboration it didn't get.

**Free Phase 6 finding, from idle observation:** at `-c 2048` auto-fit already puts layer 24 on CUDA1, which **disables Flash Attention** (`resolve_fused_ops: layer 24 is assigned to device CUDA1 but Flash Attention is assigned to device CUDA0`). Without FA the KV cache is plain f16 — so the fp8-KV savings in child E's table are *not being realised at all* on the current 2-card split, and the split/`-sm` choice rather than context size is what gates them. Also: `llama-perplexity` has a footgun — `-c 512` silently becomes a 4-slot 2048-token context (`perplexity.cpp:2012` computes `n_parallel = n_batch / n_ctx`, `:2014` multiplies back), so **`-b` must equal `-c`** or the run is wrong. My 14 KB long-prompt file was also mis-sized as 2.8 KB in my own notes.

## 2026-11-21 11:30 — Phase 4 CUDA tensor-core path landed: prefill 484.50 → 506.95 t/s

`c39ab0a00 ggml-cuda : add an MMQ tile path for MXFP8` + `6c1424737 tests : account for fp8 activations in the MXFP8 MMQ cases` (10 commits on `mxfp8-weights`, tree clean).

**Tim's steer was the win here.** Two instructions — “compare, we may not need to rewrite this” and “the codebase may already have tile loading we can extend” — turned a port into a five-file widening. The stale branches carried a whole parallel implementation (`SRAM_LAYOUT_FP8`, `block_fp8_mmq`, a `CASE_MXFP` macro whose `J` sits in argument position **1** where master’s `CASE` has it in position **5**, plus a hand-written loader); none of it was needed. Master already serves two fp types from one `LAYOUT_FP4` that is `static_assert`-stride-identical to `Q8_1`, and MXFP8 slots into it because `block_mxfp8` and `block_mxfp4` share the scale encoding and values-per-block. Full reasoning and the per-site table: `docs/mxfp8/18-mma-mxfp8.md`; the branch facts worth keeping (instruction encoding, the `sa >> 8*frag` scale-packing rule, `tidx_A`, the arch guard + fallback policy) are in `docs/mxfp8/02-prior-art-branches.md` with unfiltered diffs in `logs/priorart-*.diff` and code excerpts in `docs/mxfp8/ref/`.

**Numbers:** `test-backend-ops -p mxfp8` **237/237 on CUDA0 and CUDA1**, CPU still **306/306**, `test-quantize-fns` exit 0, and the lossless file still answers `68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` on both backends. **Prefill +4.6%** (484.50 → 506.95 t/s @2817 tokens); **decode unchanged** at ~22 t/s because single-token MoE still takes the vector path — decode acceleration is a separate item, not delivered by this commit.

**Two honesty notes recorded rather than smoothed over.** (1) `GGML_CUDA_FORCE_MMQ` is *not read by the CUDA backend on current master* (only MUSA/SYCL reference it), so the forced-vs-auto A/B I first ran proved nothing; the route is instead evidenced by behavior — the `type_b=f32` MUL_MAT cases moved from under 5e-4 to 5.5e-4..7.4e-4, which is the new e4m3 activation quantizer running and is impossible on the `mmvf` route. (2) The lane edited `quantize.cu/.cuh` **outside the fence I gave it**, and was right to: master has no e4m3+ue8m0 activation quantizer and the fp4 one can’t be reused. My fence was wrong, not the agent.

**Known sharp edge, deliberate:** MXFP8 pins `prec_src1` to the native format and ignores a requested `GGML_PREC_Q8`, so there is no A16 MXFP8 on Blackwell and the generator emits no `Q8_1` instance. Flagged in the commit message for revisiting. The 5e-3 test tolerance follows master’s own `F8_E4M3` + `NATIVE_FP8` precedent (measured 7.4e-4, so ~7x headroom — not tuned to the observation).

## 2026-11-21 02:40 — WORKING. Upstream `690b87d88` was the bug; lossless model runs correctly on CUDA

Tim pointed at fresh upstream commits. `git fetch upstream master` (29b52d398..e50c254a2, 21 commits) contained **`690b87d88 model : inkling2_loop share the loop slots after the scale pass (#32704)`**, whose message is precisely our symptom:

> “The loop slots copied the physical layers at the end of load_arch_tensors, **before the generic pass attaches the optional .scale tensors, so the later passes through the core ran NVFP4 experts without their per-expert scales.** Add an optional post_arch_tensors hook that runs after the scale pass and share the slots there.”

Cherry-picked clean as `489106434` (4 files, +7 lines; touched nothing we own). It explains every measurement in the elimination table: the `q8_0`/IQ4_NL controls were right because they carry **no** sidecars; the Qwen3.8 NVFP4 control was right because it is **not a loop model** so nothing aliases layers before the scale pass; MXFP8 was irrelevant because it needs **no** sidecar; and CPU vs CUDA differed because the first pass through the core was correctly scaled while the replayed passes were not — a graph-ordering fault, which is why no amount of byte-level verification could ever have found it.

**Gate results, all green:**

| check | result |
|---|---|
| NATIVE on CUDA, `17 times 4` | `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` — **identical to the reference file's output** (`logs/native-fixed-gpu.log`) |
| NATIVE on CPU (`-ngl 0`) | same string ⇒ CPU/CUDA agreement restored (`logs/native-fixed-cpu.log`) |
| `The capital of France is` | ` Paris.` |
| `test-backend-ops -b CPU -p mxfp8` | 306/306 |
| `test-backend-ops -b CUDA0 -p mxfp8` | 237/237 |
| prefill 2817 tok | NATIVE **484.50 t/s** vs IQ4_NL 492.32 t/s (−1.6%) |
| decode 2817 tok | NATIVE **22.06 t/s** vs IQ4_NL 23.62 t/s (−6.6%) |

**So the project goal is met on the correctness axis**: `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` (18,974,614,688 B) is a bit-exact repackage of the original NVFP4+MXFP8 weights, it loads, it generates the same text as the reference, and it runs prefill and decode on both cards at within a few percent of a lossy quantization. What remains is speed and scope, not correctness: Phase 4 (`mxf8f6f4` W8A8 + the `mmq` prefill path, since MXFP8 prefill currently rides the `mmvf` dequant route), Phase 5 (PPL/KL), Phase 6 (tuned 2-card split + fp8 KV at long context), and MTP.

**Process lesson, recorded so it costs us less next time:** when a symptom is *backend-dependent*, suspect graph/load ordering and aliasing before byte conventions. I spent two child-lanes and a long stretch of my own effort on repack conventions for a fault that lived in `load_arch_tensors` ordering — and my own verifiers could not have caught it, because they validate bytes, not the order in which the runtime attaches scales to aliased layers. Checking for upstream movement earlier would also have been cheaper than the elimination campaign.

## 2026-11-21 02:20 — ROOT CAUSE NARROWED TO ONE THING: native byte convention, not values

**Positive control that settles it:** converting the same checkpoint with our tree but **without** `--native-experts` at `--outtype q8_0` (27.8 GB, `logs/deq-q80-convert.log`) and running it on CUDA produces the **correct** answer: `Q: What is 17 times 4? A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` (`logs/deq-q80-run.log`). Same converter, same values — only the native expert bytes differ, and those are wrong on both backends.

**Row order also cleared, quantitatively.** Decoding `blk.0.ffn_gate_up_exps.weight` expert 0 from the NATIVE file (with its `.scale` applied) against the *working* IQ4_NL file's same tensor:

| row mapping | correlation |
|---|---|
| **identity** | **+0.9974** |
| un-interleave `2*(r%1152)+r//1152` | +0.0011 |
| de-interleave | +0.0010 |
| swap halves | −0.0066 |

+0.9974 (not 1.0) is exactly the residual you expect from IQ4_NL's own loss — our layout is right and every alternative is noise.

**So, by elimination:** converter math ✅, converter row order ✅, MXFP8 ✅ (device matrix), NVFP4-in-general ✅ (Qwen3.8 NVFP4 control), `.scale` consumed by both backends ✅ (patching all 40 to 1.0 corrupts both). The fault is **the NVFP4 block byte layout our repack writes vs what ggml's C decoder reads** — nibble pairing inside the 32 payload bytes, which 16-element group each of `d[0..3]` covers, and/or signed-e4m3 vs ue4m3.

**Epistemic trap I fell into and must not again:** `verify_nvfp4_gguf.py`/`layout.py` were written from the same reading of `conversion/base.py:_nvfp4_pack` as the converter itself, and they cross-check *each other*; their agreement — and the byte-match against unsloth's unvalidated public file — proved nothing about the runtime. `0.000e+00` relative error was real but measured against a shared assumption. Ground truth has to come from the C decoder. Brief `O` now targets exactly that.

Also note `N`'s earlier `corr = 0.0000` for `blk.0 gate_up` was an artifact of its own decode path, not a property of the files — the same comparison done correctly is +0.9974. Two of my conclusions this session came from harness bugs of my own making (byte-vs-element `ReaderTensor.data` views for BF16, and a 3-D byte-shape reshape); both are recorded in `14-native-numeric-bug.md`/`15-*` so nobody chases them again.

## 2026-11-21 01:30 — night close-out: state, and the one experiment to run next

**Kept:** `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` (18,974,614,688 B) — the lossless artifact, byte-verified. **Deleted:** the two oversized intermediate builds (48.3 GB MXFP8-only, 48.3 GB ablation) on Tim's standing instruction; all their logs are kept. Working tree clean at 7 commits; the temporary `# ABLATION` edit to `conversion/inkling2.py` was reverted and verified reverted (`git diff` empty).

**What is proven, to a standard I'm comfortable with:**
- The MXFP8 type, its CPU path and its CUDA path are correct in isolation (`test-quantize-fns` incl. byte-exact fixed point; 306/306 CPU ops; 237/237 CUDA0 ops, 226 matmul-family).
- The **file** is correct: experts decode to exactly the checkpoint's values (worst relative error `0.000e+00`, both gate and up halves), non-experts are bit-exact BF16 (max abs diff 0), `.scale` sidecar values are right, and the graph applies them once, before the gate/up split.
- Inkling-2 itself runs on this box: G0 baseline 589.72 t/s prefill / 22 t/s decode (long ctx) on 2 cards.

**What is NOT working, precisely:** the lossless file generates wrong text on both backends, and CPU and CUDA disagree with each other on the same bytes. Everything I could verify about the file and the generic matmul paths says they are right, so the fault is in **how a native NVFP4 expert tensor is executed in this model's MoE**, most plausibly the interaction of the per-expert `.scale` multiply with Inkling's **clamped SituGLU** (`situ_beta 4.0`, `situ_linear_beta 25.0`): scaling must be visible to the *pre-activation* values, and the CUDA-only scale-node fusion (`ggml-cuda.cu:2432/2453/5149`) plus `GGML_PREC_BF16` for NVFP4 (`llama-graph.cpp:1900-1902`) are the two places where that ordering could differ per backend. Note this is upstream NVFP4 execution code — our MXFP8 work is not implicated by any measurement so far.

**Ablation attempted and abandoned:** I converted a MXFP8-native / NVFP4-dequantized variant (48.3 GB) to isolate the group, but neither `-ngl 0` nor auto-fit can run it — auto-fit asks for a 24.5 GB compute buffer on CUDA0 and `-ngl <exact>` disables fitting in both directions (`logs/ablate-cpu-run.log`, `logs/ablate-gpu-run.log`). `-dev CPU` is not a valid device token in this build.

**Next experiment (cheapest decisive one, do this first):** the *opposite* ablation — **NVFP4 native + MXFP8 dequantized** (`native = self.native_experts and nvfp4`). It lands near ~23 GB so it actually fits on the two cards, and if it produces wrong text while the MXFP8 side is exact, native-NVFP4 execution is confirmed as the sole fault and the search collapses to the `.scale` × SituGLU ordering. Then: build a synthetic 2-tensor MoE (n_ff=1152, E=128, top-8, SituGLU + per-expert scale) and compare CPU vs CUDA vs a python reference for (a) NVFP4 weights with `.scale`, (b) BF16 weights with the same `.scale` — if (a) diverges and (b) does not, that is a reproducible upstream bug worth a PR.

**Queue after that:** Phase 4 (`mxf8f6f4` W8A8 for MXFP8 — the prior-art index for it has now timed out 3× and must be done inline, not delegated), Phase 5 (PPL/KL vs the dequantized reference), Phase 6 (2-card split + fp8 KV; note the compose fix for the crash loop is one line: drop the pinned `-ngl`), and MTP (re-scoped: needs real `dflash`/`dspark` arch work, not a converter hook).

## 2026-11-21 01:20 — Phase 2 + CUDA Tier A landed; the numeric bug is NOT in our file

**Lossless artifact produced and byte-proven.** `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` = 18,974,614,688 B (17.66 GiB), 744 tensors: NVFP4 40 (10.68 GiB) + MXFP8 8 (3.92 GiB) + BF16 178 (2.93 GiB) + F32 518 (0.14 GiB, incl. the 40 `.scale`), arch `inkling2_loop`, file_type 39. Regenerated in ~50 s with `--native-experts` (`logs/phase2b-convert.log`).

**File is numerically correct on BOTH sides:**
- experts: `verify_mxfp8_gguf.py` 320 rows byte-identical (`logs/g2b-verify-mxfp8.log`); `verify_nvfp4_gguf.py` byte-identical to the public native file, `.scale == amax/2688` (`logs/g2b-verify-nvfp4.log`).
- child M's **three-way value check**: decoding the file exactly as ggml's C `to_float` does and applying `.scale` as the graph does gives **worst relative error 0.000e+00** over 8 expert tensors × 3 experts × gate *and* up halves ⇒ packing, nibble order, scale placement, gate/up pairing, sidecar values and the ue4m3/e4m3/ue8m0 conventions are all clean. Suspects A/B/D refuted.
- non-expert tensors: `blk.0.attn_q`, `blk.0.attn_output`, `token_embd`, `output` reinterpreted from BF16 → **max abs difference exactly 0** vs the checkpoint (an initial `DIFFER` flag was my own byte-vs-element comparison bug: `ReaderTensor.data` for BF16 is a *byte* view, so element count is 2×).
- graph plumbing: `llama-graph.cpp:1904-1911` applies `w_s` correctly (reshape → `get_rows(ids)` → `mul`, broadcast over `n_ff`), once, before the gate/up split — no double-multiply (`:2588` is only a `cb()` callback).

**CUDA Tier A landed and green.** `test-backend-ops -b CUDA0 -p mxfp8` → **237/237** (226 of them `MUL_MAT`/`MUL_MAT_ID`/`MUL_MAT_FUSION`), CPU still **306/306**, route = `mmvf` + on-the-fly dequant (mmvq deliberately untouched: its dot is `dp4a` over int8 pairs and e4m3 has no integer form). `SET_ROWS`/`GET_ROWS_BACK` remain unsupported, which is upstream behaviour for *every* quantized type (`GET_ROWS_BACK` reports unsupported on CPU too).

**Recovery note:** `L` timed out leaving the tree **non-compiling** — `GGML_ASSERT(stride_row % 2 == 0 || std::is_same_v<T, block_mxfp8>)` fails because the preprocessor reads the comma inside `std::is_same_v<...>` as a macro-argument separator; parenthesised. Also: my earlier probe of `L`'s work used a *stale* binary and reported a meaningless FAIL — rebuild before judging a child's work.

**Open defect (now precisely framed).** The lossless file still generates wrong text, **and CPU vs CUDA produce *different* wrong text** for the same file/seed/prompt (`logs/g2b-cpu-run.log` vs `logs/g3-native-gpu.log`) while the lossy IQ4_NL file answers correctly (`logs/g2b-cpu-run-iq4nl.log`). Since file + graph are proven correct and both backends pass synthetic MXFP8 matmul tests, the divergence must be in how a *native NVFP4* tensor is executed per-backend. Live lead: `llama-graph.cpp:1900` forces `GGML_PREC_BF16` accumulation for NVFP4, and `ggml-cuda.cu:2432/2453/5149` **fuse** the NVFP4/F8_E4M3 scale node into the matmul on CUDA only — a fusion with no CPU equivalent is exactly the shape of this symptom. Note this is upstream NVFP4 execution code, not our MXFP8 code, and the public native file has never been numerically validated.

## 2026-11-21 00:45 — Gate **G0 CLOSED**: Inkling-2 runs locally on both cards, baseline captured

`llama-completion` (our `build-cuda`, master + MXFP8 work) with `/home/tim/models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf` (17.63 GB, arch `inkling2_loop`):

| run | ctx | prefill | decode | load | log |
|---|---|---|---|---|---|
| short prompt | 20 tok / 95 gen | 168.71 t/s | **32.00 t/s** (31.25 ms/tok) | 1482 ms | `logs/g0-baseline-run.log` |
| long prompt | 2817 tok / 31 gen | **589.72 t/s** (1.70 ms/tok) | **21.94 t/s** (45.59 ms/tok) | 1412 ms | `logs/g0-baseline-prefill.log` |

Output is coherent and on-topic (start of capture: *“Block-scaled FP8 weights exploit the hardware’s native support for FP8 arithmetic and memory bandwidth”*), CUDA graphs reused 94× with 0 reallocs, and VRAM returns to 13 MiB/card after exit (no leak). **This validates master’s 1-day-old arch support as the regression reference for every later phase.**

**Three operational findings, each cost me a failed attempt — recorded so they are not re-learned:**
1. **An explicit `-ngl <number>` disables memory fitting.** `common_fit_params` bails with `n_gpu_layers already set by user to 99, abort` and then the loader tries the whole 15861 MiB on device 0 → OOM. This is the **root cause of the docker container’s 48-restart crash loop**, not a config typo. Omit `-ngl` and the fitter places the model across both cards.
2. `-no-cnv` is no longer a `llama-cli` flag — master scopes `-cnv/-no-cnv` to `LLAMA_EXAMPLE_COMPLETION` only (`common/arg.cpp:1893`); `llama-cli` now offers `--single-turn`. Use `llama-completion` for raw prompting.
3. `llama-bench` **never** runs `common_fit_params` and defaults `--tensor-split` to `0`, so it OOMs on any model larger than one card regardless of `-ngl 99`/`-ngl -1`/`-sm layer -ts 1,1` (tried all three: 15.86 GiB, then 16.33 GiB, then 17.12 GiB requested on device 0). For 2-card Inkling numbers, measure with `llama-completion` (auto-fit) instead of `llama-bench`.

## MTP: unsloth’s draft is NOT a llama.cpp artifact (scout K, plan-changing)

`dspark-Inkling-2-Mini-Q8_0.gguf` (arch `dflash`, 54 tensors) **cannot be loaded by master’s `dflash`/`dspark`** — its Inkling-specific names (`dflare*`, `attn_k_ctx*`, `dspark_fuse`, `dspark_hidden_out`, `joint_proj`) have zero hits anywhere in `src/ ggml/ common/ tools/ conversion/ gguf-py/ tests/`, `markov_w2.weight` is required but absent, and `joint_proj.weight` is `{5120,768}` where `dspark.cpp:91` expects `{2816,768}`. So it is another stack’s convention. **Value: a decode key for `mtp.safetensors`, not an output target.** Full analysis: `~/.pi/agent/sessions/--home-tim-code-services--/subagent-artifacts/outputs/ac1b435c-.../context.md`. MTP therefore needs real arch work in `dflash.cpp`/`dspark.cpp` (or a name/shape mapping layer), not just a converter hook — I have re-scoped task #16 accordingly and it is **not** on the critical path to a lossless running model.

## 2026-11-21 00:20 — GPU access granted; `llama-cpp` container REMOVED

Tim: "you can stop the llama-cpp docker service by running `docker compose down llama-cpp`" — ran it verbatim, `EXIT=0`.

- Effect: the container was **removed**, not stopped (absent from `docker ps -a`). Restore path is a recreate: `docker compose up -d llama-cpp`.
- `spare-terminal-2`, `ts-spare-terminal-2`, `ts-spare-cpp-2` all still `Up (healthy)`; Compose attempted to remove the shared `services_tailscale-sidecar` network and Docker refused ("Resource is still in use"), so nothing else was disturbed.
- Caveat for restore: `ts-spare-cpp-2` uses `network_mode: "service:llama-cpp"`, and its namespace provider was removed underneath it — expect that a plain `up -d llama-cpp` may need the sidecar recreated too (the `/proc/.../ns/net` failure mode documented in `services/AGENTS.md`).
- **The crash loop will come back** on the next reboot (`docker-compose-stack.service` runs `compose up -d`) or on a battery→line transition (`ups-power-mgr.service` runs `compose up -d llama-cpp`) until the `Inkling-2-Mini-IQ4_NL.gguf` placement is fixed (17.63 GB fully offloaded to one 16.31 GB card). Root cause + the fix arithmetic are now available: a lossless ~18 GB file needs an explicit 2-card split, and the measured fp8 KV costs (296/552/1064/2088 MiB at 32k/64k/128k/262k) leave ≥12.4 GiB of slack on 2×16 GB.
- **Both GPUs free** (13 MiB / 16311 MiB) ⇒ G0 baseline and all GPU gates are unblocked. Queued behind the two running children so we do not contend for disk/CPU with a full-model conversion.

## The two public GGUFs are from **different naming generations** (affects G0 and our target)

Read the metadata with `gguf-py` (this session):

| file | `general.architecture` | `block_count` | loop keys | expert tensor naming | ftype |
|---|---|---|---|---|---|
| `ggml-org/…-IQ4_NL.gguf` | **`inkling2_loop`** | 24 (physical) | `inkling2_loop.loop.{count,start,end,varying}` | `loop_ffn_ctrl_*`, `loop_k_ctrl_*` | 25 = IQ4_NL |
| `unsloth/…-UD-Q4_K_XL.gguf` | **`inkling2`** | 48 (unrolled) | `inkling2.num_loops`, `inkling2.loop.core_start/core_end` | `loop_attn_down/forget/gain`, `loop_k_*` | 7 = Q8_0 |

Both have 24 `blk.N.` tensor groups (N=0..23); the loop is expressed differently — the unsloth file is from the **pre-rename** scheme that master's #32694 replaced ("Adopt naming scheme from #32588", "Separate model_type", "Rename to varying"). Consequences:
- Our lossless file comes from master's converter ⇒ arch `inkling2_loop`, `block_count = 24`, current key names. **The ggml-org file is the right structural reference, not unsloth's.**
- unsloth's native-NVFP4 bytes are still the proof that a bit-exact NVFP4 repack is shippable, but that file may not load on current master — untested (GPU work is parked pending Tim's docker decision).
- The docker crash-loop is loading the **`inkling2_loop`** file (correct arch), so that failure is purely a VRAM/placement issue, not a naming one.

## 2026-11-20 22:40 — Child E delivered the byte-level contract (26/26 invariants, exit 0)

Findings in `docs/mxfp8/06-lossless-repack-design.md` + `scripts/inkling_native/{layout,check_layout}.py` and `check_layout.log`. The five that move us:

1. **NVFP4 is already solved by the public file.** `blk.0.ffn_gate_up_exps.weight` in unsloth's GGUF is **byte-identical** (2,985,984/2,985,984; 0 differing scale bytes, 0 differing payload bytes) to what `layout.pack_nvfp4_blocks()` builds from the HF tensors, and its `.scale` is bitwise `== amax/outer_scale_denominator` for 128/128 experts. ⇒ no new ggml type, no new packing logic, no new runtime support needed for the 20 NVFP4 layers — we only need master's converter to emit what unsloth's out-of-tree tooling already proved correct.
2. **Gate/up row permutation known and verified**: GGUF row `r` ← checkpoint row `2*(r % 1152) + (r // 1152)` for `gate_up`; `down` is identity. Scales take the same row index vector. Identity order matched only 1.7% — this permutation is genuinely required and was the top silent-corruption risk.
3. **MXFP8 repack bitwise exact and byte-count neutral** (33 B / 32 elements == the checkpoint's own bytes): 0 differing of 47,775,744 sampled values vs master's dequant reference; harness also asserts the negative case for `GGML_E8M0_TO_FP32_HALF` (exactly 2× wrong).
4. `amax/denom` never a power of two (0/128; `denom` = 2688.0 = 6×448) ⇒ F32 `.scale` sidecar mandatory for NVFP4; MXFP8 needs no sidecar.
5. **Corrected size budget: lossless GGUF body = 18,014.3 MiB (~18.0 GB), not the ~20.4 GB in DR-6.** fp8 KV with the 20 SWA layers capped at 1024: 296/552/1064/2088 MiB at 32k/64k/128k/262k ⇒ fits 2×16 GB with ≥12.4 GiB slack. **f16 KV uncapped at 262k = 42,590 MiB, over by 9,822 MiB — the SWA cap is load-bearing.**

## 2026-11-20 22:10 — Phase 1 gate **G1 (CPU) CLOSED**; 4 commits on `mxfp8-weights`

```
c547f808f tests : add MXFP8 round-trip and decode checks
15b1f4412 llama : add the MXFP8 file type
a7be012d6 ggml-cpu : route MXFP8 through the generic quantized op paths
2823e8a8d ggml : add the MXFP8 quantization type
```

**Tests green:** `test-quantize-fns` (exit 0; `Testing mxfp8` is now in the automatic `GGML_TYPE_COUNT` sweep, plus the new `test_mxfp8` property checks), `test-backend-ops -b CPU -p mxfp8` **306/306**, `test-llama-archs` **587/587**, `test-quant-type-selection` **11/11**.
Note: `test-quantize-fns` prints bare type names at the end (`q2_K q4_K q5_K q4_1 q5_1`) — that is `test_quantize_imatrix_degenerate` listing what it exercises, **not** failures. Verified `EXIT_CODE=0`.

**Two findings that changed the plan:**
1. **CPU op coverage was silently incomplete.** Adding the type alone was not enough: `ggml-cpu/ops.cpp` keeps explicit lists of quantized types routed to the generic `out_buffer`/`to_float` path in `ADD`, `ADD1`, `ACC`, `OUT_PROD`, `SET`, `GET_ROWS`, `CLAMP` — 7 places. Because `ggml_backend_cpu_supports_op` does **not** gate these on type, `GET_ROWS` on an MXFP8 tensor reached a `GGML_ABORT("fatal error")` at compute time instead of being reported unsupported. Caught by the full op sweep (`logs/g1-backend-ops-cpu-all-mxfp8.log`), fixed in `a7be012d6`. **This is risk R3 materialising exactly as predicted** — any future type addition must run the whole `-p <type>` sweep, not just `MUL_MAT`.
2. **NVFP4 for this checkpoint is NOT self-describing** (child E's corrected measurement, confirmed independently): `outer_scale_denominator` = **2688.0 = 6 × 448** for all 40 tensors, per-expert `amax` ∈ [0.10, 0.64], and **`amax/denom` is a power of two for 0 of 128 experts** (128 distinct factors per tensor). So the packed `block_nvfp4` bytes alone cannot reproduce the weights — the per-expert global factor must be carried as the F32 `.scale` sidecar and applied by a `MUL` in the graph (DR-3 confirmed; this is what unsloth's file does). Good news for losslessness: every NVFP4 block scale byte in this checkpoint has its **sign bit clear** (check (a): "SIGN-BIT scale bytes 0"), so master's `& 0x7F` e4m3→ue4m3 strip at `conversion/base.py:1324` is **exactly lossless here**.

**MXFP8 group losslessness (child E, `scripts/inkling_native/check_layout.log`):** pack→unpack→dequant **bitwise-identical** to master's reference over 15.9M/7.96M sampled values per tensor (0 differing); no byte-255 scales; e8m0 exponents in −14..−9; and it flags that `GGML_E8M0_TO_FP32_HALF` would be wrong for MXFP8 (matches the C-side note above).

## 2026-11-20 21:30 — Phase 1 commit 1 done (`GGML_TYPE_MXFP8` core type)

**Committed:** `2823e8a8d ggml : add the MXFP8 quantization type` — 137 insertions / 10 files on `mxfp8-weights`.
`block_mxfp8 = { uint8_t e /*ue8m0*/; uint8_t qs[32] /*e4m3*/ }` = 33 B / 32 elems (8.25 bpw); `GGML_TYPE_MXFP8 = 44`, `GGML_TYPE_COUNT = 45`, `GGML_FTYPE_MOSTLY_MXFP8 = 30`, gguf-py `MXFP8 = 44` + `QUANT_SIZE_MAP (32, 1+32)` + `MOSTLY_MXFP8 = 43`; reference `dequantize_row_mxfp8` / `quantize_row_mxfp8_ref` / `quantize_mxfp8`; CPU traits `vec_dot_type = Q8_0` + portable `ggml_vec_dot_mxfp8_q8_0`; MXFP8 added to `all_types` + `base_types` in `test-backend-ops`.

**Gate G1 — partial PASS (CPU):**
- `test-quantize-fns -v` → `mxfp8 absolute quantization error: ok (0.000559)`, `reference implementation error: ok (0.000000)`, `dot product error: ok (0.008979)`, **0 tests failed**. (That test iterates `GGML_TYPE_COUNT`, so the new type is covered with no test edit — `tests/test-quantize-fns.cpp:320`.) Log: `logs/g1-quantize-fns.log`.
- `test-backend-ops -b CPU -o 'MUL_MAT.*' -p mxfp8` → **240/240 OK**, including `MUL_MAT_FUSION(..., use_id=1, with_gate=1)` — i.e. the fused MoE/GLU path takes a brand-new quantized weight type with no allow-list change.
- `test-backend-ops -b CPU -o 'MUL_MAT_ID.*' -p mxfp8` → **73/73 OK**.
- `test-backend-ops -b CUDA0 -o 'MUL_MAT.*' -p mxfp8` → every case **`not supported [CUDA0, CUDA0]`**, test reports FAIL for "no test case ran". That is the desired behaviour pre-Phase-3: loud skip, **no silent fallback and no numeric corruption**. Log: `logs/g1-backend-ops-cuda0-mulmat-mxfp8.log`.

**Design note that matters (independently confirmed):** MXFP8 must decode its scale with `GGML_E8M0_TO_FP32`, **not** the `GGML_E8M0_TO_FP32_HALF` that MXFP4/NVFP4 use (MXFP4 doubles its `kvalues` table, so it halves the scale). Using `_HALF` here would be a silent 2× error on every weight.

## Checkpoint facts corrected this session (supersedes earlier notes)

- **`w*_weight_outer_scale_denominator` = 2688.0 for all 40 NVFP4 tensors** (measured directly: layers 0–19 × {w13,w2}). 2688 = 6 × 448 = (max E2M1) × (max E4M3) — the NVFP4 two-level recipe constant. `w13_weight_amax` per expert ∈ [0.1053, 0.3311], so the global factor `amax/denom` ∈ [3.92e-5, 1.23e-4]: **finite, positive, never zero, and NOT a power of two** ⇒ DR-3 stands — the NVFP4 global factor must be a per-expert F32 `.scale` sidecar (exactly what unsloth's file has); it cannot be folded into the e4m3 block scales without requantizing them.
- Child E reported "denominator = 0.0" as a real finding — **that was a bug in its own harness** (its "amax" dump is full of 61/62-valued bytes in 0–255, i.e. it was reading the e4m3 `*_weight_scale` tensor, not the F32 `*_amax`, and read zeros for the denominator). Its checks (a) and (b) are sound and useful; check (c) is void. Verified by direct measurement rather than trusting the report.
- Child E's MXFP8 invariants (all PASS, `scripts/inkling_native/check_layout.log`): e8m0 scale bytes in 113..118 (= exponents −14..−9), **zero** byte-255 (NaN) scales, zero NaN payloads, and pack→unpack→dequant **bitwise-identical** to master's `conversion/inkling2.py:55-62` reference over 15.9M + 7.96M sampled values per tensor (0 differing). That is the DR-4 proof for the MXFP8 group on sampled experts {0,64,127} × 128 rows.

## Blockers / open items

- ⚠️ **`llama-cpp` docker container is crash-looping** (RestartCount=57 pre-reboot, 48 since), unrelated to this session's edits: compose now points it at `ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf` and it OOMs —
  `allocating 15861.32 MiB on device 0: cudaMalloc failed: out of memory` → `failed to allocate CUDA0 buffer of size 16631803392` → `unable to allocate CUDA0 buffer` (a 17.63 GB model being fully offloaded to one 16.31 GB card). Port 9931's tailnet endpoint has therefore been down since that config change. **Decision pending from Tim:** stop the container while we test locally (it grabs/releases VRAM every ~60 s, which makes timing runs unreliable), or leave it crash-looping.
- **G0 baseline is NOT run yet** and is now known to be blocked on two things: the docker decision below, and the fact that the unsloth file may be pre-rename arch (test the ggml-org `inkling2_loop` file first, with a placement that actually fits — see `07-vram-placement.md`, still to be written).
- Phase 1 (core type + CPU) is **done and committed** (`2823e8a8d`, `a7be012d6`); remaining Phase-1 work is a C-side byte-identity test and the CUDA paths (Phase 3). Children B/G (touchpoints, plan review) are no longer running — their questions are being re-derived in-session.

## Measured numbers log

| date | what | value | command |
|---|---|---|---|
| 2026-11-20 21:30 | mxfp8 quantization error (fns test) | abs 0.000559, ref 0.000000, dot 0.008979 | `./build-cuda/bin/test-quantize-fns -v` |
| 2026-11-20 | free VRAM per card | 16311 MiB | `nvidia-smi --query-gpu=index,memory.total --format=csv` |
| 2026-11-20 | master build wall time | ~9 min (full clean CUDA build, `-j 22`) | `logs/build-master.log` |
| 2026-11-20 20:30 | VRAM free per card (post-reboot) | 15792 MiB of 15941 MiB | `llama-cli --list-devices` |
| 2026-11-20 20:30 | llama-cpp container restarts since boot | 48 | `docker inspect llama-cpp --format '{{.RestartCount}}'` |
| _(fill in after G0)_ | decode t/s, unsloth UD-Q4_K_XL, 2 cards | | |

---

## 2026-11-21 — close-out of the overnight session (supersedes the fill-ins above)

The `_(fill in after G0)_` row and everything before it is stale history; the living state is `01-roadmap.md` §0 and `README.md` (index). This entry records what became true and the numbers that are now the baseline.

**Delivered and committed — branch `mxfp8-weights`, 16 commits, rebased clean onto `origin/master` @ `cbb1c1c50`, tree clean:**

| area | outcome |
|---|---|
| core type | `GGML_TYPE_MXFP8 = 44` (`{ue8m0 e; uint8_t qs[32]}`, 33 B / 32 vals) + CPU paths; 306/306 CPU ops; byte-exact requantization fixed point |
| file type + tooling | `LLAMA_FTYPE_MOSTLY_MXFP8`; **`llama-quantize` wiring still missing → #23** |
| lossless converter | `--native-experts` emits both groups: **19.0 GB / 744 tensors**, 40 NVFP4 + 8 MXFP8 byte-proven vs the checkpoint |
| CUDA | Tier A (dequant `mmvf`) then native `mxf8f6f4` W8A8 `e4m3.e4m3` MMQ + a new e4m3/ue8m0 **activation quantizer** (master had none); 241/241 both cards |
| decode route | implemented, measured **slower**, reverted; kept the `QI_MXFP8` 2→8 fix (see doc 19) |
| validation | PPL 5.5757 (lossless) vs 5.5636 (lossy IQ4_NL); KL resolves loss 173x noise; sensitivity control (broken `.scale`) → PPL 5.26e6 |
| vision | works end to end, incl. colour-swap negative control; needs `--jinja` |
| **memory (the big one)** | alias-aware placement (+78% prefill / +82% decode, peer traffic −85%) → causal-mask fold (ctx slope 0.1698→0.0728, 128K unlocked) → fitter zero-span guards (SIGFPE→legible) → same-type f16 gather (chain 3.5→1.0 units, −11.2% compute @128K, +27.4% auto-fit) |
| **capacity today** | **pinned `-c 262144 -ub 128` runs: 753–771 t/s prefill / 39 t/s decode; compute 8,395 + KV 4,196 MiB ≈ 12.6 of 13.4 GiB free; retrieval verified with the needle at ~230K depth** |

**Baselines to compare future work against (2×5060 Ti, lossless file, `-sm layer`, FA on, `-b 2048`) — current as of `84d9bc0dc` (adopting upstream #32808):**

| metric | value |
|---|---|
| artifact | **`Inkling-2-Mini-FULL.gguf`, 19,610,478,848 B, 798 tensors = 744 trunk + 54 embedded drafter** |
| prefill / decode @8K, `-ub 512` | **1,091.0 / 40.9 t/s** (history: 507 → 901 aliasing → 906 MMQ → 851 after rebase → **1,091** with the indexed-bias FA) |
| compute buffer @8K (final reserve) | **3,568 MiB** total (CUDA0 549.31 + CUDA1 2,882.79 + Host 136.58) |
| PPL, 64k slice (16,705 tok), `-fa on` | **5.5656 ± 0.14398** (was 5.5787 with our F16-folded table — theirs sums the bias in F32) |
| @128K **with** drafter, `-ub 256`, f16 target cache | 1,024 t/s prefill / 50.1 t/s decode, acceptance 0.625 |
| **@256K with drafter — recommended config** | **`-c 262144 -ub 256 -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 --spec-draft-n-max 2 --spec-draft-n-ubatch 64`** → **1,027 t/s prefill / 52.1 t/s decode**, acceptance 0.73301 (151/206), **4 repeats identical to <1%**, zero fitter retries, all 17.8 GiB resident |
| cache type matters a lot | the target cache had been **f16** all session (`-ctkd` only sets the *draft's*); quantizing it freed ~2 GiB and is what made 256K + drafting fit. Quality is unaffected: PPL f16 5.5656 / f8_e4m3 5.5714 / q5_1 5.5740 / **q4_1 5.5783** / q4_0 5.5837 — all inside the ±0.144 CI. **q4_1 is dominated by q4_0** (same bytes, extra per-block min, 45-47 t/s vs 52). **f8_e4m3 is refused for the draft** — `llama-context.cpp:4496` denylists FP8 KV for `LLM_ARCH_DFLASH` (Inkling's target arch is not on that list). Doc 41 |
| auto-fit ceilings | **re-measure (#56)**. Pre-merge: `-ub 128` → 258,048 · `-ub 256` → 143,616 · `-ub 512` → 47,104 |
| greedy oracle | `Q: What is 17 times 4? A:` → `68 …` on CUDA **and** CPU; note the post-rebase model enters its thinking template, so compare CPU↔CUDA agreement rather than the old literal string |
| ops suites | mxfp8 CPU + CUDA0 + CUDA1 OK · **`test-flash-attn-indexed-bias` OK on both cards incl. `rows=1`** · archs **591/591** · `test-dflare-mix` 4/4 (+ python re-derivation) · `test-jinja` OK |
| cross-arch regression | Qwen3.8-27B + z-lab DFlash2: draft statistics identical to the PR worktree build (`0.74510`, 152/204, mean len 2.49) |

**Retractions and self-corrections, recorded so they are not silently carried forward:**
1. "The file is proven bit-exact, so the numeric bug is impossible" — the verifiers were written from the same reading of the source as the converter; agreement between them was circular. The real bug was upstream's loop-slot/scale ordering (`690b87d88`), invisible to any byte-level check.
2. "The relative bias costs 3.5 f32 copies, so narrow it" — a fit, not a mechanism. Falsified by intervention: `ggml_get_rows` hard-codes an F32 return, so dtype upstream of a gather is inert.
3. "Inkling's MTP head is unretrievable / needs a new architecture" — **wrong**. DFlare is published (arXiv 2606.02091) and the reference implementation is in vLLM (`models/inkling2/nvidia/dspark.py`, `ops/dspark.py`). The clue (`dflare_norm`) was in a table I had written myself. Blueprint: doc 33.
4. "Speculative decoding is output-invariant, so a wrong draft is safe to try" — true mathematically, false numerically: four `--spec-type` values produced four different greedy strings, and unsloth documents the same. **Acceptance rate is the only real gate.**
5. Process errors worth not repeating: I claimed `--native-experts` was an upstream hook (a child invented it; corrected in the next commit's message); I raised an alarm about a deleted `inkling2-vit.py` that existed in no commit; I wasted ~90 min on two 48 GB ablation files that were unrunnable; I twice mis-sized subagent timeouts by passing `timeout` instead of `options.timeoutMs`, and I talked myself out of continuing on the false belief that context was short when it was at 30%.
6. "The flat compute term is the 40-tensor SWA bias chain, so folding it frees ~1.5 GiB" — **wrong twice over.** Summing per-tensor peaks is not the live set (the allocator reuses one span across sequential same-shape nodes: the fold measured −36 MiB), and summing peaks *across reserve passes* is meaningless. Scoped to a single pass, CUDA1 covered 613 MiB inside a 3,205 MiB span — the reserve is an address-span property, which is why three node-deletion attempts all measured neutral. Doc 40.
7. "`-ts` does not help the 256K drafter gap" — I reported failures that were **argument errors**: `llama-cli` rejects `-ts` and `-ot` outright (and `llama-speculative` needs `-ns` removed and then fails independently of my work). Verify a hypothesis' instrument before concluding from its output. #57.

**Open, ranked** — *the numbered task list is live state, not durable documentation; ids in older
sections of this file are stale. Read the harness task list (or `01-roadmap.md` §0) for what is open
now. The substantive thread in order:* #55 (last ~300 MiB so 256K runs with the drafter), #56
(re-baseline acceptance / retrieval / CPU-vs-CUDA on the merged tree), #21 (Inkling CPU batched-eval
nan), #27 (cache-type × context × `-ub` matrix), #23 (`llama-quantize` cannot emit MXFP8), #18, #28,
and the drafter chain.

**Superseded by #32808 adoption:** our fold chain (28c8acdd5 dense −INF bucket, 8def03281 window fold, e1f2296fa same-type gather) — removed in `84d9bc0dc`. `ggml_get_rows_same_type` remains as an API with op coverage but no longer has a consumer in Inkling, so #41 must either justify or drop it. Their FA kernel wants an **F32** bias table, so narrowing that table is off our path now.

---

## Close-out entry — 2026-11-23 (late, spare-parts-2)

Branch `mxfp8-weights` at `91b753d02`, 33 commits since upstream/master (verified with `git rev-list --count`), tree clean (docs/scripts/logs stay untracked by
design). **Rebased onto 28 new upstream commits**, one of which (`#32740`) took over the interface layer
of the indexed-bias feature we had adopted from `#32808` — and whose unconditional CUDA decline
auto-merged cleanly on top of our dispatch, costing 23% of prefill until measured and fixed. Doc 55 has
the collision, the A/B, and the two silent-success bugs it exposed in my own gate harness. A second
rebase the same day added 5 more commits (`metal: add NVFP4 support` and friends) with zero overlap on
shared files — details and the re-verified capability numbers are in doc 55. The
`llama-cpp` container was stopped and restored repeatedly for measurement and is currently
**healthy, `RestartCount=0`**.

**Where the model runs now** (lossless `Inkling-2-Mini-FULL.gguf`, ctx 262,144, `-ub 256`, `q4_1`
caches, `draft-dspark,ngram-mod,ngram-map-k`):

| config | prefill | decode |
|---|---:|---:|
| deployed (FULL + `q4_1` + chain) | ~1,030 | **52.0** |
| same, `draft-dspark` alone | ~1,030 | 48.2 → the chain is **+7.9%** ✓ reconfirmed |
| **variant** (q8_0 vocab) + `q4_1` + chain | ~1,030 | **55.1 (+14.3%)** — needs your call on the two vocab tensors |
| variant + `f8_e4m3` target | 196,608 ctx | 56.2 |
| variant + `f8_e4m3` target | 131,072 ctx | 60.6 |

Session arc on the complete model: **507/22 → 1,030/52 t/s**. Deployed knobs are optimal — `n_max 3`
costs 2.6%, `--spec-draft-n-ubatch` 32 vs 64 is a wash (doc 47 §4). Retrieval verified at **231,241
tokens** on both the deployed file and the variant (docs 50 §6, 51 §6).

**The night's real finding** (doc 54): decode is neither compute- nor bandwidth-bound and — contrary to
what I asserted an hour earlier — **not** launch-bound either. CUDA graphs *are* capturing
(`GGML_CUDA_GRAPH_DEBUG=1` shows warmup complete; an instrumented
`ggml_cuda_graph_check_compability` recorded **zero** disqualifying `MUL_MAT_ID` nodes, so the 2.03 M
direct launches are capture-time traffic). The cost is **16-17 CUDA0↔CUDA1 crossings per pass carrying
~179 tiny index/mask copies each** (0-16 KB: `attn_inp_k_idxs`, `attn_inp_v_idxs`, `inp_pos`,
`rs_s_copy`, `out_ids`), which is exactly the 13,845 `cudaEventSynchronize` + 47,837 peer copies and the
~65% device idle. Cause: the loop model replays 24 physical layers into 48 slots, so slot order
interleaves the two cards' layer ranges and every replayed pass re-crosses the boundary.

**Retractions added:** (8) "graphs are bypassed — 99.7% of launches are direct" — the launch count was
capture traffic; disproven by instrumenting the disqualification predicate. (9) "`f8_e4m3` blows up the
compute buffer via the dense-mask fallback" — no such fallback fired; it is the fitter's split explosion
(2 → 1,719 splits) inflating a span while weights leave the device. (10) "the flat 4.5 GiB term is the
SWA bias chain" — deleted the chain's nodes, moved 36 MiB; peak-sums are not the live set.

**Process fixes shipped into the harness** (not just intentions): `measure_variant.sh` now polls until
the GPUs are actually free and **exits 3** rather than measuring against a foreign PID — it produced four
confident garbage numbers tonight before it could refuse. New reusable modes: `cache`, `retrieval-sized`,
`f8probe` (env `CTX`), `f8fitoff`, `scheddump` (env `SCHEDDBG`), `graphdbg`, `shiptune`.
**Skills distilled from this session** — first as four model-specific pages, then generalized into a
layered set so the method transfers to any engine or kernel task:
`kernel-work-principles` (shared rules: measure the share, ablate don't fit, verify the instrument,
silent fallback is the worst failure, byte-exactness needs an independent fixture),
`perf-attribution` (classify compute/bandwidth/launch/sync-bound, kernel share, address-span vs peak-sum,
ablation protocol, settings-sweep table), `cuda-kernel-audit` (arch guards, PTX/SASS confirmation,
vectorization/alignment, fusion differences, determinism), `quant-type-integration` (dtype contract,
hardware ceilings, the convention traps that produce plausible wrong output, validation ladder),
`lossless-model-repack` (completeness inventory, size sanity, container record rules, oracle + mutation,
never overwrite the canonical file), `spec-decode-validation` (acceptance as the only metric, loud
refusal, cross-family regression gate), `engine-prior-art` (the source ladder: upstream, PRs, other
engines, forks as fixtures, papers from tensor names), and `inkling-native-inference` kept as the
box/model coordinates page. A Subagent-lanes section was added to `fleet-principles`. Details were
recovered with `tuck_recall`, not memory; the three narrow predecessors are archived in place as
`SKILL.md.retired-20261123`.

