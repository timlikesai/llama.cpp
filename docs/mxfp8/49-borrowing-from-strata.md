# 49. What llama.cpp should borrow from Strata (and what measurement already rejected)

Prompt: *"is there anything we can or should borrow to accelerate our llama.cpp code?"* — answered by
reading `repos/Strata` (MIT, HEAD `73d9b…`-era `73d4f4196`) against our own profile, then measuring the
cheap candidates before recommending anything.

Companion: `48-strata-native-mxfp8-contribution.md` (what *we* can give them).

## 0. First, the corrected picture of where our decode time actually goes

New decode-only nsys capture (`logs/decode-only.nsys-rep`, ctx 4096, drafting engaged, acceptance
0.619, 48.3 t/s). Share of **device** time:

| kernel | share | calls | avg | median | max |
|---|---:|---:|---:|---:|---:|
| `mul_mat_vec_f<bf16, float, 1, 256, 0, 0>` | **36.0%** | 798 | 258 µs | 14.6 µs | **2,187 µs** |
| `mul_mat_f<bf162, 32, 3, 6>` | **23.9%** | 1,165 | 118 µs | 13.3 µs | **2,202 µs** |
| `mul_mat_q<40>` (NVFP4, MMQ/TC) | 5.9% | 176 | 191 µs | 188 µs | 418 µs |
| `mul_mat_f<bf162, 32, 3, 8>` | 3.4% | 300 | 65 µs | 38 µs | 107 µs |
| `mul_mat_vec_q_moe<40>` | 2.7% | 108 | 141 µs | 142 µs | 212 µs |
| `mul_mat_q<44>` (**MXFP8, our path**) | 1.8% | 16 | 639 µs | 614 µs | 880 µs |
| `mul_mat_vec_q<40>` | 1.3% | 142 | 53 µs | 61 µs | 63 µs |

**~63% of decode device time is bf16 dense matmuls — the non-expert part of the model.** The two
2.19 ms outliers are the vocab matrices (`token_embd.weight`, `output.weight`, 883 MB of bf16 each);
the rest is attention projections and the router (209 BF16 tensors). **The MXFP8 expert path I spent
the session building is 1.8% of decode device time.** That is not a complaint — it is the reason
`mmvq`-style kernel swaps on MXFP8 measured neutral (TODO #6) and why "optimize the fp8 kernels" is the
wrong next target.

Roofline sanity check: 2 × 883 MB per token at 20.7 ms/token = 85 GB/s, i.e. ~10% of the ~900 GB/s
aggregate — but a single tensor lives on *one* card, so the honest ceiling is ~440 GB/s and those two
kernels run at roughly half of it. Little headroom without sharding a tensor across cards.

Everything below is ranked by what this table plus their source actually implies.

## 1. Worth borrowing — ranked

### 1a. Expert-grouped (token-sorted) MoE tiling for prefill — biggest structural idea
`engine/src/inkling2/cuda/ink_moe_fp4_sm120.cuh:26-35` defines `InkMoeTables`:
`cnt[128] / off[129] / active[128] / pairs[P] (packed tok*8+slot) / tok_idx[P] / tiles[] {.x = expert_id,
.y = pair_row0}` — i.e. build a per-expert work list first, then run tiles keyed by **expert**, so an
expert's weights are streamed once for all the tokens that routed to it instead of once per pair.
Their header claim: *"TR=128 (16 warps = 2 weight halves x 8 routed-row eighths) halves expert weight
re-reads on 2K-8K prompts"* (`moe_fp8_mxf_tc.cu:6-9`).

Why it matters for us: `MUL_MAT_ID` currently iterates ids per token, so at `-ub 256` the same expert
body is re-read by ~16 different (token, slot) pairs. Whether L2 already absorbs that is the open
question — but note prefill measures 1,036 t/s while the *naive one-pass-per-pair* traffic model allows
~1,370 t/s, so we are inside ~25% of the pessimistic bound and L2 is doing heavy lifting. Grouping
converts that L2 dependence into a guaranteed 1-read property, which also frees L2 for everything else.
Cost: a real kernel + a sorting pass in `ggml-cuda/mmid.cu`, plus a device-side worklist (their
`n_tiles[1]` is a device counter, so it stays graph-capturable ✓ good detail to copy).

### 1b. Explicit small-M regime: tensor cores only above a threshold
`ink_moe_cuda.cuh:14-17` states the policy in one line — for both SF4 and MX8 experts:

> `-> Blackwell sm_120a mxf4nvf4 TC for large M, fused register-LUT/cvt.e2m1x2 for M=1..16`

llama.cpp's dispatch has no equivalent notion keyed on the *routed* row count: our `mul_mat_q<40>` runs
with med 188 µs and `mul_mat_q<44>` at **639 µs** on passes where the effective column count is 8-16
(the MMID `ids` case and draft verification). Their split says: below ~16 rows, use a vector/LUT kernel
with hardware `cvt` and never quantize the activations at all. Two consequences we can test cheaply:

- a minimum-effective-`ncols` gate for MMQ on block-scaled types inside `ggml_cuda_mul_mat_id`;
- their *"shape gate (N >= 1024) skips small-N … where X quantization exceeds GEMM savings"*
  (`gemm_mxfp8.cuh:7`) — same principle, applied at the cuBLASLt dense path.

Note the near-neighbour evidence in our own history: the `mmvq` attempt (TODO #6) changed nothing
because it swapped *which vector kernel* ran, while the real question is *which regime* should run.
This is that question, restated with their policy as the reference.

### 1c. CUDA-graph-internal host-expert execution ("doorbell") — the only design that makes CPU experts fast
`engine/src/inkling2/hybrid_doorbell.h:1-15` is worth quoting because it solves the exact problem I
recorded in TODO #14/#22 (*"llama.cpp's MoE cache uploads on a host-sync path — the serialization we
just removed"*):

1. `k_doorbell_plan_and_ring` runs **inside the captured decode graph**: stamps which experts are
   resident, records misses as promotion candidates, writes missed expert IDs + the MoE input vector `x`
   + a sequence number into mapped host structs. *"zero host `cudaStreamSynchronize` per layer!"*
2. NUMA-bound AVX-512/AVX2 worker pools **spin** on the doorbell, compute the cold experts (supporting
   SF4/NVFP4, **MX8/MXFP8**, S_K and native `GGUF_RAW` blocks) from node-local memory
   (`MPOL_BIND` + first-touch), write `y[j]` and `ack[node]`.
3. `k_doorbell_wait_and_collect` waits on the mapped `ack` words and scatters results into GPU `ey[k]`
   with non-cached `__ldcv` loads — still inside the graph.
4. Per token end, `k_decay_counts` / `k_promote_plan` / `k_promote_commit` promote hot cold experts over
   a low-occupancy background copy stream.

Invert the direction of traffic: instead of streaming 4 MB experts to the GPU on a sync path, stream one
2,304-float activation to the CPU and let the CPU stream back only the hits. On this box that is the
difference between "host experts cost PCIe × expert size per token" and "host experts cost PCIe ×
activation per miss". This is the idea that could make 512K feasible (experts off-card, VRAM for KV) —
but it is an engine feature (batch-graph integration + host worker pools + promotion policy), not a
ggml kernel drop-in. Recommend: prototype as a *research* branch behind a flag; cite the design.

### 1d. Dual-GPU expert parallelism, claimed at zero cache misses on 2×16 GB
`ep_dual_gpu.h:17-19`: *"Dual-GPU Expert Parallelism (--gpus 2, --ep-split / IK2_EP_SPLIT, P2P or pinned
fallback, **0 cache misses on 2x16 GB**)"*. Our blocker list (#19 `-sm tensor`, per-card peaks of
8,355 + 9,451 MiB with ~1 GiB margin) has the same shape: shard *experts by id* across the cards and
replicate the ~3 GiB trunk, rather than splitting by layer where one card must replay the other's
weights. Our own `e046294a7` (alias-aware placement) fixed the pathological case; EP is the principled
version of it, and their "0 cache misses" claim is a target to check against.
Also borrowable in the same file: **K8V4 / K8V8** KV modes (beyond `fp8_e4m3`), relevant to TODO #11.

### 1e. CPU codecs for the native types — directly usable, no architecture change
`sk_codecs.hpp:100-121`: `dot_sf4_f32_row` documented as *"uses 1-instruction pshufb/vtbl register LUT"*
on AVX512/AVX2/NEON, plus `dot_mx8_f32_row`, `gemm_sk_q8_1` (multi-token, *"maximizes weight reuse in
L1/L2"*), and an ISA dispatch that reports `avx512-vnni | avx2 | neon-dotprod | scalar`
(`active_sk_isa()`). llama.cpp's CPU path for MXFP4/NVFP4/MXFP8 dequantizes to f32 and does a float dot.
A pshufb-LUT for e2m1 nibbles and a byte-wise path for e4m3+ue8m0 are self-contained, testable in
isolation, and would speed `--cpu-moe`, the MoE cache's CPU share, and — usefully — our own A/B loop,
since CPU runs are how we cross-check CUDA numerics.

### 1f. Determinism technique
*"Exact deterministic 8-slot weighted reduction (zero atomic float non-determinism -> bitwise parity
across runs)"* (`ink_moe_cuda.cuh:9-10`). Our top-8 reduction goes through `MUL_MAT_ID`'s accumulation;
if it uses atomics, run-to-run bit-parity is luck, and draft near-ties can flip. Cheap to audit in
`mmid.cu`; valuable because our whole drafter debugging method assumes a fixed greedy string.

### 1g. Test and benchmark methodology
`engine/tests/quant/` contains `quant_parity.cpp`, `quant_bench_ggml.cpp`, `quant_selftest.cpp`,
`golden_x86.txt` + `make_golden.sh` + `run_oracle.sh`, and `check_no_ggml.sh` — they *benchmark and
parity-check against ggml on purpose*, and keep a golden-output oracle. Their
`strata_v1_1_inkling2_test.cpp` asserts bit-exactness across SF4/MX8/S_K **including** loop/SITU/AttnRes/
DSpark paths. That is the discipline this session kept needing (my `test-dflare-mix` + mutation test is
the same idea); worth adopting as a standing per-type parity harness rather than per-case tests.

## 2. Cross-validation we got for free (no code, real value)

`ink_moe_cuda.cuh:6-10` describes their router as *"Exact Top-8/128 sigmoid routing with F32
score-correction bias (bias for selection only, weights from sigmoid, normalized by sum + 1e-20,
multiplied by route_scale = 8.0)"*. Three implementations agree:

| source | selection | weights |
|---|---|---|
| HF `modeling_inkling2.py:707-716` | `topk(scores + e_score_correction_bias)` | `scores.gather(idx) / (sum + 1e-20) * routed_scaling_factor`, all in fp32 |
| Strata (above) | same | same |
| llama.cpp `llama-graph.cpp:2384-2402` (`LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID` + `ffn_exp_probs_b`) | comment: *"add experts selection bias … leave probs unbiased as it's later used to get expert weights"* | `norm_weight=true`, `expert_weights_scale` |

So our Inkling integration's router semantics are independently confirmed — which matters because PPL
alone would not have caught a subtle selection/weight mixup (it would have looked merely "a bit worse").

## 3. Measured tonight and rejected — do not spend time on these

| idea | test | result |
|---|---|---|
| Shard the two 883 MB vocab matrices across cards (`-ot token_embd.weight=CUDA0,output.weight=CUDA1`) | 2 reps each, ctx 4096, code prompt, drafting on | **50.09 vs 52.37 t/s = −4.4%** ✗ (the output projection's result must cross back for sampling). Rejected. |
| Use the smaller **q8_0** DSpark draft (402 MB vs 648 MB bf16) to cut draft weight reads | 2 reps | **51.89 t/s (−0.9%)** and acceptance **0.66055 vs 0.68224** ✗. Draft bytes are not the wall; bf16 stays. |
| `q4_1` vs `f16` **draft** KV at 4096 | 2 reps | 52.37 vs 52.53 — wash (draft KV is tiny at short ctx; it matters at 256K, see `41-…`) |
| Unified memory (`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`) | full server sweep | **−33% prefill / −20% decode** ✗ (`47-…`) |
| `GGML_CUDA_MMQ_PREC=q8` (widen FP4 activations) | full server sweep | **−15% prefill** ✗ (confirms W4A4 is the live path) |
| `GGML_CUDA_DISABLE_GRAPHS`, `--threads 16/24`, dropping checkpoint/caching flags | full server sweep | all wash-or-worse (`47-…`) |
| MXFP8 MMQ tile-loader rewrite for coalescing (their *"4 threads per weight row, 16 weights/thread, 32-bit coalesced loads, branchless E4M3 XOR sign bit"*) | **not measured** — deliberately last | Our MXFP8 MMQ is **1.8%** of decode device time; even a perfect 2× is worth <1% of decode. Only worth doing for prefill shapes, where it is also a small slice (4 of 24 layers, per `44-…`). **Deprioritized** — this is where I would have wasted the evening. |

Still-open, with the profile now pointing the right way: quantizing the two **vocab** matrices (bf16 →
q8_0 halves ~21% of decode device time; needs `llama-quantize --tensor-type` support and *breaks the
"unadulterated weights" premise for those two tensors only*, so it is Tim's call, not an optimization to
slip in), and 1a/1b above.

## 4. Two process lessons this turn earned

- **My error-extraction regex mislabeled a benign log line as a fatal one.** All six "failed" runs
  succeeded; `llama_init_from_model: failed to initialize the context: dflash requires ctx_other to be
  set (this warning is normal during memory fitting)` contains the words I grep for. Test for
  `exiting due to` / `GGML ASSERT` / `error: out of memory` instead. Caught only because I read a log
  instead of trusting the table.
- **Two consecutive runs burned on an invented flag** (`-no-cnv` is not a `llama-cli` argument in this
  build; `--help` shows only `-st, --single-turn`). Standing rule: capture `<tool> --help` to a file
  once and cite it, never recall flag names across examples — the arg sets differ per example.
