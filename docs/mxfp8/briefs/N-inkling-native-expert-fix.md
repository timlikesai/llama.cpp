# Brief N — Fix Inkling-specific native-expert execution so the lossless file runs correctly on CUDA (prefill AND decode)

## Control evidence that narrows this enormously (measured by parent, do not re-derive)

| model | native types | backend | result |
|---|---|---|---|
| `~/models/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF/*-MID-HIGH.gguf` | 448 `NVFP4`, 168 `.scale` named `blk.0.ffn_gate.scale`, `blk.0.ffn_down.scale` (**unfused**) | CUDA | **coherent**: "To calculate 17 times 4, you can break it down into simpler parts: 1. Multiply" (log `logs/nvfp4-control.log`) |
| `~/models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf` | none (all dequantized→IQ4_NL) | CUDA + CPU | **correct**: `A: 68 B: 64 C: 72 …` |
| `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf` (ours) | 40 `NVFP4` fused `ffn_gate_up_exps` + 40 `.scale`, 8 `MXFP8` | CUDA | wrong: `4.0 17.0 17.0 …` |
| same | same | CPU (`-ngl 0`) | wrong, **differently**: `17 times 4. 4. 5. 5. 6. 7. 18.` |

Device matrix (`logs/m2-device-matrix.log`, script `scripts/inkling_native/probe_device_matrix.sh`): moving **MXFP8** to CPU gives output *identical* to all-GPU ⇒ MXFP8 placement is irrelevant. Putting **NVFP4 experts** on CPU is *still* wrong. CPU and CUDA disagree with each other on identical bytes.

**Therefore: native NVFP4 works in general (Qwen3.8 proves it). The fault is specific to Inkling's expert path.** The three things Inkling has that the working control does not:
1. **Fused** `ffn_gate_up_exps` with a single per-expert `.scale` (loaded into `layer.ffn_up_exps_s` at `src/llama-model.cpp:1920-1921`) instead of separate gate/up scales.
2. The **MoE cache / slots** path: `build_lora_mm_id` at `src/llama-graph.cpp:1894-1896` uses `slots == nullptr ? mul_mat_id(w,…,ids) : mul_mat_id(cached ? moe_cache->get_experts(w) : w, …, slots)`.
3. **Clamped SituGLU** (`inkling2.activation.situ_beta = 4.0`, `situ_linear_beta = 25.0`) — an activation whose clamp interacts with where the per-expert scale is applied.

## What is already proven correct (do NOT redo; do not "verify" these again)

- Our file's bytes: experts decode to exactly the checkpoint (worst relative error `0.000e+00`, both gate and up halves) via `scripts/inkling_native/verify_nvfp4_gguf.py` / `verify_nvfp4_values.py`; `.scale == amax/2688` bitwise.
- Non-expert tensors are bit-exact BF16 vs the checkpoint (max abs diff 0).
- `GGML_TYPE_MXFP8`: CPU `test-backend-ops -b CPU -p mxfp8` 306/306; CUDA `-b CUDA0 -p mxfp8` 237/237; `test-quantize-fns` includes a byte-exact requantization fixed point.
- `llama-graph.cpp:1904-1911` applies `w_s` once, on the fused result, before the gate/up split (the `cb()` at :2588 is only a callback).
- `ggml_ue4m3_to_fp32` (CPU) and `ggml_cuda_ue4m3_to_fp32` (CUDA) both return `raw*0.5`, cancelling `kvalues_mxfp4`'s doubling — symmetric, so the 0.5/2.0 asymmetry hypothesis is dead.

## Files you own (hard fence)

`src/llama-graph.cpp`, `src/llama-model.cpp`, `src/models/inkling2*.cpp`, `ggml/src/ggml-cuda/**` **except `mmq*.{cu,cuh}` and `template-instances/**`** (another agent is implementing the MXFP8 mmq fast path there — do not touch), plus `docs/mxfp8/15-inkling-native-expert-fix.md` and `logs/`.
Do not modify `conversion/inkling2.py` semantics — if you need a variant file for an experiment, generate it into a scratch name and **revert any temporary converter edit, verified with `git diff`**, before finishing.

## Task, in order

1. **Write `docs/mxfp8/15-inkling-native-expert-fix.md` within 4 tool calls** (this table + your hypotheses), and update it as you measure. Partial-on-disk beats perfect-and-lost; three sibling tasks died at the timeout having written nothing.
2. **Localize to one of the three Inkling-specific features above**, using the cheapest decisive probe rather than reading everything:
   - **MoE cache probe (start here, it explains CPU-vs-CUDA disagreement best):** find what `moe_cache->get_experts()` does to a quantized weight (repack? contiguous copy by slots?) and whether it is type-aware. A repack that assumes power-of-two `type_size` would corrupt a **33-byte MXFP8** block and/or an NVFP4 tensor while leaving IQ4_NL fine — and would plausibly differ between backends. Check `src/llama-moe.c*` / `llm_moe_cache*` and how `GGML_OP_MUL_MAT_ID` + `slots` is implemented on CUDA (`ggml/src/ggml-cuda/moe*.cu` if present).
   - **Fused-scale probe:** force `moe_cache` off (`--no-cached-moe` if such a flag exists; check `common/arg.cpp` for the cache toggle) and/or run the NATIVE file with `-ot` to move experts to CPU, and see which knob flips correctness.
   - **SituGLU probe:** compare `--override-tensor` variants; and check whether the SituGLU clamp constants are applied before or after the scale multiply.
3. **Fix it.** If the defect is in our converter's *choice* (e.g. we should not rely on `.scale` for fused gate_up, or must not emit fused+scale for cached MoE), fix in the converter and regenerate. If it is in llama.cpp's graph/CUDA, fix it there with a minimal, upstream-shaped patch and keep the tests green.
4. **Gate (all must pass, with logs):**
   - `llama-completion -m <NATIVE> -c 4096 -p "Q: What is 17 times 4? A:" -n 24 --temp 0 -no-cnv --seed 42` on **CUDA auto-fit** yields `68` (and sane continuations on 3 more prompts from `scripts/inkling_native/probe_device_matrix.sh`).
   - The same file's output agrees with the **lossy IQ4_NL reference** on all 4 prompts, and CPU-vs-CUDA agree within tolerance (this was the original red flag).
   - `test-backend-ops -b CUDA0 -p mxfp8` still 237/237 and `-b CPU -p mxfp8` still 306/306.
   - **Prefill and decode both measured** on the lossless file with both cards, reported as t/s alongside the G0 baseline (589.72 prefill / 21.94 decode long-ctx / 32.00 decode short-ctx). Use `llama-completion` (auto-fit); **never** pass an exact `-ngl` (it disables fitting and OOMs), and don't use `llama-bench` for this model (no fitter, tensor-split default 0).
5. Report in the doc: root cause in one paragraph, the minimal patch, which of the three Inkling features was responsible, whether it is ours or upstream, and the before/after matrix.

## Environment rules

- Build `cmake --build build-cuda -j 22`, tee to `logs/`. Do not reconfigure. Read complete build output; do not filter away errors.
- Both GPUs are free (the `llama-cpp` container was removed by Tim's order). **Never** run docker/systemctl. Never delete any file or log.
- The NATIVE file is 19.0 GB at `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf`; regeneration takes ~50 s: `python3 convert_hf_to_gguf.py /home/tim/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337 --native-experts --outfile <path>`. Keep disk under control: 297 GB free, and **delete nothing you didn't create**.
- If you find the root cause but can't finish the fix, leave the build green and write the exact minimal patch into the doc.
