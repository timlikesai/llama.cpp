# Brief M — Debug: our bit-exact NATIVE Inkling file is numerically WORSE than the lossy IQ4_NL file

**Symptom (reproduce exactly; both runs are CPU, same seed, same prompt, temp 0, our `build-cuda/bin/llama-completion`):**

```
./build-cuda/bin/llama-completion -m ~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf        -ngl 0 -c 4096 -p "Q: What is 17 times 4? A:" -n 24 --temp 0 -no-cnv --seed 42
  -> "Q: What is 17 times 4? A: 17 times 4. 4. 5. 5. 6. 7. 18."                      # WRONG
./build-cuda/bin/llama-completion -m ~/models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf  (same flags)
  -> "Q: What is 17 times 4? A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:"                 # CORRECT
```
Logs: `logs/g2b-cpu-run.log`, `logs/g2b-cpu-run-iq4nl.log`. A **lossy** file answers correctly and our **bit-exact** file does not ⇒ a bug in the native path, not a quantization-quality difference. Do not "explain it away" — find it.

**Deliverable:** `docs/mxfp8/14-native-numeric-bug.md` with the diagnosis, the discriminating experiment, the fix (or the precise upstream defect if it is not ours), and the before/after reproduction. Write this file within 5 tool calls and keep updating it.

## The two files

| | NATIVE (ours, broken) | IQ4_NL (reference, works) |
|---|---|---|
| experts L0-19 | 40 `NVFP4` tensors + 40 F32 `[128]` `.scale` | `IQ4_NL` (global factor baked in by dequant) |
| experts L20-23 | 8 `MXFP8` tensors, no sidecar | `IQ4_NL` |
| rest | BF16 (178) + F32 (518) | Q8_0 (186) + F32 (478) |
| arch/loop meta | `inkling2_loop`, block_count 24, `loop.start/end/count/varying` | same |

Both were produced by **this branch's** `conversion/inkling2.py`; NATIVE used `--native-experts`. Regenerate either with:
`python3 convert_hf_to_gguf.py /home/tim/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337 [--native-experts] --outfile ...` (~50 s, 19 GB; 297 GB free. Keep every log; never delete.)

## Already eliminated (do not redo)

1. **MXFP8 bytes are right.** `scripts/inkling_native/verify_mxfp8_gguf.py` → 320 rows across all 8 tensors byte-identical to the checkpoint, exit 0. MXFP8 CPU support → `test-quantize-fns` incl. `test_mxfp8` (byte-exact fixed point) and `test-backend-ops -b CPU -p mxfp8` 306/306 vs reference.
2. **NVFP4 bytes match an independent implementation.** `verify_nvfp4_gguf.py` says our `blk.0/blk.17` tensors are byte-identical to the public `~/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf` NVFP4 tensors, and `.scale == amax/2688`. **Caveat: that public file is hours old and nobody has validated its numerics — "matches unsloth" is evidence our packing agrees with theirs, NOT proof it is correct.**
3. **Row order is not it.** `_expert_row_slices` emits expert → half → 128-row blocks, giving [all gate rows][all up rows], which is exactly what `llama-graph.cpp:2599-2602` expects (`gate` = first `n_ff` rows, `up` = second half). The dequantized IQ4_NL path uses the identical slice helper, so ordering cannot differ between the two files.
4. **`.scale` plumbing is present and single-applied.** `llama-model.cpp:1920-1921` loads `ffn_gate_up_exps.scale` into `layer.ffn_up_exps_s`; `inkling2.cpp:410-412` passes it to `build_moe_ffn`; `llama-graph.cpp:2585` passes it into `build_lora_mm_id` (applied to the whole fused result, once). The extra `if (up_exps_s) { cb(...) }` at :2588-2590 is only a graph callback, not a second multiply.

## Prime suspects, in priority order

- **A. NVFP4 scale/element convention mismatch between the *file* and ggml's decoder.** `ggml_ue4m3_to_fp32` (`ggml-impl.h:521-535`) deliberately returns **`raw * 0.5`** to pair with `kvalues_mxfp4` (which are 2× the true E2M1 values), and `block_nvfp4.d` is stored as **UE4M3 = e4m3 with the sign bit stripped**. Check `dequantize_row_nvfp4` (CPU) and the CUDA mmq/mmvq NVFP4 paths against the *actual* ModelOpt recipe: per-16-element scale is **e4m3 (signed)** and the global factor is fp32. If the 0.5-vs-2.0 cancellation is applied asymmetrically (e.g. correct for `to_float` but not for the fused `MUL_MAT_ID`/mmvq path, or vice versa) every expert output is off by a constant factor per sub-block — small-looking but model-breaking. **Quantify it: compute, for one expert, the true value vs what each runtime decode returns, and report the ratio.**
- **B. Is `.scale` actually applied on the path the CPU/MUL_MAT_ID takes?** `build_lora_mm_id` may fold `w_s` only for some backends/ops. Instrument: patch a debug print or build a tiny 1-layer graph, or temporarily write a NATIVE file with `.scale` forced to 1.0 and see whether output changes at all. **If output is unchanged, the sidecar is being ignored — that is the bug.** (This is the single most decisive cheap experiment; do it early.)
- **C. `w_scale`/`expert_weights_scale` (8.0) interaction** with the per-expert `.scale` in `build_moe_ffn` — check the multiply order and whether one replaces the other.
- **D. MXFP8 in the *fused/quantized MoE* path on CPU** (the 306/306 tests exercised `MUL_MAT_ID` generically, but check `use_id=1` with `n_expert=128` and the actual n_ff=1152 shape and block-32 alignment of a 2304-wide row).
- **E. Attention dtype**: our BF16 (exact from checkpoint) vs reference Q8_0. Verify a BF16 source is written with correct endianness/type by comparing one attention tensor's dequantized values against `mtp`-free HF weights. (Low prior, but cheap to rule out.)

## Method rules

- **Bisect by construction, not by guessing**: make minimal variants and compare outputs — e.g. (i) `.scale` = ones, (ii) experts all dequantized + MXFP8-only native, (iii) NVFP4-only native. A variant that flips the answer localizes the fault to that group. Keep each conversion log.
- Use a **fixed multi-prompt suite** (at least 4 prompts: arithmetic, definition, list, code) rather than one prompt, and record the outputs verbatim in the doc; a single prompt is anecdote.
- Cross-check decode math in Python with `gguf-py`'s own dequantize helpers (they mirror ggml) *and* with the HF reference in `conversion/inkling2.py:_dequant_expert_chunk`. Three-way agreement is the standard.
- Do not touch `ggml/src/ggml-cuda/*` (another agent is implementing CUDA Tier A there). You own `conversion/`, `scripts/inkling_native/`, `docs/mxfp8/14-*`, `logs/`. If the root cause turns out to be in `ggml/` or `src/`, **write the diagnosis + the exact minimal patch into the doc and stop** — the parent will land it.
- No docker/systemctl. No deleting files or logs. 30-minute budget; finalize the doc by 25.

---

## UPDATE FROM PARENT — new decisive evidence (do not skip)

1. **The bug is in the file/converter or a backend convention, NOT in the new CUDA MXFP8 code.** Running the same NATIVE file on GPU after landing CUDA Tier A:
   - CPU (`-ngl 0`): `"A: 17 times 4. 4. 5. 5. 6. 7. 18."`
   - CUDA (`-ngl auto`): `"A: 4.0 17.0 17.0 4.0 17.0 17.0"`
   - reference lossy IQ4_NL on CUDA: `"A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:"`
   **Same file, same seed, two backends give DIFFERENT wrong answers.** A correct file must agree across backends to within fp tolerance. So the two backends disagree about the native encoding. Logs: `logs/g2b-cpu-run.log`, `logs/g3-native-gpu.log`, `logs/g2b-cpu-run-iq4nl.log`.
2. CUDA Tier A for MXFP8 is now committed and green (`test-backend-ops -b CUDA0 -p mxfp8` 237/237, 226 of them MUL_MAT family; CPU still 306/306). Do not edit `ggml/src/ggml-cuda/*` numerics — instead **use the CPU/GPU disagreement as the oracle**: whichever side is wrong tells you where the convention diverges.
3. Sharpened hypotheses, in priority order:
   - **H1: the per-expert `.scale` sidecar is applied by one backend and ignored/misapplied by the other.** Test cheaply: build a variant GGUF where every `.scale` is 1.0 (patch the tensors in place with a small python rewrite of the existing file — you do NOT need a full reconversion; nbytes and shape are unchanged) and check whether CPU and GPU outputs *change*. If one backend's output is identical with scale=1.0, that backend ignores the sidecar.
   - **H2: the ue4m3 half/double convention.** `ggml_ue4m3_to_fp32` returns `raw * 0.5` to cancel `kvalues_mxfp4`'s doubling (`ggml-impl.h:521-535`). Verify CPU `dequantize_row_nvfp4` and the CUDA NVFP4 path (mmq-instance-nvfp4 / mmvq `vec_dot_nvfp4_q8_1` at `vecdotq.cuh:334`) both apply exactly one cancellation, and that `block_nvfp4.d` really holds **ue4m3** (sign stripped) as the converter assumes. A single stray factor of 2 per 16-element sub-block would produce exactly this kind of confident nonsense.
   - **H3: NVFP4 global factor semantics.** Confirm from the checkpoint what `amax/2688` should multiply (weights only? or is the e4m3 sub-block scale already normalized such that the sidecar must be applied to the *output*?): our verifier asserts "decode == HF values × sidecar", which is consistent, but re-derive it from `conversion/inkling2.py:_dequant_expert_chunk` order of operations, and check the runtime applies `.scale` at the same point in the math (it is applied to the whole fused gate_up result **before** the gate/up split and the SiTU activation — see `llama-graph.cpp:2585-2602`; if the reference dequant scales *after* the activation, or applies it per-half, the fused case is wrong).
4. **The single most valuable artifact you can produce**: a tiny numeric harness that, for one expert of one layer, computes the MoE FFN output three ways — (a) HF reference dequant × sidecar, (b) ggml CPU decode, (c) ggml CUDA decode — and prints the max relative error of each pair. That localizes the divergence to a specific function instead of a symptom. Budget 10 minutes for it; a wrong answer with that table is far more useful than a guess.
5. Time discipline: 30-minute hard budget, write `docs/mxfp8/14-native-numeric-bug.md` in your first 5 tool calls, and land a partial diagnosis with the measurement table rather than nothing.
