# Brief S — Write the missing Inkling-2 **vision** (mmproj) converter so the model is actually multimodal

## Why this task exists

`thinkingmachines/Inkling-2-Mini` is natively multimodal: a 23-layer ViT encodes images and a projector maps them into the text hidden space, processed jointly by the decoder. Measured facts:

- **422 BF16 tensors, 636.5 MB** inside `model-00001-of-00001.safetensors`, all dtype BF16 (no block quantization — so this repack is dtype-preserving and carries none of the NVFP4/MXFP8 convention risk).
- Names: `model.visual.backbone.*` (420) + `model.visual.proj_in.weight [4096, 4096]` + `model.visual.proj_out.weight [2304, 4096]`.
- Inside `backbone`: `embeddings.{class_embedding[1024], cls_position_embedding[1024], patch_embedding.weight[1024,1200], position_embedding[1024,1024]}`, `layers.0..22.{ln_1,ln_2,attn.{qkv_proj,out_proj},mlp.{c_fc,c_proj}}` (verify the exact leaf names yourself), `ln_post.*`, `ln_pre.*`.
- `config.json.vision_config`: `model_type: "inkling2_vision"`, `hidden_size 1024`, `num_hidden_layers 23`, `num_attention_heads 16`, `mlp_ratio 4.0`, `native_patch_size 20`, `adapter_merge_size 2`, `temporal_patch_size 2`, `projection_hidden_dim 4096`, `projector_bias false`, `posemb_grid_size 32`, `text_hidden_size 2304`, `layer_norm_eps 1e-5`. (1200 = 3·20·20 confirms the patch embedding.)

**The runtime already exists and is complete** — this is the missing producer, not a missing consumer:
- `tools/mtmd/models/inkling2v.cpp` — the ViT/projector graph builder
- `tools/mtmd/mtmd-image.cpp:2152` — `mtmd_image_preprocessor_inkling2v::preprocess`
- `tools/mtmd/mtmd.cpp:1150` — wires that preprocessor
- `tools/mtmd/mtmd-impl.h:736` — `{ PROJECTOR_TYPE_INKLING2V, "inkling2v" }`
- `conversion/inkling2.py:190` currently **drops** every `model.visual.` tensor, and there is **no** vision/mmproj converter for this model anywhere in the repo (checked: no `tools/mtmd/*.py`; only `conversion/{qwen3vl,bailingmoe3,exaone}.py` reference `model.visual`).

## Deliverables

1. **`docs/mxfp8/17-vision-mmproj.md` within 4 tool calls**, then keep it updated: what the runtime expects (exact tensor names + hparams keys, quoted from `inkling2v.cpp`), what you emit, and the test results. Three sibling tasks timed out this session having written nothing — do not be a fourth.
2. **A converter that produces an mtmd-compatible projector GGUF** for Inkling-2's vision tower. Decide and *justify* in the doc between: (a) extending `conversion/inkling2.py` with an opt-in flag (e.g. `--vision`/`--mmproj <out>`) that emits a separate projector file, or (b) a new `conversion/inkling2v.py` registered for `vision_config.model_type == "inkling2_vision"`. Pick whichever matches how the repo actually builds projector files for the other multimodal models — **find that first** and imitate it (`grep -rn "PROJECTOR_TYPE\|mmproj\|is_multimodal\|visual" conversion/ tools/mtmd/*.cpp | head -40` and read one working example end to end).
3. The emitted file must carry the metadata the graph reads (architecture/projector name `inkling2v`, layer count, hidden size, head count, patch size, merge size, `image_mean`/`image_std` if the preprocessor requires them — check `mtmd_image_preprocessor_inkling2v::preprocess` for which hparams/fields it consumes) and the 422 BF16 tensors **unchanged in value** (BF16 passthrough; no requantization).

## Ground rules

- **DO NOT RUN ANY BUILD.** Another agent (`P`) is building `build-cuda` concurrently and a second `cmake --build` in the same tree will corrupt both. Use the **existing** binaries: `build-cuda/bin/llama-mtmd-cli`, `llama-mtmd-debug`, `test-mtmd-impl` (all present, built 10:00). If you conclude a rebuild is required, write the exact needed change into your doc and stop — the parent lands it.
- You own: `conversion/**` (create your converter; keep any `inkling2.py` edit behind a flag that is **off by default** so the already-proven text path cannot regress), plus `docs/mxfp8/17-*`, `logs/`, and a scratch output file. Do NOT touch `ggml/**`, `src/**`, `tools/mtmd/**`, `tests/**`, `scripts/inkling_native/**`.
- **Regression gate for the text path** (because you may edit `conversion/inkling2.py`): after your change, regenerate the text file and confirm it is unchanged —
  `python3 convert_hf_to_gguf.py <snapshot> --native-experts --outfile logs/regress-native.gguf` then `python3 scripts/inkling_native/verify_mxfp8_gguf.py --gguf logs/regress-native.gguf` must exit 0, and compare the tensor inventory (40 NVFP4 / 8 MXFP8 / 178 BF16 / 518 F32, 744 total) against `~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf`. Delete only the file you created for this check, never `Inkling-2-Mini-NATIVE.gguf` or any log.
- Snapshot: `/home/tim/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337`. ~270 GB free. No docker/systemctl.

## Validation (what "working" means)

1. The projector GGUF loads: `./build-cuda/bin/llama-mtmd-debug --model ~/models/timlikesai/Inkling-2-Mini-NATIVE.gguf --mmproj <yours>` (check `--help` for the real flag names on this build) reports the `inkling2v` projector and sane hparams.
2. An actual image→text round trip with `llama-mtmd-cli` (generate a small test image with PIL from the venv, e.g. a red square with a shape, and ask "What color is this?"). Keep the full log. If the runtime rejects something, that is a *finding* — record the exact error and what it implies about expected names/keys.
3. Document the remaining gap honestly if the tower cannot be made to run without a C++ change (e.g. a missing hparams key or tensor name mismatch between `inkling2v.cpp` and the checkpoint) — a precise mismatch table is a successful outcome for this task.

## What is already true, do not re-derive

The text backbone is done and correct: `Inkling-2-Mini-NATIVE.gguf` is 18,974,614,688 B, bit-exact (40 NVFP4 + 8 MXFP8 experts + BF16/F32 rest), generates identical output on CPU and CUDA (`A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:`), prefill 484.50 / decode 22.06 t/s at 2817 tokens, and the backend op suites are green (306/306 CPU, 237/237 CUDA0 for `-p mxfp8`). The root cause of the earlier wrong-output problem was upstream `690b87d88` (loop slots aliased before the `.scale` pass), cherry-picked as `489106434`.

Budget 30 minutes; finalize the doc at 25.
