# 17 - Inkling-2 vision (mmproj) converter

Status: **done, and the premise of the task is wrong** - the converter already exists at HEAD of
this branch, it converts cleanly (422 tensors / 641.8 MB), and the full multimodal path works with
the prebuilt binaries: `llama-mtmd-cli --jinja ... --image <png>` answers a colour question
correctly for both a red-background and a blue-background test image.
Owner task: brief `briefs/S-vision-mmproj.md`. Nothing in `conversion/` needed to change.

## 0. Headline correction

The brief's core premise ("there is no vision/mmproj converter for this model anywhere in the repo")
is false **at the current HEAD**. `git log --oneline -- conversion/inkling2.py`:

```
c30ccd689 convert : keep Inkling-2 NVFP4 experts as native bytes too
7e282624e convert : keep Inkling-2 MXFP8 experts as native bytes
29b52d398 model : Inkling 2 Mini (#32694)
d998c676f convert: Fix inkling 2 mmproj conversion (#32430)
5c008e000 feat: add Inkling 2 support (text + vision) (#32217)
```

`conversion/inkling2.py:455-501` is `class Inkling2VisionModel(MmprojModel)`, registered
`@ModelBase.register("Inkling2ForConditionalGeneration", "Inkling2LoopForConditionalGeneration")`
- shipped by upstream `5c008e000`, corrected by `d998c676f`. `git status --porcelain -- conversion/`
is empty, so this is committed state, not a sibling agent's in-flight edit. `conversion/inkling2.py:190`
is expert-quantization companion checking, not a `model.visual.` drop.

So this task reduced to: prove the existing producer + existing runtime work end to end, and record
the contract. Everything below is that validation.

## 1. What the runtime expects

Two consumers, both already in the tree.

**Graph** `tools/mtmd/models/inkling2v.cpp` (`mtmd_graph_inkling2v::build`, declared
`tools/mtmd/models/models.h:985`, dispatched `tools/mtmd/mtmd-graph.cpp:828`). Tensors are resolved
in the projector-specific block `tools/mtmd/mtmd-model.cpp:2139-2150` plus the generic CLIP loader:

| runtime slot | GGUF tensor | note |
|---|---|---|
| `model.class_embedding` | `v.class_embd` | required, `throw`s if null (mtmd-model.cpp:2142) |
| `model.class_position_embedding` | `v.class_position_embd.weight` | required |
| `model.position_embeddings` | `v.position_embd.weight` | required; bilinear-resized (`inkling2v.cpp:15`) |
| `model.pre_ln_w` / `pre_ln_b` | `v.pre_ln.weight` / `.bias` | required (`ln_pre` in the checkpoint) |
| patch embed | `v.patch_embd.weight` | conv2d layout `{ps, ps, 3, hidden}` for `build_inp()` |
| 23 ViT blocks | `v.blk.{0..22}.{ln1,ln2,attn_q,attn_k,attn_v,attn_out,ffn_up,ffn_down}.{weight,bias}`, `{ls1,ls2}.weight` | `ls*` are the checkpoint's layer-scale gammas, applied as `MUL` on both residual branches |
| `model.mm_1_w` / `mm_1_b` | `mm.1.weight` / `mm.1.bias` | bias optional (`get_tensor(..., false)`); `proj_in` |
| `model.mm_2_w` / `mm_2_b` | `mm.2.weight` / `mm.2.bias` | bias optional; `proj_out` |

`ln_post` does **not** exist in this checkpoint (5.7): the tower is `ln_pre` -> 23 blocks, so the
graph ending at `build_ffn` after the CLS drop covers every norm the reference has. HF -> GGUF
tensor counts are 422 -> 422, nothing dropped.

**Metadata** read by `case PROJECTOR_TYPE_INKLING2V` (`tools/mtmd/mtmd-model.cpp:828-836`) and the
common vision loader: `clip.vision.projector_type = "inkling2v"` (enum `tools/mtmd/mtmd-impl.h:636,736`),
`clip.vision.spatial_merge_size` -> `hparams.n_merge` (**strict** `get_u32`, see 5.2),
`clip.vision.patch_size`, `clip.vision.embedding_length`, `clip.vision.block_count`,
`clip.vision.attention.head_count`, `clip.vision.attention.layer_norm_epsilon`,
`clip.vision.projection_dim`, `clip.vision.image_mean` / `image_std`, `clip.vision.image_size`.
`rope_theta` (10000) and `ffn_op = FFN_GELU_ERF` are **hard-coded by the runtime**, not read
(mtmd-model.cpp:829-832), and the converter asserts the checkpoint agrees
(`conversion/inkling2.py:473-476`). Preprocessing
(`mtmd_image_preprocessor_inkling2v::preprocess`, `tools/mtmd/mtmd-image.cpp:2152-2185`) consumes
`patch_size`, `n_merge`, `image_max_pixels`, `image_mean/std` (`*255`, pad value `-1.0f`,
`align = patch_size * n_merge`). Output tokens = `nx/(patch_size*n_merge) x ny/(patch_size*n_merge)`
(mtmd-model.cpp:4640,4675); budget `set_limit_image_tokens(1, 4096)`, warmup capped at 32x32.

## 2. How this repo builds projector files (precedent)

Projector files are **not** a separate script. They are `ModelType.MMPROJ` subclasses of
`MmprojModel` (`conversion/base.py:3174`) registered on the *conditional-generation* architecture
name and selected when the user passes the global `--mmproj` flag
(`convert_hf_to_gguf.py:117,265-271`). The same module holds the text model: `Inkling2LoopModel`
(`conversion/inkling2.py:393`, `TextModel`) and `Inkling2VisionModel` (`MmprojModel`). 40+ other
models follow this shape (`conversion/qwen3vl.py`, `bailingmoe3.py`, `exaone.py`, ...).

## 3. Design decision

Neither option (a) `--vision` flag nor (b) `conversion/inkling2v.py`: the repo idiom (and what
already exists) is a `MmprojModel` subclass in the model's own module, reached through
`ModelBase.register` + `--mmproj`. Option (b) as the brief phrased it would not even be reachable -
`convert_hf_to_gguf.py` never keys a class off `vision_config.model_type`. Because the text model and
the projector are separate classes and separate invocations, the already-proven text path is
untouched by the vision path (5.4).

## 4. What `--mmproj` emits for this checkpoint

Full dump: `logs/s-mmproj-inventory.log`. 422 tensors, 641.8 MB, `general.type = mmproj`,
`general.architecture = clip`, `general.file_type = 32` (F32/BF16 mix), 24 KV pairs:

```
clip.vision.attention.head_count         = 16            clip.vision.patch_size         = 20
clip.vision.block_count                  = 23            clip.vision.projection_dim     = 2304
clip.vision.embedding_length             = 1024          clip.vision.spatial_merge_size = 2
clip.vision.feed_forward_length          = 4096          clip.vision.image_size         = 640
clip.vision.attention.layer_norm_epsilon = 1e-05         clip.vision.image_mean/std     = [0.5,0.5,0.5]
clip.vision.projector_type               = 'inkling2v'   clip.has_vision_encoder        = True
```

Tensor patterns: 23 x {`attn_q`,`attn_k`,`attn_v`,`attn_out`,`ffn_up`,`ffn_down`}.`{weight,bias}`,
23 x {`ln1`,`ln2`}.`{weight,bias}`, 23 x {`ls1`,`ls2`.`weight`, plus
`v.class_embd`, `v.class_position_embd.weight`, `v.patch_embd.weight`, `v.position_embd.weight`,
`v.pre_ln.weight`, `v.pre_ln.bias`, `mm.1.weight`, `mm.2.weight`.

Shapes: `v.patch_embd.weight {20,20,3,1024}`, `v.position_embd.weight {1024,1024}` (= 32x32 grid,
`posemb_grid_size * patch_size = 640 = image_size`), `mm.1.weight {4096,4096}`,
`mm.2.weight {4096,2304}` (4096 = vit 1024 x merge 2^2).

### 4.1 dtype is **not** all-BF16 (contradicts the brief)

140 BF16 (type 30) + 282 F32 (type 0). Every 1-D tensor (biases, norms, `ls1/ls2`, `class_embd`, both
position embeddings) and the 4-D `v.patch_embd.weight` are emitted as F32; the 2-D attention/MLP
weights and `mm.{1,2}.weight` stay BF16. That is the base `MmprojModel` dtype policy, and it is
widening only - see the value check in 5.6.

### 4.2 name/layout mapping done by the converter (`conversion/inkling2.py:485-501`)

- `model.visual.proj_in` -> `mm.1.weight`, `proj_out` -> `mm.2.weight` through an explicit
  `_projector_layers` table, because the `V_MMPROJ` block index cannot be derived from these names.
- the linear patch embedding `[1024, 1200]` (1200 = 3 x 20 x 20) is reshaped+permuted to conv2d
  `[20,20,3,1024]` to match `build_inp()`.
- bare params (`class_embedding`, `cls_position_embedding`, `position_embedding`) and the layer-scale
  `.gamma` / `.lambda_1` / `.lambda_2` get a synthetic `.weight` suffix so the tensor map matches.
- `get_vision_config()` (`conversion/inkling2.py:458-470`) translates the checkpoint's names to the
  keys the base class wants: `native_patch_size` -> `patch_size`, `adapter_merge_size` ->
  `spatial_merge_size`, `mlp_ratio` -> `intermediate_size`, `posemb_grid_size * patch_size` ->
  `image_size`; it hard-fails if the checkpoint ever stops being RGB-input / exact-GELU / axial-RoPE-10000.

## 5. Validation

All runs use the **prebuilt** binaries in `build-cuda/bin`; no build was run, no GPU was used
(`-dev none`). Driver: `logs/s-validate.sh`. Logs: `logs/s-convert-dryrun.log`,
`logs/s-convert-real.log`, `logs/s-mmproj-inventory.log`, `logs/s-mtmd-debug-encode.log`,
`logs/s-mtmd-cli-roundtrip.log`, `logs/s-mtmd-cli-roundtrip-nojinja.log`,
`logs/s-mtmd-cli-blue.log`, `logs/s-test-mtmd-impl.log`, `logs/s-mtmd-debug-nomerge.log`,
`logs/s-mmproj-value-check.log`. Test images: `logs/s-test-red-blue.png`, `logs/s-test-blue-red.png`.
Scratch outputs: `logs/inkling2-mmproj-F16.gguf` (641.8 MB), `logs/inkling2-mmproj-nomerge.gguf`
(negative control, kept as evidence).

### 5.1 producer - PASS

`python3 convert_hf_to_gguf.py <snap> --mmproj --outfile logs/inkling2-mmproj-F16.gguf` -> exit 0,
`n_tensors = 422, total_size = 641.8M`. (Dry-run also exit 0.)

### 5.2 projector load - PASS

`llama-mtmd-debug -m Inkling-2-Mini-NATIVE.gguf --mmproj logs/inkling2-mmproj-F16.gguf -dev none
-p encode -n 128 --image red` exits 0 and dumps a forward graph that is unmistakably
`mtmd_graph_inkling2v::build`: `IM2COL(v.patch_embd.weight{20,20,3,1024}, inp_raw{128,128,3,1})` ->
`ADD(patch_embd, pos_embd)` -> `CONCAT(ADD(v.class_embd, v.class_position_embd.weight){1024,1}, ...{1024,36})`
(CLS prepend, 37 = 36+1) -> 23 x {`attn_out_scaled = MUL(attn_out, v.blk.N.ls1.weight)`,
`ffn_out_scaled = MUL(ffn_out, v.blk.N.ls2.weight)`} -> `MUL_MAT(mm.2.weight{4096,2304}, ffn_gelu_erf{4096,9})`.
128px aligns to 120px = 6x6 patches -> 3x3 = 9 image tokens, i.e. the `nx/(patch*n_merge)` rule is live.

### 5.3 image -> text round trip - PASS (one real finding)

**Finding: `llama-mtmd-cli` aborts without `--jinja`.** `terminate called after throwing an instance
of 'std::runtime_error' what(): this custom template is not supported, try using --jinja`, thrown from
`common_chat_templates_apply_legacy` inside `mtmd_cli_context`'s constructor
(`logs/s-mtmd-cli-roundtrip-nojinja.log`). Inkling-2's chat template needs the Jinja engine, so
`--jinja` is mandatory with this model. Not a converter defect; a usage requirement.

With `--jinja`, 200x200 PNGs (red background + blue square, and the colour-swapped control) on
`-p 'What color is the big background? Answer in one word.' -n 64 --temp 0 -dev none`:

| image | model output (tail) |
|---|---|
| red bg, blue square | `...the background is clearly red, with a blue square in the center. The answer should be one word: red.` -> `red` |
| blue bg, red square | `...there's a large blue background with a red square in the center. The big background is blue.` -> `Blue` |

Both correct, so the tower is reading pixels, not a prior. 200x200 -> 10x10 patches -> 5x5 = 25 image
tokens; vision encode 189-212 ms on CPU; 64 text tokens in ~10 s.

### 5.4 text-path regression - not applicable, no converter change made

`git status --porcelain -- conversion/` is empty; `conversion/inkling2.py` is identical to HEAD, so
`Inkling-2-Mini-NATIVE.gguf` and its 744-tensor inventory cannot have moved. The regression gate in
the brief only existed because the brief assumed I would edit that file. I did not, and did not
regenerate the 19 GB text file.

### 5.5 existing unit tests + negative metadata control - PASS

`./build-cuda/bin/test-mtmd-impl`: **19 tests / 294 assertions / 0 failures**, including
`test_image_preprocessor_inkling2v` (38 assertions, `[PASS]`).

Negative control: a copy of the mmproj with the `clip.vision.spatial_merge_size` key renamed
(single byte, same length, so tensor offsets are untouched) fails to load with
`mtmd_init_from_file: error: Key not found: clip.vision.spatial_merge_size` ->
`Failed to load vision model`. That key is read only inside `case PROJECTOR_TYPE_INKLING2V`, so this
also proves the inkling2v hparams branch is the one executing.

### 5.6 value preservation - EXACT

`logs/s-mmproj-value-check.log`: aggregating every HF `model.visual.*` tensor against every GGUF
tensor (BF16 widened to F32 on both sides):

```
hf   : tensors=422 n=318256128 sum=-31923.335715 sumsq=414282.711479 min=-11.375000 max=12.500000
gguf : tensors=422 n=318256128 sum=-31923.335715 sumsq=414282.711479 min=-11.375000 max=12.500000
EXACT_MATCH
```

Element count, sum, sum of squares, min and max all agree - no requantization, consistent with
"BF16 passthrough", though it is a global aggregate rather than a per-tensor byte diff.

### 5.7 checkpoint-side inventory (name reconciliation)

`logs/s-hf-vision-names.log` - HF `model.visual.*`: 422 tensors, matching GGUF 422 tensors 1:1.
Real leaf names differ from my brief: `backbone.embeddings.{class_embedding, cls_position_embedding,
patch_embedding.weight, position_embedding}`, `backbone.ln_pre.{weight,bias}` (**no `ln_post`**),
`backbone.layers.N.{ln_1,ln_2}.{weight,bias}`, `backbone.layers.N.{ls_1,ls_2}.gamma`,
`backbone.layers.N.self_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}` (separate q/k/v, not a
fused `qkv_proj`), `backbone.layers.N.mlp.{c_fc,c_proj}.{weight,bias}`, `visual.proj_in.weight`,
`visual.proj_out.weight`. All 422 are BF16 in the checkpoint (the brief is right about that); the
emitted file re-expresses the 1-D and 4-D ones as F32 (4.1) with values unchanged (5.6).

## 6. Corrections to my brief

1. **"no vision/mmproj converter exists"** - false at HEAD; see section 0.
2. **"conversion/inkling2.py:190 currently drops every `model.visual.` tensor"** - that code is the
   expert-quantization companion check; `Inkling2VisionModel.filter_tensors`
   (`conversion/inkling2.py:481-485`) selects `model.visual.` / `model.vision_tower.` prefixes.
3. **"422 BF16 tensors"** - emitted as 140 BF16 + 282 F32 (4.1); still value-exact (5.6).
4. **"decide between (a) `--vision` flag on inkling2.py or (b) new `conversion/inkling2v.py`"** -
   neither is how this repo does it, and (b) is unreachable through `convert_hf_to_gguf.py`.
5. **`preprocessor_config.json`** - the snapshot ships `processor_config.json` instead
   (`image_processor.image_mean/std = [0.5,0.5,0.5]`, `rescale_factor = 1/255`, `size = 40x40`,
   `image_token = <|unused_200054|>`); `conversion/base.py:3246-3254` reads the latter.
6. **`llama-mtmd-debug` does not report hparams** - mtmd's
   `LOG_INF("load_hparams: projector: ...")` lines (mtmd-model.cpp:1383+) never surface from either
   binary's output, so the projector identity had to be proven from the graph dump plus the
   missing-key control (5.2, 5.5). Why those INFO lines are suppressed is UNVERIFIED.

7. **leaf names** - `attn.{qkv_proj,out_proj}` and `mlp.{c_fc,c_proj}` with an `ln_post` do not match
   the checkpoint: it is `self_attn.{q,k,v,out}_proj` and there is no `ln_post` (5.7).

## 7. Residual questions (not blockers)

- The text GGUF must carry a media marker for `mtmd_helper_eval`; the NATIVE file works, but I did
  not check which token it is. `logs/s-mtmd-cli-roundtrip.log` shows 3 chunks encoded for 1 image, so
  the marker is present.
- `--jinja` being mandatory is a docs-level trap; consider a note in `tools/mtmd/README.md` (owned by
  another task, not touched here).
