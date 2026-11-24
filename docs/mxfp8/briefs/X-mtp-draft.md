# Brief X — Inkling-2 `mtp.safetensors` → a working speculative-decoding draft (decode is the real bottleneck)

## Why this is the highest-value item left

Decode is ~22 t/s and it is **not** bandwidth- or compute-bound — a sibling lane is profiling why (`docs/mxfp8/22-prefill-perf.md`). What speculative decoding changes is the quantity that matters: tokens per wall-clock second. Inkling ships a real drafter — **54 BF16 tensors, 635,801,416 B, 3 layers, `block_size 8`** — and it fits in the ~12 GB of VRAM slack for free. Getting it wired is worth more than any vector-kernel swap (the `mmvq` attempt measured *slower*: `docs/mxfp8/19-decode-mxfp8.md`).

## Hard constraint: DO NOT BUILD. Another lane owns `build-cuda`

Use the existing binaries in `build-cuda/bin` (`llama-completion`, `llama-perplexity`, `llama-quantize`, `llama-mtmd-cli`). If the draft cannot load without a C++ change, that is expected — produce the exact minimal patch (files, functions, line anchors) in your doc and stop. **Python-only converter work + measurement is the deliverable; the runtime patch is the parent's to land.**

## What the checkpoint ships (measured, don't re-derive)

`mtp.safetensors`, 54 tensors, **all BF16**, and `hf_quant_config.json`'s `mtp` section confirms it: `compute_dtype bfloat16`, `output_dtype bfloat16`, `weight_transform: cast`, `weight_block_shape: []`, `required_activation_quantization: None`. **So the draft repack is dtype-preserving — none of the NVFP4/MXFP8 block-scale convention risk that cost this project a whole night.**

- `fuse_weight [4,4]` — mixes exactly the 4 target layers from `mtp_config.target_layer_ids [5, 23, 41, 48]`
- `mask_embed [2304]`, `norm [2304]` — block-diffusion style drafting control inputs
- `layers.{0,1,2}.` — `input_layernorm`, `post_attention_layernorm`, **`dflare_norm`**, `self_attn.{q,k,v,o}_proj` (3072/1024/1024/2304 shapes), `self_attn.{q,k}_norm`, **`self_attn.k_proj_ctx`**, **`v_proj_ctx`** [1024,2304], **`k_norm_ctx`**, `mlp.{gate,up}_proj [9216,2304]`, `mlp.down_proj [2304,9216]`
- `markov_head.` — `markov_w1 [201024, 256]` (102.9 MB), `rnn_proj [768, 5120]` + bias [768], `hidden_out [2304, 256]` + bias, `dflare_norm`
- `mtp_config`: `hidden_size 2304`, `num_hidden_layers 3`, `num_attention_heads 24`, `num_key_value_heads 8`, `head_dim 128`, `mlp_dim 9216`, `rope_theta 1e6`, `rope_ratio 0.25`, `local_layer_ids [0, 2]`, `sliding_window 1024`, `block_size 8`, `markov_rank 256`, `markov_head_type "rnn_hidden"`, `speculators_version 0.3.0`

## What master provides, and the wall a scout already hit

Master has the runtime families: `src/models/dflash.cpp`, `src/models/dspark.cpp`, 22 `LLM_TENSOR_NEXTN*` entries, and `inkling2-loop.cpp:20` already consults `hparams.n_layer_nextn()`. `conversion/inkling2.py:190` currently **discards** every `mtp.` / `markov_head` / `nextn_predict` tensor, so nothing reaches a GGUF.

**Scout K's measured finding (treat as a starting constraint, verify if you disagree — it was one agent's reading):** the public `dspark-Inkling-2-Mini-Q8_0.gguf` is **not loadable** by master's `llama_model_dflash`:
| check | outcome |
|---|---|
| `markov_w1.weight` present → `VARIANT_DSPARK` | ok |
| `dflash.target_layers` present | ok |
| `backbone = BASE` (no `dsv4_hc_mult`, no `kv_lora_rank`, `sliding_window 1024 > 0`) | ok |
| `markov_w2.weight` required by `dspark.cpp:85` | **absent → load fails** |
| `joint_proj.weight` expected `{n_embd + 2*rank, 3*rank} = {2816, 768}`, file has `{5120, 768}` | **shape mismatch → load fails** |
and repo-wide, `dflare`, `attn_k_ctx`, `attn_v_ctx`, `dspark_fuse`, `hidden_out` have **zero hits** in `src/ ggml/ common/ tools/ conversion/ gguf-py/ tests/`.

That scout concluded the reference file is a *different implementation's* naming convention. Read it as: **the checkpoint's head is richer than master's draft arch**, and the mapping is not 1:1.

## Task

1. **Create `docs/mxfp8/23-mtp-draft.md` within your first 4 tool calls** and keep updating it — a dozen delegated tasks this session timed out with nothing on disk; that is the failure mode to avoid above all.
2. **Build the definitive mapping table**: all 54 checkpoint tensors → the `LLM_TENSOR_NEXTN*` / dflash / dspark slot each would occupy, or `NO SLOT` with the reason. Read `src/models/dflash.cpp` + `dspark.cpp` + the `LLM_TENSOR_*` enum as ground truth, and confirm against the public dspark GGUF's 54 tensor names/shapes (`~/models/unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf`) — 54 vs 54 is a meaningful coincidence worth chasing.
3. **Produce the draft GGUF anyway** — a python converter emitting the best-faith mapping in master's naming, with `general.architecture = dflash` (or `dspark` if the `markov_w1` route is reachable), correct hparams (`n_layer 3`, `n_embd 2304`, heads/`head_dim 128`, `sliding_window 1024`, `rope_theta 1e6`, target layers `[5,23,41,48]`), and **BF16 tensors unchanged in value**. ~636 MB. Validate the values with a per-tensor comparison against the safetensors source (reuse `scripts/inkling_native/` patterns; mind the gguf-py byte-view and `(C, R, E)`-reversed shape traps that bit this project three times).
4. **Try to load it with the existing binary** and record exactly what happens: `llama-completion -m <NATIVE> --model-draft <draft.gguf> -p "..." -n 64 --temp 0 -no-cnv -s 42 --draft 8`. If it loads, report **acceptance rate and decode t/s** (`llama-completion` prints draft stats; look for accepted/drawn counts) versus the 22.24 t/s no-draft baseline. If it doesn't, produce the precise failure line and the minimal C++ patch that resolves it, in the doc.
5. **Decide and recommend, explicitly**, between:
   - **(A) degrade to what master supports** — emit only the tensors that map cleanly (e.g. the 3 attention+MLP blocks), accept losing `dflare_norm`/`*_ctx`/markov-head fidelity, measure whether acceptance rate still earns the complexity; and
   - **(B) extend master's draft arch** (new/extra `LLM_TENSOR_*` + graph) to carry `k_proj_ctx`/`v_proj_ctx`, `fuse_weight`, `dflare_norm` and the markov/RNN head as designed.
   Say which you recommend and what evidence would settle it. **Also check whether `n_gpu_layers`/`--spec-type`/`--spec-draft-n-max` knobs exist on this build** (`llama-completion --help`) and what master's draft plumbing expects for target-layer hidden states (`inkling2-loop.cpp` is a loop model, so target layers 5/23/41/48 are *replay* indices — that's a subtlety worth confirming: which physical layer feeds each).
6. Note for later, don't fight it now: `master..HEAD` also has upstream `c2e70e6ec spec: use draft models with SWA as dense` and `70af88f41 spec: avoid allocating unused K/V cache` (both Inkling-inclusive) — relevant to how a SWA draft behaves, and reason to rebase soon.

## Fences and rules

- You own: `conversion/**` (new draft converter; any `inkling2.py` edit must be additive and default-off so the verified text path cannot regress), `scripts/inkling_native/`, `docs/mxfp8/23-*`, `logs/`, and the draft file under `~/models/timlikesai/`.
- **Do not touch `ggml/**`, `src/**`, `tools/**`, `tests/**`, `common/**`** — read-only for those. **No builds.** If you edit `conversion/inkling2.py`, re-verify the text artifact's inventory is still 40 NVFP4 + 8 MXFP8 + 178 BF16 + 518 F32 = 744 and that `scripts/inkling_native/verify_mxfp8_gguf.py --gguf <file>` exits 0.
- **Never delete a file or log.** Nothing in `~/models/timlikesai/` gets removed. Disk ~250 GB; the draft is 0.64 GB.
- No docker/systemctl. A profiling lane holds the GPU for prefill work — keep your runs short, and if a decode number looks off, note the contention instead of reporting it.
- Snapshot: `/home/tim/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337` (`mtp.safetensors` + `config.json.mtp_config` + `hf_quant_config.json`).
- 40-minute budget; finalize the doc at 34 even if incomplete.
