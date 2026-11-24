> [!NOTE]
> Archived 2026-11-21: superseded by `../33-drafter-reference-semantics.md`. The synthesized-`fc` idea assumed
> Inkling's head had no retrievable semantics; the reference implementation is in vLLM
> (`vllm/models/inkling2/nvidia/dspark.py`, `ops/dspark.py`) and `fuse_weight [4,4]` is DFlare's layer-wise mixing
> (`alpha = softmax(fuse_weight)`), so no invented projection is needed. Its failure inventory is still useful.

# 30 - Synthesize dflash `fc.weight` from Inkling `fuse_weight`, measure acceptance vs the 40 t/s baseline

Status: **running.** Lane for `docs/mxfp8/23-mtp-draft.md` §5d path (1): the plain-dflash draft
failed to load with `check_tensor_dims: tensor 'fc.weight' not found`; this lane builds that
missing tensor from the checkpoint and measures what the degraded draft is worth.

## 0. The reference implementation was found (this changes §5d's premise)

`23-mtp-draft.md` §5d assumed `fuse_weight` is used raw. It is **not** - and there is a complete
reference forward for this exact head on Hugging Face:
`Lewfkrad/Inkling-2-Mini-DSpark-sm120-vllm-patch:dspark.py` (vLLM, Apache-2.0, mirrors the
training-side reference). Saved to `docs/mxfp8/ref/dspark.py` (with the wheel sources
`speculators-0.3.0`, `speculators-0.9.0` for the upstream dflash/dspark contract).

Key lines (verbatim from `dspark.py`):

- `:447` `self.fuse_weight = nn.Parameter(torch.zeros(num_hidden_layers + 1, len(target_layer_ids)))`
  -> the `[4,4]` is **3 draft layers + 1 markov row** x 4 target layers, not a square layer-mix.
- `:550` `fuse_alpha.copy_(torch.softmax(self.fuse_weight.float(), dim=-1))`
  -> the parameter is a **logit** matrix; the mixing coefficients are its row-wise **softmax**.
- `:466` `mix(features, row, norm) = mix_rmsnorm(targets.view(n, -1, hidden), fuse_alpha[row], norm.weight, norm.eps)`
  -> mix the 4 target features with one softmax row, then RMSNorm with **that consumer's own
  `dflare_norm`**.
- `:568` per layer `i`: `context = mix(context_states, i, layer.dflare_norm)` -> `store_context_kv`
  -> `:300` `k, v = linear(context, kv_proj_ctx.weight)`, `k = k_norm_ctx(k)`, RoPE, write to cache.
- `:578` markov head: `mix(features, -1, markov_head.dflare_norm)` - the 4th row feeds the RNN, not the KV.
- `:461` `embed_block`: slot 0 = `embed_norm(embed_tokens(anchor))`, slots 1..7 = `mask_embed` (one
  shared vector, not per-slot).
- `:521` `num_speculative_tokens must == block_size - 1` (= 7).

Consequence for §5d(1): **the block-form `fc` is valid, but must be built from `softmax(fuse_weight)`,
not from the raw parameter**, and one `fc` can only be faithful to one of the 4 rows.

TODO: table of what maps / what is lost, with the two kv-path variants.

## 1. Converter defect found while building it: every tensor was transposed

TODO (details: `GGUFWriter.add_tensor`'s `raw_shape` is the *torch* order - `gguf_writer.py:272`
writes `shape[n_dims-1-j]`, i.e. it reverses - so passing `reversed(hf_shape)` double-reverses).

## 2. What master's dflash actually needs (measured from the loader, not inferred)

TODO

## 3. Artifacts built

TODO

## 4. Numerical verification

TODO

## 5. Acceptance + decode measurement vs the 40 t/s baseline

TODO

## 6. Verdict and recommendation

TODO

## 7. Corrections to my brief

TODO
