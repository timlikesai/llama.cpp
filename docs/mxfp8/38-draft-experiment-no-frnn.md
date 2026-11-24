# 38 — Drafting the Inkling stages without `f_rnn`: what it measured

Purpose: turn "the drafter is blocked on a cross-pass channel" into a number, without shipping
anything unfaithful. Method: temporarily route `VARIANT_INKLING` to a draft graph (stages + injection +
recurrent markov head) with `f_rnn` supplied as a **zero block**, run the published
`dspark-Inkling-2-Mini-Q8_0.gguf` against the lossless target, measure, then revert. The revert is
done — `VARIANT_INKLING` refuses to build a graph again, verified.

## Result

| config | prefill | decode |
|---|---|---|
| target only, no drafter (ctx 4096) | 805 t/s | **39.8 t/s** |
| + degraded drafter (f_rnn = 0) | 805.5 t/s | **14.6 t/s** |

So it ran end to end — injection pass, three stages, ctx K/V injection, DFlare per-stage mix, recurrent
head, `hidden_out` readout — and was **2.7× slower than not drafting**. A drafter that accepts nothing
costs its own forward plus a wider verification batch, so this is a direct measurement of two things:

1. **The plumbing is sound.** Nothing in the dflash machinery objected to the Inkling stages once two
   real gaps were closed (below). The remaining distance to drafting is the head's conditioning, not
   the architecture.
2. **`f_rnn` is load-bearing, quantitatively.** Zeroing it destroys acceptance. That retires any idea
   of "ship the simple version first" — a target-context-free Inkling head is not a degraded draft, it
   is a negative optimization. It also says the DFlare design (condition the draft on the target's
   hidden states, not just its own) is doing real work on this model.

## The two real bugs this surfaced (both fixed and committed)

- **`4bad7e855` — the target never published its tapped layers.** First run died at
  `GGML_ASSERT(src != nullptr && "layer input tensor is null")` in `build_layer_inp_dev`. Inkling had no
  equivalent of the `res->t_layer_inp[il] = ...` recording the other drafter-backed arches do. Fixed in
  `build_layer` (the layer's attention input) and in `build_output` (tap id == applied-layer count,
  which is how id 48 reads: the final hidden state before the output norm). Keyed on **slot** index,
  because ids above the physical count name a replayed pass, not a file block.
- **`e30de4d61` — draft logits spanned the padded vocabulary.** Second run died at
  `get_sub_batch: failed to add token 200426 ... (error -2)`. The draft borrows the target's output
  projection (201024 columns for 200058 tokens) and, unlike the Inkling target itself
  (`inkling2.cpp:606`), the dflash path never truncated. Fixed there, plus the markov head's bias width.
  This is not Inkling-specific: any dflash-family draft against a padded target can propose an id the
  tokenizer does not have. Verified no regression on the drafter that works today (z-lab DFlash2 +
  Qwen3.8-27B: byte-identical output).

## What remains for the real head

`f_rnn = mix_rmsnorm(target features of the last committed token, fuse row -1, markov dflare_norm)`.
The draft needs it **in the token pass**, where target features are not currently available. Facts
established by reading the two candidate channels, so the next attempt starts here:

- `cparams.embeddings_layer_inp` is sized `hparams.n_layer() + 1` (`llama-context.cpp:206`) and
  `res->t_layer_inp` is `LLAMA_MAX_LAYERS + 1` — so "the final hidden state" is a first-class tap.
- `supports_embd_row()` (which reads the tap buffer) is gated to `model.arch == LLM_ARCH_DFLASH`
  (`llama-context.cpp:1458`) and works only on **embd** batches: "All entries of the batch must be of
  this kind" (`include/llama.h:1068`). A token batch cannot carry tap rows today.
- `llama_batch_ext_set_embd_state_row` **does** coexist with tokens ("All entries of a batch have a
  state row, or none", `llama.h:1073`), consumed by `build_inp_embd_state()` from
  `ctx_state_src->get_h_nextn_dev()`, but `supports_state_row()` requires
  `ctx_type == LLAMA_CONTEXT_TYPE_MTP` and the width is fixed to `hparams.n_embd_out()`
  (`llama-batch.cpp:1214-1217`) — one vector, not four taps.
- `common.cpp:1256` already creates an MTP-type draft context, and `LLAMA_CONTEXT_TYPE_MTP` maps to
  `LLM_GRAPH_TYPE_DECODER_MTP`, so the embedded path has a home.

Two viable shapes, in order of how much they respect the existing design:

1. **Widen the state channel for this model**: let `llama_batch_ext_select_n_embd_state` return
   `n_embd_out() * n_taps` for a model that declares `mtp.target_layer_ids`, have the Inkling target
   publish the concatenated taps through `res->t_h_nextn`, and let the draft read tap rows in *both*
   passes. Then `f_rnn` and the DFlare mix come from the same tensor, and the drafter is genuinely
   embedded — no separate draft file, no per-model `LLM_TENSOR` fork.
2. **Allow tap rows on token batches** for a DFLASH-arch draft whose `ctx_other` publishes
   `layer_inp_dev`: smallest change to the buffers, but it bends the "all entries or none" batch rule
   and touches `get_sub_batch`, which every speculative path shares.

Either way the validation gate is unchanged and non-negotiable: a numeric oracle against the reference
formulas plus a mutation proof (the `test-dflare-mix` pattern), and **measured acceptance rate** —
never "it generated text", because the target verifies every token and a wrong head is invisible in
output. The 27B reference gives the target to beat: 23.1 → 70.7 t/s (3.06×).

## How to reproduce

```
M=~/models/timlikesai/Inkling-2-Mini-FULL.gguf
D=~/models/unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf
./build-cuda/bin/llama-cli -m $M -md $D --spec-type draft-dspark --spec-draft-n-max 7 \
    -c 4096 -b 2048 -ub 512 -fa on --temp 0 -s 42 --single-turn --no-warmup \
    -f logs/g0-prompt-long.txt -n 64
```
Requires the temporary routing described above; on the current tree it correctly exits 1 with
`Inkling DSpark drafts need the rnn_hidden markov head, which is not implemented`.
Logs: `logs/draft-exp-1.log` (assert on taps), `logs/draft-exp-2.log` (assert on vocab),
`logs/draft-exp-3.log` (ran, 14.6 t/s). Nothing deleted.
