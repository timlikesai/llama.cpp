# 33 — Inkling DSpark drafter: reference semantics and the implementation blueprint

**Status: fully specified, not yet implemented.** This supersedes the "park MTP — the reference implementation is unretrievable" conclusion in `23-mtp-draft.md` §8, which was wrong. Ground truth found by `gh api search/code`: **vLLM ships the reference implementation** at `vllm/models/inkling2/nvidia/dspark.py` + `ops/dspark.py` (also `ollama`, `sglang/srt/speculative/dspark_components`, `modular/max`, `jundot/omlx`). Copies of both files are in the session search cache; the math is reproduced below so this doc stands alone.

Correction of record: I searched `transformers` and `vllm-project/speculators` tags, found `rnn_hidden` nowhere, and concluded the semantics were unrecoverable — while holding a tensor named `dflare_norm`. The name was the pointer: **DFlare** is a published method (arXiv 2606.02091, `W_fuse ∈ R^{D×T}`, `α⁽ⁱ⁾ = softmax(W_fuse, i, :)`), and DSpark is DFlash + a Markov logit-bias head. Lesson logged in `00-status.md`: a name I cannot explain is a search term, not a dead end.

## 1. What it is

> *"an anchor plus mask block run through a small dense transformer over DFlare-mixed target-feature K/V, then a recurrent (`rnn_hidden`) Markov head"*

3 dense draft layers (hidden 2304, 24 q heads / 8 kv heads, head_dim 128, MLP 9216, no MoE), drafting a block of 8 in parallel, conditioned on target features tapped at layers `[5, 23, 41, 48]` and mixed **per draft layer** (DFlare), plus a GRU-style rank-256 recurrent head that biases logits per block step. `embed_tokens`, `embed_norm` and `lm_head` are **shared with the target** (hence absent from the draft weights).

## 2. Shape reconciliation — every checkpoint tensor now has a declared role

| checkpoint (`mtp.safetensors`) | vLLM module | GGML shape (in, out) | role |
|---|---|---|---|
| `fuse_weight [4,4]` | `fuse_weight` → `fuse_alpha = softmax(fp32)` row per draft layer (row `-1` = markov head) | `[4,4]` F32 | **DFlare layer-wise mixing** of the 4 target taps |
| `mask_embed [2304]` | `mask_embed` | `[2304]` | block slots 1..7 input (slot 0 = anchor token) |
| `layers.N.dflare_norm` | `Inkling2DraftLayer.dflare_norm` | `[2304]` | RMSNorm on that layer's mixed context |
| `layers.N.{k,v}_proj_ctx`, `k_norm_ctx` | `store_context_kv` path | `[1024,2304]`, `[128]` | target-context K/V injected into the draft cache |
| `markov_head.markov_w1 [201024,256]` | `nn.Embedding(padded_vocab, rank)`, **frozen** | `[256, 201024]` | previous-token → rank state |
| `markov_head.rnn_proj [768,5120]` | `ReplicatedLinear(2r + 2h, 3r)` | `[5120, 768]` | **== unsloth's `joint_proj`** (rename only) |
| `markov_head.hidden_out [2304,256]` | `ReplicatedLinear(r, h)` | `[256, 2304]` | latent → hidden, **folded into `lm_head`** |
| `markov_head.dflare_norm` | `Inkling2MarkovHead.dflare_norm` | `[2304]` | norm for `f_rnn` context |

`5120 = 2·rank + 2·hidden = 512 + 4608` ✓ and `768 = 3·rank` ✓ — the two mismatches I earlier read as "different architecture" were `rnn_proj`↔`joint_proj` naming and `2·hidden` vs upstream's `n_embd`.

## 3. Exact math (from the reference kernels)

**Mixing** (`mix_rmsnorm`, fp32 accumulate, bf16 out):
```
mixed = Σ_taps alpha[tap] * x[tap]        # taps = 4 target features, rows of (B, taps, hidden)
out   = mixed * rsqrt(mean(mixed²) + eps) * weight
```

**Block input** (`embed_block`): `x = mask_embed` broadcast to `(rows, block, hidden)`; `x[:, 0] = embed_norm(embed_tokens(anchor))`.

**Draft layer**: `input_layernorm` → attention (per-head Q/K RMSNorm rounded to bf16, **neox** RoPE on leading dims; context K/V precomputed once per layer from mixed features) → MLP `silu(gate)*up` in fp32, rounded once → `post_attention_layernorm` with residual. Final: `norm(x, residual)`.

**Recurrent head** — context computed once per step-batch from `[hidden, f_rnn]`:
```
context = rnn_bias + [hidden ‖ f_rnn] @ rnn_proj.weight[:, 2r:].T          # fp32
```
then per block step, with state initialized to zeros:
```
inputs  = [state ‖ markov_w1(prev_token)]                                   # 2r
raw     = context + inputs @ rnn_proj.weight[:, :2r].T                       # fp32, (rows, 3r)
g       = bf16(1 / (1 + expf(-bf16(raw[0:r]))))          # IEEE division
cand    = bf16(tanh(bf16(raw[r:2r])))
state'  = bf16( bf16(g*state) + bf16(bf16(1-g) * cand) )
latent  = bf16(tanh(bf16(raw[2r:3r])))
bias    = fold_bias + latent @ fold_weight.T
```
with the fold done once in **fp64**: `fold_weight = lm_head @ hidden_out.weight.T` `(vocab, r)`, `fold_bias = lm_head @ hidden_out.bias`.
Logits combine via softcap: **`out = softcap * tanh(bf16(base + bias) / softcap)`** — this is `final_logit_softcapping = 30.0` in the draft metadata.

Numerics notes that matter for validation: gate ops round to bf16 at each torch step (`_round_bf16` does the RN-even bit trick explicitly), `rank` must be a power of two (256 ✓), and the reference keeps `rnn_proj` math in fp32 while base logits are bf16.

## 4. Gap against upstream `dflash`/`dspark` (what code must change)

Upstream already has the block-drafting loop, `mask_embd`, target-feature taps, `build_inject()` KV injection, iswa cache routing, low-rank `markov_w1` as `GET_ROWS`, `--spec-type draft-dspark`, and shared-target-embedding support via `ctx_other`. Four deltas:

| # | delta | files | size |
|---|---|---|---|
| 1 | `markov_w1` vocab: file ships padded 201024, loader expects 200058; both are declared (`dflash.vocab_size` / `dflash.vocab_size_unpadded`) | `src/models/dspark.cpp:84` | trivial |
| 2 | per-layer **context** projections: upstream injects via the layer's own `wk`/`wv`; Inkling has separate `attn_k_ctx`/`attn_v_ctx` + `attn_k_ctx_norm` | `include`(arch), `src/llama-arch.cpp/.h`, `src/models/dflash.cpp:299-320` | small |
| 3 | **DFlare fusion**: replace the single shared `fc {9216,2304}` + `output_norm_enc` with per-draft-layer `softmax(fuse[i,:])` mixing of the 4 tapped features + per-layer `dflare_norm` | `src/models/dflash.cpp:223-236` (`build_inject_inp`), `build_inp_target_features`, `src/llama-model.cpp` tensor load | medium |
| 4 | **`rnn_hidden` head**: upstream's `markov_w2` readout is replaced by the GRU-style recurrence + `hidden_out` folded into the target LM head + per-step state | new code in `src/models/dspark.cpp` graph; runtime state carry in `common/speculative.cpp` draft path | the real work |

Plus: 4-6 new `LLM_TENSOR_*`/`LLM_KV_*` entries, a `markov_head_type` hparam so `dflash`/`dspark`/`rnn_hidden` variants dispatch explicitly, and a converter emitting the contract (§5).

## 5. Converter

`scripts/inkling_native/convert_mtp_draft.py` already maps 54 tensors bit-exact BF16 and writes correct metadata (verified: `architecture=dflash`, `block_count 3`, `block_size 8`, `target_layers [5,23,41,48]`, `dspark.markov_rank 256`, `swa_pattern [T,F,T]`). It needs to emit the §2 names/axes for whichever contract we implement — and `rnn_proj` must **not** be renamed blindly to upstream `joint_proj` unless we also adopt its `[2r+2h]` input order (it is the same tensor: keep the name the loader expects).

## 6. Validation gates (a wrong head is *silent* — algorithmic output-invariance means bad drafting still yields correct text)

1. **Acceptance rate** is the primary metric: `llama-server`/`llama-completion` spec stats — accepted-per-step should approach what a trained block-8 drafter earns (start `--spec-draft-n-max 3`, compare to 7). A near-zero acceptance rate means the math is wrong, even if everything loads and generates.
2. Decode speed vs the 40.4 t/s baseline at 8K, and at 128K (where drafter KV costs ~1.5 GiB f16 / 0.75 fp8).
3. Greedy output **need not be bit-identical** — verified empirically this session across `--spec-type` values, and unsloth documents the same ("the target's batched verification can flip near-tie tokens on CUDA. It is not a drafter defect"). Use PPL/KL and the 256K needle instead.
4. Regression: `test-llama-archs` (591), `test-backend-ops -p mxfp8` on CPU/CUDA0/CUDA1, lossless-file correctness string, and the **z-lab `Qwen3.8-27B-DFlash2`** draft — it works on this box and shares `dflash`/`dspark` code paths, so it is the ready-made guard that upstream drafters keep loading and drafting.
5. Memory: at 8K there is 8.9 GiB free (CUDA0 6,582 + CUDA1 2,360); at 256K under 1 GiB, so long-context drafting needs `-ub`/`-b` cuts or fp8 draft KV.

## 7. Sequencing

(1) §4 rows 1+2 → the file at least *loads*; (2) row 3 DFlare fusion, verified against a mixed-features reference computed from the checkpoint in numpy; (3) row 4 head, with `markov_gate` reimplemented and unit-tested against the §3 formulas on random inputs (the rounding points are explicit, so this is testable without the model); (4) acceptance measurement; (5) only then judge speed. One commit per row, in repo style.

---

## 9. Resolved 2026-11-22 — what `f_rnn` is, and why `fuse_weight` is [4,4] for 3 layers

From `ref/dspark.py` (vLLM `vllm/models/inkling2/nvidia/dspark.py`), verified verbatim:

```python
def compute_f_rnn(features):                      # features = last COMMITTED token's raw target taps
    return model.mix(features, -1, model.markov_head.dflare_norm)

def precompute_and_store_context_kv(states, positions, slots):
    for i, layer in enumerate(model.layers):
        context = model.mix(states, i, layer.dflare_norm)      # per-stage injection
        layer.self_attn.store_context_kv(context, positions, ...)
```

Consequences, each of which removes a previous unknown:

1. **`fuse_weight` row 3 belongs to the markov head, not to a draft stage.** That is why the
   matrix is `[4,4]` (4 draft-layer rows x 4 target taps) while `block_count == 3`. Steps 1/2 use
   rows 0..2 for the stages; row -1 is the head's mix. The extra `dflare_norm` in
   `model.markov_head` is this row's norm, which previously looked redundant.
2. **`f_rnn` is target-derived, not a carried draft hidden state.** It is the DFlare mix
   (`softmax(fuse[3])` + `markov_head.dflare_norm`) of the *same* tap features used for
   injection, evaluated at the last committed token.
3. **Placement follows from (2):** anywhere the draft graph can see the current token's taps, it
   can compute `f_rnn` — the injection pass qualifies. No new cross-pass channel is needed, which
   retires the "route A/B/C" discussion in doc 36 and the widening of `h_nextn` contemplated for
   the embedded route.
4. The head's `context` argument is `rnn_proj.weight[:, 2*rank:]` applied to
   `[hidden, f_rnn.expand_as(hidden)]` (fp32) — note `expand`, so both halves of the `2*hidden`
   slice receive the *same* `f_rnn`, matching the stored `[768, 5120]` shape where
   `5120 = 2*rank + 2*n_embd`.

## 10. Packaging decision (2026-11-22)

Measured reference for a working embedded nextn draft: qwen35 27B, `--spec-type draft-mtp`,
**25.1 -> 43.9 t/s decode (1.75x)**, prefill 157.4 -> 139.7 (-11%).

Two packaging routes, both with the same model-side head work:

| route | host changes needed | why it is/isn't chosen |
|---|---|---|
| separate dflash-arch draft file + `--spec-type draft-dspark` | **none** — `common/speculative.cpp:1119` already enables target tap export from the draft's declared `target_layer_ids`, and the dflash impl already runs injection + block stages + sampling | chosen: it exercises machinery proven by the z-lab drafts (3.06x measured) and isolates risk to `src/models/` |
| embedded nextn + `--spec-type draft-mtp` | yes — tap enablement and the injection pass exist only in the EAGLE3/DFlash draft classes (`:604`, `:1119`), not in the MTP class (`:3115` sets `ctx_other` but never enables taps) | deferred: needs a second graph pass per draft step from the host, i.e. a `speculative.cpp` feature. FULL.gguf keeps the embedded copy either way (verified byte-exact), so nothing is lost by drafting from the separate file first |
