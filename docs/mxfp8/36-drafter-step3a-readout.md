# 36 — Drafter step 3a: the `rnn_hidden` readout, and what `f_rnn` still needs

Part of #39, split deliberately. **Shipped here:** the markov latent's path to the vocabulary
(`dspark_hidden_out` → shared target LM head), which replaces the `markov_w2` matmul Inkling does
not have. **Still blocked, and why:** the recurrent state needs `f_rnn`, which requires a
cross-pass data channel that does not exist yet (§4). The Inkling draft therefore still refuses to
draft — correctly, and that refusal is a gate in `test.sh`.

Files: `src/models/dspark.cpp` (+22/−2). Reference semantics: `33-drafter-reference-semantics.md` §3.

## 1. The change

`build_dspark_markov_head` asserted `w1 && w2`. Inkling has `w1` and **no `w2`** — its latent reaches
the logits through `hidden_out` folded into the target's LM head. Two edits:

```cpp
// the assert becomes conditional: w2 is required only when there is no readout to replace it
GGML_ASSERT(w1 && (w2 || model.dspark_hidden_out) && "DSpark markov weights not loaded");

// and the bias:
if (model.dspark_hidden_out) {
    lat  = build_lora_mm(dspark_hidden_out, markov)      // [rank -> n_embd]
    lat += dspark_hidden_out_b
    bias = build_lora_mm(get_lm_head(...), lat)           // [n_embd -> n_vocab], shared with target
} else {
    bias = build_lora_mm(w2, markov, dspark_markov_w2_s)  // unchanged
}
```

The reference pre-computes `fold_weight = lm_head @ hidden_out.weight` in fp64 and applies it in one
matmul. `l @ Hᵀ @ Uᵀ == l @ (U @ H)ᵀ`, so contracting in two steps is **the same map** — and it needs
no fold at load, no extra 103 MB table, and no change to the target. `get_lm_head` already resolves
the shared-with-target weight (and asserts `ctx_other`), which is how Inkling's draft borrows the
target's head.

*(My first attempt wrote `lm_head @ hidden_out.T`, transcribed literally from the reference line —
that line reflects a differently-stored view. Shapes caught it immediately; the algebra is above.)*

## 2. Measured, because "same map" is a claim about numerics

`scripts/inkling_native/check-readout-fold.py` — real `markov_head.hidden_out` {2304,256} and
`model.llm.unembed` {201024,2304} from the checkpoint, tanh-bounded latents, fp64 exact vs the
graph's two-step f32 path, 1024 rows:

| | mean abs err | max abs err | argmax flips |
|---|---|---|---|
| **two-step f32 (shipped)** | 8.3e-4 | 6.4e-3 | **0 / 1024** |
| fold-then-round-to-bf16 (declined) | 1.0e-3 | 1.4e-2 | 0 / 1024 |

Bias std is 2.14, so the error is **0.3% of the signal spread**, and the **smallest observed top-2
margin is 0.029 — 4.6× the maximum error**, with zero margins falling below it. So a flip is not
merely unobserved, it is margin-protected.

The comparison row is the actual decision: folding at load and storing the table like the other
weights (bf16) is **1.2× worse** than not folding. Two-step is the more accurate choice, not just
the simpler one. (`allow_bf16_reduced_precision_reduction = False`, so this is not a tf32 artifact.)

## 3. Regression evidence, and its honest limit

- The non-Inkling branch is **identical by construction**: `git diff` shows the only change to that
  statement is `ggml_tensor * bias = ` → `bias = `, same expression and args; and with
  `dspark_hidden_out == nullptr` the assert reduces to exactly `w1 && w2`.
- **z-lab DFlash2 on Qwen3.8-27B** (the drafter production actually uses): re-ran the recorded
  command — **generated text byte-identical**, only the perf line differs (71.3 vs 70.7 t/s).
- `test-llama-archs` 591/591; `test-backend-ops -p mxfp8` 3/3 on CPU and CUDA0.
- Inkling draft still refuses with the narrowed reason: `rnn_hidden markov head, which is not
  implemented`.

**Limit, stated rather than buried:** *nothing on this box executes the `markov_w2` path* — the only
DSpark file here is Inkling's, and it refuses before drafting. So `w2` coverage rests on the
mechanical argument above, not on a run. Fix in #41 PR prep: a synthetic GGUF that reaches
`build_dspark_markov_head` both ways, in the style of `tests/test-dflare-mix.cpp`.

## 4. Why `f_rnn` is not a local edit

The head's context input is `[hidden ‖ f_rnn]`, where `f_rnn = mix_rmsnorm(target features of the
last committed token, fuse row -1, dflare_norm_head)`. `hidden` is the draft's own last-layer state —
available in the token pass. **Target features are not**: they arrive only in the *injection*
(`embd`) pass, via `add_embd_row` / `build_inp_features`, and the row index is the token's row in the
target's **last batch**, which only `common/speculative` knows.

Candidate channels:

| route | how | cost |
|---|---|---|
| **A. state-row reuse** | `llama_batch_ext_set_embd_state_row` → `h_nextn_src` → `build_inp_embd_state()` (`llama-graph.cpp:2911-2917`) is exactly a "read row r of another context's device buffer" mechanism, and `llama.h:1208-1209` documents rows `n_batch + s` as *"free for the caller, e.g. for seq s"* | `supports_state_row()` is gated to `LLAMA_CONTEXT_TYPE_MTP`, and its width must match `n_embd_state`. Needs the gating widened + the draft's `t_h_nextn`, **which DSpark already occupies for confidence output** |
| **B. read target features in the token pass** | `layer_inp_src` *is* already available to the draft graph in general (`llama-graph.cpp:1723`, wired at `llama-context.cpp:3067`) — only the row index is missing | add one int per block (the anchor's feature row) to the draft token batch; smallest true change, but it extends the batch ABI |
| **C. compute `f_rnn` in the injection pass and stash it** | injection already has the features | no existing store survives between passes except the KV caches and `t_h_nextn` — i.e. it degenerates into A |

**Recommendation: B.** It keeps `t_h_nextn` free for confidence, it is one index rather than a new
tensor channel, and it makes the dependency explicit ("the drafter needs the anchor's feature row")
instead of piggybacking on an MTP-only path. It also matches how the reference actually reads:
`f_rnn` is a *constant per block*, entering `context` as `f_rnn @ W_f` — an additive term, which
means a later optimization could fold it into `dspark_joint_proj_b` per block, though not before the
index exists.

Whatever route is taken, the gate is the same as #38's: **a numeric test against a transcription of
the reference, plus a mutation proving the test catches a wrong value.** The recurrence itself is
already nearly free — upstream's `is_rnn` path (`dspark.cpp:216`) implements the reference's
`state' = g·state + (1−g)·cand` as `cand + g·(state − cand)` (algebraically identical), zero-inits
state per block, and takes `latent = tanh(chunk₂)`; `build_dspark_markov_head` is called with
`hidden = cur` at `dspark.cpp:895`, which is the precedent for passing the draft hidden in.

## 5. Harness traps hit while doing this (also in `README.md`)

- **`llama-completion` + Inkling throws at startup**: `terminate called ... this custom template is
  not supported, try using --jinja`, from `common_chat_format_example` in the completion setup path.
  **`--no-warmup` does not prevent it** (it is not the warmup call). Use `llama-cli --jinja`, which
  works — but note it puts Inkling into its thinking preamble, so the historic raw-completion oracle
  `A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:` is **not reproducible through the jinja path**. The
  oracle is therefore now a *blessed hash of a fixed command*, not a remembered string — which is
  what `test.sh` implements.
- `--warmup` takes no argument; it is the `--warmup` / `--no-warmup` bool pair.
- A `/tmp`-free rule applies to this work: gate logs live in `logs/gates/`, and `test.sh` never
  deletes them.
- Comparing runs by hashing a *window of lines* is fragile — my first pass included the perf line and
  reported a false "DIFFERS". `test.sh` extracts only the response text and keeps the raw log.
