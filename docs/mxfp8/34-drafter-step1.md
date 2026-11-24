# 34 — Drafter step 1 (todo #37): variant-gated loader + per-layer ctx tensors

Blueprint: `33-drafter-reference-semantics.md` §4 rows 1+2, sequenced by §7 step 1.
Goal: `dspark-Inkling-2-Mini-Q8_0.gguf` gets PAST tensor loading (`check_tensor_dims`)
without changing behaviour for any existing draft (z-lab DFlash2 is the regression guard).

## Plan (done — implementation summary in Result)

1. [x] Facts: dump every metadata key + tensor name/shape/type of the Inkling draft file, and the
      DFlash2 draft file, with `gguf-py`. -> `docs/mxfp8/drafter-draft-tensors.txt` (note: `GGUFReader.Tensor.shape`
      **is** the GGML `ne` order; reversing it gives the torch order, which is what doc 33 §2 quotes).
2. [x] Variant discriminator: inferred from tensor presence, no new KV key. `dspark_hidden_out.weight`
      present + `markov_w1.weight` present => `VARIANT_INKLING` (`src/models/dflash.cpp:45`).
3. [x] Loader under the Inkling variant: `markov_w1` at the stored (padded) rows, `markov_w2` NOT_REQUIRED,
      `joint_proj` at `{2*rank + 2*n_embd, 3*rank}` (`src/models/dspark.cpp:82-116`).
4. [x] Per-layer context tensors `attn_k_ctx` / `attn_v_ctx` / `attn_k_ctx_norm` (+ per-layer `dflare_norm`):
      new `LLM_TENSOR_*`, name table, `llama_layer` fields, declared NOT_REQUIRED in
      `load_tensors_layers_base`, and `graph_base::build_inject()` prefers them, falling back to `wk`/`wv`
      (`src/models/dflash.cpp:164-168`, `315-334`).
5. [x] Loud failure: `build_arch_graph` throws for `VARIANT_INKLING` before any graph type is built, so the
      encoder graph is refused too (`src/models/dflash.cpp:554-559`). Verified below.
6. [x] Gates: all green, see Result.

## Facts (parent, verified — do not re-derive)

1. **The file has NO markov-head-type key.** Full KV dump: `dflash.{attention.{causal False, head_count 24, head_count_kv 8, key_length 128, value_length 128, layer_norm_rms_epsilon ~1e-6, sliding_window 1024, sliding_window_pattern [True,False,True]}, block_count 3, block_size 8, context_length 1048576, dspark.markov_rank 256, embedding_length 2304, feed_forward_length 9216, final_logit_softcapping 30.0, has_confidence_head False, rope.{dimension_count 128, freq_base 1e6}, sample_from_anchor False, target_layers [5,23,41,48], vocab_size 201024, vocab_size_unpadded 200058}` + `general.{architecture dflash, basename/name Inkling-2-Mini, file_type 7, size_label 318M, type model, quantization_version 2}`.
2. **=> the variant must be INFERRED from tensor presence**, not metadata. Recommended discriminator: `ml->get_tensor_meta("dspark_hidden_out.weight") != nullptr` (or `dspark_fuse.weight`) => Inkling `rnn_hidden` head. Adding a new KV key would not make the *published* file load, which is the gate.
3. **Why upstream expects 200058**: `src/models/dflash.cpp:502` uses `model.vocab.n_tokens()` — the draft borrows the TARGET's vocab (200058 unpadded). But vLLM's reference builds `markov_w1 = nn.Embedding(config.padded_vocab_size, rank)` => the stored table is 201024. **Accepting the stored (padded) size is correct, not a hack**: token ids (< 200058) are a strict subset of those rows. So size `markov_w1` from `ml->get_tensor_meta("markov_w1.weight")->ne[1]` the same way the existing code already reads `ne[0]` for the rank at `src/models/dspark.cpp:82` — no slicing, no reinterpretation.
4. `hparams.n_vocab_unpadded` / `n_vocab_padded` are an **Inkling-target-specific** pair (read in `src/models/inkling2.cpp:19`, validated at `:41`, sliced at `:455`); `LLM_KV_VOCAB_SIZE_UNPADDED` exists in the arch table (`src/llama-arch.cpp:223`). The dflash arch loader does not consult it.
5. `n_vocab_draft` in `dflash.cpp:95-100` equals `n_vocab` unless a `d2t` tensor exists — Inkling's file has **no `d2t`**, so `n_vocab_draft == n_vocab` (200058). Upstream's required `markov_w2 {rank, n_vocab_draft}` has no counterpart in Inkling's head (see §3 of doc 33: `hidden_out` is folded into the LM head instead).
6. Upstream's `dspark` tensor set is only `markov_w1 / markov_w2 / conf_proj / joint_proj / mask_embd` (`src/llama-arch.cpp:910-914`). Inkling's file additionally carries `dspark_fuse`, `dspark_norm`, `dspark_hidden_out{.weight,.bias}` and per-layer `dflare_norm` + `attn_{k,v}_ctx` + `attn_k_ctx_norm` — these are currently **undeclared**, so declare them (as NOT_REQUIRED outside the variant) or the unused-tensor path will complain once loading proceeds.

## Design decisions (parent)

- Variant inference per fact 2; **no new required metadata**, so the published file loads unmodified.
- Shapes for the Inkling variant come from the stored tensor metadata (facts 3, 6): `markov_w1 {256, 201024}`, `joint_proj {5120, 768}` = `[2*rank + 2*n_embd, 3*rank]` per doc 33 §2 — do NOT rename or transpose anything.
- `markov_w2` NOT_REQUIRED under the variant (fact 5).
- Per-layer ctx tensors become new `LLM_TENSOR_*` entries used in `graph_base::build_inject()` when present, falling back to `wk`/`wv` otherwise.
- **Fail loudly where the head is not yet implemented** (DFlare fusion #38, recurrence #39): load may succeed, but drafting under the Inkling variant must abort at graph-build with a message naming the missing step. A silently-wrong drafter is the failure mode to avoid, because speculation hides it (doc 33 §6).

## Result

**Done.** `dspark-Inkling-2-Mini-Q8_0.gguf` now loads all 54 tensors, and refuses to draft with a named
reason. Existing drafts are provably untouched (measured, see gates).

### What changed (7 files, +99/-12)

| file | change |
|---|---|
| `src/llama-arch.h` | 7 new `llm_tensor`s: `ATTN_K_CTX`, `ATTN_V_CTX`, `ATTN_K_CTX_NORM`, `DSPARK_FUSE`, `DSPARK_NORM`, `DSPARK_HIDDEN_OUT`, `DFLARE_NORM` |
| `src/llama-arch.cpp` | name table (`blk.%d.attn_k_ctx`, `blk.%d.attn_v_ctx`, `blk.%d.attn_k_ctx_norm`, `dspark_fuse`, `dspark_norm`, `dspark_hidden_out`, `blk.%d.dflare_norm`) + `LLM_TENSOR_INFOS` (MUL_MAT for the projections, MUL for the norms) |
| `src/llama-model.h` | `llama_layer.attn_k_ctx/attn_v_ctx/attn_k_ctx_norm/dflare_norm`; `llama_model.dspark_fuse/dspark_norm/dspark_hidden_out/dspark_hidden_out_b` |
| `src/models/models.h` | `variant_type::VARIANT_INKLING` |
| `src/models/dflash.cpp` | variant inference from `dspark_hidden_out.weight`; `fc` + `enc.output_norm` NOT_REQUIRED under the variant; Inkling per-layer tensors declared; `build_inject()` uses the ctx K/V when present; `build_arch_graph` throws for the variant |
| `src/models/dspark.cpp` | `markov_w1` rows from stored meta, `markov_w2` NOT_REQUIRED, `joint_proj` `{2r+2h, 3r}`, and the 5 Inkling head tensors declared under the variant |
| `src/llama-model.cpp` | `llama_model_dflash_variant` maps `VARIANT_INKLING -> LLAMA_DFLASH_VARIANT_DSPARK` (the public enum is unchanged; `--spec-type draft-dspark` selects it and then hits the refusal) |

### Decisions I made where the plan was silent

- **`fc` and `enc.output_norm` had to become optional.** They are required today and the Inkling file has
  neither, so the load failed on `fc.weight` immediately after `markov_w1` was fixed. Not in the plan's fact list.
- **`mask_embd` had to be declared for the base backbone.** It is only declared in the Mistral Large 4 path
  (`src/models/dspark.cpp:159`), and `llama_model_loader::done_getting_tensors()` (called with `partial = false`
  from `src/llama-model.cpp:2042`) throws unless *every* tensor in the file is accounted for. The Inkling file
  stores it 1-D as `{2304}` (one shared mask vector), not `{n_embd, block_size-1}`.
- **Inkling-only head tensors are REQUIRED inside the variant, NOT_REQUIRED outside it.** Their presence is what
  defines the variant, and a missing `dspark_fuse` should fail with "missing tensor", not as a null deref in #38/#39.
- **Shapes come from hparams where doc 33 §2 gives a formula** (`dspark_fuse {n_taps, n_layer+1}`, `hidden_out
  {rank, n_embd}`, norms `{n_embd}`), and from stored meta only for `markov_w1.ne[1]`, per fact 3.
- **Refusal lives at the top of `build_arch_graph`**, not in the decoder switch, so `LLM_GRAPH_TYPE_ENCODER`
  (which would deref the now-null `fc`) is refused as well. `throw std::runtime_error` rather than `GGML_ABORT`:
  `llama_init_from_model` catches it and the tools exit with a readable error and code 1.
- Known interaction, left alone: a file that is *both* Mistral-Large-4-backbone and Inkling-head would declare
  `mask_embd` twice and trip "too many tensors created". Impossible today (the two discriminators are exclusive),
  and the failure is loud.

### Facts corrected / added

- `GGUFReader.Tensor.shape` is GGML `ne` order (doc 33 §2 quotes the torch order; both are right).
- The Inkling draft file carries the full tokenizer (200058 tokens), so `n_vocab` in the loader is
  `vocab.n_tokens()` = **200058** (`src/llama-model.h:1178`), while `dflash.vocab_size` says 201024. That is the
  whole `markov_w1` mismatch; nothing about it needs a new KV key.
- `unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf` does **not** load on this branch at all
  (`check_tensor_dims: tensor 'blk.24.attn_q.weight' not found` - its `inkling2` metadata advertises 48 layers
  with `loop.{start 6,end 18,count 3}` but ships 24 physical layers). Pre-existing, unrelated to this step, and
  already flagged "stale revision, unloadable" in `code/services/docker-compose.yml`. Use the NATIVE lossless file
  as the Inkling target.
- `timlikesai/Inkling-2-Mini-MTP-DFLASH.gguf` (our own converter output, 47 tensors) has **no** `markov_w1`,
  `joint_proj`, `dspark_fuse` or `dspark_hidden_out`, so it is still `VARIANT_DFLASH` and still fails on the
  required `fc.weight`. Unchanged by this step - it is the #38 (DFlare) contract question, not #37.

### Gates (all on the final build)

| gate | result | evidence |
|---|---|---|
| build | green, no new warnings (`cmake --build build-cuda -j 16`, exit 0) | `build-drafter-step1.log`, `build-drafter-step1-rebuild.log` |
| Inkling draft loads | `load_tensors_dspark: DFlash with DSpark markov head (rank = 256)`, `CPU_Mapped ... 372.26 MiB` = every tensor in the file; no shape error, no "wrong number of tensors" | `docs/mxfp8/log-drafter-step1-inkling-draft-verbose.txt` |
| Inkling refuses to draft | `llama_init_from_model: failed to initialize the context: Inkling DSpark drafts need DFlare tap mixing and the rnn_hidden markov head, which are not implemented`, tool exits 1 | `docs/mxfp8/log-drafter-step1-inkling-pair.txt` |
| baseline (pre-change) Inkling load | `check_tensor_dims: tensor 'markov_w1.weight' has wrong shape; expected 256, 200058, got 256, 201024` | `docs/mxfp8/log-drafter-baseline-inkling-draft-load.txt` |
| **DFlash2 regression** (z-lab `Qwen3.8-27B-DFlash2-Q4_K_M` + `Qwen3.8-27B-UD-Q4_K_XL`, `-sm layer -fa on -ngl 99 -c 8192`, KV q4_1, `--spec-type draft-dflash2 --spec-draft-n-max 7`, temp 0, seed 42, 128 tokens) | before **and** after: `draft acceptance = 0.72603 (106 accepted / 146 generated), mean len = 6.05`, 97.6 vs 97.6 t/s, greedy output sha1 identical (`07c3d88b8094f06b71a94d71733baa33db9c0b73`) | `docs/mxfp8/log-drafter-{baseline,step1}-dflash2-{server,request}*.txt` |
| `test-llama-archs` | `Summary: all 591 test(s) passed` | `docs/mxfp8/log-drafter-step1-test-llama-archs.txt` |
| `test-backend-ops -p mxfp8` | CUDA0 237/237, CUDA1 237/237; `-b CPU` 306/306 | `docs/mxfp8/log-drafter-step1-backend-ops-cpu*.txt` |
| Inkling lossless file still answers | `Inkling-2-Mini-NATIVE.gguf` generates (needs `--jinja`; its custom template aborts without it, pre-existing) | `docs/mxfp8/log-drafter-step1-inkling-lossless-answer.txt` |

The before/after numbers came from rebuilding the tree with the change stashed (`git stash push -- src`),
measuring, then popping; the restored diff is byte-identical to `docs/mxfp8/drafter-step1.patch`.

### Ready for #38 / #39

Step 2 replaces `build_inject_inp` (`fc` + `output_norm_enc`) with per-stage `softmax(dspark_fuse[i,:])` mixing
+ `layer.dflare_norm`; step 3 replaces `build_dspark_markov_head`'s `markov_w2` readout with the recurrence over
`[state; prev; hidden; f_rnn]` (`joint_proj` is already declared at that width) and the `hidden_out` fold; both
then delete the refusal at `src/models/dflash.cpp:554-559`. The ctx-KV path step 2/3 needs is already live in
`graph_base::build_inject()`.

## Parent verification (independent re-run of every gate, 2026-11-21)

Committed as **`6e89360b6 model : load the Inkling DSpark draft`** (17th on the branch), after confirming the working
tree was byte-identical to `drafter-step1.patch` and that the lane's stash/unstash cycle left the earlier #30/#32/#31
work intact.

| gate | parent's own run |
|---|---|
| build | incremental no-op, green |
| Inkling draft | **no `check_tensor_dims` error**; loads every tensor (`markov head (rank = 256)`), then refuses at graph build with the intended message |
| correctness | `A: 68 B: 64 C: 72 D: 70` (CUDA) and `...E: 66 F: 69 G:` (CPU) — unchanged |
| `test-llama-archs` | 591/591 |
| `test-backend-ops -p mxfp8` | CPU OK, CUDA0 OK |
| **DFlash2 regression** | z-lab `Qwen3.8-27B-DFlash2-Q4_K_M` + `Qwen3.8-27B-UD-Q4_K_XL`, `-ngl 99 -sm layer -fa on`: **drafts fine** |

**Note on the `dflash requires ctx_other to be set` error line**: it appears at the *start* of every dflash-draft run,
including the healthy DFlash2 one — it is llama.cpp's first-pass probe before `ctx_other` is attached, not a failure.
It initially read as a regression; the before/after comparison is what settled it.

**The number that reframes #38/#39 priority** (`logs/dflash2-regression.log` + the `--spec-type none` control, same
model, same flags, same prompt):

| | prompt t/s | **decode t/s** |
|---|---|---|
| Qwen3.8-27B, no draft | 207.3 | **23.1** |
| Qwen3.8-27B + DFlash2 draft | 174.7 | **70.7** |

**3.06x decode** from a working block drafter on this hardware class, at a small prefill cost. Inkling's decode is
launch-bound for the same structural reason (~384 tiny expert GEMVs per token), which is the strongest argument yet
for finishing the drafter rather than treating it as a stretch goal. Inkling's head is 3 layers / 636 MB BF16 vs the
27B's own drafter, and it borrows the target's embeddings and LM head — so the same multiplier is a reasonable
expectation, to be confirmed by measured acceptance rate in #40, not assumed.
