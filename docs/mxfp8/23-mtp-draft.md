# 23 - Inkling-2 MTP head -> loadable speculative-decoding draft

Status: **artifact built and load-tested by the parent; §2-§4 left PENDING by the lane, §5/§5b/§5c/§6 filled in from the artifact and the measured failure.**

## 1. Sources on disk

- Checkpoint head: `~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/mtp.safetensors`
- Reference draft: `~/models/unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf`
- Runtime ground truth: `src/models/dflash.cpp`, `src/models/dspark.cpp`, `src/llama-arch.cpp` (`LLM_TENSOR_NEXTN*`)

## 2. Tensor mapping table

Authoritative source is the code: `LAYER_MAP` / `ROOT_MAP` in `scripts/inkling_native/convert_mtp_draft.py` (~lines 30-62). Reconstructed and verified against the 54 BF16 tensor names in the emitted file:

| checkpoint (`mtp.safetensors`) | GGUF name | note |
|---|---|---|
| `layers.N.self_attn.{q,k,v,o}_proj.weight` | `blk.N.attn_{q,k,v,output}.weight` | 3072x2304 / 1024x2304 / 1024x2304 / 2304x3072 |
| `layers.N.self_attn.{k,v}_proj_ctx.weight` | `blk.N.attn_{k,v}_ctx.weight` | 1024x2304 — the target-context K/V path |
| `layers.N.self_attn.{q_norm,k_norm,k_norm_ctx}.weight` | `blk.N.attn_{q,k}_norm`, `attn_k_ctx_norm` | 128 each |
| `layers.N.mlp.{gate,up,down}_proj.weight` | `blk.N.ffn_{gate,up,down}.weight` | 9216x2304 / 9216x2304 / 2304x9216 |
| `layers.N.{input,post_attention}_layernorm`, `dflare_norm` | `blk.N.attn_norm`, `ffn_norm`, `dflare_norm` | 2304 each |
| `fuse_weight` [4,4] | `dspark_fuse.weight` | **no fitting counterpart** — §8 blocker 1 |
| `mask_embed`, `norm.weight` | `mask_embd.weight`, `output_norm.weight` | 2304 each |
| `markov_head.markov_w1.weight` [201024,256] | `markov_w1.weight` | **layout mismatch** — §8 blocker 2 |
| `markov_head.rnn_proj.{weight,bias}` [768,5120] | `joint_proj.{weight,bias}` | **concat mismatch** — §8 blocker 3 |
| `markov_head.hidden_out.{weight,bias}` [2304,256] | `dspark_hidden_out.{weight,bias}` | no required counterpart |
| `markov_head.dflare_norm.weight` | `dspark_norm.weight` | 2304 |

## 3. Metadata checklist

Verified by reading the artifact back: `general.architecture = dflash`, `general.file_type = 32` (a **llama ftype**, not a ggml type), `general.name`, plus `dflash.{block_count 3, block_size 8, target_layers [5,23,41,48], dspark.markov_rank 256, embedding_length 2304, feed_forward_length 9216, attention.{head_count 24, head_count_kv 8, key_length 128, value_length 128, sliding_window 1024, sliding_window_pattern [T,F,T], causal False, layer_norm_rms_eps ~1e-6}, rope.{freq_base 1e6, dimension_count 128}, context_length 1048576, vocab_size 201024, final_logit_softcapping 30.0, sample_from_anchor False, has_confidence_head False}` and non-standard `inkling.mtp.markov_head_type = rnn_hidden`. Tokenizer and a few `general.*` keys are copied verbatim from the reference dspark draft (`COPY_FROM_REF`) because llama.cpp validates some on load. Note this `sliding_window_pattern` is the **draft's own** 3-entry pattern derived from `mtp_config.local_layer_ids [0,2]` — unrelated to the target's 24-entry array.

## 4. Draft converter

`scripts/inkling_native/convert_mtp_draft.py` (written by the MTP lane; the parent added `--no-markov`).

```
python3 scripts/inkling_native/convert_mtp_draft.py --out ~/models/timlikesai/Inkling-2-Mini-MTP-BF16.gguf
python3 scripts/inkling_native/convert_mtp_draft.py --no-markov --out ~/models/timlikesai/Inkling-2-Mini-MTP-DFLASH.gguf
```

Defaults: `--hf-dir` = the Inkling-2-Mini snapshot; `--ref-gguf` = unsloth's `dspark-Inkling-2-Mini-Q8_0.gguf` (metadata/tokenizer only). Tensors are copied **bit-exact BF16** via `raw_shape=tuple(reversed(shape))`, on the reasoning that GGML stores (in, out) so the HF row-major buffer is already the on-disk buffer. **Neither variant loads today** — see §5 and §8.

## 5. Load test / acceptance measurement — PARENT RAN IT (lane timed out before doing so)

Flag discovery first: `--spec-draft-model`/`-md` is gated to `{SPECULATIVE, SERVER, CLI, IMATRIX}` (`common/arg.cpp:4336`) — **`llama-completion` rejects it with `error: invalid argument: -md`**. Same example-gating trap as `-no-cnv` earlier tonight. Use `llama-cli` (or `llama-server`).

Command (`logs/x-mtp-load-2.log`):

```
llama-cli -m Inkling-2-Mini-NATIVE.gguf -md Inkling-2-Mini-MTP-BF16.gguf \
          -c 8192 -p "Q: What is 17 times 4? A:" -n 64 --temp 0 --seed 42 --single-turn
```

Result — **fails to load the draft**, and the reason is an architectural contract mismatch, not a packaging error:

```
E llama_model_load: check_tensor_dims: tensor 'markov_w1.weight' has wrong shape;
                    expected 201024, 200058, got 201024,    256,      1,      1
E common_speculative_init_result: failed to load draft model
```

**Interpretation.** master's `dspark` markov head declares `markov_w1` as a **`[n_vocab, n_vocab]` bigram transition table** (201024 x 200058 — 200058 is the *unpadded* vocab, the same number lane U hit in the `.kld` files), whereas Inkling's checkpoint ships `markov_head.markov_w1` as **`[201024, 256]`** — a rank-256 embedding, matching `mtp_config.markov_rank: 256` and `markov_head_type: "rnn_hidden"`. The two implementations reuse one name for different mathematics. Combined with scout K's findings (missing `markov_w2`, `joint_proj` `{5120,768}` vs expected `{2816,768}`): **the public dspark GGUF and master's loader disagree with each other, and the checkpoint agrees with neither.**

Secondary, worth not losing: in the same run, before the draft error, the *target* hit `allocating 19222.61 MiB on device 0: cudaMalloc failed: out of memory`. With a draft configured, target auto-fit asked 19.2 GB of one card. Unverified whether `n_gpu_layers_draft`/fit interacts badly — belongs with the FA/KV memory work (#22).

## 5b. What the artifact proves (reconstructed by the parent; lane left §2-§4 PENDING)

`~/models/timlikesai/Inkling-2-Mini-MTP-BF16.gguf` — 647,729,056 B, **54 BF16 tensors** (source: 54 BF16, 635,801,416 B), `general.file_type = 30`, `general.name = "Inkling-2-Mini MTP (rnn_hidden draft, BF16)"`. It *is* translated into master's vocabulary, with metadata showing real thought:

| key | value |
|---|---|
| `general.architecture` | `dflash` |
| `dflash.block_count` / `block_size` | 3 / 8 |
| `dflash.target_layers` | `[5, 23, 41, 48]` (verbatim from `mtp_config`) |
| `dflash.dspark.markov_rank` | 256 |
| `dflash.attention.sliding_window` / `_pattern` | 1024 / `[True, False, True]` |
| `dflash.attention.{head_count,head_count_kv,key_length,value_length}` | 24 / 8 / 128 / 128 |
| `dflash.rope.freq_base`, `context_length`, `vocab_size` | 1e6, 1048576, 201024 |
| also | `inkling.mtp.markov_head_type = rnn_hidden` |

Names produced include `dspark_fuse.weight`, `blk.N.dflare_norm.weight`, `markov_w1.weight`, `joint_proj.{weight,bias}`, `mask_embd.weight`, `output_norm.weight` — so `fuse_weight -> dspark_fuse`, `mask_embed -> mask_embd`, plus a synthesized `joint_proj`. The 54-vs-54 tensor count against the reference file is the coincidence scout K flagged; it is probably meaningful (same head, different serialization contract).

## 5c. Next experiment — the (A) vs (B) decision, now grounded

**Try (A) properly:** emit a *plain dflash* draft — `general.architecture = dflash` with the markov/dspark tensors **omitted** (no `markov_w1`, no `joint_proj`, no `dspark_fuse`, nothing that sets `VARIANT_DSPARK`), keeping the 3 attention+MLP blocks and norms. If it loads we get what actually decides the question: **acceptance rate and decode t/s delta**. If (A) still fails, its failure list *is* the specification for (B) — extending master's draft arch to carry `k_proj_ctx`/`v_proj_ctx`, `fuse_weight`, `dflare_norm` and the rank-256 rnn head.


## 5d. Option (A) attempted and measured — **it cannot be satisfied without fabricating weights**

`scripts/inkling_native/convert_mtp_draft.py` gained a `--no-markov` mode (drops `markov_head.*` + `fuse_weight`, omits the `dspark.markov_rank` key, takes `vocab_size` from the reference draft). It produced `~/models/timlikesai/Inkling-2-Mini-MTP-DFLASH.gguf` (535,749,632 B, 49 tensors = 54 - 4 markov - 1 fuse). Loading it (`logs/ad-dflash-load-1.log`, `llama-cli -md ...`, since **`-md` is rejected by `llama-completion`**):

```
E llama_model_load: check_tensor_dims: tensor 'fc.weight' not found
```

The markov failure is resolved; the next requirement is structural and deeper. `src/models/dflash.cpp`:

- `:55` — `hparams.n_embd_inp_enc_impl = target_layer_ids.size() * hparams.n_embd` -> **4 x 2304 = 9216**
- `:110` — `fc = create_tensor(tn(LLM_TENSOR_FC, "weight"), { n_embd_inp, n_embd }, 0)` -> **required, `{9216, 2304}`**
- `:176-178` — the graph feeds `inp_target` as an `[n_embd_inp, n_tokens]` tensor, i.e. the four target hidden states **concatenated then linearly projected** into draft space

Inkling's head does the opposite: `fuse_weight` is **`[4,4]`** — a scalar mixing matrix over four same-width (2304) target hidden states, with **no projection matrix at all**, plus `dflare_norm` and the `*_ctx` K/V path. dflash's fusion is `W_fc · concat(h_5,h_23,h_41,h_48)`; Inkling's is `sum_k fuse[k,*] (x) h_k`. Same purpose, different mathematics and a different parameter count (9216x2304 = 21.2M params vs 16).

**So option (A) as "degrade and load" is closed.** Two remaining paths:

1. **Synthesize `fc` from `fuse_weight`** — possible *only if* Inkling's mixing is expressible as a linear map on the concatenation. A `[4,4]` mix of four 2304-wide vectors **is** a linear map on their concatenation, so a `9216 x 2304` matrix exists in principle (block structure `fuse[k,l] * I_2304`), and it costs 42 MB in BF16. **This is the promising route and it must be verified numerically, not assumed**: build the block-form `fc`, run dflash's graph on a fixed prompt, and compare draft hidden states / logits against the HF reference forward for the same inputs. If the semantics of `fuse_weight` are not plain layer-mixing (e.g. per-slot scalars applied after a norm, or combined with `dflare_norm` in a way that is not linear), the comparison fails and this path dies. Also unaddressed either way: `attn_k_ctx`/`attn_v_ctx` (whether dflash's `ctx` slots can carry them) and the markov head, which has **no** representable form (rank-256 vs vocab x vocab).
2. **Option (B): extend master's draft arch** to express layer-mixing fusion + dflare norms + ctx K/V + a rank-r rnn markov head. The measured failure list above is its specification: `markov_w1 {n_vocab, rank}` instead of `{n_vocab, n_vocab}`, `fuse {n_target, n_target}` instead of `fc {n_target*n_embd, n_embd}`, plus `dflare_norm`, `attn_*_ctx`, `k_norm_ctx`.

## 6. Recommendation: (A) degrade vs (B) extend — SUPERSEDED by §8

**Updated after §5d: neither pure option survives.** Plain degrade is closed — `fc.weight {9216,2304}` has no counterpart in the checkpoint. The recommendation is therefore: **(1)** attempt the synthesized-`fc` route above as a bounded experiment (a day of work at most, and its success criterion is a numerical match against the HF forward, not "it loaded"); if the mixing is not expressible, **(2)** go to (B), for which we now have an exact, measured specification rather than a guess. Note the markov head is *not* representable in either path — it is the expensive part (102.9 MB of the 636 MB head) and a rank-256 rnn correction is a different mechanism from master's vocab-by-vocab transition table.

**Context that changed the calculus:** decode is now **~40 t/s** after `e046294a7` (alias-aware device placement), not ~22. Speculative decoding still helps, but the headroom that motivated it halved, and the remaining decode cost is dominated by the MoE expert gather — where a draft that itself has full 9216-wide MLPs (the draft's own MLPs are 3 x ~97M params) is not cheap. Measure acceptance *and* draft cost before investing in (B).

## 8. Reference-implementation investigation (parent, from primary sources) — **why neither (A) nor a transpose can work**

Evidence chain, each item verified rather than assumed:

1. `transformers` (5.23.0, system install) ships `models/inkling2/` and `models/inkling/`, but `modeling_inkling2.py` contains **no** `fuse_weight`, `markov_w1`, `rnn_proj`, `hidden_out`, `k_proj_ctx` or `dflare` at all — only base config, with `num_mtp_layers`, `chain_hidden_post_norm`, **`mtp_hidden_states_first`**, `mtp_local_layer_ids`, `mtp_layer_types` present as `AttributeError()`/`raise` stubs in `modular_inkling2.py:159-177`. So HF has the *skeleton* of an MTP config and not the implementation.
2. The real family lives in **`vllm-project/speculators`** (the repo renamed from huggingface/speculators; checkpoint says `markov_head_type: rnn_hidden`, `markov_rank: 256`, `block_size: 8`). Its **v0.7.0+** releases added DSpark — "extends DFlash with a low-rank Markov logit-bias head for intra-block token dependency, a per-position confidence head" — and **v0.8.0 shipped "feat: add Inkling weight aliases and training script" (#989)**, which is only `llm.embed.weight` / `llm.unembed.weight` aliases for verifier loading plus `examples/train/dspark_inkling_small_nvfp4.sh`.
3. **But `speculators` v0.3.0 — the version line the checkpoint's `mtp_config` sits nearest — contains only `eagle3`** (tree listing: `models/{__init__,eagle,eagle3/*,independent,mlp}.py`, no `dflash/`, no `dspark/`, no `mtp/`), and **`rnn_hidden` appears nowhere in current `main` either** (`grep -rn rnn_hidden` over the clone: 0 hits; `dspark/config.py:48` enumerates `Literal["vanilla", "gated", "rnn"]`). So the head's own reference implementation is not retrievable from either the pinned-era or the current package.

**What that means for each concrete blocker, with the code:**

| # | blocker | evidence | verdict |
|---|---|---|---|
| 1 | `fc.weight` `{9216, 2304}` is **required** | `dflash/core.py:120-121` `self.fc = nn.Linear(len(target_layer_ids) * hidden_size, ...)`; master `dflash.cpp:110` `create_tensor(..., { n_embd_inp, n_embd }, 0)` (0 = required) | Inkling has `fuse_weight [4,4]` and **no projection matrix**. A block-diagonal fc *can* be constructed (`fuse[k,l] · I_2304`), but it is only faithful if Inkling's fusion really is a plain scalar layer-mix — **unverifiable with no reference implementation available**, and `mtp_hidden_states_first` hints the concat order/semantics differ. Guessing here produces a draft that loads and quietly drafts badly. |
| 2 | `markov_w1` shape | `dspark.cpp:82` `dspark_markov_rank = get_tensor_meta("markov_w1.weight")->ne[0]` then requires `{rank, n_vocab}`; PyTorch `markov_w1 = nn.Embedding(verifier_vocab_size, markov_rank)` (`model_definitions.py:35`) stores `[vocab, rank]` | **A genuine layout mismatch** — this single error *is* fixable by transposing (and would make master read rank = 256 correctly). Verified: it is exactly why the message said "expected 201024, 200058". |
| 3 | `markov_w2` absent | `model_definitions.py:37` `markov_w2 = nn.Linear(markov_rank, draft_vocab_size)`; master `dspark.cpp:86` requires it | Inkling has **no** `markov_w2`; it has `rnn_proj [768, 5120]` + `hidden_out [2304, 256]`. And 5120 = 2·2304 + 2·256 ≠ master's `n_embd + 2·rank = 2816`, while 768 = 3·256 does match `3*rank`. So the RNN variant feeds a different concatenation than `joint_proj` expects. |
| 4 | head type name | `dspark/config.py:48` accepts only `vanilla/gated/rnn`; checkpoint says `rnn_hidden` | Not in the current registry, not in v0.3.0. Either a pre-rename name or a thinkingmachines-original variant. |

**Conclusion (replaces §6's tentativeness):** the shipped drafter requires a **new draft architecture in llama.cpp** — option (B) — implementing the D-Flare block-diffusion head: `fuse [n_target, n_target]` mixing instead of `fc [n_target·n_embd, n_embd]`, `dflare_norm`, `attn_{k,v}_ctx` + `k_norm_ctx`, `mask_embed` for block queries, and a rank-256 `rnn_hidden` logit-bias head. §5d's degraded attempt is closed, and *not* because the mapping was done badly — the target format simply does not describe this head.

**One caution I want on the record:** items 2 and 3 are individually fixable and tempting. Do **not** ship a draft assembled from transposed `markov_w1` plus a synthesized `fc` without a numerical check, because speculative decoding is output-invariant — **a wrong draft can never produce wrong text, only slow acceptance.** So "it loads and generates" is worthless as evidence here; the only meaningful success signal is a measured acceptance rate (accepted/drafted) close to what `block_size 8` could plausibly deliver. If we ever build (B), acceptance rate is also the metric that proves the fusion semantics were guessed right.

**Practical recommendation:** park MTP. Decode is ~40 t/s after `e046294a7` (was ~22), and this head needs either its unretrieved reference implementation or a from-scratch draft arch plus a numerical acceptance experiment. Better ROI right now: #29 (the ~0.174 MiB/token compute term that gates 256K/512K), #21 (Inkling CPU batched-eval nan, which affects *every* quantization and so is a real user-facing bug), and #23 (`llama-quantize` MXFP8 wiring) — all three are self-contained, and #21/#23 are PR-worthy upstream.

## 7. Corrections to my briefs

1. **Brief X's option (b) — "a new `conversion/inkling2v.py` registered on `vision_config.model_type`" — was unreachable**: `convert_hf_to_gguf.py` never keys a class off vision config. Projector files are `MmprojModel` subclasses reached via the global `--mmproj` flag (`docs/mxfp8/17-vision-mmproj.md` §2-§3).
2. **"The converter discards MTP tensors" is by design, not a defect** — llama.cpp consumes drafts as a separate `--model-draft` file, so dropping `mtp.`/`markov_head`/`nextn_predict` from the *target* GGUF is correct. The actual gap is that no producer exists for a draft file this loader can read.
3. **The `speculators_version: 0.3.0` I recorded earlier could not be re-confirmed.** `mtp_config`'s real keys are `block_size, head_dim, hidden_size, intermediate_size, local_layer_ids, markov_head_type, markov_rank, max_position_embeddings, num_attention_heads, num_hidden_layers, num_key_value_heads, o_bias, q_bias, rms_norm_eps, rope_theta, sliding_window_size, target_layer_ids, torch_dtype` — **no `speculators_version`**. Treat that attribution as unverified; §8's conclusion does not rest on it.
4. **The 200058 in master's expected `markov_w1` shape is the *unpadded* vocab** (padded = 201024) — relevant to any future draft metadata.


PENDING
