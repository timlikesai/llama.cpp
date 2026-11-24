# 39 — the Inkling drafter runs; acceptance is the remaining gap

Status: **the head executes end to end and is measured.** Speculative decoding is not yet a win
(0.092 acceptance, decode 18.4 t/s against 40.2 without a drafter). This document records what was
built, what the ablations proved, and what is now ruled out — so the next pass starts here.

Commits: `614758f25` (state channel), `734e6632c` (the head), `42518136f` (anchor position).
Supersedes the "park MTP" recommendation in doc 23 and the packaging conclusion in doc 38.

---

## 1. Packaging — settled, and independently validated

`scripts/inkling_native/make_dspark_draft.py` derives a `dflash`-arch draft file **from the nextn
tensors already inside `Inkling-2-Mini-FULL.gguf`**, adopting the public
`dspark-Inkling-2-Mini-Q8_0.gguf` as the name/metadata contract. 635.9 MB of tensor data =
exactly `mtp.safetensors`, byte-count neutral, and:

> 27 same-type tensors **byte-identical** to the public file, 1 exactly equal across dtype, 0 differing
> (the remaining 27 are their Q8_0 lossy choice, not comparable bitwise).

Two things this caught that "it loaded" did not:

- **Orientation.** Every 2-D tensor in the earlier hand-made `Inkling-2-Mini-MTP-BF16.gguf` was
  untransposed (`markov_w1 [201024,256]` where the reference stores `[256,201024]`). The loader
  *derives the markov rank from `ne[0]`, so a transposed file passes its own shape check* — gate 1
  ("no tensor error") was circular, exactly the failure mode doc 0 §cautions warns about.
- **`dspark_fuse` must be F32.** `build_dflare_alpha` asserts it, because the reference computes
  `softmax(fuse.float())`. The checkpoint stores BF16; the widening is exact (asserted in the script)
  and the result matches the public file's values to the bit.

`scripts/inkling_native/convert_mtp_draft.py` is **dead**: its main write loop was
`w.add_tensor(...) if False else None`. The artifact on disk predated that edit, which is how
"the file exists" hid it.

## 2. What was missing in the graph (three semantic errors, all reference-confirmed)

| fix | evidence | acceptance effect |
|---|---|---|
| block slots 1..n take the learned `mask_embed` **vector**, not the vocab mask token | `ref/dspark.py:468 embed_block` | 0.0264 → 0.0753 (**2.9×**) |
| slot 0's embedding is normalized with the **target's** `tok_norm` (reference borrows it at load: `model.embed_norm = target_model.model.embed_norm`, hence absent from the draft file) | `ref/dspark.py:474,535` | 0.0753 → 0.0947 (**1.26×**) |
| `f_rnn` = DFlare mix of the anchor's taps, `mix(features, row=-1, markov_head.dflare_norm)`, delivered through the new batch state channel | `ref/dspark.py:576` | ≈ 0 (see §4) |

`fuse_weight` being `[4,4]` for **3** draft stages is explained by the same line: row 3 is the
markov head's own mix. That resolves the shape that looked like an off-by-one in doc 34.

## 3. Ablation table (2026-11-22, `--spec-draft-n-max 4`, long baseline prompt, 192 tokens, 2×5060 Ti)

Measured with temporary env gates, since removed; the gates were shape-exact (zero the *result*, not
the input) so each row isolates information, not arithmetic.

| config | acceptance | per-position |
|---|---|---|
| all on | **0.09222** (51/553) | (0.223, 0.101, 0.022, 0.022) |
| `f_rnn` zeroed | 0.09222 | unchanged |
| stage context features zeroed | **0.01849** | (0.062, 0.011, 0, 0) |
| mask vector off (token-id embedding) | 0.02639 | (0.081, 0.023, 0, 0) |

Read-outs:

- **The injected context is the dominant signal** — zeroing it costs 5×. So the tap export, the DFlare
  mix, the `attn_k_ctx`/`v_ctx` injection and the RoPE'd context KV are all *working*.
- **The markov head is not the lever.** See §4 for why that is a property of the weights, not a bug.
- **Nothing here is a quantization effect**: the same run against the public **Q8_0** draft gives
  *identical* statistics (0.09472, 52/549, 172 calls) — a lossy file and a bit-exact one agreeing
  step-for-step.

## 4. Why `f_rnn` cannot move acceptance (measured from the checkpoint)

`rnn_proj.weight` is `[768, 5120]`, split `[state(256) | w1_prev(256) | hidden(2304) | f_rnn(2304)]`:

| block | frobenius | rms | share of total |
|---|---|---|---|
| state | 16.66 | 0.0376 | 60.1% |
| w1_prev | 14.45 | 0.0326 | 52.1% |
| hidden | 15.75 | 0.0118 | 56.8% |
| **f_rnn** | **5.84** | **0.0044** | **21.0%** |

(`shares sum >100% because the blocks partition columns of a 5120-wide matrix.)`

`f_rnn` is the weakest of the four by 2.7× on rms. Zeroing it leaving acceptance *bit-identical* is
therefore consistent with the weights. I still fixed the anchor lookup (`42518136f`: the ring was
being read at `dp.pos0 - 1`, one token before the anchor): a small-magnitude input is not a licence
to feed a wrong one.

## 5. Ruled out, with the evidence that ruled it out

| hypothesis | how it died |
|---|---|
| tap 48's source (nextn vs layer-input, i.e. post- vs pre-output-norm) | both routings give byte-identical statistics (0.07534, 44/584). Kept the pre-norm routing as more consistent with the other three taps. |
| `block_uniform` mask for the windowed draft cache | no-op: 0.09222 → 0.09222. Not in the tree. |
| injection never runs | refuted by instrumentation: 9 inject passes, `dflare=1, n_layer=3`, and `f_rnn` built with `n_embd_enc=9216`. (My earlier "0 occurrences of `Kcur_injected` in the log" proved nothing — that log lists *input tensors*, not nodes.) |
| BF16 vs Q8_0 draft quality | identical acceptance (see §3). |
| `markov_w1` padded-vocab / orientation | loads clean; the repack is byte-identical to the public file. |

## 6. Next step — a numeric oracle for the whole draft forward, not another hypothesis

Three semantic fixes each moved acceptance, and the residual shape (pos1 0.22, pos3+ ~0.02) does not
match any single remaining candidate. Guessing has been superseded twice tonight by measuring; the
remaining gap is worth a harness rather than another reading pass:

1. Build a numpy forward of the draft from the checkpoint tensors (embed_block → 3 stages with
   DFlare-mixed context KV → final norm → shared lm head → markov chain with the exact bf16 rounding
   points from `ref/dspark.py`).
2. Feed it the *same* anchor tokens, taps and positions as a llama.cpp run of 8–16 tokens, and compare
   **draft logits per slot**, not sampled text.
3. Any slot where the logits disagree localizes the remaining semantic error the way the
   device-matrix localized the loop-slot bug — and unlike acceptance, it distinguishes "wrong" from
   "correctly weak".

Worth re-checking in the same pass: the reference's bf16 rounding points in `markov_gate` and the
`30·tanh(bias/30)` softcap (my graph keeps these in f32 — more accurate, but *different*, and the
chain is greedy so a rounding difference compounds across a block).

## 7. Traps recorded on the way

- `--draft-max` is **removed**; use `--spec-draft-n-max` (the error message says so — read it).
- The `draft acceptance` / `acc per pos` lines come from the embedded server's slot report: **only
  printed with `-v`**.
- An empty `-f` prompt file produces empty output *silently*. `logs/g0-prompt.txt` was 0 bytes and
  cost me two "regression" reads. The oracle is `-p`/`-f` with exactly `Q: What is 17 times 4? A:`;
  the historical `A: 68 B: 64 C: 72 …` string is **pre-rebase** and no longer reproducible because
  Inkling now enters its thinking template (`[Start thinking]`). CPU ≡ CUDA agreement is the
  invariant to use post-rebase.
- `llm_graph_input_embd_h::build_inp_embd_state` **returns** `h`; the caller must assign
  `inp->h = ...`. Relying on it to store leaves a null tensor and a segfault, not an assert.
- One ugraph serves both the pp and the tg ubatch, so a graph input's `can_reuse` must compare
  **presence**, not just width — fixed in `614758f25`; it was a latent crash for every state-reading
  graph, not just Inkling's.

---

## 8. RESOLVED — upstream PR #32218 gave the answer (2026-11-22)

`ggml-org/llama.cpp#32218` "feat: Inkling 2 DSpark support" (pwilkin) is an independent
implementation of this drafter, touching the same 16 files. Two lines of its diff located the real
bug, and #32773 (same head, measured on a 16 GB card) supplied the operating guidance.

**The bug: tap ids count layer *visits*, not cache slots.** Their loop hunk says so in one line
(`// DSpark target features are indexed by layer visit, not by cache slot`). My `4bad7e855` had
deliberately keyed by slot and argued it in the commit message - true of the slot numbering, wrong
about the taps, and from the first replayed pass the two diverge (slot 35 is visit 13). Fixed in
`3ac4e7873`, which also adopts their post-output-norm final tap and retracts my "pre-norm is more
consistent" reasoning.

**That explains both null results from §5.** With the index wrong, the *source* of one tap (nextn vs
layer-input) could not change anything - hence byte-identical acceptance for two different routings.
And `block_uniform` measured as a no-op for the same reason: a null result is only evidence about the
hypothesis when the rest of the path is correct. Both were re-adopted (`e716be5b2`) because two
independent implementations set them; `softcap after bias` came from `add_softcap(base, bias, cap)`
in the reference, which my §6 note had already flagged as unverified.

### Measured after the fixes (2×5060 Ti, lossless FULL target, my bit-exact BF16 draft)

| prompt | config | acceptance | decode | vs no-draft |
|---|---|---|---|---|
| code continuation, ctx 8192 | none | – | 40.2 t/s | – |
| code | n_max 2, draft KV q4_0 | **0.673**, len 2.34 | **49.4 t/s** | **+23%** |
| code | n_max 4, draft KV q4_0 | 0.458, len 2.83 | 45.6 t/s | +13% |
| long technical (repeated), ctx 4096 | n_max 4 | 0.255, pos (0.564, 0.255, 0.138, 0.064) | 32.9 t/s | −18% |
| long technical | n_max 2 | – | 41.7 t/s | +4% |

Gates unchanged by the target-side edit: CPU ≡ CUDA greedy output, archs, `mxfp8` ops on CPU and
CUDA0, `test-dflare-mix`, and **z-lab DFlash2 on Qwen3.8-27B identical at 0.60000 (81/135)** —
the check that matters, since the tap buffers are shared machinery.

### Operating rules taken from #32773 (measured by their author, RX 9070 XT 16 GB)

- **Acceptance is domain-dependent, not a single number:** code 68–81%, English prose 18–21%, French
  prose 9–14%, reasoning 30–44%. Their n_max 4 on code: 80.2 t/s vs 47.6 (+68%). So a low acceptance
  on a repetitive technical prompt is expected and #50's prompt-suite requirement is right.
- **Use a fixed draft length.** Any change in the verification batch size rebuilds the graph — Inkling
  has ~12 800 nodes, and `ggml_backend_sched_alloc_graph` alone is 14–20 ms. Their n_max 7 collapses
  to 17.6 t/s *despite 50% acceptance* for this reason; my n_max 7 (26.9) is the same mechanism.
- **Draft KV defaults to f16 at full context** and silently oversubscribes VRAM on a 16 GB card
  (their 48-token prompt went 245 ms → 1550 ms). `-ctkd q4_0 -ctvd q4_0` fixed it: 22.7 → 79.0 t/s.
- `--spec-draft-p-min` is slower than plain n_max 4; combining n-gram with DSpark only added cost.
- The floor with the head loaded but drafting paused is ~-7%, which they attribute to layer-input
  extraction on the target path — consistent with my measured −15% prefill on the host-copy route.

### Still open

- Their format differs from the published file: per-layer `blk.N.ctx_mix/ctx_norm` + `markov_ctx_mix`
  instead of a shared `dspark_fuse`, and mask_embd as `{n_embd, block_size-1}` rows. Mine keeps the
  original `fuse_weight`/single mask vector and is byte-comparable to unsloth's; the per-layer form
  removes the `[4,4]` axis ambiguity, which is worth stealing as a check.
- Their in-graph `draw()` reproduces the *sampling* distribution (top-64 candidates, argsort,
  softmax(vals/temp), cumsum against a uniform carried in the state rows). Mine chains greedy
  argmax, which is right at T=0 and wrong at T>0 - the reason their acceptance holds up at T=1.
- Gap to their code numbers at n_max 4 (0.458 vs 0.68) is unexplained: could be prompt, target quant,
  or the sampling chain. Calibration: build #32218 in a worktree on plain upstream and run the same
  prompts on this box.

## 9. Head-to-head against the independent implementation (pr-32773, same box)

`pr-32773` is not stacked on `#32218` — it carries its own Inkling2 implementation
(`0a9105de4 llama : implement Inkling2 DSpark block drafting`) that reads **the same published
`dspark_fuse {n_taps, n_layer+1}` layout** as my repack, so both binaries can run the identical
target file and the identical draft file. Their PR needs `#32218`'s converter (`--dspark` on the loop
arch is rejected in this branch); building their branch directly is the calibration.

Their build could not run with `-ngl 99` (`common_fit_params: ... n_gpu_layers already set by user
to 99, abort` — the trap recorded in doc 0), so the comparison uses the fitter on both sides.

**IQ4_NL target, code prompt, ctx 8192, draft KV q4_0, n_extract 4 — no-draft baseline 46.6 t/s:**

| build | n_max | acceptance | mean len | decode | vs no-draft |
|---|---|---|---|---|---|
| mine (block = n_max+1) | 2 | 0.647 | 2.29 | **50.8 t/s** | **+9.0%** |
| mine | 4 | 0.506 | 3.02 | **55.5 t/s** | **+19.1%** |
| theirs (block = full 8) | 2 | 0.770 | 2.54 | 38.8 t/s | −16.7% |
| theirs | 4 | 0.640 | 3.54 | 43.8 t/s | −6.0% |

So their acceptance rate is higher and mine is 24–28% faster; on this hardware only mine beats
no-speculation. **Their +68% headline was measured against a 47.6 t/s baseline on an RX 9070 XT
(Vulkan/RDNA) with a 12.1 GB target — where drafting pays because the base decode is slow enough for
a draft step to be cheap relative to it.** A drafter's value is a ratio, not a property of the head.

### Their architecture, in the parts that matter

- `build_inkling2_block` asserts `n_tokens == dflash_block_size` (always the full 8-slot block) and
  **unrolls the head chain in-graph**: for each output slot it runs `inkling2_head_step`, then
  `inkling2_draw`/`inkling2_argmax`, and feeds the drawn token as `prev` into the next step.
- The head's `features = concat(hidden_row_of_slot_{i+1}, context)` where `context` is the single
  f_rnn vector fed as a 1-D input — same semantics as my `[hidden, f_rnn]` concat.
- Their state row is `[f_rnn (n_embd) | flag | uniform | 1/temp]` (narrower than mine: I ship the
  raw taps, 4·2304 floats, and mix in the graph). `e4b1cb531` is literally "fix draft state width".
- They pin the chain's nodes to the output device with
  `ggml_backend_sched_set_tensor_backend`, because the unrolled chain is many tiny ops — the same
  cross-device stall class that cost this model 27% of prefill before `e046294a7`. Worth stealing.

### Two of their design choices, measured here and NOT adopted

- **Full 8-slot block.** Reverted after measuring: acceptance unchanged within noise (n_max 2:
  0.684 full vs 0.673 narrow; n_max 4: 0.466 vs 0.458) while decode dropped ~20% (39.6 vs 49.3 t/s
  on the lossless target). Their reason for it is graph-reuse stability, which matters for a policy
  that *varies* the draft length; my block width is already constant within a run.
- **`sliding_window = size + 1`.** Their converter writes 1025 for `mtp_config.sliding_window_size
  = 1024` ("the block sees W positions before the anchor, llama.cpp SWA sees n_swa - 1"). A draft
  file rebuilt with 1025 gave byte-identical acceptance (0.68372 / 0.46629) — kept the published
  1024.
- And a correction to my own reasoning: `softcap(base + bias)` vs `cap(base) + bias` **cannot** move
  greedy acceptance — a softcap is monotone, so argmax is invariant. It matters only for T>0
  rejection sampling and for `p_min`. Keeping the change is right (it matches the reference), but it
  was never going to show up at T=0, which is why it didn't.

### Lossless model, all fixes in, no-draft baseline 40.1–40.2 t/s

| prompt | n_max 2 | n_max 4 |
|---|---|---|
| code continuation | 0.673, **49.3 t/s (+23%)** | 0.458, 45.6 t/s (+14%) |
| short question | 0.741, **53.1 t/s (+32%)** | 0.557, 53.4 t/s (+33%) |
| repetitive technical | 0.469, 40.7 t/s (+1%) | 0.266, 33.4 t/s (−17%) |

Domain dependence of exactly the shape #32773 reported (their French prose was −35%): the drafter is
a real win on predictable text and a loss on text the head cannot predict. `n_max 2` is the safe
default; `n_max 4` wins when acceptance stays above ~0.45.
