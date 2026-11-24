# Rebase onto master (3 commits, incl. GDN replay) — 2026-11-24

## What landed upstream

| commit | files | overlap with my 33 |
|---|---:|---:|
| `c74c71a6f` fix editor config in master (#32914) | 1 | 1 (`ggml.c`, whitespace) |
| `38b9cad71` quantize: make local const static for MSVC (#32902) | 1 | 0 |
| `d35dc4d9e` **GDN: use replay instead of rollback (#31669)** | **52** | **15** |

Rebase: **zero conflicts**, 33 commits replayed.

### What `d35dc4d9e` actually does

Adds `cparams.rs_replay` for gated-delta-net models (`rs_replay = n_rs_seq > 0 && …`), drops the
graph replay fallback for `delta_net`, and — the part that is *not* GDN-specific — fixes
`find_slot` swapping `pos/src/rs_pend/seq_id` of a cell **without `rs_max`**, so a sequence moved by
`find_slot` inherited the rollback depth of the cell's previous owner. Their note: a smaller value
made `seq_rm` refuse the rollback (#31656), a larger one made snapshot mode restore the previous
owner's snapshots.

**Relevance to Inkling:** none directly — Inkling's state op is `GGML_OP_SSM_CONV` (`k_sconv`), not
the `GDN_*` ops, and the replay path is gated on delta-net. But `rs_max` is the generic rollback-slot
bookkeeping that the server's seq-removal capability depends on, and seq-removal is what gates
speculation (`ctx_tgt_seq_rm_type == NO` → "speculative decoding not supported"). So this is a
plausible upstream fix for checkpoint/cache behaviour we exercise.

## Verification — capability, not just a green build

Applying the rule from the previous rebase (a conflict-free auto-merge is not a semantic pass), the
check was the number:

| | prefill | decode | acceptance | PPL | fitter retries |
|---|---:|---:|---:|---:|---:|
| FULL, ctx 8192 | 1030.67 / 1030.92 | 48.57 / 48.56 | 0.65455 | — | 0 |
| variant, ctx 8192 | 1026.61 / 1030.05 | 55.25 / 55.20 | 0.69811 | — | 0 |
| PPL (16,705 tok) | | | | **5.5656 ± 0.14398** | identical to pre-rebase |

Prefill in the 1,026–1,031 band means the CUDA indexed-bias dispatch is live (the decline path
measures 838–840). Gates **13/13** including `flash-cuda`, `ops-mxfp8-cuda0/cuda1/all`,
`cpu-vs-cuda` (identical reasoning stream), `dflare`, `archs`, `quantize`.

A presence audit of my own hunks in the 15 overlapped files initially reported three "GONE" — all
three were **my wrong needle names**, not lost code: `mtp_head_taps`/`build_dflare_*` live in
`src/models/dflash.cpp` (not `llama-graph.cpp`), and `get_rows_same_type` was deliberately retired
when the indexed-bias path deleted the gather it existed for. The indexed-bias call is at
`src/llama-graph.cpp:3224`.

## Superseded / closed by this round

- **#32752 "[Inkling 2] Build mask in chunks"** — attacks large compute buffers by building the
  attention mask in chunks with existing ops (and compares against #32883). It targets the
  `n_kv`-wide mask tensor that the indexed-bias path **eliminates entirely**, so it is not
  applicable to this branch. This also closes the mask-chunk line of attack generally: it is why
  several of my node-deletion attempts measured neutral — the tensor was already gone.
- **#32218's head structure** — reading their current implementation refuted my "slot-specific
  context" hypothesis (see below).

## #32218 now implements the rnn_hidden head — and it refutes my lead

Their recent commits: `unify rnn condition with markov_head_type`, `dspark: fix Inkling2 draft state
width, reseeding and draw order`, `spec: align Inkling2 DSpark with n_max draft tokens`,
`handle temp > 0`.

Their `build_dspark_markov_head` is **structurally equivalent to mine**: per-slot `hidden_i` (a
strided view of the draft hidden), `f_rnn` appended as the 4th concat block and carried in the
**anchor row** (so anchor-only, as in mine), `state = cand + gate*(state − cand)`,
`markov = tanh(chunk(2))`, `bias = build_lora_mm(w2, markov)`, `col = softcap(base_i + bias)` with
`base_i` a strided view of `res->t_logits`. So the uniform acceptance gap is not the head shape.

Remaining differences, in order of suspicion:

1. **`f_rnn` provenance.** They compute it on the **host** from target features and pass a
   `n_embd_dec`-wide vector through the state channel; I ship 9216 raw taps and mix in-graph. If
   their host mix uses a different tap set, order, or norm, that is a uniform per-slot quality
   difference — exactly the shape the per-position decomposition showed.
2. **Folded `w2`.** They fold `hidden_out` into the LM head at load (`fold_weight = lm_head @
   hidden_out.T`) and store it as `markov_w2`; I keep the two-step contraction. I measured these as
   numerically ~equivalent (0/1024 argmax flips), so this is low.
3. **State width** `n_embd_dec` (2304) vs my 9216 — mine may disable a device path.

Two of their fixes worth taking regardless:

- **`begin()` reseeds the draw RNG per request, not per round** — *"or two identical requests
  differ"*. **Checked against my tree: already handled.** My reseed is guarded by
  `dflash2_seed[seq_id] != dp.seed` and gated on `sample = params.probabilistic && dp.result_q`, so
  greedy never touches the RNG; and the ngram impls have no RNG at all. So my measured variance is
  **not** RNG — which promoted #8 (taps missing on prompt-cache hits) to the top of the list.
- **Draft-context length guard** — `dp.pos0 + block_size > llama_n_ctx_seq(ctx_dft)` → skip
  drafting. Adopted as TODO #47: my block is numbered `[pos0, pos0 + n_block_tokens)`, so the guard
  uses the actual block width. Implemented this round.

## The draft-context length guard: ported, measured, reverted

Upstream #32218 adds, in the dspark draft loop:

```cpp
// the full block of a markov context draft has to fit in the draft context
if (has_markov_context && (uint64_t) dp.pos0 + block_size > llama_n_ctx_seq(ctx_dft)) continue;
```

Ported it (+10 lines, `common/speculative_impl_draft_dspark::draft`), sized to my actual block width
(`n_block_tokens`, not the trained `block_size`), and tested it by driving generation to the context
edge. Getting a reachable condition took three attempts, each of which was its own lesson:

| attempt | what happened |
|---|---|
| `-c 512 -f prompt-code.txt` | the prompt is 2,527 tokens, so the server **rejected the request** and nothing ran — a vacuous "0 hits, 0 crashes" |
| `-c 128` | `--fit on` (in `SHIPPED`) auto-inflated the context; with `--fit off`, `-c 128` still **rounds up to `n_ctx_seq = 256`** |
| `-c 128 --fit off -n 112` | stopped on `max_tokens` at position 139 — edge never approached |
| `-n 250` | reached it exactly: 27 prompt + 229 predicted = **256**, `stop_type: limit` — and still **0 hits**, correctly: at `n_max 2` the block is 3 wide and the last anchor sits at `pos0 = n_ctx − 3`, so 3 tokens fit exactly |
| `--spec-draft-n-max 8` | **4 hits** at pos 250/251 with an 8-token block against a 256 draft ctx ✓ |

Then the counterfactual, one binary with a temporary `LLAMA_NO_CTXGUARD` switch so both arms shared
everything else:

| arm | guard hits | acceptance | mean len | crash |
|---|---|---|---|---|
| guard on | 4 | **0.21916 (135/616)** | 2.53 | 0 |
| guard off | 0 | **0.21916 (135/616)** | 2.53 | 0 |

**Byte-identical.** The over-boundary drafts were already harmless — llama.cpp's own context
management clamps them, and the draft simply yields nothing that gets used. So the guard is dead code
*in my design*: it is necessary for the **full-block** design (always decoding `block_size` slots,
which upstream uses and which I measured as ~20% slower here and reverted), where the window is
`block_size − n_max` wide instead of essentially zero.

Reverted rather than kept "defensively": a branch that fires and changes nothing is the shape of code
that later reads as coverage. Recorded so that if the full-block design is ever revisited, the guard
comes with it.

### Also: a false alarm worth recording

`strings build-cuda/bin/llama-cli | grep "no room for a"` returned 0, which I first read as "my change
was never compiled". It was not: **`llama-cli` links `libllama`/`libcommon` dynamically**, so `strings`
on the binary finds *no* source strings at all — including `no target taps recorded`, which I have
watched fire. The object file (`build-cuda/common/CMakeFiles/llama-common.dir/speculative.cpp.o`) did
contain my string. The control that caught me: check a string you already know is live, in the same
artifact, before believing a negative.

### And a process failure

`pgrep -f "measure_variant.sh ctxend" | while read p; do kill "$p"; done` matched its own `bash -c`
command line and killed my shell (exit 143) mid-edit — the third time this session I have hit the
self-match trap. Use explicit PIDs from `ps` with a pattern that cannot match the killing command.
