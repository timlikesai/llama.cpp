# 46 — Device-row taps attempted, measured, and deliberately reverted

**Date:** 2026-11-23 · **Outcome:** change **not kept**; `#3` de-scoped to low priority on evidence.
Attempt preserved at `logs/dev-taps-attempt.patch` (182 lines) — do not re-attempt without reading §4.

## 1. What was tried

Goal from `45-…`: stop the drafter shuttling the anchor's target taps through host RAM. Three edits,
all reverted:

1. `src/llama-graph.cpp` `build_inp_embd_state` — choose the gather source by width: `h_nextn_src`
   (n_embd_out, the MTP drafts) else `layer_inp_src` (`n_taps × n_embd`, i.e. Inkling's 9,216-float
   concatenated taps). Preferring h_nextn keeps every existing draft byte-identical.
2. `common/speculative.cpp` — delete the `n_embd_state == 0` precondition that forced Inkling onto host
   copies; record the tap *row* per position in the device injection branch; hand the drafting batch
   `set_embd_state_row()` instead of 9,216 floats.
3. `src/llama-context.cpp` `supports_state_row()` — it hard-required `LLAMA_CONTEXT_TYPE_MTP` plus
   `ctx_state_src`'s `h_nextn` buffer. Generalized it to also accept `ctx_other`'s `layer_inp_dev`,
   mirroring `supports_embd_row()`.

## 2. Measured: it works in the server, crashes in the CLI, and buys ~nothing

**Server (profile_shipped.sh, chain config).** The new path engaged — `init: draft markov head takes the
anchor's target taps (9216 floats) from device rows`, `DFlash target features via device buffer` — and I
initially read acceptance 0.609 → 0.806 as a win. **That reading was wrong**: the same log contains
**36** lines of

```
E decode: this context cannot read state rows (needs an MTP context and an h_nextn device buffer in ctx_state_src)
```

i.e. `llama_process` returned −1 on most draft rounds. The "improved" acceptance was computed on rounds
that *failed to draft*, and the ngram chain had already made per-task numbers non-comparable. The
memcpy histograms settle it, and they are the decisive measurement of this whole exercise:

| | H2D 2,359,296 B | D2H 2,359,296 B |
|---|---|---|
| before (host copies) | 14,381 / 33.93 GB | 14,260 / 33.64 GB |
| after (device rows) | 14,570 / 34.38 GB | 14,468 / 34.13 GB |

**Unchanged.** And `logs/ps-both-kernels`-style totals show the new path did cut the *tap* transfers it
owned (the trace saw `inp_target_features` at 387 copies = 0.91 GB), so the honest conclusion is not
"the change failed to engage" but **"the taps were never the dominant traffic"** — they are ~9% of
steady-state bytes and ~2.5% of the session total.

**CLI (one-shot, `--spec-type draft-dspark`).** After generalizing `supports_state_row`, the same
configuration **segfaulted** (exit 139, core dumped) immediately after `... from device rows`. The gate
said the buffer was usable; the graph still could not service it for that context. A half-extended
contract is worse than an unextended one — the previous code failed loudly with a clear message, mine
crashed.

**Reverted.** Tree at HEAD, rebuilt, `git status` clean of tracked modifications.

## 3. What the ceiling actually is

Two independent placements put tensors in host RAM, at *every* context size tested:

```
done_getting_tensors: tensor 'token_embd.weight' (bf16) cannot be used with preferred buffer type
                      CUDA_Host, using CPU instead
done_getting_tensors: tensor 'mask_embd.weight' (f32)  ... using CPU instead      (the draft's)
```

`token_embd` is on CPU at 262144, 229376, 196608 **and 163840** — so this is a buffer-type *policy*
(the tensor prefers `CUDA_Host`, i.e. unified/managed memory, and falls back to plain CPU when that is
unavailable), **not** a capacity spill. I had earlier attributed it to the fitter spilling at 256K; that
was wrong and the ctx sweep disproves it.

Forcing it onto the GPU removes it from the host list but not the problem:

| config | prefill | decode | acceptance | tensors on CPU |
|---|---|---|---|---|
| plain, `-c 262144 -ub 256`, draft on | 1025.37 | 45.08 | 0.56425 (101/179) | `mask_embd`, `token_embd` |
| `-ot token_embd.weight=CUDA0` | 1022.00 | 46.18 | 0.56425 (101/179) | `mask_embd` |

Prefill is noise, decode +2.4%, acceptance bit-identical. So the host embedding is **not** the ceiling
either. Untried knobs that follow from this: `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (make `CUDA_Host`
succeed as intended rather than fall back) and `--spec-draft-override-tensor mask_embd.weight=CUDA0`
(`arg.cpp:358`) for the draft's tensor.

Meanwhile the numbers that *have* never moved across every experiment: **device busy ~33%, 1.23 M kernel
launches in 39 s (821 k of them direct), `cpy_scalar`/`concat`/`scale_f32` in 5–20 µs shots.** That is a
launch- and occupancy-bound profile — which is what `#6` (MoE decode batching) and `#4` (tiny copy
kernels) target, and both now outrank the tap work.

## 4. Rules this cost me

- **A change that engages is not a change that works.** The log line proved the new branch ran; only
  grepping for *errors* revealed 36 rejected rounds. Check the error count in the same pass that you
  check the success marker — and never compare acceptance across configs whose round counts differ.
- **`-v` is not optional for measurement in this build.** Without it `llama-cli`/`llama-server` write
  only warnings to their log; four of my runs produced 155-byte files that I first read as "no
  amplification found" rather than "no data". With `-v`, the `[ Prompt: x | Generation: y ]` summary
  disappears and `print_timing` lines appear instead — two different parsers, both now documented.
- **Loosen a gate only together with the code it guards.** `supports_state_row` + `build_inp_embd_state`
  are one contract; extending one half turned a clean error into a segfault.
- **Prefer de-scoping to salvaging.** The remaining prize for #3 is ≤2%, so the correct action is to
  demote it, not to debug the crash.
