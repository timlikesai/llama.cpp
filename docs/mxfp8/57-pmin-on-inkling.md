# 57. `--spec-draft-p-min` on Inkling: what actually happens

Started as "verify my own claim". I claimed twice, wrongly, and the measurements settled it. Both wrong
hypotheses are recorded here because each looked plausible from reading the code.

- **Hypothesis 1** (mine, from grepping): "`p_min` is only enforced on the host sampling path, so it is
  inert for dspark, which sets `backend_sampling = true`."
- **Hypothesis 2** (after finding the real code): "it *is* implemented, via `llama_get_embeddings_nextn`,
  and better than the upstream-PR approach."

Neither survived contact. The truth is in `common_speculative_impl_draft_dspark`'s constructor:

```cpp
if (this->params.p_min > 0.0f) {
    char buf[16] = {};
    const bool has_conf =
        llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
        std::strcmp(buf, "true") == 0;
    if (!has_conf) {
        throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
    }
}
```

`Inkling-2-Mini-DSpark-BF16.gguf` declares `dflash.has_confidence_head bool = false` and contains no
confidence tensors, so the guard fires:

```
E srv load_model: failed to initialize speculative decoding context: DSpark draft has no confidence head:
                  please set --spec-draft-p-min 0
D slot launch_slot_: ... "speculative":false
```

**The guard is correct and the behaviour is the good kind** — a loud, actionable refusal at load, not a
silent degradation. (I had worried the key was invisible because `gguf_kv` is built from `ml.metadata`;
it isn't filtered — `metadata` is the full `gguf_init_from_file` context, so `dflash.*` is present.)

## The one real cost: the draft model stays loaded after its init fails

Same file, same flags, ctx 32,768, code prompt, 384 tokens, zero allocator retries, `measure_variant.sh
pminsweep` plus a control run:

| configuration | decode t/s |
|---|---:|
| working draft (`p_min 0`) | **48.24** |
| `p_min 0.3 / 0.6 / 0.9` — init refused, draft model still loaded, `speculative:false` | **36.49 – 36.57** |
| clean no-draft run (no `--spec-type` at all) | **38.46** |

So a failed speculative init is **4.9% slower than never configuring a drafter** — the draft weights are
committed to VRAM and the fitter planned around them, but nothing uses them. All three `p_min` values
land within 0.08 t/s of each other, which is the signature of "the knob is not doing anything different"
rather than "the gate is tightening" — a useful negative control to have measured.

Fix direction (small, upstream-shaped): when `common_speculative_init` fails, free the draft model (or
fail the load outright) instead of continuing with it resident. Filed as a task.

## Also worth keeping from this

- The `p_min` sweep mode is now in the harness (`measure_variant.sh pminsweep`, env `PMIN_N`).
- Baseline correction: no-draft decode on this box at ctx 32,768 with the code prompt is **38.46 t/s**,
  not the 40.7–41.9 figure I had been quoting (that came from a different context/prompt combination).
  Quote the config with the baseline.
- Two confident wrong claims in one investigation, both corrected by measurement rather than by more
  reading. The pattern that keeps biting: reading code tells you what *can* happen, not what *does*.

## Correction: the control was wrong, and the penalty is real

My first "clean no-draft" comparison was invalid in two ways, both worth writing down:

1. `measure_variant.sh` builds `SHIPPED` from the compose `command:` — **which already contains
   `--spec-type`**. So a run that "omits the draft flags" but uses `"${SHIPPED[@]}"` is still a fully
   drafting run. My `nodraft` label measured 48.13 t/s with `draft acceptance` present in the log.
   To get a genuine no-draft run you must strip the spec flags from `SHIPPED`, not just skip adding them.
2. The single hand-written control run (38.46 t/s) was one sample.

Re-run paired, 3 reps each, same target flags, ctx 32,768, code prompt, 384 tokens, zero allocator
retries (`measure_variant.sh pminctl`):

| configuration | decode t/s | median | init refused | drafting |
|---|---|---:|---|---|
| working draft | 48.10 / 48.13 / 48.16 | 48.13 | no | yes |
| refused init (`p_min 0.6`) | 36.53 / 36.53 / 36.60 | **36.53** | yes | no |
| true no-draft (no `--spec-type` at all) | 38.46 (1 rep) | — | — | no |

**The 5.0% penalty is real and reproducible.** And it is *not* what I guessed: the memory breakdowns are
effectively identical between a refused init and a clean run —

```
refused : CUDA0 9151 = 8355 model + 247 ctx + 548 compute | CUDA1 11358 = 9451 + 193 + 1713 | Host 1094
clean   : CUDA0 9147 = 8355 model + 243 ctx + 548 compute | CUDA1 11355 = 9451 + 189 + 1713 | Host 1094
```

— so the draft model is genuinely freed (`spec_init.reset()` in the `else` branch of
`server-context.cpp:1535`, which I confirmed before writing this up), the fitter's plan is unchanged,
and the difference is ~4 MiB of context. The cause is **not memory and not a leak**; it is per-step work
somewhere in the path that a configured-but-failed speculator still triggers. I have not attributed it,
so it stays an open item rather than a claim.

What is solid regardless: configuring a drafter that the runtime refuses costs about 5% versus not
configuring one, so the loud error message should also be a hard failure — or the server should
re-initialize as if no speculation had been requested.
