# 50. KV cache type, the FA kernel list, and why `-ub` moves draft acceptance

Measured 2026-11-23 (late) on the lossless file `Inkling-2-Mini-FULL.gguf` (19.61 GB, 798 tensors),
2x RTX 5060 Ti, drafting with the published DSpark BF16 head, `-sm layer -fa on -b 2048`,
`--spec-draft-n-max 2 --spec-draft-n-ubatch 64`, `--fit on`, `-ngl` unset (as deployed), prompt =
`logs/prompt-code.txt` (2,527 tokens), `--temp 0 -s 42`. Harness: `scripts/inkling_native/measure_variant.sh`
(modes `cache`, `retrieval-sized`, `cpuprobe`, …); it stops the container and restores it on exit.

## 1. The finding that started it: a warning in the deployed config's own log

```
W ggml_cuda_flash_attn_ext_vec: no FlashAttention vector kernel compiled for K/V types q4_1-q4_1,
  converting K and V to f16 instead (slow). Add "q4_1-q4_1" to GGML_CUDA_FA_QUANTS to compile it.
I system_info: ... FA_QUANTS = q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;f8_e4m3-f8_e4m3
```

`ggml/CMakeLists.txt:208` sets that default list, and **`q4_1` is not in it** — so the deployed
`--cache-type-k/v q4_1` takes a per-decode-step f16 conversion path. That looked like the explanation
for q4_1 measuring ~7 t/s below q4_0 in `47-setting-sweep.md`.

**It is not.** I reconfigured the native build with `q4_1-q4_1` added, rebuilt, and re-measured:

| build | ctx | FA fallback warning | decode t/s | acceptance |
|---|---:|---|---:|---:|
| baseline (no q4_1 kernel) | 8,192 | present | 48.92 | 0.65455 |
| **rebuilt with `q4_1-q4_1`** | 8,192 | **gone** | **48.41** | 0.65455 |
| baseline | 262,144 | present | 48.24 | 0.65455 |
| **rebuilt** | 262,144 | gone | **48.37** | 0.65455 |

The warning disappears and performance does not move (both deltas inside run-to-run noise, acceptance
bit-identical). So the missing kernel is *not* the cost, and **`47-setting-sweep.md`'s causal claim for
q4_1 being slower is retracted** — see §3 for what actually explains it. I reverted the build config so
the native binary matches the container again.

## 2. What does matter: the target cache *type*, at equal `-ub`

| target / draft cache | ctx | prefill | decode | acceptance | fits? |
|---|---:|---:|---:|---:|---|
| `f8_e4m3` / `q4_0` | 8,192 | 1,034.7 | **55.80** | **0.794** | yes |
| `f8_e4m3` / `q8_0` | 8,192 | 1,028.2 | 55.90 | 0.794 | yes |
| `q4_0` / `q4_0` | 8,192 | 1,029.5 | 49.02 | 0.667 | yes |
| `q4_1` / `q4_1` | 8,192 | 1,031.3 | 48.92 | 0.655 | yes |
| `f8_e4m3` / `q4_0` | 131,072 | 1,034.9 | **55.35** | **0.794** | yes |
| `q4_0` / `q4_0` | 131,072 | 1,039.8 | 48.92 | 0.667 | yes |
| `f8_e4m3` / `q4_0` | 196,608 | 1,030.9 | **55.34** | **0.794** | yes |
| `f8_e4m3` / `q4_0` | 196,608 (`-ub 256` rerun) | 1,028.2 | 54.41 | 0.770 | yes |
| `q4_0` / `q4_0` | 196,608 | 1,039.2 | 48.87 | 0.667 | yes |
| `f8_e4m3` / `q4_0` | 229,376 | — | — | — | **CUDA OOM** at `-ub 256` |
| `f8_e4m3` / `q4_0` | 262,144 | 782.3 | 46.19 | 0.556 | only at `-ub 128` |
| `f8_e4m3` / `q4_0` | 262,144 | 586.2 | 53.30 | 0.737 | only at `-ub 64` |
| `q4_0` / `q4_0` | 262,144 | 1,031.4 | 48.73 | 0.667 | yes |
| `q4_1` / `q4_1` | 262,144 | 1,026.9 | 48.24 | 0.655 | yes |

Every run: `host = 883 MiB` (the embedding policy, constant — **no weight spill**) and **zero fitter
retries**, so these are clean comparisons per the rule in `47-setting-sweep.md`.

**`f8_e4m3` on the target is the best cache type at every context it fits: +13% to +17% decode and
+19% to +24% acceptance versus `q4_0`, and it is also the most accurate** (`41-kv-cache-type-study.md`:
PPL +0.10% over f16 vs q4_0's +0.33%). It costs ~2x the KV bytes of a 4-bit cache, which is what stops
it at 262,144 with `-ub 256`.

Why fp8 beats q4_0 on a *decode* path that is not bandwidth-bound: the fp8 vector kernel converts a byte
to a float, while q4_0 unpacks 4-bit integers and applies a scale per block. Same lesson as the
intrinsics audit (`44-…`): on this model decode is dominated by per-element work and launches, not bytes.

## 3. The confound that explains the q4_1-vs-q4_0 gap: `-ub` moves acceptance

Acceptance is not a property of the drafter alone — it depends on the target's **micro-batch size**:

| config | decode | accepted / drafted |
|---|---:|---|
| `q4_0`, ctx 196,608, **`-ub 256`** | 46.61 | 52 / 84 = **0.619** |
| `q4_0`, ctx 196,608, **`-ub 128`** | 41.65 | 47 / 95 = **0.495** |
| `q4_0`, ctx 262,144, `-ub 128` | 41.71 | 47 / 95 = **0.495** |

Context size is irrelevant (identical 0.495 at 196,608 and 262,144 ✓ good sanity check), but halving
`-ub` drops acceptance by 20% relative and decode by 11%. The likely mechanism is the one already
documented for speculative decoding: verification recomputes the target's logits for a 3-token batch,
which is a **different reduction shape** than the target's own prefill batch, so near-ties at
`--temp 0` flip — and the batch shape is set by `-ub`.

**Methodological consequence, and it applies to numbers earlier in this doc set:** acceptance
comparisons are only valid at a fixed `-ub` (and fixed `-b`, prompt, `n_max`, seed). Some of the
"acceptance gap" I have been chasing in TODO #9 may be measurement shape rather than drafter quality.
The 0.655 that the deployed `q4_1` config reports is measured at `-ub 256`, same as the 0.667 for
`q4_0` — so **q4_1 vs q4_0 is a ~1% acceptance difference, not the 13% the server sweep suggested**;
that sweep also had the ngram chain and vision in play, which is a separate variable.

## 4. What I would change in the deployment, and what I did not

| option | decode | prefill | acceptance | capacity | note |
|---|---:|---:|---:|---|---|
| **as deployed**: `q4_1` both sides, `-ub 256`, ctx 262,144 | 48.2 | 1,027 | 0.655 | fits | current |
| `q4_0` both sides, `-ub 256`, ctx 262,144 | 48.7 | 1,031 | 0.667 | fits | +1%, slightly less accurate |
| **`f8_e4m3` target + `q4_0` draft, `-ub 256`, ctx 196,608** | **55.3** | 1,031 | **0.794** | 196K | **+15% decode, +21% acceptance, best PPL** |
| `f8_e4m3` target + `q4_0` draft, `-ub 64`, ctx 262,144 | 53.3 | 586 | 0.737 | 256K | +10% decode but **−43% prefill** |

If 256K is a hard requirement, the deployed config is right and `q4_0` is marginally better than
`q4_1`. If ~196K is acceptable, `f8_e4m3` on the target is strictly better on speed, acceptance *and*
quality. **I did not change the compose file** — that is a context-length tradeoff for Tim to choose,
and the numbers are here to make it.

Adding `q4_1-q4_1` to `GGML_CUDA_FA_QUANTS` in `llama-cpp.Dockerfile` would silence the warning at no
runtime cost, but it buys nothing measurable (§1), so it is not worth the extra compile time.

## 5. Open items this raised

- **`-ub` → acceptance coupling** deserves its own investigation (TODO #33): if the mechanism is
  verify-batch numerics, a fixed-shape verification path could recover several points of acceptance,
  which is worth more than any cache type. It also means my drafter tuning must pin `-ub`.
- `f8_e4m3` at 229,376 OOMs at `-ub 256` but the same shape fits at 196,608 — the gap is ~1 GiB, which
  is what TODO #34 (the flat batch-proportional compute term) would buy.
- The vocab-q8_0 variant experiment (TODO #32) is still open, but its builder needs a rewrite:
  llama.cpp's GGUF reader requires tensor offsets to be **strictly contiguous in table order**
  (`gguf_init_from_reader: tensor 'token_embd_norm.weight' has offset 16597299200, expected
  15670980608`), so the "keep every other offset, append the new payloads" design cannot work. The
  table must be rewritten packed, which means a full data-section rewrite via `GGUFWriter`.

## 6. Follow-ups measured 2026-11-23 (late)

> The `TODO #NN` references in §5 are dead — the task list was slimmed and renumbered twice since.
> Lesson: cite the finding in the doc, not a volatile task id.

### Retrieval on the q8_0-vocab variant — PASS

`needle-fit.txt` (587,380 chars ≈ **231,241 tokens**), ctx 262,144, `q4_1` caches, DSpark drafting on:
the model's own reasoning quotes the fact planted at position ~0 —

```
...a "launch code". They haven't asked a question explicitly. Let me look at the structure.
The "IMPORTANT FACT: the launch code is 4-2-9-7-1." This seems like a...
```

So a 4-bit-cache variant still retrieves at 231k depth. **The second hit in that log is the request echo,
not the model** — count hits carefully before claiming retrieval.

Same run gives the honest *deep-context* throughput, which is far below the short-prompt numbers I have
been quoting all session: **prefill 442.7 t/s, decode 19.7 t/s, acceptance 0.492** at a filled 231k
window (vs 1,030 / 55 with a short prompt at the same `n_ctx`). Both are real; they answer different
questions. Quote which one it is.

### `f8_e4m3` target cache at 262,144 fails — and not for the reason I guessed

It OOMs at every `-ub` (128/192/256 → 14,190 / 17,388 / 20,586 MiB requested on CUDA0). My first guess
was the dense-mask fallback (that scaling looks exactly like `n_kv × n_ubatch`). **Wrong** — no
`dense_bias_mask` or conversion warning appears in the log. The reserve sequence shows the actual
mechanism: as the fitter moves layers off CUDA0, **`splits` climbs from 2 to 1,719** and each split adds
a cross-device copy buffer, so the compute ask inflates while the *weights* leave that device.

Total demand is only ~28 GiB of the 32 GiB available (17.6 weights + 8.0 f8 KV + ~2.5 compute), so this
is a planning pathology, not a capacity wall → testing `--fit off -ngl 99` in the same session.

### The runtime `FA_QUANTS` string can be stale

`ggml-cuda.cu:6975` prints the **compile-time macro**, so an incremental rebuild that adds
`fattn-vec-instance-q4_1-q4_1.cu.o` without recompiling `ggml-cuda.cu` keeps reporting
`FA_QUANTS = q4_0-q4_0` while the kernel is in fact linked. Verify capability behaviourally (is the
`converting K and V to f16 ... (slow)` warning present?) rather than trusting that system_info line.
The image build sets `-DGGML_CUDA_FA_QUANTS=all` (`llama-cpp.Dockerfile:54`).

### Process hygiene, again — my third and fourth self-inflicted incidents

1. A leaked `llama-server` from a profiling script that was **sourced** (it has a top-level body, so
   sourcing runs it) survived `tmux kill-session` and held 15 GiB. Four "measurements" taken against it
   were pure OOM noise, and the rule for it existed in the `perf-attribution` skill but the harness
   didn't enforce it.
   `measure_variant.sh` now polls until the cards are free and **exits 3** rather than measuring.
2. `pkill -f "build-cuda/bin/llama-cli"` matched my own `bash -c` command line and killed the shell
   (exit 143) — twice. Enumerate the GPU PIDs from `nvidia-smi --query-compute-apps=pid` and kill by
   explicit PID after checking `/proc/<pid>/cgroup` for docker, so the container is never targeted.

The `docker compose stop` inside the harness can also race the `start` from a previous run's restore
trap; that is why the poll loop re-issues the stop instead of doing it once.
