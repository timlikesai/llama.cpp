# 51. Quantizing the two vocab matrices: +14% decode at the deployed config, and it stacks

Measured 2026-11-23 (late) on this box (2x RTX 5060 Ti, `Inkling-2-Mini-FULL.gguf` 19.61 GB lossless).
Setup and motivation follow from `50-cache-type-and-ubatch.md` and the decode profile in
`49-borrowing-from-strata.md`: **~21% of decode device time is two bf16 tensors** — `token_embd.weight`
and `output.weight`, 926,318,592 B each (2304 x 201024), showing up as the two 2.19 ms outliers in
`mul_mat_vec_f<bf16>`.

## 1. The variant, and how it was made trustworthy

`scripts/inkling_native/make_vocabq_variant.py` produces
`Inkling-2-Mini-FULL-vocabq8.gguf` by re-encoding **only those two tensors** bf16 → q8_0
(926,318,592 B → 492,106,752 B each; file 18.264 → 17.455 GiB). The q8_0 payloads come from gguf-py's
`Q8_0`, whose own docstring states it is bit-exact with `ggml-quants.c`, so the variant matches what
llama.cpp's own quantizer would emit.

`--verify` is the part worth trusting, and it now checks three things:

```
tensor shapes identical                  PASS  (798 tensors compared)
metadata values identical                PASS (66 entries)
untouched tensors bit-identical          PASS  (796 tensors, 16.53 GiB hashed)
  replaced output.weight            type 30 -> 8, 926318592 B -> 492106752 B
  replaced token_embd.weight        type 30 -> 8, 926318592 B -> 492106752 B
RESULT: ALL PASS - only the intended tensors changed
```

**The 40 NVFP4 and 8 MXFP8 expert tensors are bit-identical to the original file**, which is the
project's premise, so the trade is confined to two tensors the checkpoint itself ships as bf16.

Three format bugs the builder had to die for (all now asserted, not assumed):

1. **llama.cpp requires tensor offsets strictly contiguous in table order.** My first design kept every
   untouched tensor at its original offset and appended the new payloads — the loader rejected it:
   `check_tensor_dims: tensor 'token_embd_norm.weight' has offset 16597299200, expected 15670980608`.
   The data section must be rewritten packed, so the file goes through `GGUFWriter`.
2. **`GGML_QUANT_SIZES` is `(elements, BYTES)`** — not `(bytes, elements)`. Q8_0 → `(32, 34)`,
   MXFP8 → `(32, 33)`, NVFP4 → `(64, 36)`. My first reading was backwards; the cross-check assert
   against gguf-py's own `n_bytes` is what caught it.
3. **For quantized types `GGUFWriter` wants the shape in BYTES.** `add_tensor_info` runs
   `quant_shape_from_byte_shape` whenever the array is `uint8` and a `raw_dtype` is given. Passing the
   element shape (or nothing) produced `expected 2304, 201024, got 463159296, 1` at load time — and
   **my own `--verify` passed that broken file**, because it compared tensor *contents* and never
   *shapes*. The shape gate exists now because a check that cannot see the failure is worse than none.

Also: GGUF puts the tensor-info table immediately after the KV section with **no padding**; alignment
padding comes between the table and the data.

## 2. Result — the deployed configuration, one change at a time

All runs: `-sm layer -fa on -b 2048 -ub 256 --spec-type draft-dspark --spec-draft-n-max 2
--spec-draft-n-ubatch 64 --jinja --fit on --temp 0 -s 42`, prompt `logs/prompt-code.txt` (2,527
tokens), 128 tokens generated, `-ngl` unset (as deployed). Two reps where noted.

| model file | target cache | draft cache | ctx | prefill | decode | acceptance |
|---|---|---|---:|---:|---:|---:|
| FULL (bf16 vocab) | `q4_1` | `q4_1` | **262,144** | 1,026.9 | **48.24** | 0.655 |
| **vocab-q8_0** | `q4_1` | `q4_1` | **262,144** | 1,037.1 | **55.13** | **0.698** |
| FULL | `q4_1` | `q4_1` | 8,192 | 1,029.0 / 1,035.5 | 48.79 / 48.67 | 0.655 |
| vocab-q8_0 | `q4_1` | `q4_1` | 8,192 | 1,035.3 / 1,034.0 | **55.44 / 55.41** | 0.698 |
| FULL | `f8_e4m3` | `q4_0` | 131,072 | 1,032.0 | 55.67 | 0.794 |
| vocab-q8_0 | `f8_e4m3` | `q4_0` | 8,192 | 1,029.8 | **60.71** | 0.770 |
| vocab-q8_0 | `f8_e4m3` | `q4_0` | 131,072 | 1,034.7 | **60.57** | 0.770 |
| vocab-q8_0 | `f8_e4m3` | `q4_0` | 262,144 (`-ub 192`) | 945.8 | 52.71 | 0.595 |
| vocab-q8_0 | `f8_e4m3` | `q4_0` | 262,144 (`-ub 256`) | — | **OOM** | — |

**The two independent wins stack**, on different bottlenecks:

```
deployed (FULL + q4_1, 262144)                 48.24 t/s
  + q8_0 on the two vocab tensors              55.13 t/s   (+14.3%)
  + f8_e4m3 target cache at ctx <= 131072      60.6  t/s   (+25.6% vs deployed)
```

Every run reports `retries=0` (no fitter back-off) and no weight spill; host usage *dropped*
883 → 469 MiB with the variant, because the embedding it keeps in RAM got 414 MB smaller.

## 3. Quality: not distinguishable, and one claim of mine corrected

Full WikiText-2 slice (16,705 tokens, `-c 2048 -b 2048 -ub 512`, `q4_1` caches):

| file | PPL |
|---|---|
| FULL (bf16 vocab) | **5.5783 ± 0.14412** |
| vocab-q8_0 | **5.5693 ± 0.14367** |

The difference is 0.16%, an order of magnitude inside the confidence interval — the variant is not
measurably worse (it even reads slightly better, which is within noise in both directions). I also
glanced at per-chunk values earlier and saw the variant worse on `[1] [2] [3]` by ~0.25%; the
full-corpus aggregate does not support that, so I am not claiming a quality cost **or** a gain:
this measurement cannot resolve it. Reference points: the pre-rebase baseline was 5.5757 and post-rebase
5.5671, i.e. all of these sit inside the same ±0.14 CI.

Draft acceptance *improves* with both changes (0.655 → 0.698 with the vocab change alone), which is
expected: acceptance is a property of the whole target/draft logit stream, not a pure quality metric.

## 4. What this means for the deployment — Tim's call, not mine

The project's premise is *unadulterated weights*, and that premise is about the **expert tensors** the
checkpoint ships as NVFP4/MXFP8 — all 48 of which stay bit-identical in the variant. The two vocab
matrices are bf16 in the checkpoint itself; quantizing them is a normal llama.cpp choice (`q8_0`
embeddings are common), but it is still a deviation from "byte-exact repack", so **I did not change the
compose file or the served model.** If the trade is acceptable, the one-line change is
`--model .../Inkling-2-Mini-FULL-vocabq8.gguf`, giving +14% decode at 256K with the current cache types.

If ~131K context is enough for the workloads, `f8_e4m3` target cache adds another ~9% (60.6 t/s) and is
the *most accurate* cache option measured (`41-…`: +0.10% PPL vs f16). At 262,144 it does not fit with
`-ub 256`, and at `-ub 192` its acceptance drops to 0.595, so 256K should stay on q4_x.

## 5. Follow-ups this opened

- **Retrieval on the variant** — long-context recall was verified at 231,241 tokens on the *original*
  file (TODO #11, PASS: the model reproduced the planted code `4-2-9-7-1` from position ~0, with and
  without drafting); the same test on the variant is unrun, and per an embedding change it is the
  right check before switching a served model.
- **#34 (the flat ~4 GiB batch-proportional compute term)** is what stands between `f8_e4m3` and 256K:
  the variant already returned 414 MB and `f8` at 262,144/`-ub 256` still OOMs by a few hundred MiB.
- **`-ub` moves acceptance** (0.495 → 0.619 between `-ub 128` and 256, context-independent) — see
  `50-…` §3; unresolved, and it means every acceptance number in this doc set is only comparable at
  equal `-ub`.

## 6. The resulting choice table (all measured, zero fitter retries unless noted)

| configuration | ctx | decode t/s | vs deployed |
|---|---|---:|---|
| **deployed** — FULL + `q4_1` target/draft caches | 262,144 | 48.2 | — |
| **variant** (q8_0 vocab) + `q4_1` caches | 262,144 | **55.1** | **+14.3%** |
| **variant** + `f8_e4m3` target cache | 196,608 | **56.2** | +16.5% |
| **variant** + `f8_e4m3` target cache | 131,072 | **60.6** | +26% |

`f8_e4m3` at 262,144 with the drafter **does not fit on 2×16 GB** — measured at `-ub` 128/192/256 with
`--fit on` and again with `--fit off -ngl 99` (CUDA0 then OOMs on a 1,157 MiB ask). So the top row and
the 196,608 row are the two ends of the real trade: maximum context vs maximum speed.

Caveat on the prefill column: those runs used a 47-token prompt, so "prompt eval 93 t/s" is startup
overhead, not throughput. Prefill throughput comes only from a long prompt (1,030-1,037 t/s at
`-ub 256`, and 442.7 t/s when the 231k window is actually full).
