# 31 — SIGFPE in the device fitter: zero memory span between probes — **fixed, `4a5c1fa3c`**

A crash found while chasing capacity, not while chasing correctness — and the kind of bug you only meet once
the buffers get small enough. It is an upstream latent defect that my own optimization made reachable.

## Symptom

```
llama-completion -m Inkling-2-Mini-NATIVE.gguf -c 262144 -b 2048 -ub 128 \
  --flash-attn on --cache-type-k f8_e4m3 --cache-type-v f8_e4m3
-> Floating point exception (core dumped), exit 136
```

gdb with no symbols still localized it exactly:

```
Thread 1 received signal SIGFPE, Arithmetic exception.
#0 common_params_fit_impl(...)            libllama-common.so
#1 common_fit_params(...)
#2 common_fit_apply(...)
#3 common_init_result::common_init_result(...)
#4 common_init_from_params(...)
```

Reproduced twice; note it killed the process *before* any usable diagnostic, so a user sees a core dump
instead of "this does not fit".

## Root cause — two interpolations that assume strictly increasing probe memory

`common/fit.cpp` interpolates twice, and both denominators are unguarded:

1. **context probe span** (step 2): `sum_projected_used - sum_projected_used_min_ctx` — the difference between
   the max-context and min-context probes. A max-context probe that **cannot reserve** returns the same
   footprint as the minimum ⇒ span 0.
2. **layer-fill span** (two sites, steps 3/4): `mem_high[id] - mem[id]` — assigning *more* layers to a device
   does not raise *that device's* memory use once the excess **overflows to the next device or to system
   memory** ⇒ span 0. Same bug shape at both loops.

## Fix

```c
const int64_t ctx_memory_delta = sum_projected_used - sum_projected_used_min_ctx;
if (ctx_memory_delta > 0 && sum_used_target > sum_projected_used_min_ctx) { ...interpolate... }
```
When the context span is not positive, reducing the context cannot free anything, so the minimum context is
kept (it is already assigned at that point) and the next step reduces layers — which is the existing `else`
path's semantics, so behavior for healthy models is untouched.

```c
const int64_t mem_span = mem_high[id] - mem[id];
uint32_t step_size = mem_span > 0 ? uint32_t(int64_t(delta) * (targets[id] - mem[id]) / mem_span) : delta / 2;
```
When the layer span is not positive, bisect. `step_size` is already clamped to `[1, delta - 1]`, so halving
converges exactly like the interpolation loop it replaces.

## Attribution — the honest, and more interesting, answer

Re-running the same repros with `#30` (the causal-mask fold) reverted but the fitter fix kept:

| config | **without #30** | **with #30 + fix** |
|---|---|---|
| `-c 262144 -ub 128` | exits 1, clean `failed to allocate CUDA1 buffer of size **6,745,929,728**` | exits 1, clean `... size **5,063,906,560**` (−19.4% on that device) |
| auto ctx, `-ub 128` | fits and runs, **`n_ctx = 162,304`** | fits and runs, **`n_ctx = 202,496`** (**+24.8%**) |

So pre-`#30` the fitter never lands on a zero span here: the degenerate condition **appears only after the
compute buffers shrink**, which moves the probe values into the overflow regime. Conclusion for the record:

- **The defect is upstream and pre-existing** — an unguarded division in code this branch never touched.
- **The trigger is new and ours** — `#30`'s smaller buffers exposed it.

That is the right framing for the PR: memory optimizations in this area will surface this class of latent
fitter bug, and they should not have to ship with a crash behind them.

## What this unlocked operationally

The fitter's auto path at small `ubatch` was effectively unusable for this model on this box; now it selects
and runs the largest fitting context. With the lossless 19 GB file, fp8 KV, FA on, `-b 2048`:

| `-ub` | auto `n_ctx` | exit |
|---|---|---|
| 128 | **202,496** | 0 |
| 256 | 107,776 | 0 |
| 512 | 47,104 | 0 |

Both terms scale with `ubatch` (the flat batch-proportional term and the remaining ctx term), which is why
halving `-ub` roughly doubles the reachable context — and why #34 (attributing the flat ~4.5 GiB) is the next
big lever.

## Verification

Dense control unchanged: `Qwen3.5-0.8B-Q8_0` still auto-fits to its trained `n_ctx = 262144` with `26/26`
layers offloaded and answers `68`. `test-backend-ops -p mxfp8` OK on CPU and CUDA0, `test-llama-archs`
587/587. Inkling 128K at `-ub 256` fp8 KV: **913.00 t/s prefill / 41.43 decode**; 8K baseline **901.15 / 40.37**
(no regression from the fitter change). Logs: `logs/fpe-gdb.log`, `logs/fpe-fixed-{1,2}.log`,
`logs/auto-{128,256,512}.log`, `logs/attr-{262k-ub128,auto-ub128}.log`, `logs/dense-fit.log`.

## Related: a null result worth recording (#33, reverted, not committed)

Dropping the now-unused dense `kq_mask` allocation entirely — skipping `build_attn_inp_kq_mask` for the base
cache when folding, deriving `n_stream` from `ubatch.n_seqs_unq` instead of the mask tensor, and null-guarding
`can_reuse` — changed **nothing measurable**: compute identical to the byte at both 8k (5,104.67) and 128K
(9,892.88), prefill 904.3/905.0/904.2 vs 906.09 (noise). The mask is a *leaf input* sitting under the
allocator's per-buffer peak, and its host fill was never a bottleneck. Reverted rather than committed: it
added a null-input path and a reuse guard in exchange for zero benefit. Recorded so nobody re-tries it.
