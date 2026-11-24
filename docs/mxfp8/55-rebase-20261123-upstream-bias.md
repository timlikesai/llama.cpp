# 55. Rebasing onto the upstream that took over our feature (2026-11-23)

28 new upstream commits. One of them replaced a third of this branch, and another nearly cost us a 23%
win without any build or test failing.

## What upstream now owns

`239242322 ggml: compact bias on the existing FLASH_ATTN_EXT (#32740)` is the **interface layer of the
feature we adopted from #32808** — the same design: `ggml_flash_attn_ext_add_indexed_bias(a, table,
indices)`, op-param slot 5, `src[6]`/`src[7]`, the same asserts. It also brought the CPU reference
implementation, the per-backend `supports_op` guards (cann, et, hexagon, opencl, openvino, sycl,
virtgpu, webgpu, zdnn, vulkan, metal, backend-meta) and a `test_flash_attn_ext_indexed_bias` case in
`test-backend-ops`.

Consequences:

- our merge commit went from **37 files to 16** on rebase;
- the compile-unverified cross-backend guard worry is now upstream's problem, not ours;
- their `has_indexed_bias` is slightly better than ours — `op == FLASH_ATTN_EXT && param != 0` instead
  of an assert, so it is safe to call on any tensor. Their version won in every conflict.

Still ours, because #32740 explicitly "moved the CUDA/HIP and Inkling 2 graph changes out": the CUDA
kernels (`fattn-tile/vec/mma-f16/common`), the Inkling graph wiring, the kv-cache changes, and
`tests/test-flash-attn-indexed-bias.cpp`.

## The near-miss: a clean auto-merge that silently killed a 23% win

#32740 also added, inside the kernel selector:

```c
if (ggml_flash_attn_ext_has_indexed_bias(dst)) {
    return BEST_FATTN_KERNEL_NONE;      // correct upstream, wrong here
}
```

Right in a tree with no CUDA implementation. Ours has one, ~90 lines further down. Git found two hunks
in two places, merged them **without a conflict**, and produced a function that returns early — our
dispatch became unreachable dead code. Build green. Tests green. Nothing warned.

I caught it by chasing a number I first mis-read twice (once as "the new test is CPU-gated", once as
"my kernel is broken"): the upstream test reported `0/0 tests passed` on CUDA. The measurement that
settled it was an A/B on the deployed configuration:

| | prefill | decode |
|---|---:|---:|
| upstream's early return | 838.66 – 840.41 | 45.1 – 47.9 |
| our dispatch restored | **1,033.4 – 1,035.0** | **48.7 – 55.4** |

**+23% prefill riding on one statement's position.** Fixed in `88a27bf91`.

The rule, which this branch has now paid for twice: *a clean auto-merge across a feature that exists on
both sides is not a semantic pass.* After any rebase touching a capability, verify the capability —
measure the thing it was supposed to make fast, or prove the code path is reachable — not just that it
builds.

## Other overlaps, resolved

| commit | verdict |
|---|---|
| `46f4da217` zero blocks in few-row MMQ | same divide guard our MXFP8 activation path already has; ours is stricter (`!(amax > 0.0f)` also catches NaN). No action |
| `30ec35ae2` quantize: auto-pick types to hit a target bpw | doesn't know MXFP8/NVFP4, so the `llama-quantize` gap stays open — but this is now the right home for it, and a better tool than my hand-built vocab variant |
| `f4f6f2e7d` generic `ggml_backend_tensor_memset` fallback | replaces host-side `llama_clear_tensor_data` in `state_clear` — device-side zeroing of SSM/conv state, the same "stop shuttling to host" theme, done upstream |
| `87f816e87` metal BF16/F32 src1 | collides with #32808's Metal port at `FWD_TMPL`/`FWD_ARGS`. **Dropped our Metal port** (reverted `ggml-metal/*` to upstream): merging two divergent refactorings of a backend I cannot compile here is how you ship a broken build, and upstream's own guard says Metal has no indexed bias. Side effect: our one-line `get_rows` Metal guard went with it — safe, it protects an API with no callers. Revisit if #32808's Metal work merges |
| `dd53f70d0` inkling parser/grammar | no file overlap; may change the `content` vs `reasoning_content` split we documented — re-check the server probe |

## Gates after the rebase — 13/13

`measure_variant.sh gates` is now a standing mode; it used to be an ad-hoc chain, and converting it
immediately exposed two silent-success bugs in my own harness:

- `test-backend-ops -d <backend>` is **not a flag** — it prints usage and **exits 0**, so a whole session
  of "ops green on CUDA0/CUDA1" could have been nothing running. Plain `-p <regex>` (no `-b`) iterates
  every backend and is the invocation that actually proves coverage.
- `-b CUDA0` still prints "3/3 backends passed" while skipping two, so that line is not evidence. The
  gate now greps for `Backend …: <name>` and the absence of `Skipping <name>`.
- `llama-cli -dev` takes a **name** (`CUDA0`, `none`), not an index — `-dev 0` is `invalid device: 0`.
- The three greedy probes only proved each backend *ran*. Inkling spends a short token budget inside its
  thinking block, so `content` is empty on all three and an equality check on it passes vacuously; the
  gate now compares `reasoning_content` across CPU/CUDA0/CUDA1.

Result: 13/13 green — flash-attn indexed bias vs the CPU reference including `kv_rows`, `test-backend-ops
FLASH_ATTN_EXT`, mxfp8 proven to execute on both cards and across all backends, archs, the dflare oracle
plus its python re-derivation, quantize round-trip, cross-backend greedy agreement, and
**PPL 5.5656 ± 0.14398 — identical to pre-rebase**, so the numerics survived.

## Second rebase the same day: 5 more commits (`c212d25ad`..`ab789a246`)

Overlap analysis first, using `git show --name-only` per commit — **not** `git diff --name-only <commit>`,
which compares against the working tree and made every commit look like it touched all 68 of my files.

| commit | files | overlap with mine |
|---|---:|---:|
| `c212d25ad` server: preserve a prompt-cache entry until restores succeed | 1 | 0 |
| `fd24ee26a` opencl: Adreno binary kernel library at configure time | 4 | 0 |
| `8355b8862` ui: remove the Matte theme | 16 | 0 |
| `51831d47d` **metal: add NVFP4 support** | 11 | **0** |
| `ab789a246` metal: batch concat rows, copy in vector chunks | 2 | 1 (`tests/test-backend-ops.cpp`) |

`metal: add NVFP4 support` touches no shared type tables, so our MXFP8/NVFP4 work is untouched. Two
conflicts still appeared, both in `ggml-metal-device.m`, both from commits that carried Metal hunks I had
already decided to drop — resolved to upstream without hesitation.

One of them was instructive: my side excluded `GGML_TYPE_NVFP4` from Metal `get_rows`, and upstream's new
commit **removed** that exclusion because Metal now supports it. Taking my version would have silently
reverted someone else's feature. The rule held: when you cannot compile or test a backend, you do not
own its file.

### Capability re-verified, not assumed

Applying the lesson from the first rebase, the check after this one was the *measurement*, not the build:

| config | prefill | decode | acceptance | retries |
|---|---:|---:|---:|---:|
| FULL, ctx 8,192 | 1,026.11 / 1,028.75 | 48.43 / 48.57 | — | 0 |
| variant, ctx 8,192 | 1,029.15 / 1,033.08 | 55.07 / 55.25 | — | 0 |
| FULL, ctx 262,144 | 1,024.13 | 44.45 | 0.568 | 0 |
| variant, ctx 262,144 | 1,030.55 | **52.11** | 0.646 | 0 |

Prefill in the 1,024-1,033 band means the CUDA indexed-bias dispatch is live; the decline path measures
838-840. Gates 13/13, PPL **5.5656 ± 0.14398** — identical to before both rebases.

### Worth knowing for this deployment

`c212d25ad` fixes a server prompt-cache case where an entry was dropped after a failed restore. Our
compose runs `--cache-reuse 4096`, `--ctx-checkpoints 256` and `--checkpoint-min-step 32768`, so this is
directly relevant — though note vision (`-mm`) disables `--cache-reuse` entirely, so the fix only bites
if the mmproj lines are removed.
