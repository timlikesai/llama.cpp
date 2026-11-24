# 53. Where the small kernels come from (named, not guessed)

Method: `GGML_SCHED_DEBUG=2` prints every node of every graph with its op, name, size and assigned
backend (`ggml-backend.cpp:2315`, printed by `ggml_backend_sched_print_assignments`). Harness mode:
`measure_variant.sh scheddump` (env `MODEL`, `SCHEDDBG`). Raw log `logs/mv-scheddump-*.log` (38 MB),
aggregation `logs/scheddump-analysis.txt`.

Representative graph: **8,004 nodes, 27 distinct ops.**

| op | nodes/graph | total bytes |
|---|---:|---:|
| MUL_MAT | 1,169 | 2,874 MiB |
| MUL | 946 | **4,611 MiB** |
| ADD | 824 | 1,398 MiB |
| **SCALE** | **821** | 2,023 MiB (but see below — mostly 0 K) |
| **CONCAT** | **644** | **2,212 MiB** |
| RMS_NORM | 514 | 1,546 MiB |
| TANH | 289 | 1,250 MiB |
| FILL | 288 | – |
| CPY | 312 | – |
| SUM_ROWS / DIV / CLAMP | 192 / 192 / 145 | – |

## The launch-count mass is the MoE router renorm

`SCALE` and `CPY` instances by name, across all 35 graphs:

```
378 x ffn_moe_weights_sum_    0K  CUDA0      2160 x s_slot_rev   0K  CUDA0
378 x ffn_moe_weights_scal    0K  CUDA0      1680 x s_slot_rev   0K  CUDA1
294 x ffn_moe_weights_sum_    0K  CUDA1        30 x blk.N.attn_res_query  4K
294 x ffn_moe_weights_scal    0K  CUDA1        30 x blk.N.ffn_res_query   4K
```

Two findings:

1. **`ffn_moe_weights_scale_*` / `_sum_*` are 0 K tensors issued thousands of times.** Inkling's router
   is `sigmoid -> top-8 -> divide by the sum -> x route_scale 8.0` (confirmed three ways: HF
   `modeling_inkling2.py:707-716`, `llama-graph.cpp:2384-2402`, and Strata's `ink_moe_cuda.cuh`), and
   llama.cpp implements that renormalization as `SUM_ROWS` + `DIV` + `SCALE` **per layer per graph**.
   With 48 applied slots that is ~800 SCALE + ~200 SUM_ROWS + ~200 DIV nodes every pass, each operating
   on eight floats. They cannot be a bandwidth problem; they are ~1,200 kernel launches per step.
2. **`s_slot_rev` CPYs (0 K, 3,840 of them)** are the loop model's slot-reversal index copies — again
   pure launch overhead, and again per slot rather than once.

`blk.N.attn_res_query` / `ffn_res_query` CPYs at 4 K are the convolution state reads (`k_sconv`), 30 per
graph, which is proportional to `-ub` and fine.

## Bytes, on the other hand

The largest byte consumers are `MUL` (4.6 GiB), `CONCAT` (2.2 GiB) and `SCALE` (2.0 GiB). The `MUL`
volume is the SituGLU activation chain (`tanh`+`sigmoid`+two `MUL`s per expert, 289 TANH nodes/graph)
across 8-of-128 experts x 48 slots — intrinsic to the architecture. `CONCAT` at 2.2 GiB deserves a
follow-up: the names are anonymous (`node_90` 68 K, `node_118` 36 K), which means they come from graph
composition rather than a model-side helper, so attribution needs `GGML_OP`-level naming or the alloc
dump. Not yet done.

## Why this matters more than it looks

The shipped-config profile said **32.7% / 35.6% device busy** with **1,229,891 launches in 39 s**, of
which 821,508 were *direct* `cudaLaunchKernelExC` rather than graph replays. If graphs were covering
the decode step, ~1,200 zero-byte launches per step would cost nothing. So the question this doc hands
to #6 is not "how do we speed these kernels up" (they are already trivial) but **why the captured
graphs do not cover them** — see `54-graph-coverage.md`.
