# 37 — Prefill kernel profile, before/after the f16 mix, and what is left

Measured with `nsys` on 2× RTX 5060 Ti using the complete lossless file (`Inkling-2-Mini-FULL.gguf`,
weights + embedded drafter), ctx 8192, `-b 2048 -ub 512`, flash attention on, prompt
`logs/g0-prompt-long.txt`. Reports: `logs/prefill-kern-sum.txt` (before), `logs/prefill-kern-sum2.txt`
(after). Raw captures `logs/prefill-prof*.nsys-rep` kept.

## 1. The finding that redirected the work

I went looking for a slow **activation quantizer** (the MXFP8 e4m3+ue8m0 kernel is scalar-per-thread,
while the NVFP4 one uses `float8` loads and a two-level reduce — a legitimate criticism on paper).
The profile says `quantize_mmq_mxfp8` does not appear among the top kernels at all. Rewriting it would
have been churn against a kernel that costs nothing measurable, so it stays as it is. That is the
answer to #45, and it is a *negative* answer, which is the useful kind.

What actually dominated was not tensor-core math:

| kernel | share before | what it is |
|---|---|---|
| `mul_mat_q<NVFP4>` | 11.6% | the expert MMQ path — the real arithmetic, fine |
| `k_get_rows_float_vec<float>` | 11.1% | f32 gathers (400 calls) |
| `mul_mat_vec_f<float,float>` | 10.3% | **f32 batched matmuls — the depth mix** |
| `cpy_scalar<float,float>` | 9.5% | 2,960 copy calls |
| `mul_mat_q<MXFP8>` | 4.0% | my MXFP8 MMA path — confirmed live in prefill |
| `cpy_scalar_transpose<float>` | 4.3% | **the mix's materialized transpose** |

## 2. What the f16 mix removed (commit `f3c27a4c3`)

`mix()` assembles all earlier block outputs into one `states` tensor and walks it in f32: a broadcast
copy of the query, a materialized transpose (CUDA has no TN form for the block-axis contraction), and
two batched matmuls. Running it in f16 for batched evaluation:

| | before | after |
|---|---|---|
| prefill @8192 | 812.8 t/s | **883.9 t/s (+8.7%)** |
| prefill @262144 | 740.3 t/s | **795 t/s (+7.3%)** |
| decode @8192 | 39.8 t/s | 39.7–39.8 (unchanged; single tokens keep the f32 path) |
| PPL (wikitext-2, 8 chunks) | 5.4373 ± 0.13969 | 5.4354 ± 0.13916 |

Gone from the table: `mul_mat_vec_f<float,float>` (10.3%) and `cpy_scalar_transpose<float>` (4.3%).
The f16 matmuls appear as `cutlass_80_tensorop_s16816gemm_bf16` / `cpy_scalar_contiguous<float,__half>`.

Two things this change taught me, both recorded because they generalize:

- **`GGML_PREC_F32` is not the default on those matmuls.** My first version asserted in a comment that
  "accumulation stays f32" — unverified, and wrong: with the default precision CUDA may accumulate in
  f16 while the CPU widens to f32, and greedy output **diverged between backends at 16 tokens**. I
  A/B-ed the divergence against a stashed tree (so I knew it was mine, not the pre-existing CPU
  defect) and fixed it with `ggml_prec_set_acc(..., GGML_PREC_F32)` on both matmuls.
- **`ggml_cast` never no-ops.** Casting a tensor to the type it already has emits a CPY, so the
  decode path and the prefill path are written out separately instead of unified behind a type
  variable — and decode is left on f32 on purpose, because there the extra cast nodes cost more than
  the bytes they save (measured: −3.5% when applied unconditionally).

## 3. What is left — attributed to tensors, not guessed

Graph attribution with `GGML_SCHED_DEBUG=2 -v` through `scripts/inkling_native/parse_graph.py`
(`logs/graph-prefill.log`), one prefill of 2048 tokens in 4 ubatches:

| input tensor | nodes | total elements | approx bytes moved |
|---|---|---|---|
| `attn_inp_kq_mask` (f16, CPY) | 480 | 126,320,640 | ~252 MB |
| `attn_inp_rel_pos_swa` (i32, GET_ROWS) | 480 | 126,320,640 | ~505 MB |
| `attn_inp_rel_pos` (i32, GET_ROWS) | 96 | 33,685,504 | ~135 MB |
| `rs_s_copy` (i32, GET_ROWS) | 576 | 576 | negligible (12 nodes × 48 slots of scalars) |

Reading: the mask and the relative-position indices are materialized **once per applied layer slot**
(48 slots ⇒ 480 nodes across 4 ubatches plus the loop's second pass), not once per ubatch. The i32
index tensors cost more bytes than the f16 masks they feed. That is the same class of finding as
`e046294a7` (code iterating applied slots that should be shared) and it is the biggest remaining
non-MMA traffic in prefill: roughly 890 MB of index/mask traffic per 2048-token prompt.

Two candidate fixes, both needing care because `llm_graph_input_kq_mask` is shared by every hybrid
model:

1. **Compute `rel_pos` once per ubatch and share the tensor across slots.** For a single sequence the
   distances depend only on positions and the window, not on the layer, so the per-slot i32 gather is
   redundant. This is the larger win (505 MB) and the smaller code change if the reuse path
   (`can_reuse`) tolerates a shared tensor.
2. **Skip the per-slot CPY of the mask** by letting consumers take a view. Note #33 already tried
   exactly this shape of change for the *dense* mask and measured **zero** benefit (a leaf input sits
   under the allocator's peak), so this must be measured the same way and reverted if it does not pay.

Also still open from #30: giving `FLASH_ATTN_EXT` a bias operand (#35) would delete the gathered bias
entirely, which subsumes item 1 for the bias but not for the mask.

## 4. How to reproduce

```
M=~/models/timlikesai/Inkling-2-Mini-FULL.gguf
nsys profile -t cuda --force-overwrite=true -o logs/prefill-prof ./build-cuda/bin/llama-cli \
    -m $M -c 8192 -b 2048 -ub 512 -fa on --temp 0 -s 42 --single-turn --no-warmup \
    -f logs/g0-prompt-long.txt -n 32
nsys stats -r cuda_gpu_kern_sum logs/prefill-prof.nsys-rep

GGML_SCHED_DEBUG=2 ./build-cuda/bin/llama-cli -m $M -ngl 99 -c 2048 -b 2048 -ub 512 -fa on \
    --temp 0 --single-turn --no-warmup -f logs/g0-prompt-long.txt -n 2 -v > logs/graph-prefill.log
python3 scripts/inkling_native/parse_graph.py logs/graph-prefill.log
```

Note: `--pg` is not a flag in this build and `-v5` is invalid (verbosity is `-v` / `--verbose N`);
`llama-completion` without `--jinja` still aborts on Inkling's chat template.
