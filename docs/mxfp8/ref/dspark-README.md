---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
tags:
  - inkling2
  - inkling-2-mini
  - vllm
  - speculative-decoding
  - dspark
  - blackwell
  - sm_120
  - rtx-pro-6000
---

# DSpark speculative decoding for Inkling 2 Mini on sm_120 (RTX PRO 6000 / RTX 50xx)

This is a one-file patch for the `vllm/vllm-openai:inkling-2-mini` image. With it, the Inkling 2 Mini DSpark draft (`mtp.safetensors`) runs on sm_120 GPUs. Without it, any `--speculative-config '{"method":"dspark",...}'` launch dies during warmup:

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
  File ".../vllm/models/inkling2/nvidia/dspark.py", line 209, in forward
    out = q.new_zeros((q.shape[0], self.q_size))
```

The traceback line is misleading. `new_zeros` is fine. The error is raised asynchronously by the kernel that ran just before it.

## Root cause

The draft's attention layers write their KV with `FlashAttentionImpl.do_kv_cache_update`, which calls `reshape_and_cache_flash` from `vllm._C`. The draft cache is bf16 with layout `(blocks, kv_heads, page, 2 * head_dim)` and is passed as the head-strided view `kv_cache.transpose(1, 2).split(head_dim, -1)`. On this image's build, `reshape_and_cache_flash` has no sm_120 kernel for that call. The target model's own layers never hit this path, so plain serving works and only DSpark crashes.

`probes/repro_reshape_and_cache_flash.py` reproduces the failure in about 10 lines, outside the server.

How it was found: `CUDA_LAUNCH_BLOCKING=1` did not pin it. Calling `cudaPeekAtLastError()` through ctypes on the already-loaded `libcudart` after each step of the draft forward did. The first non-zero result came right after `_write_kv`.

## The fix

`dspark.py` is the image's file plus a small Triton kernel, `_sm120_write_kv`, that does the same paged K/V write. `_write_kv` uses it only when all three are true: the GPU is sm_120, `kv_cache_dtype == "auto"`, and the cache dtype matches K (bf16). Every other case still calls `do_kv_cache_update` unchanged. `dspark-sm120.patch` is the same change as a unified diff (69 lines).

`probes/verify_sm120_write_kv.py` checks it against a reference scatter. Result: K and V max error 0.0, and slots of -1 are skipped.

## Use

Mount the file over the image's copy:

```bash
-v $PWD/dspark.py:/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2/nvidia/dspark.py:ro
```

`run-vllm.sh` is the full launcher we run in production:

```bash
MODEL_DIR=/path/to/Inkling-2-Mini   # HF checkpoint dir that contains mtp.safetensors
PORT=8000 SPEC=7 GMU=0.60 ./run-vllm.sh
```

Notes:
- `num_speculative_tokens` must be 7. The checkpoint's block size fixes it, and 3, 4 or 5 refuse to boot.
- The patch replaces the whole `dspark.py`. Re-check it against any newer image before mounting, because a newer upstream file would be overwritten.
- Tested with image digest `sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e`, driver 615.71.09, on an RTX PRO 6000 Blackwell Workstation Edition (96 GB).
- `chat_template_medium.jinja` is optional. It is the stock template with the default reasoning effort lowered from 0.9 to 0.7. At 0.9 the model can spend a long-document answer's whole token budget on thinking.

## Results

All runs: temperature 0, `reasoning_effort: low`, NVFP4/MXFP8 HF checkpoint, fp8 KV for the target, 600 max tokens. `bench_sweep.py` produced the table and `bench_results.jsonl` is its raw output.

| Config | Prose | Code | List | Agent | 1-user mean | 4 users, per stream |
|---|---|---|---|---|---|---|
| No speculation | 168 | 168 | 168 | 168 | 168 | 98 |
| DSpark, probabilistic draft (patched) | 156 | 372 | 225 | 231 | 246 | 153 |
| DSpark, greedy draft | 156 | 372 | 225 | 231 | 246 | 152 |
| DSpark, CUDA graphs up to 128 tokens | 156 | 371 | 224 | 230 | 246 | 152 |

All numbers are tok/s. Mean acceptance length was about 2.7 to 3.1 tokens per step. Novel prose drafts poorly and runs about 7% slower than no speculation. Code and structured output gain the most.

Also checked on the patched production server (262K context):
- 5 of 5 tool calls came back with valid JSON arguments.
- An 89K-token needle-in-a-haystack test answered correctly in 14.4 s.
- The same prompt gives identical output run to run. The text is not token-identical to the no-speculation server, because the draft sampling path differs. Answers checked were correct on both.

## Files

| File | What it is |
|---|---|
| `dspark.py` | Patched drop-in for `vllm/models/inkling2/nvidia/dspark.py` |
| `dspark-sm120.patch` | Same change as a diff against the image's file |
| `run-vllm.sh` | Docker launcher (env: MODEL_DIR, PORT, SPEC, GMU, SEQS, MNBT, KVD, DSM, CGMAX, MAXLEN) |
| `chat_template_medium.jinja` | Optional template with default effort 0.7 |
| `probes/repro_reshape_and_cache_flash.py` | Minimal repro of the sm_120 failure |
| `probes/verify_sm120_write_kv.py` | Exactness check of the Triton writer (run inside the image with the patch mounted) |
| `bench_sweep.py`, `bench_results.jsonl` | Benchmark harness and raw results |

## License

Apache 2.0, the same as vLLM. The patch modifies a vLLM source file and keeps its SPDX header.
