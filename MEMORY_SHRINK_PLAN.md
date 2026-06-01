# Continuous GPU Memory Shrink — Implementation Plan

**Branch:** `cuda-memory-shrink`
**Goal:** Evolve idle-only pool shrinking into a continuous background process that keeps memory pressure down during long inference sessions, preventing mid-stream OOM.

---

## Lessons Learned

### Continuous Pool Shrink During Active Inference Is Unsafe

`llama_decode()` submits CUDA kernels **asynchronously** to inference streams, then returns. The VMM pool shrink calls `cuMemUnmap()` which runs on the **default stream (stream 0)**. There is no ordering guarantee between the inference stream and stream 0 — the GPU can still be reading from memory we just unmapped → **illegal memory access**.

This was confirmed by a crash during MTP speculative decoding:
```
CUDA error: an illegal memory access was encountered
... common_speculative_impl_draft_mtp::process → llama_get_embeddings_pre_norm
```

### Why Stream Synchronization Doesn't Help

To make pool shrink safe during inference, we'd need to `cudaStreamSynchronize()` all inference streams before calling `cuMemUnmap()`. This blocks for the full kernel execution latency (milliseconds per decode step), completely negating the benefit of proactive shrinking.

### What Works

- **Idle-only shrink:** Safe — all kernels are complete, streams are quiescent
- **VMM pool shrink:** Safe when no inference is active — releases freed tail region
- **Async allocator trim:** Safe when no inference is active — releases cached pages
- **Pool stats API:** Useful for monitoring, no safety concerns

---

## Problem

With 2 parallel slots serving continuous requests, the system almost never reaches "all idle". The VMM pool and CUDA async allocator grow to peak allocation and hold it forever, causing memory pressure to creep upward during long inference sessions (e.g., coding agents with 262K context).

## Current State (commit `a7a740b6f`)

- `ggml_backend_cuda_pool_shrink_all()` — shrinks VMM pools + trims async allocator
- Server calls `ggml_cuda_shrink_if_needed()` only when **all slots are idle**
- 30-second minimum interval between shrinks
- Pool metrics exposed via `/metrics`

---

## Phase 1 — Decouple VMM Pool Shrink from Async Allocator Trim (DONE)

**Risk: Low | Effort: Small | Impact: High**

The VMM pool's `shrink()` only releases already-freed tail memory. The async allocator trim
(`cudaMemPoolTrimTo`) is NOT safe during active inference — it can cause driver re-allocation latency spikes.

### Changes (committed)

**`ggml/src/ggml-cuda/ggml-cuda.cu`:**
- Split `ggml_cuda_device_shrink_all()` into `ggml_cuda_device_shrink_pools()` + async trim
- Exported `ggml_backend_cuda_pool_shrink_all()` (pools only)
- Exported `ggml_backend_cuda_shrink_all()` (pools + async trim)
- Exported `ggml_backend_cuda_async_trim_all()` (async trim only)
- Added `ggml_backend_cuda_get_pool_stats()` for monitoring

**`ggml/include/ggml-cuda.h`:**
- Updated declarations for split APIs

**`tools/server/server-context.cpp`:**
- Runtime-resolved symbols for split APIs
- Idle path uses `ggml_backend_cuda_shrink_all()` (pools + async trim)

---

## Phase 2 — Continuous Pool Shrink in Main Loop (ABANDONED)

**Status: ABANDONED — unsafe during active inference**

`llama_decode()` submits kernels asynchronously. Pool shrink calls `cuMemUnmap()` on
stream 0, racing with in-flight kernels on other streams → illegal memory access.
Stream synchronization would block for full kernel latency, negating the benefit.

## Phase 2 (Revised) — Background Shrink Thread With Stream Synchronization

**Risk: Medium | Effort: Medium | Impact: Medium**

A dedicated background thread that waits for stream quiescence before shrinking.
Instead of synchronizing (which blocks), it polls stream activity and shrinks only
when no kernels are in flight.

### Changes

**`tools/server/server-context.cpp`:**
- `std::thread g_pool_shrink_thread` — low-priority background thread
- Poll `cudaStreamQuery()` on all inference streams before shrinking
- Only shrink when all streams report complete (no in-flight kernels)
- 5-second sleep interval between checks

**`ggml/src/ggml-cuda/ggml-cuda.cu`:**
- Export `ggml_backend_cuda_stream_query(device, stream_idx)` — non-blocking query
- Or expose `streams()` accessor on `ggml_backend_cuda_context`

---

## Phase 3 — Background Shrink Thread (Deferred Until Phase 2 Revised)

**Risk: Medium | Effort: Medium | Impact: Medium**

Dedicated low-priority thread that shrinks pools only when inference streams are quiescent.
Depends on Phase 2 revised (stream polling API).

### Changes

**`tools/server/server-context.cpp`:**
- `std::thread g_pool_shrink_thread` — low-priority background thread
- `pool_shrink_loop()`: sleep 5s → poll streams → shrink if quiescent → repeat
- Thread started at server init (`init()`), joined at shutdown
- Thread set to lowest nice value (`setpriority(PRIO_PROCESS, 0, 19)`)
- Guard against concurrent shrink with atomic flag

---

## Phase 4 — Configuration and Observability

**Risk: Low | Effort: Small | Impact: Nice-to-have**

### Changes

**Server CLI flags:**
- `--pool-shrink-interval` (default: 5 seconds)
- `--pool-shrink-threshold` (default: 0.3 — shrink when >30% of pool is unused)

**Logging:**
- `SRV_DBG` messages for each shrink event with bytes released
- Prometheus metrics: `llamacpp:gpu_pool_shrink_bytes_total` counter, `llamacpp:gpu_pool_shrink_interval_seconds` gauge

**`tools/server/server-common.h`:**
- Add params fields for shrink configuration

---

## File Change Summary

| File | Phase | Changes |
|------|-------|---------|
| `ggml/include/ggml-cuda.h` | 1,2 | Split API declarations, add pool stats |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 1,2 | Split shrink functions, add pool stats API |
| `ggml/src/ggml-cuda/common.cuh` | — | No changes needed |
| `tools/server/server-context.cpp` | 1,2,3 | Decouple trim, continuous shrink, background thread |
| `tools/server/server-common.h` | 4 | Config parameters |
| `tools/server/CMakeLists.txt` | — | No changes (dl already linked) |

---

## Execution Order

1. ~~Phase 1 (decouple) — DONE~~
2. ~~Phase 2 (continuous main loop) — ABANDONED (unsafe)~~
3. Phase 2 revised (background thread with stream polling) — pending
4. Phase 3 (background thread) — pending (depends on Phase 2 revised)
5. Phase 4 (config) — polish

## Current State

- Pool + async trim APIs split cleanly in CUDA backend
- Idle path uses `ggml_backend_cuda_shrink_all()` (pools + async trim)
- Pool stats API available for monitoring
- Continuous shrink removed from main loop (documented why unsafe)
