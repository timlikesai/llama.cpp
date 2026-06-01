# Continuous GPU Memory Shrink — Implementation Plan

**Branch:** `cuda-memory-shrink`
**Goal:** Evolve idle-only pool shrinking into a continuous background process that keeps memory pressure down during long inference sessions, preventing mid-stream OOM.

---

## Problem

With 2 parallel slots serving continuous requests, the system almost never reaches "all idle". The VMM pool and CUDA async allocator grow to peak allocation and hold it forever, causing memory pressure to creep upward during long inference sessions (e.g., coding agents with 262K context).

## Current State (commit `a7a740b6f`)

- `ggml_backend_cuda_pool_shrink_all()` — shrinks VMM pools + trims async allocator
- Server calls `ggml_cuda_shrink_if_needed()` only when **all slots are idle**
- 30-second minimum interval between shrinks
- Pool metrics exposed via `/metrics`

---

## Phase 1 — Decouple VMM Pool Shrink from Async Allocator Trim

**Risk: Low | Effort: Small | Impact: High**

The VMM pool's `shrink()` only releases already-freed tail memory — safe during active inference.
The async allocator trim (`cudaMemPoolTrimTo`) is NOT safe — it can cause driver re-allocation latency spikes.

### Changes

**`ggml/src/ggml-cuda/ggml-cuda.cu`:**
- Split `ggml_cuda_device_shrink_all()` into two functions:
  - `ggml_cuda_device_shrink_pools(device)` — pool shrink only (safe during inference)
  - `ggml_cuda_device_shrink_all(device)` — pools + async trim (idle-only)
- Export `ggml_backend_cuda_pool_shrink_all()` to shrink pools only (rename for clarity)
- Add `ggml_backend_cuda_async_trim_all()` as new API for async trim (idle path)

**`ggml/include/ggml-cuda.h`:**
- Update declarations for the split APIs

**`tools/server/server-context.cpp`:**
- Add runtime-resolved `g_cuda_async_trim_all` symbol
- Rename `g_cuda_pool_shrink_all` usage to `g_cuda_pool_shrink` (pools only)
- In `update_slots()` all-idle path: call pool shrink + async trim
- In `ggml_cuda_shrink_if_needed()`: call pool shrink only (no async trim)

---

## Phase 2 — Continuous Pool Shrink in Main Loop

**Risk: Low | Effort: Small | Impact: Medium**

Call pool shrink on every `update_slots()` iteration, not just idle. Guarded by utilization check and interval.

### Changes

**`tools/server/server-context.cpp`:**
- Reduce `POOL_SHRINK_INTERVAL_MS` from 30000 to 5000 (5 seconds)
- Move `ggml_cuda_shrink_if_needed()` call from inside `if (all_idle)` to after it — runs every iteration, gated by interval timer
- Add utilization-aware check: resolve `ggml_backend_cuda_get_pool_stats(device)` to get per-pool `pool_size`/`pool_used`, skip shrink if utilization > 70%
- Add `ggml_backend_cuda_get_pool_stats()` API in CUDA backend that returns per-pool metrics

**`ggml/src/ggml-cuda/ggml-cuda.cu`:**
- Add `ggml_backend_cuda_get_pool_stats(device, &total_size, &total_used)` — aggregates across all contexts/streams for a device

---

## Phase 3 — Background Shrink Thread

**Risk: Medium | Effort: Medium | Impact: High**

Dedicated low-priority thread that shrinks pools independently of the inference loop. Provides finer-grained control and doesn't add latency to decode.

### Changes

**`tools/server/server-context.cpp`:**
- Add `std::atomic<bool> g_shutdown` + `std::thread g_pool_shrink_thread`
- `pool_shrink_loop()`: sleep 5s → check utilization → shrink if needed → repeat
- Thread started at server init (`init()`), joined at shutdown (`server_context_free`)
- Thread set to lowest nice value (`setpriority(PRIO_PROCESS, 0, 19)`)
- Guard against concurrent shrink with atomic flag or mutex

**`tools/server/server-context.h`:**
- Add thread member to `server_context` class (or keep global with shutdown guard)

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

1. Phase 1 (decouple) — safest, no behavioral change for idle path
2. Phase 2 (continuous main loop) — adds pool shrink during active inference
3. Phase 3 (background thread) — adds independent shrink thread
4. Phase 4 (config) — polish

Each phase is independently buildable and testable.
