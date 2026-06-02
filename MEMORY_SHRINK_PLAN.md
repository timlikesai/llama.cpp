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

### Why cudaStreamQuery() Is Insufficient (TOCTOU Race)

The background thread uses `cudaStreamQuery()` to check if streams are idle. This is a
**point-in-time snapshot** — it returns `cudaSuccess` if no work is currently in flight on
that stream. However, between the query returning and `cuMemUnmap()` executing:

1. Inference thread can call `llama_decode()` → `graph_compute_async()` → submit new kernels
2. Background thread calls `cuMemUnmap()` → frees VMM tail memory
3. Newly submitted kernel reads/writes to now-unmapped memory → crash

The `g_cuda_ctx_mutex` is released **before** `cuMemUnmap()` runs (it is held only inside
`ggml_cuda_are_streams_quiescent()`), so it does not protect the critical section. The
check and the action are not atomic.

**Crash reproduction:**
```
gpu pool shrink (bg): size 234 -> 0 MiB, freed 234 MiB
CUDA error: an illegal memory access was encountered
  → ggml_backend_cuda_buffer_set_tensor (cudaMemcpyAsync D→D, cudaStreamSynchronize)
  → common_speculative_impl_draft_mtp::process (MTP draft, device 1)
```

18ms between shrink and crash: the GPU was executing a kernel on memory that was just
unmapped. The kernel was submitted after `cudaStreamQuery` said "idle" but before
`cuMemUnmap` ran.

### What Works

- **Idle-only shrink:** Safe — all kernels are complete, streams are quiescent
- **VMM pool shrink:** Safe when no inference is active — releases freed tail region
- **Async allocator trim:** Safe when no inference is active — releases cached pages
- **Pool stats API:** Useful for monitoring, no safety concerns
- **Lock-protected shrink:** Safe — mutex prevents new work submission during shrink
- **Event-based shrink:** Safe — `cudaEventQuery` provides causal ordering, not just snapshot

---

## CUDA Event Mechanisms Available in Codebase

The codebase already has rich CUDA event/stream infrastructure that can be leveraged:

| Mechanism | Current Usage in llama.cpp | Relevance |
|-----------|---------------------------|-----------|
| `cudaEventRecord` / `cudaStreamWaitEvent` | Cross-GPU tensor copies, concurrent MUL_MAT streams, split backends, peer-to-peer copies | **Core primitive** — mark "compute submitted" then wait for it |
| `cudaEventSynchronize` | Backend event API (`ggml_backend_event_synchronize`) | Host-side wait for GPU completion |
| `cudaEventQuery` | Not currently used | **Key addition** — non-blocking: "has this event fired?" — solves TOCTOU |
| `cudaStreamQuery` | Current (broken) quiescence check | Fragile — point-in-time snapshot, no causal guarantee |
| `cudaStreamSynchronize` | Buffer ops, device memory queries | Too heavy for our use case |

**Existing event pattern for cross-device copies** (ggml-cuda.cu:3367-3373):
```cpp
cudaEventRecord(event, src_stream);       // mark work submitted on src
cudaStreamWaitEvent(dst_stream, event);   // dst waits for src to complete
```

This is exactly the pattern we need: record event after `graph_compute_async` →
background thread checks event with `cudaEventQuery` → shrink only when event has
fired (all prior work GPU-complete).

### Why Events Solve the TOCTOU Problem

`cudaStreamQuery` is a snapshot — "are streams idle right now?" New work can be submitted
immediately after.

`cudaEventQuery` is causal — "has the work recorded at this event completed?" The event
is recorded **at the exact moment work is submitted**, creating a causal chain:

```
graph_compute_async() → cudaEventRecord(event, stream)
                                   ↓
                          (GPU executes all prior work)
                                   ↓
                     cudaEventQuery(event) == cudaSuccess
                                   ↓
                          Safe to shrink — all prior work done
```

If a new `graph_compute_async` hasn't been called yet, there is no new event to check —
the old event still represents the last submitted work. If the old event fired, all that
work is done. If a new `graph_compute_async` is called, a new event is recorded — the
old event does not retroactively cover it.

The background thread approach using events:
1. After each `graph_compute_async`, record an event on each device's inference stream
2. Background thread: `cudaEventQuery(event)` — if all events fired → safe to shrink
3. If events haven't fired → sleep and retry
4. No mutex needed between check and shrink — the event provides the guarantee

---

## Problem

With 2 parallel slots serving continuous requests, the system almost never reaches "all
idle". The VMM pool and CUDA async allocator grow to peak allocation and hold it forever,
causing memory pressure to creep upward during long inference sessions (e.g., coding
agents with 262K context).

---

## Bugs Found in Current Implementation

### Critical

| Bug | Location | Impact | Fix |
|-----|----------|--------|-----|
| Background thread TOCTOU race | `ggml_cuda_pool_shrink_loop()` | **Crash** — `cuMemUnmap` races with inference kernel submission | Remove background thread; use lock-protected inline shrink |
| `g_last_pool_shrink_ms` data race | `server-context.cpp:33` | UB (rare crash on some compilers/architectures) — plain `int64_t` accessed from 2 threads | `std::atomic<int64_t>` |

### Moderate

| Bug | Location | Impact | Fix |
|-----|----------|--------|-----|
| Legacy pool `shrink()` frees ALL buffers | `ggml_cuda_pool_leg::shrink()` → `clear_pool()` | Performance: forces driver reallocation on next inference | Only free buffers above a size threshold, or skip if recent allocation |
| Missing `ggml_cuda_free_runtime()` on shutdown | `server_context_impl::destroy()` | Resource leak: `dlclose` and thread join from phase 1 never fires | Call `ggml_cuda_free_runtime()` from `destroy()` |
| `ggml_cuda_init_runtime()` on every metrics request | `server_context_impl::handle_task_metrics()` | Wasted CPU cycles (benign due to early return guard) | Call once at init, store result |

### Low / Cosmetic

| Issue | Location | Impact |
|-------|----------|--------|
| VMM destructor `cuMemUnmap` uses reduced `pool_size` after shrink | `~ggml_cuda_pool_vmm()` | Works correctly per CUDA docs (unmap before release), but confusing | Add comment explaining |
| `std::remove` + `erase` in context unregister | `ggml_backend_cuda_free()` | Correct, but could use `erase(it)` directly | Minor cleanup |

---

## Current State (after crash analysis)

- Pool + async trim APIs split cleanly in CUDA backend
- Idle path uses `ggml_backend_cuda_shrink_all()` (pools + async trim, 30s interval)
- **Background thread: PROVEN UNSAFE — causes crashes**
- `ggml_backend_cuda_are_streams_quiescent()`: TOCTOU — insufficient guarantee
- Pool stats API available for monitoring
- Thread started at server init, joined at destroy (but `free_runtime` not called)

---

## Phase 1 — Decouple VMM Pool Shrink from Async Allocator Trim (DONE)

**Risk: Low | Effort: Small | Impact: High**

The VMM pool's `shrink()` only releases already-freed tail memory. The async allocator trim
(`cudaMemPoolTrimTo`) is NOT safe during active inference — it can cause driver re-allocation
latency spikes.

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

---

## Phase 2 (Revised) — Background Shrink Thread (ABANDONED)

**Status: ABANDONED — TOCTOU race with cudaStreamQuery, crash confirmed in production**

`cudaStreamQuery()` provides a point-in-time snapshot, not a causal guarantee.
Between "query says idle" and `cuMemUnmap` executes, new inference work can be
submitted → crash confirmed in production.

**See "Why cudaStreamQuery() Is Insufficient" above for detailed analysis.**

---

## Phase 3 (New) — Lock-Protected Inline Shrink (IMPLEMENT NOW)

**Risk: Low | Effort: Small | Impact: High**

Remove the background thread entirely. Replace with a mutex-protected inline shrink
called at safe points in the main loop.

### Why a Lock Solves the Problem

`cuMemUnmap` is a **synchronous CPU-to-driver call** — it completes before returning.
If we hold a mutex that prevents new inference work from being submitted during the
shrink, then:

1. Kernels submitted before the lock was acquired are already using mapped memory
2. No new work is submitted during the lock hold (the mutex blocks it)
3. After lock release, new work can allocate from the pool (which grows via `cuMemMap`)

The lock prevents the TOCTOU race entirely. It does not need to wait for GPU completion
because `cuMemUnmap` only releases **freed tail memory** (above `pool_used`).

### Lock Design

Single mutex (`g_pool_shrink_mutex`) that covers:
- **Inference path:** held around `llama_decode()` call in the main loop (specifically
  around `graph_compute_async`)
- **Shrink path:** held around `ggml_cuda_device_shrink_pools()` + `cuMemUnmap`

The inference path holds the lock only during the async submit (microseconds), not
during kernel execution (milliseconds). The shrink path holds the lock during the
synchronous `cuMemUnmap` call (also microseconds).

### Changes

**`ggml/src/ggml-cuda/ggml-cuda.cu`:**
- Add `std::mutex g_pool_shrink_mutex` (global, in the CUDA backend)
- `ggml_cuda_device_shrink_pools()` acquires this mutex around `cuMemUnmap`
- Export `ggml_backend_cuda_pool_shrink_all_safe()` — lock-protected version
- Remove `ggml_backend_cuda_are_streams_quiescent()` (no longer needed)
- Remove `g_cuda_contexts` tracking (only needed for background polling)

**`ggml/include/ggml-cuda.h`:**
- Add `ggml_backend_cuda_pool_shrink_all_safe()` declaration
- Remove `ggml_backend_cuda_are_streams_quiescent()` declaration

**`tools/server/server-context.cpp`:**
- Remove background thread entirely (`g_pool_shrink_thread`, `ggml_cuda_pool_shrink_loop`, etc.)
- Remove `ggml_cuda_start_pool_shrink_thread()` / `ggml_cuda_stop_pool_shrink_thread()`
- Add inline shrink call in main loop after `update_slots()` (between decode steps, when
  no inference is in flight)
- Call `ggml_cuda_free_runtime()` from `destroy()`
- Fix `g_last_pool_shrink_ms` to use `std::atomic<int64_t>`

---

## Phase 4 (New) — Event-Based Background Shrink (FUTURE)

**Risk: Medium | Effort: Medium | Impact: Medium**

Replace the lock with CUDA event-based synchronization. Zero CPU contention.

### Design

1. **Record event after each `graph_compute_async`:**
   - Per-device `cudaEvent_t last_compute_event` on `ggml_backend_cuda_context`
   - `cudaEventRecord(event, stream)` after each compute submission
   - Export `ggml_backend_cuda_record_compute_event(int device)` for server to call

2. **Background thread checks events, not streams:**
   - `cudaEventQuery(event)` — non-blocking: "has prior compute completed?"
   - If all device events fired → safe to shrink (all prior work GPU-complete)
   - If any event still pending → sleep and retry
   - No mutex needed — the event provides causal ordering

3. **Export new API:**
   - `ggml_backend_cuda_record_compute_event(int device)` — server calls after decode
   - `ggml_backend_cuda_compute_event_done(int device)` — background thread polls

### Changes

**`ggml/src/ggml-cuda/common.cuh`:**
- Add `cudaEvent_t last_compute_event = nullptr` to `ggml_backend_cuda_context`

**`ggml/src/ggml-cuda/ggml-cuda.cu`:**
- Implement `ggml_backend_cuda_record_compute_event()` — records event on current stream
- Implement `ggml_backend_cuda_compute_event_done()` — queries all device events
- Background thread uses `compute_event_done` instead of `are_streams_quiescent`

**`ggml/include/ggml-cuda.h`:**
- Add `ggml_backend_cuda_record_compute_event()` declaration
- Add `ggml_backend_cuda_compute_event_done()` declaration

---

## File Change Summary

| File | Phase | Changes |
|------|-------|---------|
| `ggml/include/ggml-cuda.h` | 1,3,4 | Split API declarations, lock-protected shrink, event API |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 1,3,4 | Split shrink functions, pool stats API, mutex, event recording |
| `ggml/src/ggml-cuda/common.cuh` | 4 | Add `last_compute_event` to CUDA context |
| `tools/server/server-context.cpp` | 1,3,4 | Decouple trim, remove background thread, inline shrink, event recording |
| `tools/server/server-task.h` | 1 | Pool metrics in response |
| `tools/server/server-task.cpp` | 1 | Serialize pool metrics |
| `tools/server/CMakeLists.txt` | 1 | Link `dl` on Linux |
| `MEMORY_SHRINK_PLAN.md` | — | This plan |

---

## Execution Order

1. Phase 1 (decouple) — DONE
2. Phase 2 (continuous main loop) — ABANDONED (unsafe)
3. Phase 2 revised (background thread with stream polling) — ABANDONED (TOCTOU crash)
4. **Phase 3 (lock-protected inline shrink) — IMPLEMENT NOW**
5. Phase 4 (event-based background shrink) — FUTURE (after Phase 3 proves stable)
6. Phase 4 config (CLI flags, observability) — polish (optional)

---

## Current State

- Pool + async trim APIs split cleanly in CUDA backend
- Idle path uses `ggml_backend_cuda_shrink_all()` (pools + async trim, 30s interval)
- Background thread: **PROVEN UNSAFE, must be removed**
- Crash confirmed: TOCTOU between `cudaStreamQuery` and `cuMemUnmap`
- Pool stats API available for monitoring
- Next step: implement Phase 3 (lock-protected inline shrink)