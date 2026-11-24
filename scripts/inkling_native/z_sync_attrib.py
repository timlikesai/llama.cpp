#!/usr/bin/env python3
"""Attribute cudaStreamSynchronize calls in an nsys sqlite export to the op that precedes them.

Usage: z_sync_attrib.py logs/w-nsys-prefill.sqlite
Output: plain-text sections on stdout (tee to logs/).
"""
import sqlite3
import sys
from collections import Counter, defaultdict

db = sys.argv[1] if len(sys.argv) > 1 else "logs/w-nsys-prefill.sqlite"
c = sqlite3.connect(db)

S = dict(c.execute("select id, value from StringIds"))
LAUNCH_APIS = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchKernelExC_v11060",
               "cuLaunchKernel", "cudaGraphLaunch", "cudaGraphLaunch_v10000",
               "cudaMemsetAsync", "cudaMemcpyAsync", "cudaMemcpyPeerAsync", "cudaMemcpy2DAsync")

import re

def strip_ver(n):
    return re.sub(r"_v\d+$", "", n)

# correlationId -> kernel short name (device side)
corr2kern = {}
for cid, sn in c.execute("select correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    corr2kern[cid] = S.get(sn, "?")
corr2graph = {cid: gid for cid, gid in c.execute(
    "select correlationId, graphId from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null and graphId != 0")}

# runtime records per thread, in time order
api = []
for start, end, tid, corr, nid in c.execute(
        "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start"):
    api.append((start, end, tid, corr, S.get(nid, "?")))

# 1. sync attribution: what API call immediately precedes each sync (same thread)
last_on_thread = {}
prev_of_sync = Counter()
prev_kernel_of_sync = Counter()
sync_time_by_prev_kernel = defaultdict(int)
sync_time_by_prev_api = defaultdict(int)
patterns = Counter()
sync_rows = []
n_sync = 0
for i, (start, end, tid, corr, name) in enumerate(api):
    short = strip_ver(name)
    prev = last_on_thread.get(tid)
    if short == "cudaStreamSynchronize":
        n_sync += 1
        prev_api = prev[4] if prev else "NONE"
        prev_kern = corr2kern.get(prev[2], "") if prev else ""
        prev_of_sync[strip_ver(prev_api)] += 1
        sync_time_by_prev_api[strip_ver(prev_api)] += end - start
        key = prev_kern or strip_ver(prev_api)
        prev_kernel_of_sync[key] += 1
        sync_time_by_prev_kernel[key] += end - start
        nxt = api[i + 1][4] if i + 1 < len(api) else "END"
        patterns[(strip_ver(prev_api), strip_ver(nxt))] += 1
        sync_rows.append((start, end, tid, key, end - start))
    last_on_thread[tid] = (start, end, corr, name, name)

print("== total cudaStreamSynchronize runtime records:", n_sync)
print("\n== sync count / time by PRECEDING API call on the same thread")
print(f"{'prev api':38s} {'n':>7s} {'total ms':>10s}")
for k, n in prev_of_sync.most_common(15):
    print(f"{k:38s} {n:7d} {sync_time_by_prev_api[k]/1e6:10.1f}")

print("\n== sync count / time by DEVICE WORK the preceding launch enqueued (kernel short name)")
print(f"{'preceding kernel or api':38s} {'n':>7s} {'total ms':>10s} {'avg us':>9s}")
for k, n in prev_kernel_of_sync.most_common(40):
    t = sync_time_by_prev_kernel[k]
    print(f"{k[:38]:38s} {n:7d} {t/1e6:10.1f} {t/n/1e3:9.1f}")

print("\n== (prev api -> next api) around syncs, top 15")
for (p, nx), n in patterns.most_common(15):
    print(f"{p:24s} -> {nx:24s} {n:7d}")

# 2. kernels: in-graph vs not, per device
print("\n== kernels: launched inside a CUDA graph vs bare")
tot = Counter()
for gid, cnt in c.execute("select case when graphId is null or graphId=0 then 'bare' else 'in-graph' end, count(*) from CUPTI_ACTIVITY_KIND_KERNEL group by 1"):
    tot[gid] = cnt
print(dict(tot))

print("\n== top kernels by total device time")
print(f"{'kernel':44s} {'n':>7s} {'ms':>9s} {'%':>6s}")
rows = c.execute("select shortName, count(*), sum(end-start) t from CUPTI_ACTIVITY_KIND_KERNEL group by shortName order by t desc limit 25").fetchall()
allt = c.execute("select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
for sn, n, t in rows:
    print(f"{S.get(sn,'?')[:44]:44s} {n:7d} {t/1e6:9.1f} {100*t/allt:6.1f}")
print(f"{'TOTAL':44s} {'':7s} {allt/1e6:9.1f}")

print("\n== kernels by count")
rows = c.execute("select shortName, count(*) n, sum(end-start) t from CUPTI_ACTIVITY_KIND_KERNEL group by shortName order by n desc limit 20").fetchall()
for sn, n, t in rows:
    print(f"{S.get(sn,'?')[:44]:44s} {n:7d} {t/1e6:9.1f}")

# 3. memcpy breakdown
print("\n== memcpy activity by copyKind")
kind = dict(c.execute("select id, name from ENUM_CUDA_MEMCPY_OPER"))
for r in c.execute("select copyKind, count(*), sum(bytes), sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind order by 2 desc"):
    print(f"{kind.get(r[0], r[0]):26s} n={r[1]:7d} bytes={r[2]:15d} ms={r[3]/1e6:9.1f}")

# 4. memcpy attribution: which sizes move host<->device, and when
print("\n== H2D/D2H activity: size histogram")
corr2api = {}
for corr, name in c.execute("select correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    corr2api[corr] = strip_ver(S.get(name, "?"))
for kind_name, cd in (("H2D", 1), ("D2H", 2), ("D2D", 3)):
    hist = Counter()
    tot = Counter()
    for b, in c.execute("select bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=?", (cd,)):
        hist[b] += 1
        tot[b] += b
    if not hist:
        continue
    print(f"-- {kind_name}: {sum(hist.values())} copies, {sum(tot.values())/1e9:.2f} GB total")
    print(f"   {'bytes':>12s} {'count':>7s} {'GB':>9s}")
    for b, n in hist.most_common(10):
        print(f"   {b:12d} {n:7d} {tot[b]/1e9:9.3f}")

print("\n== biggest single transfers")
for r in c.execute("select start, bytes, copyKind, end-start, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY "
                   "order by bytes desc limit 12"):
    print(f"start={r[0]} bytes={r[1]:12d} kind={r[2]} ms={r[3]/1e6:8.2f} api={corr2api.get(r[4], '?')}")

# 5. graph trace
print("\n== CUDA graph executions on device")
g = c.execute("select count(*), count(distinct graphId), sum(end-start) from CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchone()
print(f"executions={g[0]} distinct graphs={g[1]} device time={g[2]/1e6:.1f} ms")

# 6. timeline: 100 ms windows, to separate load / prefill / decode phases
print("\n== per-100ms window: syncs, sync ms, memcpy count, kernel busy ms")
min_t = c.execute("select min(start) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
w_sync_n, w_sync_t, w_memcpy, w_kern = Counter(), Counter(), Counter(), Counter()
for start, end, name in c.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    if strip_ver(S.get(name, "?")) == "cudaStreamSynchronize":
        w = (start - min_t) // 100_000_000
        w_sync_n[w] += 1
        w_sync_t[w] += (end - start) / 1e6
for start, in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMCPY"):
    w_memcpy[(start - min_t) // 100_000_000] += 1
for start, end in c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL"):
    w_kern[(start - min_t) // 100_000_000] += (end - start) / 1e6
print(f"{'win':>4s} {'syncs':>6s} {'sync ms':>8s} {'memcpy':>7s} {'kern ms':>8s}")
for w in sorted(set(w_sync_n) | set(w_memcpy) | set(w_kern)):
    print(f"{w:4d} {w_sync_n[w]:6d} {w_sync_t[w]:8.1f} {w_memcpy[w]:7d} {w_kern[w]:8.1f}")

# 7. threads
print("\n== runtime API records per thread (top 8)")
for r in c.execute("select globalTid, count(*) n, sum(end-start) t from CUPTI_ACTIVITY_KIND_RUNTIME group by globalTid order by n desc limit 8"):
    tn = c.execute("select s.value from ThreadNames t join StringIds s on s.id=t.nameId where t.globalTid=? limit 1", (r[0],)).fetchone()
    print(f"tid={r[0]} n={r[1]:7d} api_ms={r[2]/1e6:9.1f} name={(tn[0] if tn else '?')[:40]}")
