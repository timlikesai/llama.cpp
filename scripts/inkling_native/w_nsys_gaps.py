#!/usr/bin/env python3
"""Gap / attribution analysis of an nsys sqlite export for Inkling prefill.

Usage: w_nsys_gaps.py <export.sqlite> [--window start_ns end_ns] [--top N]

Prints, per CUDA device: kernel+memcpy busy time, wall span of the window,
idle ratio, the biggest idle gaps and the kernel that precedes each gap, and a
per-kernel aggregate. Used to separate "GPU is doing slow math" from "GPU is
waiting for the host".
"""
import argparse
import sqlite3
import sys
from collections import defaultdict


def load(db, lo, hi):
    per_dev = defaultdict(list)
    rows = db.execute("""
        select k.start, k.end, k.deviceId, s.value
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on s.id = k.demangledName
    """).fetchall()
    for st, en, dev, name in rows:
        if en < lo or st > hi:
            continue
        per_dev[dev].append((max(st, lo), min(en, hi), name))
    mem = db.execute("""
        select m.start, m.end, m.deviceId, m.copyKind, m.bytes
        from CUPTI_ACTIVITY_KIND_MEMCPY m
    """).fetchall()
    for st, en, dev, kind, nbytes in mem:
        if en < lo or st > hi:
            continue
        per_dev[dev].append((max(st, lo), min(en, hi), "memcpy[%d] %d B" % (kind, nbytes)))
    return per_dev


def busy(intervals):
    """merge overlapping intervals -> total busy ns and list of gaps"""
    if not intervals:
        return 0, []
    iv = sorted((a, b) for a, b, _ in intervals)
    total = 0
    cur_a, cur_b = iv[0]
    merged = []
    for a, b in iv[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            total += cur_b - cur_a
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    total += cur_b - cur_a
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]
    return total, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--window", nargs=2, type=int, default=None)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    db = sqlite3.connect(a.sqlite)
    if a.window:
        lo, hi = a.window
    else:
        lo, hi = db.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()

    span = hi - lo
    per_dev = load(db, lo, hi)
    print("window: %.3f s (%d .. %d)" % (span / 1e9, lo, hi))

    # kernel->end lookup for "what ran just before the gap"
    all_iv = [iv for lst in per_dev.values() for iv in lst]
    all_iv.sort(key=lambda x: x[1])
    ends = [x[1] for x in all_iv]

    import bisect
    for dev, lst in sorted(per_dev.items()):
        b, gaps = busy(lst)
        print("\n== device %s: busy %.3f s of %.3f s window (%.1f%%), %d intervals, %d gaps, idle %.3f s"
              % (dev, b / 1e9, span / 1e9, 100.0 * b / span, len(lst), len(gaps),
                 sum(g[1] - g[0] for g in gaps) / 1e9))
        gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
        for st, en in gaps[: a.top]:
            i = bisect.bisect_left(ends, st)
            prev = all_iv[max(0, i - 1)][2][:70] if i > 0 else "-"
            nxt_i = bisect.bisect_left([(x[0]) for x in all_iv], en)
            nxt = all_iv[min(nxt_i, len(all_iv) - 1)][2][:70]
            print("   gap %8.3f ms after %-70s | before %s" % ((en - st) / 1e6, prev, nxt))

    print("\n== per-kernel totals (all devices, windowed)")
    agg = defaultdict(lambda: [0, 0])
    for lst in per_dev.values():
        for st, en, name in lst:
            agg[name][0] += en - st
            agg[name][1] += 1
    for name, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[: a.top]:
        print("  %8.1f ms  %6d  %6.1f us/call  %s" % (t / 1e6, n, t / 1e3 / n, name[:90]))


if __name__ == "__main__":
    sys.exit(main())
