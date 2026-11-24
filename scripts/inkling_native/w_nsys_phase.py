#!/usr/bin/env python3
"""Attribute GPU idle time in an nsys export to the activity that surrounds it.

Usage: w_nsys_phase.py <export.sqlite> [--split-first-logits]

Stages the trace into (a) everything, (b) prefill only - the window from the
first kernel after the last >100 MB HtoD (weight upload) up to the first DtoH
of >=100 kB (the logits read that starts sampling). For each window: per-device
busy %, gap-time bucketed by what the gap is waiting on.
"""
import argparse
import bisect
import sqlite3
import sys
from collections import defaultdict

KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}


def events(db):
    ev = []
    for st, en, dev, name in db.execute(
        "select k.start,k.end,k.deviceId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id=k.demangledName"
    ):
        ev.append([st, en, dev, name])
    for st, en, dev, kind, nbytes in db.execute(
        "select start,end,deviceId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        ev.append([st, en, dev, "memcpy[%s] %.2f MB" % (KIND.get(kind, kind), nbytes / 1e6)])
    for st, en, dev in db.execute("select start,end,deviceId from CUPTI_ACTIVITY_KIND_MEMSET"):
        ev.append([st, en, dev, "memset"])
    ev.sort(key=lambda e: e[0])
    return ev


def window(ev):
    """prefill window: after the last big HtoD, before the first logits DtoH"""
    big = max((e[1] for e in ev if e[3].startswith("memcpy[HtoD] ") and float(e[3].split()[1]) > 100), default=None)
    if big is None:
        return ev[0][0], ev[-1][1]
    start = min((e[0] for e in ev if e[0] > big), default=ev[-1][0])
    end = max((e[1] for e in ev if e[3].startswith("memcpy[DtoH] ") and float(e[3].split()[1]) > 0.1 and e[0] > start),
              default=ev[-1][1])
    return start, end


def report(ev, lo, hi, top):
    w = [e for e in ev if e[1] > lo and e[0] < hi]
    span = hi - lo
    print("\n== window %.3f s" % (span / 1e9))
    for dev in sorted({e[2] for e in w}):
        d = [e for e in w if e[2] == dev]
        d.sort()
        busy = 0
        cur_a, cur_b = d[0][0], d[0][1]
        merged = []
        for e in d[1:]:
            if e[0] <= cur_b:
                cur_b = max(cur_b, e[1])
            else:
                merged.append((cur_a, cur_b))
                cur_a, cur_b = e[0], e[1]
        merged.append((cur_a, cur_b))
        busy = sum(b - a for a, b in merged)
        # gaps: idle between end of merged[i] and start of merged[i+1]; attribute to the first event after
        cat = defaultdict(lambda: [0, 0])
        gaps = []
        for i in range(len(merged) - 1):
            g0, g1 = merged[i][1], merged[i + 1][0]
            j = bisect.bisect_left([e[0] for e in d], g1)
            after = d[min(j, len(d) - 1)][3]
            key = after.split(" ")[0] + (" " + after.split(" ")[1] if "memcpy" in after else "")
            cat[key][0] += g1 - g0
            cat[key][1] += 1
            gaps.append((g1 - g0, d[min(j, len(d) - 1) - 1][3] if j else "?", after))
        idle = sum(g[0] for g in gaps)
        print("  dev%d: busy %.3f s (%.1f%%), idle %.3f s over %d gaps; %d activity"
              % (dev, busy / 1e9, 100 * busy / span, idle / 1e9, len(gaps), len(d)))
        for k, (t, n) in sorted(cat.items(), key=lambda kv: -kv[1][0])[:top]:
            print("     idle %7.1f ms in %5d gaps before %-34s (%.1f us/gap)"
                  % (t / 1e6, n, k, t / 1e3 / n))
        gaps.sort(reverse=True)
        for g, prev, after in gaps[:top]:
            print("     big gap %7.1f ms  after %-52s before %s" % (g / 1e6, prev[:52], after[:52]))

    # aggregate kernel time
    agg = defaultdict(lambda: [0, 0])
    for st, en, dev, name in w:
        agg[name][0] += en - st
        agg[name][1] += 1
    print("  -- top activities by total time")
    for name, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
        print("     %7.1f ms %6d %8.1f us  %s" % (t / 1e6, n, t / 1e3 / n, name[:78]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    ev = events(db)
    lo, hi = window(ev)
    print("first activity %d, last %d, prefill window %d..%d" % (ev[0][0], ev[-1][1], lo, hi))
    report(ev, ev[0][0], ev[-1][1], a.top)
    report(ev, lo, hi, a.top)


if __name__ == "__main__":
    sys.exit(main())
