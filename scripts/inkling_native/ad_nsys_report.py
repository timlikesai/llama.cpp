#!/usr/bin/env python3
"""Phase + kernel report for an nsys sqlite export of an Inkling run (doc 28).

Stages the trace into three windows and reports each one:
  whole    - everything captured
  prefill  - after the last >100 MB HtoD (weight upload) up to the first DtoH >= 0.1 MB (logits)
  decode   - that first logits read to the last activity

Per window: per-device busy/idle with the gap attribution, top kernels by total device time,
a rollup by op class, and the CUDA runtime API counts that matter for cross-device work.

Usage: ad_nsys_report.py <export.sqlite> [--tokens-prompt N] [--tokens-decode N] [--top N]
"""
import argparse
import bisect
import sqlite3
from collections import defaultdict

KIND = {1: "HtoD", 2: "DtoH", 3: "HtoH", 4: "DtoD", 8: "MTP2P", 9: "MTDP", 10: "PtoP"}

# kernel-name -> op class, so the report says what kind of work dominates rather than only which symbol
CLASS = [
    ("mul_mat_id", "moe-gemm"),
    ("mul_mat_q", "gemm-quant"),
    ("mul_mat_vec", "gemv-quant"),
    ("mmv_", "gemv-quant"),
    ("mmq", "gemm-quant"),
    ("gemm", "gemm-f32"),
    ("magma", "gemm-f32"),
    ("cutlass", "gemm-f32"),
    ("flash_attn", "attention-fa"),
    ("soft_max", "attention-dense"),
    ("k_bin_bcast", "attention-mask"),
    ("get_rows", "gather"),
    ("set_rows", "scatter"),
    ("cpy_", "copy/convert"),
    ("convert_", "copy/convert"),
    ("dup_", "copy/convert"),
    ("rms_norm", "norm"),
    ("rope", "rope"),
    ("ssm_conv", "conv-state"),
    ("quantize", "quantize"),
    ("moe_weighted", "moe-combine"),
    ("mm_ids", "moe-route"),
    ("concat", "concat"),
    ("scale_f32", "eltwise"),
    ("unary", "eltwise"),
    ("bin_bcast", "eltwise"),
    ("reduce", "reduce"),
    ("clamp", "eltwise"),
    ("fill", "eltwise"),
    ("softmax", "attention-dense"),
]


def classify(name):
    n = name.lower()
    for pat, cls in CLASS:
        if pat in n:
            return cls
    return "other"


def strip_ver(s):
    return s.split("_v")[0]


def load(db):
    ev = []
    for st, en, dev, name in db.execute(
        "select k.start,k.end,k.deviceId,s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id=k.demangledName"):
        ev.append((st, en, dev, name))
    for st, en, dev, kind, nbytes in db.execute(
        "select start,end,deviceId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
        ev.append((st, en, dev, "memcpy[%s] %.2f MB" % (KIND.get(kind, kind), nbytes / 1e6)))
    for st, en, dev in db.execute("select start,end,deviceId from CUPTI_ACTIVITY_KIND_MEMSET"):
        ev.append((st, en, dev, "memset"))
    ev.sort(key=lambda e: e[0])
    return ev


def api_summary(db):
    S = dict(db.execute("select id, value from StringIds"))
    cnt, dur = defaultdict(int), defaultdict(float)
    for nid, start, end in db.execute(
            "select nameId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME"):
        n = strip_ver(S.get(nid, "?"))
        cnt[n] += 1
        dur[n] += (end - start) / 1e6
    return cnt, dur


def windows(ev):
    first, last = ev[0][0], ev[-1][1]
    big_h = max((e[1] for e in ev if e[3].startswith("memcpy[HtoD] ")
                 and float(e[3].split()[1]) > 100), default=None)
    logits = min((e[0] for e in ev if e[3].startswith("memcpy[DtoH] ")
                  and float(e[3].split()[1]) >= 0.1 and (big_h is None or e[0] > big_h)), default=None)
    pre = (big_h if big_h is not None else first, logits if logits is not None else last)
    dec = (logits if logits is not None else last, last)
    return {"whole": (first, last), "prefill": pre, "decode": dec}


def busy_gaps(d, lo, hi):
    """d: events on one device sorted by start, clipped to window -> (busy_ns, merged intervals)"""
    iv = sorted((max(e[0], lo), min(e[1], hi)) for e in d)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    busy = sum(b - a for a, b in merged)
    return busy, merged


def report(ev, lo, hi, label, top, tok):
    w = [e for e in ev if e[1] > lo and e[0] < hi]
    span = hi - lo
    print("\n===== %s: %.3f s wall, %d activity records%s" % (
        label, span / 1e9, len(w),
        "" if not tok else ", %.1f ms/token at %d tokens" % (span / 1e6 / tok, tok)))
    for dev in sorted({e[2] for e in w}):
        d = [e for e in w if e[2] == dev]
        busy, merged = busy_gaps(d, lo, hi)
        gaps = [(merged[i + 1][0] - merged[i][1], merged[i][1], merged[i + 1][0])
                for i in range(len(merged) - 1)]
        idle = sum(g[0] for g in gaps)
        starts = [e[0] for e in d]
        cat = defaultdict(lambda: [0, 0])
        for g, g0, g1 in gaps:
            j = bisect.bisect_left(starts, g1)
            after = d[min(j, len(d) - 1)][3]
            key = after.split(" ")[0] + (" " + after.split(" ")[1] if "memcpy" in after else "")
            cat[key][0] += g
            cat[key][1] += 1
        print("  dev%d busy %.3f s (%.1f%%), idle %.3f s in %d gaps" % (
            dev, busy / 1e9, 100.0 * busy / span, idle / 1e9, len(gaps)))
        for k, (t, n) in sorted(cat.items(), key=lambda kv: -kv[1][0])[:min(top, 6)]:
            print("     idle %8.1f ms in %5d gaps before %-36s (%.1f us/gap)" % (
                t / 1e6, n, k[:36], t / 1e3 / n))
        for g, _g0, _g1 in sorted(gaps, reverse=True)[:min(top, 5)]:
            print("     biggest gap %8.1f ms" % (g / 1e6))

    agg = defaultdict(lambda: [0, 0])
    for st, en, dev, name in w:
        agg[name][0] += min(en, hi) - max(st, lo)
        agg[name][1] += 1
    kern = {k: v for k, v in agg.items() if not k.startswith("memcpy") and k != "memset"}
    tot = sum(v[0] for v in kern.values())
    print("  -- kernels: %.3f s of device time over %d launches" % (tot / 1e9, sum(v[1] for v in kern.values())))
    for name, (t, n) in sorted(kern.items(), key=lambda kv: -kv[1][0])[:top]:
        print("     %8.1f ms %6d %9.1f us %5.1f%%  %s" % (t / 1e6, n, t / 1e3 / n, 100.0 * t / tot, name[:70]))
    cls = defaultdict(lambda: [0, 0])
    for name, (t, n) in kern.items():
        cls[classify(name)][0] += t
        cls[classify(name)][1] += n
    print("  -- by op class")
    for c, (t, n) in sorted(cls.items(), key=lambda kv: -kv[1][0]):
        print("     %8.1f ms %6d %5.1f%%  %s" % (t / 1e6, n, 100.0 * t / tot, c))
    mem = defaultdict(lambda: [0, 0, 0])
    for st, en, dev, name in w:
        if name.startswith("memcpy"):
            k = name.split("]")[0] + "]"
            mem[k][0] += min(en, hi) - max(st, lo)
            mem[k][1] += 1
    print("  -- memcpy activity")
    for k, (t, n, _) in sorted(mem.items(), key=lambda kv: -kv[1][0]):
        print("     %8.1f ms %6d  %s" % (t / 1e6, n, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--tokens-prompt", type=int, default=0)
    ap.add_argument("--tokens-decode", type=int, default=0)
    ap.add_argument("--window", default="both", choices=("both", "whole"),
                    help="both = prefill+decode windows (needs a run that contains both); "
                         "whole = the whole trace (use for single-phase profiles)")
    a = ap.parse_args()

    db = sqlite3.connect(a.sqlite)
    ev = load(db)
    cnt, dur = api_summary(db)
    print("== %s" % a.sqlite)
    print("== CUDA runtime API (count, total ms) - cross-device + sync relevant")
    for name in ("cudaMemcpyPeerAsync", "cudaMemcpyAsync", "cudaStreamSynchronize",
                 "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaStreamWaitEvent",
                 "cudaEventRecord", "cudaGraphLaunch", "cudaLaunchKernelExC", "cudaMalloc",
                 "cudaFree", "cudaMemGetInfo"):
        if cnt.get(name):
            print("   %-24s n=%-7d %9.1f ms" % (name, cnt[name], dur[name]))
    print("   launches total: %d, graph launches: %d" % (
        sum(v for k, v in cnt.items() if k.startswith("cudaLaunchKernel")), cnt.get("cudaGraphLaunch", 0)))

    ws = windows(ev)
    print("== windows: whole %.3f s, prefill %.3f s, decode %.3f s" % (
        (ws["whole"][1] - ws["whole"][0]) / 1e9, (ws["prefill"][1] - ws["prefill"][0]) / 1e9,
        (ws["decode"][1] - ws["decode"][0]) / 1e9))
    if a.window == "whole":
        report(ev, ws["whole"][0], ws["whole"][1], "WHOLE", a.top,
               a.tokens_prompt or a.tokens_decode)
    else:
        report(ev, ws["prefill"][0], ws["prefill"][1], "PREFILL", a.top, a.tokens_prompt)
        report(ev, ws["decode"][0], ws["decode"][1], "DECODE", a.top, a.tokens_decode)


if __name__ == "__main__":
    main()
