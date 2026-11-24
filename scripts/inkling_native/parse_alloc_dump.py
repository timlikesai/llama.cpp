#!/usr/bin/env python3
"""Rank compute-buffer allocations from a GGML_ALLOC_DUMP log.

The allocator prints one line per sized tensor per reserve pass, so the same
tensor appears many times. Peak per tensor is what the buffer must hold, so we
take the max per (backend, op, name, shape) key.

Usage:
    parse_alloc_dump.py LOG [LOG ...]     # one per context setting, compared side by side

With two logs the second is treated as the "larger context" run and each entry
is classified by how it moved: CONTEXT (scales with ctx), BATCH (fixed shape but
present), or FLAT (unchanged).
"""
import re
import sys
import collections

LINE = re.compile(
    r"ALLOCDUMP (\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\d+) B\s+\|\s+ne\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")


def load(path):
    peak = {}
    for line in open(path, errors="ignore"):
        m = LINE.search(line)
        if not m:
            continue
        buft, op, name, size = m.group(1), m.group(2), m.group(3), int(m.group(4))
        ne = tuple(int(m.group(i)) for i in range(5, 9))
        if "(leaf)" in line:
            op = op + ":leaf"
        key = (buft, name if name != "None" else op, op, ne)
        if size > peak.get(key, 0):
            peak[key] = size
    return peak


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--coverage":
        coverage(sys.argv[2])
        return 0
    logs = [load(p) for p in sys.argv[1:]]
    if not logs:
        print(__doc__)
        return 1
    if len(logs) == 1:
        total = sum(logs[0].values())
        print(f"peak-sized tensors: {len(logs[0])}, sum of peaks {total / 2**20:.0f} MiB "
              f"(overcounts: the allocator reuses addresses across layers)")
        for (buft, name, op, ne), size in sorted(logs[0].items(), key=lambda x: -x[1])[:30]:
            print(f"{size / 2**20:9.2f} MiB  {buft:<14} {op:<14} {name:<28} ne={ne}")
        return 0

    a, b = logs[0], logs[1]
    keys = set(a) | set(b)
    rows = []
    for k in keys:
        sa, sb = a.get(k, 0), b.get(k, 0)
        rows.append((sb - sa, sa, sb, k))
    rows.sort(key=lambda r: -r[0])

    print(f"{'delta MiB':>11} {'ctx1 MiB':>10} {'ctx2 MiB':>10}  backend / op / name / shape")
    shown = 0
    for d, sa, sb, (buft, name, op, ne) in rows:
        if abs(d) < 4 * 2**20 and sb < 20 * 2**20:
            continue
        tag = "GROWS" if d > 4 * 2**20 else ("SHRINKS" if d < -4 * 2**20 else "flat-big")
        print(f"{d / 2**20:11.2f} {sa / 2**20:10.2f} {sb / 2**20:10.2f}  {tag:<9} {buft:<12} {op:<14} {name:<26} ne={ne}")
        shown += 1
        if shown >= 25:
            break

    for label, log in (("ctx1", a), ("ctx2", b)):
        per = collections.defaultdict(int)
        for (buft, name, op, ne), s in log.items():
            per[buft] += s
        print(f"\n{label} sum-of-peaks by backend: " +
              ", ".join(f"{k} {v / 2**20:.0f} MiB" for k, v in sorted(per.items(), key=lambda x: -x[1])))

    # what is genuinely flat in context but large -- the #34 target
    flat = [(sa, sb, k) for k, (sa, sb) in ((k, (a.get(k, 0), b.get(k, 0))) for k in keys)
            if abs(sb - sa) < 2**20 and max(sa, sb) > 20 * 2**20]
    flat.sort(key=lambda r: -max(r[0], r[1]))
    print(f"\n### FLAT (context-independent) buffers above 20 MiB — #34 candidates")
    for sa, sb, (buft, name, op, ne) in flat[:20]:
        print(f"{max(sa, sb) / 2**20:9.2f} MiB  {buft:<12} {op:<14} {name:<26} ne={ne}")
    print(f"total flat: {sum(max(x, y) for x, y, _ in flat) / 2**20:.0f} MiB across {len(flat)} tensors")
    return 0


# (entry point at the bottom of the file, so the helpers below are defined)


#
# Address-coverage mode: per-tensor peaks overstate the reserve because the allocator
# reuses one span for sequential same-shape nodes. Printing chunk+offset lets us take the
# union of intervals, which IS what the buffer must physically hold, and name what lives
# in each part of it.
#
LINE_ADDR = re.compile(
    r"ALLOCDUMP (\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\d+) B\s+\|\s+chunk\s+(-?\d+)\s+off\s+(\d+)\s+\|\s+ne\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")


def load_addr(path):
    """-> {(buft, chunk): [(off, size, op, name, ne), ...]} taking the max size per (buft,chunk,off,name)."""
    spans = {}
    for line in open(path, errors="ignore"):
        m = LINE_ADDR.search(line)
        if not m:
            continue
        buft, op, name, size = m.group(1), m.group(2), m.group(3), int(m.group(4))
        chunk, off = int(m.group(5)), int(m.group(6))
        ne = tuple(int(m.group(i)) for i in range(7, 11))
        if "(leaf)" in line:
            op += ":leaf"
        per = spans.setdefault((buft, chunk), {})
        key = (off, name, op, ne)
        if size > per.get(key, 0):
            per[key] = size
    return {k: [(off, size, op, name, key_ne) for (off, name, op, key_ne), size in v.items()]
            for k, v in spans.items()}


def coverage(path, bucket=2**20):
    print(f"### address coverage from {path}")
    print("    (union of [offset, offset+size) per buffer chunk == bytes the buffer must hold)\n")
    for (buft, chunk), items in sorted(load_addr(path).items()):
        if not items:
            continue
        # interval union
        iv = sorted((off, off + sz) for off, sz, _, _, _ in items if sz > 0)
        covered = []
        for a, b in iv:
            if covered and a <= covered[-1][1]:
                covered[-1][1] = max(covered[-1][1], b)
            else:
                covered.append([a, b])
        total = sum(b - a for a, b in covered)
        span = max(x[1] for x in iv) - min(x[0] for x in iv)
        print(f"{buft} chunk {chunk}: {len(items)} tensors, span {span/2**20:.0f} MiB, "
              f"covered {total/2**20:.0f} MiB (reuse saves {(span-total)/2**20:.0f} MiB)")
        # who owns each bucket
        owner = collections.defaultdict(lambda: [0, collections.Counter()])
        for off, sz, op, name, ne in items:
            if sz == 0:
                continue
            b0 = off // bucket
            for bi in range(b0, (off + sz - 1) // bucket + 1):
                owner[bi][0] += 1
                owner[bi][1][f"{op} {name} {ne}"] += sz
        top = sorted(owner.items(), key=lambda kv: -sum(kv[1][1].values()))
        for bi, (_, sizes) in top[:10]:
            merged = collections.Counter()
            for off, sz, op, name, ne in items:
                if off // bucket <= bi <= (off + sz - 1) // bucket:
                    merged[f"{op}|{name}|{ne}"] += sz
            biggest = merged.most_common(2)
            print(f"    @{bi*bucket/2**20:7.0f} MiB: " +
                  "; ".join(f"{k} ({v/2**20:.0f} MiB)" for k, v in biggest))
        print()


if __name__ == "__main__":
    sys.exit(main())
