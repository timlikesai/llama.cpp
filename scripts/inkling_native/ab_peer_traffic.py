#!/usr/bin/env python3
"""Peer traffic counters from an nsys sqlite export: runtime memcpy-peer calls, device-side
memcpy activity per (src device, dst device), and stream sync count/time.

Usage: ab_peer_traffic.py <a.sqlite> [<b.sqlite> ...]
Output: plain-text table on stdout (tee to logs/).
"""
import sqlite3
import sys

COPY_KIND = {1: "H2D", 2: "D2H", 3: "H2H", 4: "D2D", 8: "MTP2P", 9: "MTDP", 10: "P2P"}


def strip_ver(n):
    return n.split("_v")[0]


def report(db):
    c = sqlite3.connect(db)
    S = dict(c.execute("select id, value from StringIds"))

    api = {}
    for (nid,) in c.execute("select nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        name = strip_ver(S.get(nid, "?"))
        api[name] = api.get(name, 0) + 1

    print("== %s" % db)
    for name in ("cudaMemcpyPeerAsync", "cudaMemcpyAsync", "cudaStreamSynchronize",
                 "cudaEventRecord", "cudaStreamWaitEvent", "cudaDeviceSynchronize"):
        print("   api %-22s %6d" % (name, api.get(name, 0)))

    # device-side copies, grouped by kind and by device pair
    cols = [r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")]
    sdev, ddev = ("srcDeviceId", "dstDeviceId") if "srcDeviceId" in cols else ("srcDevice", "dstDevice")
    rows = list(c.execute("select copyKind, %s, %s, count(*), sum(bytes), "
                          "sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY group by 1,2,3 "
                          "order by 5 desc" % (sdev, ddev)))
    print("   device-side memcpy activity (kind src->dst, count, MiB, ms):")
    tot_bytes = 0
    tot_n = 0
    for kind, s, d, n, nbytes, dur in rows:
        kind = COPY_KIND.get(kind, str(kind))
        peer = s is not None and d is not None and s != d
        tot_bytes += nbytes if peer else 0
        tot_n += n if peer else 0
        print("     %-5s %s->%s  n=%-6d %10.2f MiB  %8.1f ms%s" % (
            kind, s, d, n, nbytes / 1048576.0, dur / 1e6, "   <- PEER" if peer else ""))
    print("   peer copies: %d, peer bytes total: %.2f GiB" % (tot_n, tot_bytes / 1073741824.0))

    # size histogram of the peer copies, to tell weights (fixed per-layer sizes) from activations
    # (this nsys build leaves src/dst device empty, so classify by the runtime call instead)
    peer_corr = set(cid for (cid, nid) in c.execute("select correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME")
                    if strip_ver(S.get(nid, "?")) == "cudaMemcpyPeerAsync")
    hist = {}
    for cid, nbytes in c.execute("select correlationId, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
        if cid in peer_corr:
            hist[nbytes] = hist.get(nbytes, 0) + 1
    print("   cudaMemcpyPeerAsync device records, by size:")
    for nbytes, n in sorted(hist.items(), key=lambda x: -x[0] * x[1])[:12]:
        print("     %12d B  n=%-6d %.2f GiB" % (nbytes, n, nbytes * n / 1073741824.0))
    print("   peer MiB total: %.2f" % (sum(k * v for k, v in hist.items()) / 1048576.0))
    print()


if __name__ == "__main__":
    for db in sys.argv[1:]:
        report(db)
