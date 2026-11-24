#!/usr/bin/env python3
"""Attribute a llama.cpp graph from a GGML_SCHED_DEBUG=2 -v log: which tensors drive which ops.

The point is to answer "what are those 2,960 copy kernels?" with names and sizes instead of guesses.
Lines look like:

    ... input tensor '        attn_inp_rel_pos' [i32, ne = {  1024,  1,  1,  1 }] is used by node 'node_1170' (GET_ROWS)

usage: parse_graph.py LOG [--ops GET_ROWS,CPY,CONCAT,CAST] [--top 20]
"""
import argparse
import collections
import pathlib
import re

LINE = re.compile(
    r"input tensor '\s*(?P<name>[^']+)'\s*\[(?P<type>[a-z0-9_]+),\s*ne = \{\s*(?P<ne>[^}]*)\}\]"
    r"\s*is used by node '[^']*'\s*\((?P<op>[A-Z_0-9]+)\)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--ops", default="GET_ROWS,CPY,CONCAT,CAST,MUL_MAT,MUL_MAT_ID,SUM_ROWS")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    want = {o.strip().upper() for o in args.ops.split(",")}

    agg = collections.Counter()
    count = collections.Counter()
    total = collections.Counter()

    with open(args.log, errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            op = m.group("op")
            if op not in want:
                continue
            ne = [int(x) for x in m.group("ne").replace("\n", " ").split(",") if x.strip()]
            n = 1
            for x in ne[:4]:
                n *= max(x, 1)
            name = re.sub(r"-\d+", "-N", m.group("name").strip())  # fold per-layer/per-node suffixes
            key = (op, name, m.group("type"))
            agg[key] += n
            count[key] += 1
            total[op] += 1

    if not agg:
        print("no matching lines - is the log from a run with GGML_SCHED_DEBUG=2 and -v?")
        return 1

    print("nodes seen per op:", ", ".join(f"{k}={v}" for k, v in sorted(total.items())))
    print(f"\nby total input elements (name has per-layer suffixes folded to -N):")
    for (op, name, typ), elems in agg.most_common(args.top):
        print(f"  {elems:>15,} elems  {count[(op, name, typ)]:>6} nodes  {op:11s} {typ:5s} {name}")

    print(f"\nby node count:")
    for (op, name, typ), c in count.most_common(args.top):
        print(f"  {c:>6} nodes  {agg[(op, name, typ)]:>15,} elems  {op:11s} {typ:5s} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
