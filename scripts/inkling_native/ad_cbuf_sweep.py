#!/usr/bin/env python3
"""Compute-buffer scaling sweep (brief AD / doc 29-compute-buffer-scaling).

Discriminates context-proportional *allocation* from batch-proportional *scratch*
by sweeping n_ctx and n_ubatch independently and reading the final sched_reserve
pass of each run:

  - one `sched_reserve: <device> compute buffer size` line per device
  - BOTH `llama_kv_cache: size =` lines (iswa prints dense first, then SWA)
  - the `llama_memory_recurrent: size` line

Every full run log is kept under logs/ad-cb-<tag>.log. Nothing is deleted.

Usage: ad_cbuf_sweep.py [--quick]
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLI = os.path.join(ROOT, "build-cuda", "bin", "llama-completion")
LOGDIR = os.path.join(ROOT, "logs")
OUT = os.path.join(ROOT, "logs", "ad-cb-sweep.tsv")

INKLING = "/home/tim/models/timlikesai/Inkling-2-Mini-NATIVE.gguf"
QWEN = "/home/tim/models/timlikesai/Qwen3.5-0.8B-Q8_0.gguf"

RE_RESERVE = re.compile(r"sched_reserve: reserving \.\.\.")
RE_COMP = re.compile(r"sched_reserve:\s+([A-Za-z0-9_ ()-]+?) compute buffer size =\s+([\d.]+) MiB")
RE_KV = re.compile(r"llama_kv_cache: size =\s+([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers")
RE_RS = re.compile(r"llama_memory_recurrent: size =\s+([\d.]+) MiB")
RE_GRAPH = re.compile(r"sched_reserve: graph.*?splits = (\S+?)[ ,]")


def run(tag, model, extra):
    log = os.path.join(LOGDIR, "ad-cb-%s.log" % tag)
    cmd = [CLI, "-m", model, "-p", "Q: What is 17 times 4? A:", "-n", "1",
           "--temp", "0", "-s", "42", "-no-cnv", "-lv", "4"] + list(extra)
    print("+ %s" % " ".join(cmd[3:]), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    with open(log, "w") as f:
        f.write("$ %s\n%s" % (" ".join(cmd), out))
    if p.returncode != 0:
        print("  rc=%d (see %s)" % (p.returncode, log), flush=True)
    return out, p.returncode, log


def parse(out):
    parts = RE_RESERVE.split(out)
    if len(parts) < 2:
        return None
    last = parts[-1]  # the final reserve pass, after the fitter settled
    comp = dict((d.strip(), float(v)) for d, v in RE_COMP.findall(last))
    kv = [(float(m), int(c), int(l)) for m, c, l in RE_KV.findall(last)]
    rs = RE_RS.findall(last)
    sp = RE_GRAPH.findall(last)
    return {"comp": comp, "kv": kv, "rs": float(rs[-1]) if rs else 0.0,
            "splits": sp[-1] if sp else "?", "total": sum(comp.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        cases = [("ctx8192-ub512", INKLING, ["-c", "8192", "-ub", "512", "-fa", "on"]),
                 ("ctx32768-ub512", INKLING, ["-c", "32768", "-ub", "512", "-fa", "on"])]
    else:
        cases = []
        for ub in ("512", "2048"):
            for ctx in ("2048", "8192", "32768"):
                cases.append(("ink-ctx%s-ub%s" % (ctx, ub), INKLING,
                              ["-c", ctx, "-b", "2048", "-ub", ub, "-fa", "on"]))
        for ctx in ("8192", "32768"):
            cases.append(("ink-ctx%s-ub256" % ctx, INKLING,
                          ["-c", ctx, "-b", "2048", "-ub", "256", "-fa", "on"]))
            cases.append(("ink-ctx%s-ub512-faoff" % ctx, INKLING,
                          ["-c", ctx, "-b", "2048", "-ub", "512", "-fa", "off"]))
        for ub in ("512", "2048"):
            for ctx in ("2048", "8192", "32768"):
                cases.append(("qwen-ctx%s-ub%s" % (ctx, ub), QWEN,
                              ["-c", ctx, "-b", "2048", "-ub", ub, "-fa", "on"]))

    rows = []
    with open(OUT, "w") as f:
        f.write("tag\tctx\tubatch\tfa\tCUDA0\tCUDA1\tCUDA_Host\tcomp_total\tdense_kv\tswa_kv\tkv_total\trecurrent\tsplits\trc\n")
        for tag, model, extra in cases:
            out, rc, log = run(tag, model, extra)
            r = parse(out)
            if r is None:
                print("  NO PARSE %s rc=%d" % (tag, rc), flush=True)
                continue
            ctx = extra[extra.index("-c") + 1]
            ub = extra[extra.index("-ub") + 1]
            fa = extra[extra.index("-fa") + 1]
            dense = r["kv"][0][0] if len(r["kv"]) > 0 else 0.0
            swa = r["kv"][1][0] if len(r["kv"]) > 1 else 0.0
            row = [tag, ctx, ub, fa, "%.2f" % r["comp"].get("CUDA0", 0.0),
                   "%.2f" % r["comp"].get("CUDA1", 0.0),
                   "%.2f" % r["comp"].get("CUDA_Host", 0.0), "%.2f" % r["total"],
                   "%.2f" % dense, "%.2f" % swa, "%.2f" % (dense + swa),
                   "%.2f" % r["rs"], r["splits"], str(rc)]
            rows.append(row)
            line = "\t".join(row)
            print("  " + line, flush=True)
            f.write(line + "\n")
            f.flush()
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
