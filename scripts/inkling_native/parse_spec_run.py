#!/usr/bin/env python3
"""Summarize llama-cli / llama-server spec-decode runs from a verbose log.

Why this exists: with `-v` (needed for the "draft acceptance" line) llama-cli does NOT print
its `[ Prompt: X t/s | Generation: Y t/s ]` summary, so grepping for it silently finds
nothing. The per-slot `print_timing` lines are the reliable source.

Usage: parse_spec_run.py LOG [LOG ...]
"""
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

PROMPT_TS = re.compile(r"prompt eval time = .*?/\s*(\d+) tokens \(.*?,\s*([0-9.]+) tokens per second\)")
EVAL_TS = re.compile(r"eval time = .*?/\s*(\d+) tokens \(.*?,\s*([0-9.]+) tokens per second\)")
ACCEPT = re.compile(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)")
PER_POS = re.compile(r"per-position \( ([0-9.,\s]+)\)")
NCTX = re.compile(r"n_ctx\s*=\s*(\d+)")
KV = re.compile(r"(CUDA0|CUDA1|KV buffer size.*?)([0-9.]+) MiB")


def summarize(path):
    text = ANSI.sub("", open(path, errors="ignore").read())
    d = {}
    p = PROMPT_TS.findall(text)
    e = EVAL_TS.findall(text)
    if p:
        d["pp_tokens"], d["prefill_tps"] = int(p[-1][0]), float(p[-1][1])
    if e:
        d["decode_tokens"], d["decode_tps"] = int(e[-1][0]), float(e[-1][1])
    a = ACCEPT.findall(text)
    if a:
        last = a[-1]
        d["accept"], d["acc"], d["gen"], d["mean_len"] = float(last[0]), int(last[1]), int(last[2]), float(last[3])
    pos = PER_POS.findall(text)
    if pos:
        d["per_pos"] = pos[-1].strip()
    n = NCTX.findall(text)
    if n:
        d["n_ctx"] = int(n[-1])
    # the last KV summary line per device, if present
    for m in re.finditer(r"\|\s*(KV|compute) buffer size\s*\|\s*([0-9.]+) MiB", text):
        d.setdefault(m.group(1), []).append(float(m.group(2)))
    if "failed to allocate" in text or "out of memory" in text or "exited before becoming ready" in text:
        err = re.search(r"(failed to allocate [^\n]*|cudaMalloc failed: out of memory)", text)
        d["error"] = (err.group(1)[:66] if err else "server exited before ready")
    return d


def main():
    print(f"{'log':<38} {'n_ctx':>7} {'prefill':>8} {'decode':>7} {'accept':>7} {'acc/gen':>10} {'len':>5}  per-position")
    for path in sys.argv[1:]:
        d = summarize(path)
        name = path.split("/")[-1].replace(".log", "")
        if not d:
            print(f"{name:<38}  (empty log)")
            continue
        cells = [
            f"{d.get('n_ctx', 0):>7}",
            f"{d.get('prefill_tps', 0):>8.1f}",
            f"{d.get('decode_tps', 0):>7.1f}",
            f"{d.get('accept', 0):>7.3f}" if "accept" in d else f"{'-':>7}",
            f"{d.get('acc', 0)}/{d.get('gen', 0):<5}" if "acc" in d else f"{'-':>10}",
            f"{d.get('mean_len', 0):>5.2f}" if "mean_len" in d else f"{'-':>5}",
        ]
        line = f"{name:<38} " + " ".join(cells)
        if "error" in d:
            line += f"  ERROR: {d['error']}"
        elif "per_pos" in d:
            line += f"  ({d['per_pos']})"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
