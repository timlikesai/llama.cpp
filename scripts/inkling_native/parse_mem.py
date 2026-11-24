#!/usr/bin/env python3
"""Read llama.cpp load/reserve logs and report memory the way it is actually accounted.

Three traps this exists to avoid re-stepping, all of them hit in this project:
  * an iswa cache prints TWO "llama_kv_cache: size =" lines (dense + SWA); taking one gives a wrong
    KV total - and it also prints one DEBUG line per layer, which must not be dumped here;
  * a long-context run prints MANY sched_reserve passes (one per ubatch size tried), so summing
    every compute-buffer line gives nonsense - the value in effect is the LAST one per device;
  * a CUDA_Host model buffer is only a spill when it is non-zero.

usage: parse_mem.py LOG [LOG ...]
"""
import argparse
import pathlib
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
TS = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\s+[IDWE]:?\s*")


def clean(path: pathlib.Path) -> list[str]:
    raw = path.read_text(errors="replace")
    raw = ANSI.sub("", raw).replace("\x08", "").replace("\x07", "")
    return [TS.sub("", l).strip() for l in re.split(r"[\r\n]", raw)]


def last_per_key(lines: list[str], pattern: str) -> list[str]:
    """Last occurrence per device/label - earlier ones are probes from failed reserve passes."""
    out: dict[str, str] = {}
    rx = re.compile(pattern)
    for l in lines:
        m = rx.search(l)
        if m:
            # key on the match, not the line: log prefixes (timestamps, spinner remnants) make otherwise
            # identical lines look distinct, which defeats the whole point of collapsing passes
            key = m.group(1) if m.groups() else m.group(0)
            out[key] = l[m.start():]
    return list(out.values())


def report(path: pathlib.Path) -> None:
    lines = clean(path)
    print(f"########## {path.name}")

    kv = last_per_key(lines, r"llama_kv_cache: size = .*?\((.*?)\)")
    if kv:
        print("  KV cache (final, both iswa halves):")
        for l in kv:
            print(f"    {l}")

    rec = last_per_key(lines, r"llama_memory_recurrent: size = ")
    for l in rec:
        print(f"    {l}")

    comp = last_per_key(lines, r"sched_reserve:\s+(\S+) compute buffer size")
    if comp:
        print("  compute buffers (last value per device = the reservation in effect):")
        for l in comp:
            print(f"    {l}")

    mb = last_per_key(lines, r"load_tensors:\s+(\S+)\s+model buffer size")
    if mb:
        print("  weights:")
        for l in mb:
            print(f"    {l}")

    for l in last_per_key(lines, r"llama_context:\s+(flash_attn\s+= \w+)"):
        print(f"  {l.split(':', 1)[1].strip()}")

    perf = [l for l in lines if re.search(r"(Prompt|Generation):.*t/s", l)]
    for l in perf[-2:]:
        print(f"  perf: {l}")

    hard = [l for l in lines if re.search(r"^E |out of memory|failed to (allocate|reserve)", l)]
    for l in hard[:4]:
        print(f"  !! {l}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args()
    for name in args.logs:
        p = pathlib.Path(name)
        if not p.is_file():
            print(f"missing log: {p}", file=sys.stderr)
            return 1
        report(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
