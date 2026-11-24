#!/usr/bin/env python3
"""FA x KV-type x context memory matrix for Inkling-2-Mini (item #22 / doc 29).

Runs llama-completion once per config at -lv 4 (so the llama library INFO lines are visible),
parses the allocation log lines, and prints measured vs theoretical bytes per bucket:

  weights / attn KV (non-SWA + SWA caches) / recurrent state / compute buffer, per device.

Theoretical KV is computed from the hparams the run itself prints (n_layer slots,
n_embd_k_gqa, n_embd_v_gqa, n_swa) plus the loop aliasing rule, so the comparison cannot
drift from the model file. Each full run log is kept under logs/.

Usage: ad_fa_kv_matrix.py [--out logs/ad-fa-kv-matrix.tsv] [--quick]
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLI = os.path.join(ROOT, "build-cuda", "bin", "llama-completion")
MODEL = "/home/tim/models/timlikesai/Inkling-2-Mini-NATIVE.gguf"
PROMPT = "Q: What is 17 times 4? A:"
LOGDIR = os.path.join(ROOT, "logs")

# ggml type size in bytes per element (block types: size per element incl. scale)
TSZ = {"f16": 2.0, "bf16": 2.0, "f32": 4.0, "q8_0": 34.0 / 32.0, "q4_0": 18.0 / 32.0,
       "f8_e4m3": 1.0, "iq4_nl": 0.5625, "mxfp8": 33.0 / 32.0, "nvfp4": 0.5625}

MIB = 1024.0 * 1024.0

RE_MODEL_BUF = re.compile(r"load_tensors:\s+([A-Za-z0-9_ ()-]+?) model buffer size =\s+([\d.]+) MiB")
RE_NGL = re.compile(r"load_tensors: offloaded (\d+)/(\d+) layers to GPU")
RE_CELLS = re.compile(r"llama_kv_cache_iswa: creating (non-SWA|    SWA) KV cache, size = (\d+) cells")
RE_KV = re.compile(r"llama_kv_cache: size =\s+([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers,"
                   r"\s*(\d+)/(\d+) seqs\), K \((\S+)\):\s+([\d.]+) MiB, V \((\S+)\):\s+([\d.]+) MiB")
RE_RS = re.compile(r"llama_memory_recurrent: size =\s+([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers")
RE_COMP = re.compile(r"sched_reserve:\s+([A-Za-z0-9_ ()-]+?) compute buffer size =\s+([\d.]+) MiB")
RE_FA = re.compile(r"llama_context: flash_attn\s*=\s*(\S+)")
RE_FA_OFF = re.compile(r"Flash Attention not supported, set to disabled")
RE_FA_MISMATCH = re.compile(r"resolve_fused_ops: layer (\d+) is assigned to device (\S+) but (\S+) is assigned to device (\S+)")
RE_INFO = re.compile(r"print_info: (n_layer|n_embd_k_gqa|n_embd_v_gqa|n_swa)\s*= (\d+)")
RE_BREAKDOWN = re.compile(r"\|\s+- (\w+)[^|]*\|\s*(\d+) =\s+(\d+) \+\s*\(\s*(\d+) =\s+(\d+) \+\s+(\d+) \+\s+(\d+)\)")
RE_GEN = re.compile(r"common_perf_print:\s+prompt eval time =\s+(\d+).*/\s+(\d+) tokens \([^)]*\)\s*([\d.]+) tokens per second")


def run(tag, args, extra=()):
    log = os.path.join(LOGDIR, "ad-%s.log" % tag)
    cmd = [CLI, "-m", MODEL, "-p", PROMPT, "-n", "1", "-b", "2048", "-ub", "2048",
           "-sm", "layer", "--temp", "0", "-s", "42", "-no-cnv", "-lv", "4"] + list(extra) + args
    print("+ %s" % " ".join(cmd[1:]), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    with open(log, "w") as f:
        f.write("$ %s\n%s" % (" ".join(cmd), out))
    return out, p.returncode, log


def parse(out):
    r = {"rc_note": "", "warn": [], "info": {}}
    r["model_buf"] = {k.strip(): float(v) for k, v in RE_MODEL_BUF.findall(out)}
    r["comp_buf"] = {k.strip(): float(v) for k, v in RE_COMP.findall(out)}
    r["kv"] = [{"mib": float(a), "cells": int(b), "layers": int(c), "type_k": d, "k_mib": float(e),
                "type_v": f, "v_mib": float(g)} for a, b, c, _s, _sn, d, e, f, g in RE_KV.findall(out)]
    m = RE_CELLS.search(out)
    r["cells"] = [int(x[1]) for x in RE_CELLS.findall(out)]
    r["rs"] = [{"mib": float(a), "cells": int(b), "layers": int(c)} for a, b, c in RE_RS.findall(out)]
    r["ngl"] = RE_NGL.search(out).groups() if RE_NGL.search(out) else ("?", "?")
    r["fa"] = RE_FA.search(out).group(1) if RE_FA.search(out) else "?"
    r["fa_off"] = bool(RE_FA_OFF.search(out))
    r["fa_mismatch"] = RE_FA_MISMATCH.findall(out)
    r["info"] = {k: int(v) for k, v in RE_INFO.findall(out)}
    r["breakdown"] = {k: {"total": int(a), "free": int(b), "self": int(c), "model": int(d),
                          "context": int(e), "compute": int(f)}
                     for k, a, b, c, d, e, f in RE_BREAKDOWN.findall(out)}
    r["gen"] = RE_GEN.search(out)
    for pat in ("cache type is not supported", "not supported", "using full-size SWA",
                "failed to fit params", "out of memory", "Flash Attention"):
        for line in out.splitlines():
            if pat in line and (line.startswith(("0.", "1.", "2.", "3.", "4.", "5.")) and " W " in line or " E " in line):
                if line not in r["warn"]:
                    r["warn"].append(line)
    return r


def theoretical(info, type_k, type_v, n_ctx, n_ubatch=2048, n_seq_max=1, n_swa_window=1024):
    """bytes of attn KV, per the rules in src/llama-kv-cache-iswa.cpp:75 and the loop alias map."""
    n_slot = info["n_layer"]                 # 48 cache slots
    n_phys = 24
    core_start, core_end = 6, 18
    # is_swa per physical layer: pattern is 5-of-6 True, full at 5, 11, 17, 23
    full_phys = {5, 11, 17, 23}

    def phys(il):
        return il if il < n_phys else core_start + (il - n_phys) % (core_end - core_start)

    n_full = sum(1 for il in range(n_slot) if phys(il) in full_phys)
    n_swa_slots = n_slot - n_full

    size_base = n_ctx
    raw = min(size_base, n_swa_window * n_seq_max + n_ubatch)
    size_swa = (raw + 255) // 256 * 256

    per_tok_full = info["n_embd_k_gqa"] * TSZ[type_k] + info["n_embd_v_gqa"] * TSZ[type_v]
    base = n_full * size_base * per_tok_full
    swa = n_swa_slots * size_swa * per_tok_full
    uniq_full = len(full_phys)
    uniq_swa = n_phys - uniq_full
    uniq = (uniq_full * size_base + uniq_swa * size_swa) * per_tok_full
    return {"size_base": size_base, "size_swa": size_swa, "n_full": n_full, "n_swa": n_swa_slots,
            "theo_mib": (base + swa) / MIB, "uniq_mib": uniq / MIB}


def fmt_row(tag, cfg, r):
    if not r["kv"]:
        return "%-34s FAILED  %s" % (tag, "; ".join(r["warn"][:2]) or "no KV alloc lines (rc!=0)")
    kv_mib = sum(k["mib"] for k in r["kv"])
    cells = r["cells"] or [0, 0]
    lay = [k["layers"] for k in r["kv"]]
    t = theoretical(r["info"], cfg["tk"], cfg["tv"], cfg["c"], cfg["ub"])
    comp = sum(r["comp_buf"].values())
    rs = sum(x["mib"] for x in r["rs"])
    w = sum(r["model_buf"].values())
    dev = " ".join("%s:m%.0f/c%.0f/k%.0f" % (k.replace("CUDA", ""), v["model"], v["compute"], v["context"])
                   for k, v in sorted(r["breakdown"].items()))
    return ("%-34s w=%7.1f kv=%8.1f (th %7.1f, uniq %7.1f) cells=%s/%s lay=%s rs=%5.2f "
            "comp=%8.1f fa=%-7s %s") % (
        tag, w, kv_mib, t["theo_mib"], t["uniq_mib"], cells[0], cells[1], "/".join(map(str, lay)),
        rs, comp, r["fa"] + ("!" if r["fa_off"] else ""), dev)


CFG = []
# A) the #22 matrix at the shipped n_ubatch (512, common/common.h:480)
for c in (2048, 8192, 32768):
    for fa in ("on", "off"):
        for ty in ("f16", "q8_0", "f8_e4m3", "q4_0"):
            CFG.append({"c": c, "fa": fa, "tk": ty, "tv": ty, "ub": 512})
# B) compute-buffer sensitivity: it dwarfs the KV cache, so show how it scales with n_ubatch
for ub in (128, 512, 1024, 2048):
    for fa in ("on", "off"):
        CFG.append({"c": 8192, "fa": fa, "tk": "f16", "tv": "f16", "ub": ub})
# C) K quantized, V kept f16 - the only quantized-KV shape that is legal with FA off
for tk in ("q8_0", "f8_e4m3"):
    for fa in ("on", "off"):
        CFG.append({"c": 8192, "fa": fa, "tk": tk, "tv": "f16", "ub": 512})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(LOGDIR, "ad-fa-kv-matrix.tsv"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.out, "w") as tsv:
        tsv.write("tag\tc\tfa\ttype\tub\tweights_MiB\tkv_MiB\tkv_theory_MiB\tkv_unique_layer_MiB\t"
                  "cells_base\tcells_swa\trs_MiB\tcompute_MiB\tfa_resolved\tbreakdown\n")
        for cfg in CFG[:args.limit or len(CFG)]:
            ub = cfg["ub"]
            tag = "c%d-fa%s-%s%s%s" % (cfg["c"], cfg["fa"], cfg["tk"],
                                       "" if cfg["tv"] == cfg["tk"] else "-v%s" % cfg["tv"],
                                       "" if ub == 512 else "-ub%d" % ub)
            out, rc, log = run(tag, ["-c", str(cfg["c"]), "-fa", cfg["fa"],
                                     "-ctk", cfg["tk"], "-ctv", cfg["tv"], "-ub", str(ub), "-b", str(ub)])
            r = parse(out)
            line = fmt_row(tag, cfg, r)
            print(line, flush=True)
            print("   log: %s (rc=%d)" % (log, rc), flush=True)
            kv_mib = sum(k["mib"] for k in r["kv"]) if r["kv"] else 0.0
            t = theoretical(r["info"], cfg["tk"], cfg["tv"], cfg["c"], ub) if r["info"] else {}
            tsv.write("%s\t%d\t%s\t%s/%s\t%d\t%.1f\t%.1f\t%.1f\t%.1f\t%s\t%s\t%.2f\t%.1f\t%s\t%s\n" % (
                tag, cfg["c"], cfg["fa"], cfg["tk"], cfg["tv"], ub, sum(r["model_buf"].values()), kv_mib,
                t.get("theo_mib", 0), t.get("uniq_mib", 0),
                "/".join(map(str, r["cells"])), "/".join(str(k["layers"]) for k in r["kv"]),
                sum(x["mib"] for x in r["rs"]), sum(r["comp_buf"].values()),
                r["fa"] + ("/disabled" if r["fa_off"] else ""),
                ";".join("%s=%.0f/%.0f/%.0f" % (k, v["model"], v["context"], v["compute"])
                         for k, v in sorted(r["breakdown"].items()))))
            tsv.flush()
    print("wrote %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
