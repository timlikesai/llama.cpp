#!/usr/bin/env python3
"""Probe a running llama-server for the Inkling native model: text oracle,
drafting, and the vision path.

Written because the vision-enabled bring-up crashed the server on the first
image request ("CUDA error: out of memory" from the ViT encode) -- the GPU-only
projector did not fit. This script is the regression check for that: if it
reports VISION PASS and the container's RestartCount is unchanged afterwards,
multimodal is actually working, not merely loaded.

Discriminative by construction: two solid-color images (red and blue) are
submitted with the same prompt. A model that is guessing from a language prior
gets one of them wrong; a model that reads pixels gets both right. A blank or
unrelated answer to the text control means the endpoint is not serving Inkling.

Usage:
    probe_server_vision.py [--url http://localhost:9931] [--log FILE]

Exit status: 0 if every check passed, 1 otherwise.
"""

import argparse
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

# ---------------------------------------------------------------- image synth


def solid_png(rgb: tuple[int, int, int], w: int = 64, h: int = 64) -> bytes:
    """Minimal lossless PNG, no dependencies, one solid color."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


# ------------------------------------------------------------------- HTTP API


def post(url: str, path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def chat(url: str, content, max_tokens: int = 220) -> tuple[str, dict]:
    """Return (visible_content, timings). Inkling thinks first: the answer may
    land in `reasoning`/`reasoning_content` while `content` stays empty if the
    budget runs out inside the thinking block."""
    body = {"messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False, "timings": True}
    d = post(url, "/v1/chat/completions", body)
    m = d["choices"][0]["message"]
    vis = (m.get("content") or "").strip()
    if not vis:
        vis = (m.get("reasoning_content") or m.get("reasoning") or "").strip()
    return vis, d.get("timings") or {}


def ask_text(url: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
    return chat(url, [{"type": "text", "text": prompt}], max_tokens)


def ask_image(url: str, rgb: tuple[int, int, int], prompt: str, max_tokens: int) -> tuple[str, dict]:
    uri = "data:image/png;base64," + base64.b64encode(solid_png(rgb)).decode()
    return chat(url, [{"type": "text", "text": prompt},
                      {"type": "image_url", "image_url": {"url": uri}}], max_tokens)


# --------------------------------------------------------------------- checks

COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220)}
IMG_PROMPT = "Name the single color of this image in one word."
TEXT_PROMPT = "Q: What is 17 times 4? A:"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9931")
    ap.add_argument("--log", default=None, help="append a JSON record of the run here")
    args = ap.parse_args()

    out = {"url": args.url, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "checks": {}, "failures": []}

    def check(name, ok, detail):
        out["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            out["failures"].append(name)

    # 0 -- liveness
    try:
        h = json.load(urllib.request.urlopen(args.url + "/health", timeout=30))
        check("health", h.get("status") == "ok", h)
    except Exception as e:  # noqa: BLE001
        check("health", False, repr(e))
        print("\nserver is not answering; aborting")
        return 1

    try:
        props = json.load(urllib.request.urlopen(args.url + "/props", timeout=30))
        check("alias", props.get("model_alias") == "inkling/inkling-2-mini",
              f"alias={props.get('model_alias')} ftype={props.get('model_ftype')} "
              f"modalities={props.get('modalities')}")
        vision = (props.get("modalities") or {}).get("vision")
        check("vision_advertised", bool(vision),
              f"modalities={props.get('modalities')} (vision=True => mmproj loaded)")
    except Exception as e:  # noqa: BLE001
        check("props", False, repr(e))

    # 1 -- text control: the model must still reason and answer
    try:
        t0 = time.perf_counter()
        ans, t = ask_text(args.url, TEXT_PROMPT, 400)
        text_ms = 1000 * (time.perf_counter() - t0)
        check("text_oracle", "68" in ans,
              (f"answer={ans[:90]!r} decode_tps={t.get('predicted_per_second'):.2f} round_trip={text_ms:.0f} ms")
              if t else f"answer={ans[:90]!r}")
    except urllib.error.HTTPError as e:
        check("text_oracle", False, f"HTTP {e.code} {e.read().decode()[:160]}")
    except Exception as e:  # noqa: BLE001
        check("text_oracle", False, repr(e))

    # 2 -- vision: both colors must come back right, or it is not reading pixels
    for name, rgb in COLORS.items():
        t0 = time.perf_counter()
        try:
            ans, t = ask_image(args.url, rgb, IMG_PROMPT, 300)
            ms = 1000 * (time.perf_counter() - t0)
            check(f"vision_{name}", name in ans.lower(),
                  f"answer={ans[:90]!r} round_trip={ms:.0f} ms "
                  f"prefill_tps={t.get('prompt_per_second')}")
        except urllib.error.HTTPError as e:
            check(f"vision_{name}", False, f"HTTP {e.code} {e.read().decode()[:160]}")
        except Exception as e:  # noqa: BLE001
            # RemoteDisconnected here is what the OOM crash looked like from the client.
            check(f"vision_{name}", False, f"{type(e).__name__}: {e} "
                  "(a dropped connection means the server died mid-encode -- check RestartCount)")

    # 3 -- what does CPU encoding actually cost? max_tokens=1 makes the round trip
    #      almost entirely encode+prefill, so image_minus_text ~= the ViT forward.
    try:
        t0 = time.perf_counter()
        uri = "data:image/png;base64," + base64.b64encode(solid_png(COLORS["red"])).decode()
        post(args.url, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "OK"},
                {"type": "image_url", "image_url": {"url": uri}}]}],
            "max_tokens": 1, "temperature": 0, "stream": False}, timeout=300)
        img1_ms = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        post(args.url, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "OK"}],
            "max_tokens": 1, "temperature": 0, "stream": False}, timeout=300)
        txt1_ms = 1000 * (time.perf_counter() - t0)
        out["checks"]["encode_cost"] = {"ok": True, "detail": f"image={img1_ms:.0f} ms text={txt1_ms:.0f} ms"}
        print(f"INFO  encode_cost: image+1tok={img1_ms:.0f} ms, text+1tok={txt1_ms:.0f} ms "
              f"=> CPU ViT encode ~= {img1_ms - txt1_ms:.0f} ms")
    except Exception as e:  # noqa: BLE001
        print(f"INFO  encode_cost: not measured ({e!r})")

    # 4 -- did the server survive us?
    try:
        h = json.load(urllib.request.urlopen(args.url + "/health", timeout=30))
        check("alive_after_vision", h.get("status") == "ok", h)
    except Exception as e:  # noqa: BLE001
        check("alive_after_vision", False, repr(e))

    if args.log:
        with open(args.log, "a") as f:
            f.write(json.dumps(out) + "\n")
        print(f"\nrecord appended to {args.log}")

    print("\nRESULT:", "ALL PASS" if not out["failures"] else f"FAILURES: {out['failures']}")
    return 0 if not out["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
