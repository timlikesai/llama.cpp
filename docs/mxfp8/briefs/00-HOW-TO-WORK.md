# HOW TO WORK (read this before your brief)

You are one of several parallel research/implement children on the Inkling-2-Mini MXFP8 project. Follow your own brief file for *what*; this file is about *how*.

## Hard rules (learned the hard way)

1. **A 30-minute wall will end you.** Your parent has seen several children die at the timeout with zero output. So:
   - **Create your deliverable file within your first 6 tool calls** — skeleton with section headers + `TODO:` under each.
   - **Rewrite/extend the file after every 3–5 findings**, not at the end. Partial-but-on-disk always beats complete-but-lost.
   - Prefer breadth of *facts* over polish of prose. Bullets and tables, no narrative padding.
2. **Do not edit anything outside your assigned output paths.** Other agents are working in the same repo right now. Read-only elsewhere.
3. **Never** run `cmake`, `make`, `ninja`, `docker`, or anything that touches GPUs or services — the parent owns those. Never delete any file or log.
4. Cite `file:line` for claims about this repo. Cite URLs for external claims. Anything you inferred but did not verify gets the literal marker `UNVERIFIED`.
5. If your brief asks something impossible or you find it contradicted by the code, **say so in the doc** under `## Corrections to my brief` — do not silently answer a softer question.
6. Finish (or at 22 minutes elapsed, stop and finalize) with a short stdout summary: what you wrote, the 5 most decision-relevant findings, and the top open question.

## Repo facts you can rely on (do not spend calls re-verifying)

- Repo: `/home/tim/code/services/repos/llama.cpp`, branch `mxfp8-weights` = upstream master `29b52d398` ("model : Inkling 2 Mini (#32694)").
- Goal: add `GGML_TYPE_MXFP8` = 33 bytes per 32 elements (`{uint8_t e /*ue8m0*/; uint8_t qs[32] /*e4m3*/}`), the analogue of `block_mxfp4` at `ggml/src/ggml-common.h:215-219`, so the raw checkpoint repacks **bit-exactly**.
- Checkpoint (read-only, local): `~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/` — `model-00001-of-00001.safetensors` 19,526,019,168 B; experts layers 0–19 = NVFP4 (U8 payload, F8_E4M3 scales, 1×16 K, plus F32 `…_amax`/`…_outer_scale_denominator`), layers 20–23 = MXFP8 (F8_E4M3 payload, U8 `ue8m0` scales, 1×32 K); everything else BF16/F32; `mtp.safetensors` 635,801,416 B BF16.
- Existing GGUFs to inspect: `/home/tim/models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf` (all requantized) and `/home/tim/models/unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf` (**40 native NVFP4 expert tensors blk.0–19 + F32 `[128]` `.scale` sidecars; blk.20–23 experts are Q8_0**), both 17.63 GB. Read with `sys.path.insert(0,'gguf-py'); from gguf import GGUFReader`. Enum: F32=0, Q8_0=8, BF16=30, IQ4_NL=20, MXFP4=39, NVFP4=40, F8_E4M3=43.
- Python GGUF access pattern that works (do not re-discover):
  ```python
  import json, struct, os, glob
  S = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/*"))[0]
  p = os.path.join(S, "model-00001-of-00001.safetensors")
  with open(p, "rb") as f:
      n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
  # h[name] -> {"dtype","shape","data_offsets":[lo,hi]} ; data starts at 8+n, seek+read per tensor
  ```
  `GGUFReader` in this tree has **no `.version` attribute** and `ReaderField` has `.types`/`.parts`, not `.type` — use `.contents` (call it as `f.contents()` where applicable) or read `f.parts`.
- Big-file hygiene: 61 GB RAM. Stream with offsets/mmap; keep peak RSS < 6 GB.

## Project doc index

`docs/mxfp8/00-status.md` (live log), `01-roadmap.md` (plan + decision records DR-1…DR-6 + risks R1…R8), `briefs/{A..H}-*.md` (the briefs), and the deliverables `02-prior-art-branches.md`, `03-touchpoint-checklist.md`, `04-inkling2-master.md`, `05-hw-and-ecosystem.md`, `06-lossless-repack-design.md`, `07-vram-placement.md`, `08-plan-review.md`, `09-other-engines-mxfp8.md`.
