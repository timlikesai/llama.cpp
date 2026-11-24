# Phase 2b — Native NVFP4 expert export for Inkling-2

Progress notes, written live. Brief: `docs/mxfp8/briefs/J-phase2b-nvfp4.md`.

Status: **in progress** (skeleton written early, updated as I go).

## Goal

`--native-experts` currently keeps only the 8 MXFP8 expert tensors (layers 20-23) as native
bytes; the 40 NVFP4 expert tensors (layers 0-19, w13 + w2) still dequantize to F32 -> BF16,
which costs ~36.5 GiB of the 44.95 GiB output. Emitting them as `GGML_TYPE_NVFP4` (40) with
the F32 per-expert `.scale` sidecar should land the lossless file at ~18 GB.

## Files I own

- `conversion/inkling2.py` (extend the existing native path)
- `scripts/inkling_native/verify_nvfp4_gguf.py` (new)
- `docs/mxfp8/11-phase2b-nvfp4.md` (this file)

## Notes / findings

TODO

## Acceptance gates

TODO

## Corrections to my brief

TODO
