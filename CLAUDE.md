IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## MXFP Flash Attention — MANDATORY RULES

**Read [PLAN.md](PLAN.md) BEFORE touching any MXFP or Metal FA code. Every session. No exceptions.**

PLAN.md is the design spec. It contains the architecture, the memory layout, the pipeline, and the WHY for every decision. If code contradicts PLAN.md, the code is wrong. Fix the code, not the plan.

### SoA-Only Architecture
- MXFP Flash Attention uses **Struct-of-Arrays (SoA) memory layout ONLY**
- Layout: `[all_qs contiguous][all_e8m0 contiguous]` per full multi-head row
- This enables uint32_t-aligned loads → GPU memory pipeline parallelism → performance
- AoS (`block_mxfp4*`, `block_mxfp8*`, `block_mxfp6*` struct pointers) is FORBIDDEN in FA

### Before writing ANY Metal FA code, verify:
1. Does it use `device const char *` row pointers (NOT block_mxfp* structs)?
2. Are memory loads uint32_t-aligned from contiguous qs regions?
3. Is e8m0 loaded from the separate region at offset nblocks * QS_PER_BLOCK?
4. Does it match the pipeline in PLAN.md?

### If you see block_mxfp* in FA template params → it’s AoS → DELETE IT
### If you want to rationalize keeping AoS → STOP → re-read PLAN.md
### CUDA (reference) uses SoA exclusively. Metal must match.
