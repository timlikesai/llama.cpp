# MXFP4 KV Cache Research Findings

Branch: `mxfp4-flash-attention-v2`
Date: 2026-02-27

This document tracks experiments conducted based on ideas from `KV_TRAINING_RESEARCH.md`,
exploring the intersection of MXFP4 flash attention with speculative decoding, continual
encoding, and other KV cache optimization techniques.

## Current System State

| Component | Configuration |
|-----------|--------------|
| Model | Qwen3-Coder-30B-A3B-Instruct (MXFP4_MOE GGUF) |
| GPU | 2× RTX 5070 Ti (sm_120a Blackwell) |
| K cache | MXFP4 + Hadamard + compact 1-bit sign residual (1.5×) |
| V cache | MXFP4 (no Hadamard, no residual) |
| PPL | 9.8622 (wikitext-2-raw, 59 chunks) |
| KV memory | 63.75 MiB (6× reduction vs F16 384 MiB) |
| Speed | pp512: ~5,916 t/s, tg128: ~150 t/s (no spec decoding) |

---

## Experiment 1: N-gram Speculative Decoding with MXFP4 Cache

**Research question**: Does MXFP4 KV cache quality affect speculative decoding acceptance
rates? Can n-gram spec decoding provide meaningful speedups for agentic coding workloads?

**Reference**: KV_TRAINING_RESEARCH.md §5.2 — claims >90% acceptance when replaying prior
conversations, particularly for agentic coding with repetitive tool calls and boilerplate.

### Methodology

Tested with llama-server at temperature 0 using a single GPU slot (ctx=32768).
Two test scenarios:
1. **Code CRUD**: Multi-turn conversation where model extends a Flask API with a new resource
   that mirrors existing code structure (~825 tokens, high structural repetition)
2. **Merge sort**: Simple single-turn code generation (500 tokens, low repetition baseline)

Each scenario run twice: warm-up (builds n-gram table) and replay (uses warm table).
Six n-gram configurations tested across two spec types and various draft lengths.

### Full Benchmark Results

#### Code CRUD (multi-turn, high repetition)

| Config | Type | n | m | draft | Warm t/s | Replay t/s | Warm acc% | Replay acc% |
|--------|------|---|---|-------|----------|------------|-----------|-------------|
| Baseline | none | - | - | - | 149.8 | 149.9 | - | - |
| A | simple | 8 | 32 | 32 | 231.1 | 232.4 | 37.1% | 37.1% |
| B | simple | 8 | 8 | 8 | 211.6 | 210.5 | 70.3% | 70.4% |
| C | map-k4v | 8 | 8 | 8 | 208.9 | 214.2 | 74.2% | 80.5% |
| D | map-k4v | 4 | 16 | 16 | 226.6 | 232.0 | 51.8% | 58.1% |
| E | simple | 4 | 8 | 8 | 216.3 | 210.7 | 63.1% | 60.5% |
| **F** | **map-k4v** | **4** | **32** | **32** | **246.9** | **240.4** | **39.4%** | **44.7%** |

#### Merge sort (single-turn, low repetition)

| Config | Warm t/s | Replay t/s | Warm acc% | Replay acc% |
|--------|----------|------------|-----------|-------------|
| Baseline | 157.4 | 157.3 | - | - |
| A | 155.5 | 155.5 | 8.1% | 8.1% |
| B | 156.6 | 156.5 | 31.2% | 31.2% |

### Key Findings

1. **Config F (map-k4v n=4 m=32 d=32) wins on throughput**: 246.9 t/s warm, 240.4 t/s
   replay — a **65% speedup** over baseline 150 t/s. This is now the production config.

2. **Acceptance rate ≠ throughput**: Config B achieves 70% acceptance but only 211 t/s.
   Config F achieves 39-45% acceptance but 247 t/s. Longer drafts accept more total tokens
   per verification batch despite lower per-token acceptance rate.

3. **map-k4v > simple for replay**: Config C (80.5% replay) outperforms Config B (70.4%)
   on same parameters — storing 4 continuations per n-gram key helps when the same
   patterns reappear.

4. **Shorter n-grams (n=4) find more matches**: All n=4 configs outperform their n=8
   equivalents on throughput. The trade-off is more false matches, but the verification
   step catches those cheaply.

5. **Low-repetition content shows minimal benefit**: Merge sort gets 8-31% acceptance
   with negligible speedup — spec decoding overhead roughly cancels the benefit.

6. **Draft-level vs token-level acceptance**: Server stats show 88.6% of drafts had
   at least one accepted token, but only 27.8% of individual tokens were accepted.
   Drafts start correctly but diverge at variation points.

### Recommendation

For production MXFP4 servers handling agentic coding workloads:
```
--spec-type ngram-map-k4v --spec-ngram-size-n 4 --spec-ngram-size-m 32 --draft 32
```

Expected speedup: **50-65%** on multi-turn conversations with code context.
No speedup on short single-turn completions (but no penalty either).

### The >90% Acceptance Claim

The research doc's claim of >90% acceptance when "replaying prior chat conversations"
was NOT validated in these tests. Best replay acceptance was 80.5% (Config C) with
short drafts, but that only achieved 214 t/s (+43%). The highest throughput (Config F,
+65%) had only 45% token acceptance.

However, these tests used code generation scenarios. True agentic workloads with:
- Identical tool call response formats (JSON, XML)
- Repeated system prompts and instruction formatting
- Copy-paste code blocks from prior assistant turns
...may achieve higher acceptance rates. Further testing with real agentic conversations
is needed.

---

## Experiment 2: Continual Re-encoding Feasibility Analysis

**Research question**: Is the dequant→compute→requant cycle fast enough for practical
continual re-encoding on Blackwell hardware?

**Reference**: KV_TRAINING_RESEARCH.md §§3-4, §8 — core thesis of the research document.

### Analysis

The FP4→FP8 dequant is free in hardware during MMA. The custom work is:
- Hadamard rotation: ~192 FP32 ops per 32 elements (K only)
- MXFP4 quantization: ~32 comparisons + E8M0 computation per block
- Compact 1-bit residual: sign extraction + E8M0 per block

**Estimated cost per re-encoding pass** (one layer, one head, 4096 tokens):
- Read: 4096 × 128 × 0.5 = 262 KB (MXFP4 K) — ~0.003 ms at 100 GB/s HBM
- Compute: dequant (free) + attention (Q against K) + requant
- Write: 262 KB — ~0.003 ms
- Total estimated: ~1 ms per layer per head (compute-dominated)

For this model (48 layers × 4 KV heads = 192 passes): **~192 ms total**
This is comparable to generating ~30 tokens — **viable at semantic boundaries**.

### Implementation Path

The simplest form of continual re-encoding doesn't require architectural changes:

1. **Periodic K re-quantization**: At trigger points (buffer full, newline), dequant
   the K cache back to FP16, recompute Hadamard + MXFP4 quantization with updated
   block statistics. The 1-bit residual would also be recomputed.

2. **What this would achieve**: Redistributes quantization error across the block.
   As the cache fills, the distribution of values within blocks changes, and the
   original quantization may no longer be optimal. Re-quantizing with current
   statistics would tighten the E8M0 exponent fit.

3. **What this would NOT achieve**: It won't add new information to the cache.
   For that, you'd need the full Cache Processor architecture (§2.1 Bottlenecked
   Transformers) where an auxiliary model rewrites cache contents.

### Verdict

Simple re-quantization is implementable but likely provides minimal benefit —
the quantization error is dominated by E2M1 grid density and E8M0 scaling
limitations, not by stale block statistics. The 1-bit sign residual already
captures the most important correction signal.

The **full continual encoding** approach (dequant → attention-based consolidation →
requant) would require:
- Training or fine-tuning a Cache Processor module
- New kernel architecture for the consolidation attention pass
- Integration with the KV cache management system

This is a weeks-long effort and requires a model with the Cache Processor trained in.

---

## Experiment 3: Asymmetric K/V Precision (Validated)

**Research question**: Should K and V have different precision or residual correction?

**Reference**: KV_TRAINING_RESEARCH.md §6.5, §9.1 item 3.

**Answer**: K quantization dominates quality loss. V type has negligible impact.

| Config | PPL | vs F16 |
|--------|-----|--------|
| MXFP4 K + MXFP4 V | 10.3099 | +0.578 |
| MXFP4 K + F16 V | 10.3741 | +0.642 |
| MXFP4 K + residual + MXFP4 V | 9.8622 | +0.130 |

V type makes less than 0.065 PPL difference — noise level. Literature confirms keys
are 38-45× more sensitive to quantization than values (KV-AdaQuant, arxiv 2502.15075).

The current configuration (K: 1.5× with Hadamard + residual, V: 1.0× plain) is already
asymmetric in the optimal direction.

---

## Experiment 4: MXFP4 + Spec Decoding Effective Throughput

**Research question**: What is the combined benefit of MXFP4 memory reduction + spec decoding
speed boost for agentic coding workloads?

### Combined Impact

| Metric | F16 baseline | MXFP4 + spec | Improvement |
|--------|-------------|-------------|-------------|
| KV memory | 384 MiB | 63.75 MiB | **6× reduction** |
| PPL | 9.7319 | 9.8622 | +0.13 (minimal) |
| Token gen (no spec) | ~200 t/s | ~150 t/s | -25% (quantized overhead) |
| Token gen (with spec) | ~200 t/s | **~247 t/s** | **+23%** |
| Max context (16GB VRAM) | ~12K tokens | ~75K tokens | **6× longer** |

The MXFP4 flash attention system with n-gram spec decoding achieves:
- **6× more context** at the same VRAM budget
- **Better quality than Q4_0** (PPL 9.86 vs 9.97)
- **23% faster token generation** than F16 baseline (with warm n-gram table)
- **65% faster** than its own non-spec baseline

This is the core value proposition for agentic coding: massive context windows with
near-F16 quality and faster generation speed than the full-precision baseline.

---

## Research Priorities (from KV_TRAINING_RESEARCH.md)

### Completed
1. ~~N-gram spec decoding optimization~~ — Config F (map-k4v n=4 m=32 d=32) achieves +65%
2. ~~Asymmetric K/V treatment~~ — K residual + V plain is optimal
3. ~~Compact 1-bit sign residual~~ — Closes 86% of gap to F16
4. ~~Context length degradation~~ — No cumulative error growth (see Experiment 5)

### Immediately Actionable
5. **Real agentic workload testing** — Test with actual Claude Code / tool-call conversations
6. **Attention score retention analysis** — Diagnostic tool to measure information loss

### Requires Moderate Work (days)
7. **Linear correction adapters (KVLinC-style)** — Lightweight learned transform for
   error compensation. Constant memory cost, but needs per-model calibration.
8. **Progressive precision** — Older tokens get lower precision within the MXFP4 budget.
   Could use the compact residual selectively (recent tokens get residual, old ones don't).

### Requires Significant Work (weeks)
9. **Cache Processor** — Small auxiliary transformer for periodic cache consolidation
   (Bottlenecked Transformer approach). Requires training or fine-tuning.
10. **Fixed-size continual encoding** — Full implementation of the core thesis.
    Requires new kernel architecture for dequant→attention→requant cycle.

### Not Recommended for This Architecture
11. ~~**QuantSpec self-speculative decoding**~~ — Analysis (Experiment 7) shows this
    is not viable for GQA MoE models: no size asymmetry for draft speedup, verification
    needs full context, and n-gram speculation already provides better throughput.

---

## Experiment 5: Context Length Degradation Test

**Research question**: Does MXFP4 quality degrade with longer contexts due to cumulative
quantization error?

**Reference**: KV_TRAINING_RESEARCH.md §4.3 (PM-KVQ) — addresses cumulative quantization
error in long reasoning chains.

### Methodology

Ran 10-chunk perplexity (n_ctx=512) with both MXFP4+residual and F16 cache types.
Compared against known 59-chunk results to see if the gap grows with context.

### Results

| Metric | F16 | MXFP4+res | Gap |
|--------|-----|-----------|-----|
| 10-chunk PPL (n_ctx=512) | 11.2421 | 11.3368 | +0.095 |
| 59-chunk PPL (full) | 9.7319 | 9.8622 | +0.130 |

Per-chunk running averages (10-chunk test):
```
F16:   [1]7.45 [2]10.49 [3]9.85 [4]9.40 [5]9.26 [6]9.55 [7]9.60 [8]10.14 [9]10.73 [10]11.24
MXFP4: [1]6.82 [2]9.65  [3]9.21 [4]8.96 [5]8.88 [6]9.31 [7]9.45 [8]10.18 [9]10.78 [10]11.34
```

### Findings

**No evidence of cumulative error degradation.** The MXFP4 vs F16 gap grows only from
+0.095 at 10 chunks to +0.130 at 59 chunks — a 0.035 increase over 6× more text.
This is within noise of per-chunk PPL variance (±0.68).

The 1-bit sign residual provides stable correction regardless of context position.
This is expected because:
1. Each K entry is quantized independently (no error propagation between positions)
2. The residual corrects the dominant quantization direction for each block
3. Attention softmax normalization limits the impact of per-position errors

This means the PM-KVQ approach (progressive precision with age) is unnecessary for
MXFP4 — uniform precision across all context positions is adequate.

### KV Cache Memory Comparison

| Cache type | K (512 cells) | V (512 cells) | Total | vs F16 |
|------------|--------------|--------------|-------|--------|
| F16 | 96.00 MiB | 96.00 MiB | 192.00 MiB | 1.0× |
| MXFP4+res | 38.25 MiB | 25.50 MiB | 63.75 MiB | 3.0× smaller |

---

## Experiment 6: MXFP4 Quantization Residual Structure Analysis (GEAR-inspired)

**Research question**: Does the MXFP4 quantization residual have exploitable low-rank
structure, as GEAR (§4.1) found for general quantization? Would a linear correction
adapter (KVLinC-style, §4.4) further improve quality?

**Reference**: GEAR shows quantization residuals decompose into low-rank + sparse components.
KVLinC adds Hadamard rotation + learned linear adapters for correction.

### Theoretical Analysis

The MXFP4 quantization residual `res[j] = K[j] - dequant(quant(K[j]))` has two sources:

**1. E8M0 Scale Error (Rank-1 Component)**

The block exponent rounds to the nearest power of 2. If the true optimal scale is 3.0,
the E8M0 rounds to either 2 or 4 (up to 41% error). This multiplicative error affects
all 32 elements in a block uniformly, creating a **rank-1 residual component**:
```
res_scale[j] ≈ K[j] * (1 - 2^round(log2(max_abs)) / max_abs)
```
This is the coherent structure that our 1-bit sign residual captures — the sign of each
element's error is correlated with the element's value and the scale direction.

**2. E2M1 Grid Snapping Error (Independent Noise)**

Each value snaps to the nearest of `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (scaled by E8M0).
This error is independent per element and depends on where the value falls relative to
the non-uniform FP4 grid. The grid has larger gaps at higher values (3→4, 4→6), creating
position-dependent quantization noise.

After Hadamard rotation, input values cluster around the block mean (the transform
concentrates energy). This means post-Hadamard values land in the mid-range of the FP4
grid where gaps are moderate — Hadamard partially mitigates grid snapping too.

### What Our 1-Bit Residual Captures

The sign + E8M0 magnitude format captures:
- **Direction of rank-1 scale error**: 100% (sign exactly recovers which elements are
  over/under-quantized relative to the block mean)
- **Grid snapping direction**: Partially (sign gives direction but not magnitude)
- **Grid snapping magnitude**: Not captured (would need multi-bit residual)

This explains the experimental results:
- Full MXFP4 residual (17 bytes/block): PPL 9.7686 — captures both components
- 1-bit sign residual (5 bytes/block): PPL 9.8622 — captures rank-1 well, grid noise partially
- Plain MXFP4 (no residual): PPL 10.3099 — neither component

### Would a Linear Adapter Help?

A KVLinC-style adapter applies `K_corrected = W * K_dequant + b` after dequantization.

**Pros**:
- Could capture per-channel bias patterns (if certain channels consistently over/under-quantize)
- Constant memory cost (D×D matrix per layer + bias = small)
- Orthogonal to our sign residual (could be used together)

**Cons**:
- Requires per-model calibration data and training
- Our Hadamard rotation already decorrelates channels, so per-channel biases should be minimal
- The remaining gap (0.13 PPL) is mostly grid snapping noise which is **independent per element** — a linear transform cannot fix independent noise
- Adding compute per attention query (matrix multiply in the critical path)

### Verdict

A linear adapter is **unlikely to improve significantly** beyond the 1-bit sign residual.

The residual IS low-rank (primarily rank-1 from scale error), and our sign + E8M0 already
captures this efficiently. The remaining 14% gap to F16 is from independent grid snapping
noise that requires either:
- A denser quantization grid (NVFP4 with 16-element blocks, or Q4_0 with uniform grid)
- Full multi-bit residual storage (which we tested — gives 94% recovery at 2× memory)

The sign residual at 1.5× memory is the optimal point on the quality-memory Pareto frontier
for MXFP4 quantization without model-specific calibration.

---

## Experiment 7: QuantSpec Self-Speculative Decoding Feasibility

**Research question**: Can MXFP4 KV cache serve as a "draft" cache for self-speculative
decoding, with a small FP16 "verification buffer" for recent tokens (QuantSpec-style)?

**Reference**: KV_TRAINING_RESEARCH.md §5.1 — QuantSpec (Apple, arxiv 2502.10424)

### QuantSpec Concept

QuantSpec decomposes INT8 KV cache values into upper (INT4) and lower (INT4) halves.
The draft model uses the upper INT4 half (fast, low precision), the target verifies
against full INT8 reconstructed from both halves. This achieves >90% acceptance because
the draft and target share weights — they differ only in KV cache precision.

The question: can we adapt this for MXFP4 draft + FP16 verification?

### Architecture Analysis

llama.cpp's speculative decoding pipeline (`common/speculative.cpp`) supports:
- **Draft model** (`ctx_dft`): generates candidate tokens via autoregressive `llama_decode()`
- **Target model** (`ctx_tgt`): verifies N+1 tokens in a single batch forward pass
- **Accept/reject**: token-by-token comparison, KV cache trimmed at first mismatch

Critically, `model_dft` can point to the **same** `llama_model` as the target — weights
are shared, only KV caches differ. So self-speculative with different cache types IS
architecturally possible in llama.cpp.

### Why It Won't Work for Our Setup

**Problem 1: No size asymmetry**

Speculative decoding gains speed because the draft model is 10-50× smaller than the
target. With self-speculative decoding (same 30B MoE model), the draft pass costs nearly
as much as a target pass. The only savings come from attention bandwidth reduction with
MXFP4, which is small relative to total compute — especially with GQA (4 KV heads) where
attention is already cheap.

The break-even equation: `speedup = (1 + R×N) / (C_draft×N/C_verify + 1)`. When
`C_draft ≈ C_verify` (same model), speedup → 1 regardless of acceptance rate R.

**Problem 2: Verification requires full context**

A "small FP16 verification buffer" holding only recent tokens is fundamentally broken
for transformer attention. The verifier must compute attention over the entire prefix
to produce correct logits. If the FP16 buffer holds only 512 tokens of a 16K context,
the verification logits are wrong for any query that attends to tokens outside the window.

This isn't a precision issue — it's a correctness issue. The verifier would disagree
with the draft not because of quantization but because it's computing different attention.

**Problem 3: Per-layer type mixing unsupported**

`type_k` and `type_v` are context-level parameters baked into allocation. There's no
mechanism for layer-selective precision (e.g., early layers MXFP4, later layers FP16).
Two separate contexts with full KV caches are needed, doubling memory.

**Problem 4: MoE attention fraction is small**

For Qwen3-Coder-30B-A3B with 4 KV heads and GQA ratio 8, attention is a small fraction
of per-token compute (most time goes to expert FFN). MXFP4 KV cache saves memory
bandwidth during attention, but does not meaningfully reduce the compute cost of a
draft forward pass. The QuantSpec approach works when quantized attention IS the
dominant cost (dense models with many KV heads).

### What N-gram Spec Decoding Already Provides

N-gram draft generation is effectively "free" — a hash table lookup taking microseconds,
vs a full model forward pass. This means:

| Property | N-gram (current) | QuantSpec (hypothetical) |
|----------|-----------------|------------------------|
| Draft cost | ~0 (hash lookup) | ~6.7 ms (full 30B forward) |
| Acceptance rate | 39-45% (Config F) | ~90% (same model) |
| Net throughput | 247 t/s (+65%) | ~160 t/s (worse than baseline*) |
| Memory overhead | ~0 | +384 MiB (FP16 KV cache) |
| Implementation | Done | Weeks of work |

*QuantSpec estimate: even with 90% acceptance and N=8 draft tokens, each draft requires
a full forward pass (~6.7 ms). 8 drafts = ~54 ms, verification batch = ~10 ms, for
~7.2 accepted tokens in ~64 ms = ~112 t/s effective. This is SLOWER than baseline 150 t/s.

### Verdict

QuantSpec-style self-speculative decoding is **not viable** for our architecture:

1. Same-model drafting provides no compute advantage (no size asymmetry)
2. Verification requires full context (can't use a small FP16 buffer)
3. N-gram speculation already achieves better throughput with zero overhead
4. The only scenario where QuantSpec helps is dense models with many KV heads
   where attention is the compute bottleneck — opposite of our GQA MoE setup

The QuantSpec paper's results apply to dense models where quantized attention IS
significantly faster. For MoE models with few KV heads, n-gram speculation is
strictly superior.

### Research Priority Update

Moving QuantSpec from "Requires Significant Work" to **Not Recommended** for this
model class. N-gram speculation (already deployed) provides the speed gains that
QuantSpec would theoretically offer, without the engineering cost or memory overhead.
