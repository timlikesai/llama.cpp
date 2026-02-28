# Continual Encoding KV Cache: Research Landscape & Implementation Direction

## For Claude Code — Reference Document for MXFP4 Flash Attention Kernel Development on Blackwell

---

### Project Context

Tim is building custom MXFP4 flash attention kernels for llama.cpp targeting NVIDIA Blackwell GPUs. The kernel is stable with perplexity slightly above q4_0. A **multi-pass encoding technique** prototype (Python, built by Opus in Claude Code) showed compelling results — a second encoding pass significantly limited quantization losses. This document explores the research basis for extending that multi-pass insight into a **continual encoding KV cache**: a fixed-size cache that is iteratively refined rather than grown or evicted.

### Key Performance Observation

N-gram speculative decoding with the MXFP4 kernel achieves **>90% acceptance rates** when replaying prior chat conversations on fresh runs. This produces extremely high token generation speeds — particularly valuable for **agentic coding workloads** where tool calls, code outputs, structural tokens, and boilerplate are highly repetitive and thus trivially speculatively generated from the cache.

---

## Table of Contents

1. [Core Thesis](#1-core-thesis)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Fixed-Size Compressive Memory Architectures](#3-fixed-size-compressive-memory-architectures)
4. [Quantization Error Correction — Why Multi-Pass Works](#4-quantization-error-correction--why-multi-pass-works)
5. [Speculative Decoding Synergies](#5-speculative-decoding-synergies)
6. [Blackwell Hardware Intrinsics & FP4 Implementation](#6-blackwell-hardware-intrinsics--fp4-implementation)
7. [llama.cpp Integration Points](#7-llamacpp-integration-points)
8. [Proposed Architecture](#8-proposed-architecture)
9. [Open Questions & Experiments](#9-open-questions--experiments)
10. [Full Reference Index](#10-full-reference-index)

---

## 1. Core Thesis

Instead of the standard approach — grow the KV cache linearly, then compress/evict when memory is exhausted — we propose:

1. **Fix the KV cache to a constant size** (e.g., 2048 or 4096 slots)
2. **Store it in MXFP4** (E2M1 with block-of-32 E8M0 scaling) for minimal memory footprint
3. **Continually re-encode** the cache as new context arrives, refining the representation within the fixed budget
4. **Leverage Blackwell's native FP4→FP8 dequant path** for the re-encoding attention passes
5. **Combine with n-gram speculative decoding** to exploit the high token-repetition patterns in agentic coding workloads

This is a learned, fixed-budget context compression running at inference time — not static quantization, not eviction, but iterative refinement of what information the cache holds.

**Why this is novel**: Existing work treats the cache as either (a) a growing buffer you compress, or (b) a static representation you quantize once. Nobody has shipped a system that treats the cache as a *continually trained artifact* in FP4, optimized for Blackwell Tensor Core execution.

---

## 2. Theoretical Foundations

### 2.1 Bottlenecked Transformers — Periodic KV Cache Consolidation

**Paper**: "Bottlenecked Transformers: Periodic KV Cache Consolidation for Generalised Reasoning"
**ArXiv**: https://arxiv.org/abs/2505.16950 (May 2025)

This is the strongest theoretical validation of the continual encoding idea. The authors prove via **Information Bottleneck (IB) theory** that:

- Standard autoregressive training produces KV representations that are *minimally compressive* and *maximally reconstructive* — the cache stores enough to reproduce the input verbatim, wasting capacity on reconstruction rather than predictive/reasoning features.
- Periodically **rewriting** the cache with an auxiliary **Cache Processor** (a small Transformer) compresses away reconstruction-oriented information and retains only predictive features.
- The Cache Processor performs **non-causal, in-place KV rewrites** at reasoning step boundaries (newline-delimited).
- This is memory *reconsolidation* in the neuroscientific sense — reactivated memories enter a plastic state and can be modified before restabilizing.

**Key architectural detail**: The Cache Processor consolidates recently written KV entries and reconsolidates a small top-k attention-selected set of prior entries. It operates at step boundaries (newlines), which maps naturally to agentic coding boundaries (tool call outputs, code blocks, etc.).

**Critical insight for our work**: "In-place memory rewrites under (re)consolidation do not necessarily entail reduction in memory footprint." — We *do* want footprint reduction (fixed MXFP4), so we're combining their rewrite insight with quantization-aware compression.

**Cross-references**: [§4.1 GEAR](#41-gear--low-rank-residual-error-correction) for error structure exploited by rewrites, [§6](#6-blackwell-hardware-intrinsics--fp4-implementation) for FP4 compute path, [§8](#8-proposed-architecture) for how the Cache Processor maps to our implementation.

### 2.2 Information-Theoretic Limits of KV Cache Compression

**Paper**: "Limits of KV Cache Compression for Tensor Attention based Autoregressive Transformers"
**ArXiv**: https://arxiv.org/abs/2503.11108 (March 2025)

Establishes fundamental **space complexity lower bounds** for KV cache compression via reduction from communication complexity. Key finding: there exist information-theoretic limits on how aggressively you can compress without losing expressivity, and these limits depend on the attention mechanism variant (standard vs. tensor attention).

**Relevance**: Our continual encoding approach doesn't violate these bounds — it operates within them by choosing *which* information to preserve (predictive vs. reconstructive, per the IB theory above). The bounds help us understand the theoretical floor for perplexity at a given cache size.

**Cross-references**: [§2.1](#21-bottlenecked-transformers--periodic-kv-cache-consolidation) for IB theory context.

---

## 3. Fixed-Size Compressive Memory Architectures

### 3.1 Infini-attention — Compressive Memory with Bounded Footprint

**Paper**: "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
**ArXiv**: https://arxiv.org/abs/2404.07143 (Google, April 2024)

The closest existing architecture to our proposal. Core mechanics:

- **Fixed-size compressive memory**: A `d_key × d_value` matrix per head — constant regardless of sequence length.
- **Update rule**: Old KV states are accumulated into memory via a linear attention outer-product (delta rule). New tokens update the memory; old segments are discarded.
- **Retrieval**: Queries attend to the memory matrix to retrieve compressed historical context.
- **Gating**: A learned scalar β gates between local causal attention (current segment) and memory-retrieved content: `A = sigmoid(β) ⊙ A_mem + (1 - sigmoid(β)) ⊙ A_dot`
- **Streaming**: Processes input segment-by-segment with bounded memory and compute.

**Limitation**: The memory update is a simple additive accumulation, not a learned gradient-based refinement. It's a running summary, not an iteratively optimized representation.

**Implementation relevance**: The segment-by-segment streaming and gating mechanism are directly implementable. The memory size is fixed and knowable at compile time — ideal for MXFP4 pre-allocation.

**Key difference from our approach**: Infini-attention *accumulates*; we want to *refine*. Their update is O(1) per token; our re-encoding is more expensive but potentially much higher quality.

**Cross-references**: [§3.2 RMT](#32-recurrent-memory-transformer-rmt--learned-memory-tokens) for learned memory tokens, [§8](#8-proposed-architecture) for how we extend this with re-encoding.

### 3.2 Recurrent Memory Transformer (RMT) — Learned Memory Tokens

**Paper**: "Recurrent Memory Transformer" — NeurIPS 2022
**ArXiv**: https://arxiv.org/abs/2207.06881
**Scaling paper**: "Scaling Transformer to 1M tokens and beyond with RMT" — https://arxiv.org/abs/2304.11062
**GitHub**: https://github.com/booydar/recurrent-memory-transformer

The purest existing form of "continually trained fixed-size context":

- Adds **special memory tokens** (typically 10) to the input sequence — these *are* the fixed-size context.
- The model is trained via BPTT across segments to control both memory operations and sequence processing.
- Demonstrated storage/retrieval across sequences of up to **2 million tokens**.
- Attention patterns reveal distinct **write-to-memory** and **read-from-memory** phases.
- Uses **curriculum learning** — training on shorter sequences first, then extending.
- Can be combined with Transformer-XL caching for hybrid local+global memory.

**Key insight**: RMT proves that a tiny fixed-size memory (10 tokens!) can represent extremely long contexts if the memory is *learned*. The training loop for memory tokens is essentially our re-encoding step.

**Implementation relevance**: Curriculum learning suggests our re-encoding could benefit from a warmup phase (start with larger cache, progressively compress). The hybrid RMT+Transformer-XL approach maps to our "fixed MXFP4 cache + recent token buffer in higher precision" architecture.

**Cross-references**: [§3.3 Titans](#33-titans--test-time-memory-learning), [§8](#8-proposed-architecture).

### 3.3 Titans — Test-Time Memory Learning

**Paper**: "Titans: Learning to Memorize at Test Time" (Google, 2025)
**Semantic Scholar**: https://www.semanticscholar.org/paper/Recurrent-Memory-Transformer-Bulatov-Kuratov/a8cf0f7a20f886acfb332071c2daaf58ba86a5ca (see citing papers)

Introduces a neural long-term memory module that **learns to memorize at test time** — the memory module is updated during inference, not just during training. Three architecture variants for incorporating memory. This validates that test-time/inference-time learning of memory representations is viable.

**Cross-references**: [§3.2 RMT](#32-recurrent-memory-transformer-rmt--learned-memory-tokens) for learned memory, [§2.1](#21-bottlenecked-transformers--periodic-kv-cache-consolidation) for why rewrites improve reasoning.

### 3.4 Hierarchical Memory Transformer (HMT) — NAACL 2025

Implements **brain-inspired memory hierarchy**: learned memory tokens at multiple levels (sensory, short-term, long-term) with a retrieval mechanism determining when to write or read. Key advantage: model-agnostic, plug-and-play design applicable to existing LLMs without architectural changes. Reduces perplexity by up to 25% on long-document benchmarks.

**Relevance**: Our fixed-size MXFP4 cache could benefit from hierarchical organization — e.g., a small "hot" buffer in FP8 for recent tokens + a larger compressed cache in MXFP4 for historical context, with learned routing between them.

### 3.5 DeepSeek-V2 Multi-Head Latent Attention (MLA)

Projects keys and values into a **lower-dimensional latent space during training**, directly reducing the cached representation size. This is the "train the architecture to need less cache" approach rather than "train the cache itself" — complementary to our approach, as MLA-trained models would benefit even more from continual encoding.

### 3.6 Dynamic Memory Compression (ICML 2024)

**Paper**: Nawrot et al., "Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference"

Learns compression policies to merge KV entries as they accumulate, with compression ratios learned per-head, per-layer. Demonstrates that different heads and layers need different cache budgets — relevant for allocating our fixed MXFP4 budget across layers.

---

## 4. Quantization Error Correction — Why Multi-Pass Works

### 4.1 GEAR — Low-Rank Residual Error Correction

**Paper**: "GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM"
**ArXiv**: https://arxiv.org/abs/2403.05527
**GitHub**: https://github.com/opengear-project/GEAR

Demonstrates that quantization error in KV caches has **exploitable structure** — exactly what the second encoding pass is finding.

- Decomposes KV matrices into three components:
  1. **Quantized backbone** (~98% of entries at 4-bit)
  2. **Low-rank matrix** approximating quantization residuals (SVD, typically rank 2-5)
  3. **Sparse correction matrix** for outlier entries (~2%)
- The low-rank residual captures **coherent error structure** shared across tokens.
- **Critical ablation**: Dropping either the low-rank or sparse component degrades GSM8k from 15.7% to ~2%.
- Uses a streaming buffer — new tokens buffered in FP16, compressed when buffer fills.

**Why this validates multi-pass**: The structured nature of quantization residuals (low-rank + sparse outliers) explains why a second pass recovers perplexity. There's *learnable signal* in the quantization loss. In our continual encoding scheme, the re-encoding step learns this residual correction **in-place** rather than storing it as a separate growing matrix.

**Key difference from our approach**: GEAR stores corrections as separate matrices (growing memory overhead). We fold the correction back into the cache representation itself.

**Cross-references**: [§4.2 SQuat](#42-squat--attention-aware-quantization), [§4.4 KVLinC](#44-kvlinc--learned-linear-correction-adapters), [§8](#8-proposed-architecture).

### 4.2 SQuat — Attention-Aware Quantization

**Paper**: "SQuat: Subspace-orthogonal KV Cache Quantization"
**ArXiv**: https://arxiv.org/abs/2503.24358 (March 2025)

Multi-pass quantization that is **aware of what attention needs**:

- Iteratively quantizes block by block, updating remaining elements at each iteration.
- Ensures quantization error is **orthogonal to the query subspace** — preserving information that actually matters for attention computation.
- Not uniform error minimization; it's *attention-score-preserving* quantization.

**Implementation relevance**: When our re-encoding step rewrites the MXFP4 cache, it should prioritize minimizing error in the directions recent queries have been attending to — i.e., the re-encoding should be query-subspace-aware, not uniformly optimized.

**Cross-references**: [§4.1 GEAR](#41-gear--low-rank-residual-error-correction), [§8](#8-proposed-architecture).

### 4.3 PM-KVQ — Progressive Mixed-Precision for Long Chain-of-Thought

**Paper**: "Progressive Mixed-Precision KV Cache Quantization for Long-CoT LLMs"
**OpenReview**: https://openreview.net/forum?id=Vem6FQvRvq (2025)

Addresses **cumulative quantization error** in long reasoning chains:

- Progressive quantization strategy: gradually lower bit-width as tokens age.
- Block-wise memory allocation assigns higher precision to more sensitive transformer blocks.
- Uses positional interpolation to calibrate for long-context distributions using short-context data.
- PM-KVQ surpasses baselines by up to 6.5% on AIME-2024 and can even exceed original 16-bit performance.

**Relevance for continual encoding**: The progressive precision idea could inform our re-encoding schedule — older information might tolerate more aggressive compression (fewer bits of the MXFP4 budget), while recent information gets priority in the fixed cache.

### 4.4 KVLinC — Learned Linear Correction Adapters

**Paper**: "KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction"
**ArXiv**: https://arxiv.org/abs/2510.05373 (October 2025)

Trains **lightweight linear adapters** to compensate for attention distribution distortion from quantized keys:

- Hadamard rotation before quantization spreads outlier energy, reducing quantization error.
- Linear correction adapters learn to fix the remaining distortion.
- Adapter memory cost is **constant** — does not grow with sequence length.
- Achieves robust 2-bit KV cache across Llama-3, Qwen-2.5, and Qwen-3 families.

**Implementation relevance**: The constant-cost adapters could serve as the "re-encoding function" in our architecture. A small learned linear transform applied during the consolidation step, with Hadamard pre-rotation for the MXFP4 quantization.

**Cross-references**: [§4.1 GEAR](#41-gear--low-rank-residual-error-correction), [§6](#6-blackwell-hardware-intrinsics--fp4-implementation) for how linear transforms map to Tensor Core ops.

### 4.5 Coupled Quantization (CQ) — Exploiting Inter-Channel Dependencies

**Paper**: "KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization"
**ArXiv**: https://arxiv.org/abs/2405.03917 (NeurIPS 2024)

Key insight: KV activation channels are **highly inter-dependent**, and joint entropy grows slower than the sum of marginal entropies. This means per-channel independent quantization is sub-optimal — coupling channels together captures more information per bit. CQ preserves model quality down to **1 bit per channel**.

**Relevance**: When designing our MXFP4 block layout, we should consider coupling correlated channels within the same scaling group to maximize information density.

---

## 5. Speculative Decoding Synergies

### 5.1 QuantSpec — Self-Speculative Decoding with 4-bit KV Cache

**Paper**: "QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache"
**ArXiv**: https://arxiv.org/abs/2502.10424 (Apple, ICML 2025)

Directly relevant to our combined approach:

- Draft model shares architecture with target but uses **hierarchical 4-bit quantized KV cache and 4-bit weights**.
- Achieves **>90% acceptance rates** with up to ~2.5× end-to-end speedup.
- Hierarchical quantization: **bit-sharing** between target and draft KV caches eliminates need for separate draft memory.
- INT8 values decomposed into upper and lower INT4 components — the draft uses INT4, the target uses the full INT8 reconstructed from both halves.
- **Double full-precision cache buffer** for most recent tokens improves acceptance and eliminates wasteful quant/dequant.

**Implementation relevance**: The hierarchical bit-sharing pattern could be adapted for MXFP4 — our continually encoded cache serves as the draft, while occasional full-precision verification happens on a small recent-token buffer.

**Key finding for our work**: "In the small batch + long context regime, attention dominates latency due to expensive load-store operations for the large KV cache. KV cache quantization provides performance improvements in this regime." — This is exactly our operating regime for agentic coding.

**Cross-references**: [§5.2 N-gram spec decoding](#52-n-gram-speculative-decoding-in-llamacpp), [§7](#7-llamacpp-integration-points).

### 5.2 N-gram Speculative Decoding in llama.cpp

**Source**: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md

llama.cpp supports several n-gram speculative decoding modes:

- **`ngram-simple`**: Looks for current n-gram in token history, drafts the following m-gram if found with sufficient hits.
- **`ngram-map-k4v`**: Hash-map based, tracks up to four m-grams per key n-gram, picks the most frequent. Variable draft lengths.
- **`ngram-mod`**: Shared hash pool across all server slots — different requests benefit from each other.
- Configurable parameters: `--spec-ngram-size-n` (lookup size, default 12), `--spec-ngram-size-m` (draft size, default 48), `--spec-ngram-min-hits` (minimum occurrences, default 1), `--draft-max` (max draft tokens, default 16).
- Ideal for: **Iterating over blocks of text/code**, reasoning models repeating thinking in final answers.

**Why >90% acceptance in our case**: Agentic coding conversations have massive token-level repetition — tool call formats, code structure, XML/JSON scaffolding, repeated function signatures, etc. The n-gram hash pool accumulates these patterns, and the MXFP4 cache stores the KV states cheaply enough to hold large context windows.

**Recommended config for agentic coding** (based on observations):
```
llama-server [...] --spec-type ngram-map-k4v \
  --spec-ngram-size-n 8 --spec-ngram-size-m 8 \
  --spec-ngram-min-hits 2 --draft-max 64 \
  --flash-attn --cache-type-k q4_0 --cache-type-v q4_0
```

**Cross-references**: [§5.1 QuantSpec](#51-quantspec--self-speculative-decoding-with-4-bit-kv-cache), [§7](#7-llamacpp-integration-points).

---

## 6. Blackwell Hardware Intrinsics & FP4 Implementation

### 6.1 Fifth-Generation Tensor Core Architecture

**Key references**:
- "Microbenchmarking NVIDIA's Blackwell Architecture" — https://arxiv.org/abs/2512.02189 (Dec 2025)
- "NVIDIA Tensor Core Evolution: From Volta to Blackwell" — SemiAnalysis (June 2025)
- CUTLASS Blackwell Tutorial — https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/

**Critical architectural changes from Hopper**:

- **New PTX opcode**: `tcgen05.mma` replaces `wgmma.mma_async` (deprecated on Blackwell). In CUTLASS this is called **UMMA**.
- **Single-thread MMA dispatch**: Unlike Hopper's warpgroup-level (128 threads), Blackwell issues MMA from a single thread. Reduces scheduler stalls by 18-23% in memory-bound kernels.
- **Tensor Memory (TMEM)**: 256KB per SM, organized as 512 columns × 128 rows of 32-bit cells. Dedicated memory for MMA accumulators — frees up register file for other work.
- **Native FP4/FP6 support**: Hardware dequantization converts FP4 to FP8 or FP16 during matrix multiplication. SASS-level instructions: OMMA (FP4), QMMA (FP8), HMMA (FP16).
- **Built-in block scaling**: tcgen05.mma natively handles micro-scaled FP4 data including grouping, dynamic scaling, and 4-bit matrix operations.
- **Instruction latency**: Nearly constant ~11 cycles across all precisions. Throughput scaling is via wider datapaths, not deeper pipelining.
- **Measured throughput**: FP4 achieves **7702.5 TFLOPS** (96.3% of peak) at 2.4 GHz. FP8 achieves 3851.4 TFLOPS.
- **CTA pair execution**: Two CTAs share operands, reducing redundant data movement.

### 6.2 MXFP4 vs NVFP4 Format Comparison

| Property | MXFP4 | NVFP4 |
|----------|-------|-------|
| Data format | E2M1 (1 sign, 2 exp, 1 mantissa) | E2M1 |
| Block size | 32 elements | 16 elements |
| Scale format | E8M0 (8-bit exponent only) | E4M3 (FP8) |
| Scale granularity | 1 scale per 32 values | 1 scale per 16 values |
| Second-level scale | None | Per-tensor FP32 |
| Effective bits/value | ~4.25 | ~4.5 |
| Accuracy | Lower (coarser scaling) | Higher (finer scaling) |

**Key tradeoff**: NVFP4's finer granularity (16 vs 32) reduces quantization error but uses more memory for scales. For a fixed-size continually-encoded cache where we're re-encoding anyway, MXFP4's coarser scaling may be acceptable because the re-encoding step compensates for quantization error.

**Source**: NVIDIA blog — https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

### 6.3 The FP4→FP8 Dequant Path for Re-Encoding

The continual re-encoding step maps to Blackwell's native datapath:

```
[MXFP4 cache in HBM]
  → TMA load to SMEM
    → tcgen05.mma dequant to FP8 (hardware-automatic)
      → Attention computation in FP8 (re-encoding pass)
        → Result in TMEM (FP32 accumulator)
          → Quantize back to MXFP4 (custom kernel)
            → Store to HBM
```

The dequant-compute-requant round trip is the core loop of our re-encoding. The key optimization is that dequantization is handled by hardware during MMA, so the only custom work is the final requantization back to MXFP4.

### 6.4 Memory Block Rearrangement for MXFP4

Tile operands should be organized as **64×64 for maximal TMEM bandwidth** (per Blackwell microbenchmark recommendations). For MXFP4 with block-of-32 scaling, this means:

- Each 64-wide tile contains 2 complete MXFP4 scaling groups.
- Scale factors can be loaded alongside data via TMA with minimal overhead.
- The re-encoding kernel should prefetch tile N+1's data via `tcgen05.cp` while computing on tile N.

**Cross-references**: [§8](#8-proposed-architecture) for full kernel architecture.

### 6.5 KV Cache Quantization Sensitivity

Empirical findings from Ollama/llama.cpp community testing:

- **K cache is more sensitive to quantization than V cache** — suggests asymmetric precision allocation (e.g., MXFP4 for V, higher precision for K, or more frequent re-encoding of K).
- Using q4_0 for V cache + FP16 for everything else is more precise than q6_K with FP16 KV.
- q8_0 for K + q4_0 for V (6.5 effective bits) is more precise than q6_K weights.
- No significant quality loss from q8_0 vs FP16 for KV cache.

**Source**: https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/

**Implication for continual encoding**: The re-encoding step could use asymmetric effort — spend more compute refining K representations than V.

---

## 7. llama.cpp Integration Points

### 7.1 Existing KV Cache Infrastructure

- KV cache types supported: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`
- Flash attention kernels support on-the-fly dequantization of Q, K, V tensors.
- CUDA flash attention uses Split-K optimization for parallelism across KV sequence dimension.
- Accumulator type controlled via template parameters and `ACC_TYPE` preprocessor defines.
- Context shifting supported — both draft and main models shift KV caches independently when context fills.

**Source**: https://deepwiki.com/ggml-org/llama.cpp/7.4-flash-attention-and-optimizations

### 7.2 MXFP4 in llama.cpp

- GPT-OSS models ship natively in MXFP4 format ("roughly equivalent to Q4_0 but get to keep full quality since trained in that format").
- KV cache quantization with MXFP4 models currently performs poorly — "performance was halved and CPU got involved."
- This suggests the current FA kernels don't have optimized paths for MXFP4 KV — opportunity for our custom kernel.

**Source**: https://github.com/ggml-org/llama.cpp/discussions/15396

### 7.3 Speculative Decoding Integration

- N-gram spec decoding operates at the server level with shared hash pools across slots.
- Draft and target models maintain independent KV caches with potentially different dtypes and context sizes.
- Context shifting compatible with speculative decoding.
- The speculative infrastructure already handles accept/reject of draft tokens and KV cache rollback — our continual encoding can hook into the accept phase to trigger re-encoding.

### 7.4 Potential Integration Architecture

```
llama-server
  ├── MXFP4 flash attention kernel (our custom kernel)
  │     ├── Standard attention path (query against MXFP4 cache)
  │     └── Re-encoding path (periodic consolidation pass)
  ├── N-gram speculative decoding (existing)
  │     └── Shared hash pool benefits from fixed-size cache
  ├── KV cache manager
  │     ├── Fixed-size MXFP4 primary cache
  │     ├── Small FP8/FP16 recent-token buffer
  │     └── Re-encoding trigger logic
  └── Context management
        └── No context shifting needed (fixed cache absorbs all context)
```

---

## 8. Proposed Architecture

### 8.1 Components

1. **Fixed-size MXFP4 KV Cache**: Pre-allocated at startup. Size determined by memory budget (e.g., 4096 slots × d_model × n_layers × 2 (K+V) × 0.5 bytes).

2. **Recent-token buffer**: Small buffer (64-128 tokens) in FP8 or FP16. New tokens land here first. Serves as the "residual buffer" from GEAR's streaming strategy.

3. **Re-encoding module**: A lightweight transformation applied periodically to consolidate the recent buffer into the fixed cache. Options (in order of increasing sophistication):
   - **(a) Linear correction**: KVLinC-style Hadamard rotation + learned linear adapter. Cheapest.
   - **(b) Attention-based consolidation**: Small auxiliary attention pass (2-4 layers) like Bottlenecked Transformer's Cache Processor.
   - **(c) Full re-quantization with SQuat-style query-subspace awareness**: Re-quantize the cache to minimize attention-score error in the directions recent queries have used.

4. **N-gram speculative decoding**: Existing llama.cpp infrastructure. Benefits from fixed-size cache (no eviction = stable n-gram history).

### 8.2 Execution Flow

```
for each new token:
  1. Compute K, V projections (FP8/BF16)
  2. Append to recent-token buffer (FP8)
  3. Compute attention: Q against [MXFP4_cache | recent_buffer]
     → Uses custom MXFP4 flash attention kernel for cache portion
     → Standard FA for recent buffer portion
     → Gated combination (à la Infini-attention)
  4. Output token

  if trigger_condition():  # e.g., buffer full, newline, tool call boundary
    5. Re-encode:
       a. Dequant MXFP4 cache → FP8 (Blackwell hardware path)
       b. Apply consolidation transform (merge recent buffer info)
       c. Re-quantize → MXFP4 (custom kernel)
       d. Clear recent buffer
    6. Update n-gram hash pool with accepted tokens
```

### 8.3 Re-Encoding Trigger Strategies

- **Buffer-full trigger**: Re-encode every N tokens (simplest, predictable latency).
- **Semantic boundary trigger**: Re-encode at newlines, tool call boundaries, code block ends (matches Bottlenecked Transformer's approach).
- **Attention-entropy trigger**: Re-encode when attention entropy over the MXFP4 cache drops (indicating the cache is becoming less useful / more stale).
- **Speculative rejection trigger**: Re-encode when n-gram spec decoding acceptance rate drops below threshold (indicating the cached context is becoming misaligned).

### 8.4 The Re-Encoding Kernel on Blackwell

Core loop pseudocode for the consolidation pass:

```
// Tile-based re-encoding on Blackwell Tensor Cores
// Uses tcgen05.mma with MXFP4 inputs, FP8 compute, FP32 accumulate

for each layer L:
  for each head H:
    // 1. Load MXFP4 cache tile + recent buffer tile into SMEM via TMA
    tcgen05.cp(smem_cache, hbm_cache[L][H], tile_desc)
    tcgen05.cp(smem_recent, hbm_recent[L][H], tile_desc)

    // 2. Compute consolidation attention (recent queries attend to full cache)
    // Hardware auto-dequants MXFP4 → FP8 during MMA
    tcgen05.mma(tmem_attn_scores, smem_recent_Q, smem_cache_K)

    // 3. Apply learned correction (linear adapter or small attention)
    // This is the "multi-pass" step — refining the cache representation
    tcgen05.mma(tmem_updated_V, tmem_attn_scores, smem_cache_V)

    // 4. Merge updated representations back
    // Blend old cache with new information using learned gating
    apply_gating(tmem_updated_V, smem_cache_V, gating_params[L][H])

    // 5. Re-quantize to MXFP4 and write back
    quantize_mxfp4(hbm_cache[L][H], tmem_updated_V, scale_factors)
```

### 8.5 Memory Budget Example

For a 7B model (32 layers, 32 heads, d_head=128):
- **MXFP4 cache (4096 slots)**: 32 × 32 × 128 × 4096 × 2 (K+V) × 0.5 bytes = **512 MB**
- **FP8 recent buffer (128 slots)**: 32 × 32 × 128 × 128 × 2 × 1 byte = **32 MB**
- **FP16 baseline (4096 slots)**: 32 × 32 × 128 × 4096 × 2 × 2 bytes = **2 GB**
- **Savings**: 4× memory reduction, enabling 4× longer context or 4× more concurrent requests.

---

## 9. Open Questions & Experiments

### 9.1 Immediate Experiments (Python prototyping with Opus)

1. **Multi-pass perplexity sweep**: Vary number of re-encoding passes (1, 2, 4, 8) and measure perplexity recovery curve. Find the diminishing returns point.

2. **Re-encoding frequency**: Fix cache size, vary how often re-encoding happens (every 32, 64, 128, 256 tokens). Measure perplexity vs. latency tradeoff.

3. **Asymmetric K/V treatment**: Re-encode K cache more frequently than V cache (given K's higher quantization sensitivity). Measure quality impact.

4. **Query-subspace-aware re-encoding**: During re-encoding, weight the optimization toward the attention subspace used by recent queries (SQuat-style). Compare against uniform re-encoding.

5. **Information retained vs. discarded**: Measure what percentage of the original FP16 KV cache's attention-score variance is captured by the MXFP4 cache after 1, 2, N re-encoding passes.

### 9.2 Kernel Development Experiments

6. **MXFP4 dequant-compute-requant round-trip latency**: Benchmark the full cycle on Blackwell. Compare against simply growing the cache in FP8.

7. **TMEM utilization during re-encoding**: Profile whether the consolidation pass is compute-bound or memory-bound on Blackwell.

8. **Block scaling group alignment**: Test whether aligning MXFP4 scaling groups to attention head boundaries improves quality (ensures each head's scale is independent).

### 9.3 Integration Experiments

9. **N-gram acceptance rate with fixed vs. growing cache**: Compare speculative acceptance rates when using a fixed MXFP4 cache (no eviction, stable history) vs. standard growing+shifting cache.

10. **Agentic coding benchmark**: Run an agentic coding task (e.g., SWE-bench) with the continual encoding cache + n-gram spec decoding. Measure wall-clock time vs. baseline.

---

## 10. Full Reference Index

### Theoretical Foundations
| ID | Paper | ArXiv/URL | Year | Key Relevance |
|----|-------|-----------|------|---------------|
| T1 | Bottlenecked Transformers: Periodic KV Cache Consolidation | https://arxiv.org/abs/2505.16950 | 2025 | IB theory justification for KV cache rewrites |
| T2 | Limits of KV Cache Compression for Tensor Attention | https://arxiv.org/abs/2503.11108 | 2025 | Information-theoretic lower bounds on compression |

### Fixed-Size Memory Architectures
| ID | Paper | ArXiv/URL | Year | Key Relevance |
|----|-------|-----------|------|---------------|
| M1 | Infini-attention (Google) | https://arxiv.org/abs/2404.07143 | 2024 | Fixed-size compressive memory with gating |
| M2 | Recurrent Memory Transformer | https://arxiv.org/abs/2207.06881 | 2022 | Learned fixed-size memory tokens, 2M token scaling |
| M3 | Scaling RMT to 1M tokens | https://arxiv.org/abs/2304.11062 | 2024 | Curriculum learning for memory training |
| M4 | Titans: Learning to Memorize at Test Time | (Google, 2025) | 2025 | Test-time memory learning |
| M5 | Dynamic Memory Compression | (Nawrot et al., ICML 2024) | 2024 | Learned per-head compression ratios |
| M6 | DeepSeek-V2 MLA | (DeepSeek-AI, 2024) | 2024 | Latent-space KV projection |

### Quantization Error Correction
| ID | Paper | ArXiv/URL | Year | Key Relevance |
|----|-------|-----------|------|---------------|
| Q1 | GEAR | https://arxiv.org/abs/2403.05527 | 2024 | Low-rank residual + sparse outlier correction |
| Q2 | SQuat | https://arxiv.org/abs/2503.24358 | 2025 | Query-subspace-orthogonal multi-pass quantization |
| Q3 | PM-KVQ | https://openreview.net/forum?id=Vem6FQvRvq | 2025 | Progressive mixed-precision for long CoT |
| Q4 | KVLinC | https://arxiv.org/abs/2510.05373 | 2025 | Learned linear correction adapters, Hadamard rotation |
| Q5 | Coupled Quantization (CQ) | https://arxiv.org/abs/2405.03917 | 2024 | Inter-channel dependency exploitation, 1-bit KV |
| Q6 | KVQuant | https://github.com/SqueezeAILab/KVQuant | 2024 | Per-channel pre-RoPE key quantization, NUQ, dense+sparse |
| Q7 | KIVI | (Liu et al., 2024) | 2024 | Tuning-free asymmetric 2-bit KV quantization |

### Speculative Decoding
| ID | Paper | ArXiv/URL | Year | Key Relevance |
|----|-------|-----------|------|---------------|
| S1 | QuantSpec (Apple) | https://arxiv.org/abs/2502.10424 | 2025 | Hierarchical 4-bit KV cache for self-spec decoding, >90% acceptance |
| S2 | llama.cpp speculative docs | https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md | 2025 | N-gram spec decoding implementation details |
| S3 | LongSpec | https://arxiv.org/abs/2502.17421 | 2025 | Long-context lossless speculative decoding |

### KV Cache Compression Surveys & Benchmarks
| ID | Paper | URL | Year | Key Relevance |
|----|-------|-----|------|---------------|
| C1 | ChunkKV | https://openreview.net/forum?id=20JDhbJqn3 | 2025 | Semantic chunk-level compression, NeurIPS 2025 |
| C2 | Expected Attention | https://arxiv.org/abs/2510.00636 | 2025 | Future-query-aware KV pair importance scoring |
| C3 | MorphKV | (Ghadia et al., 2025) | 2025 | Adaptive fixed-size cache via attention pattern refinement |
| C4 | KV Cache Optimization survey | https://www.emergentmind.com/topics/kv-cache-optimization | 2025 | Comprehensive overview of compression techniques |
| C5 | Awesome KV Cache Compression | https://github.com/October2001/Awesome-KV-Cache-Compression | 2025 | Curated paper list, continuously updated |

### Blackwell Hardware
| ID | Reference | URL | Year | Key Relevance |
|----|-----------|-----|------|---------------|
| H1 | Blackwell Microbenchmarks (Jarmusch) | https://arxiv.org/abs/2512.02189 | 2025 | FP4 throughput/latency characterization, TMEM analysis |
| H2 | Tensor Core Evolution (SemiAnalysis) | SemiAnalysis newsletter | 2025 | tcgen05.mma, MXFP4/NVFP4 hardware details |
| H3 | CUTLASS Blackwell GEMM Tutorial | https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/ | 2025 | UMMA atom, TMEM layout, tile programming |
| H4 | NVFP4 KV Cache blog (NVIDIA) | https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/ | 2025 | NVFP4 KV quantization, dequant→FP8 pipeline |
| H5 | NVFP4 format introduction (NVIDIA) | https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ | 2025 | MXFP4 vs NVFP4 comparison, block scaling details |
| H6 | Blackwell NVFP4 inference benchmarks | https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/ | 2025 | RTX PRO 6000 FP4 inference benchmarks |
| H7 | Modular: Matrix Multiplication on Blackwell | https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction | 2026 | TMEM pipelining, tcgen05 vs wgmma |
| H8 | FP4 Tensor Cores survey | https://www.emergentmind.com/topics/fp4-tensor-cores | 2025 | Comprehensive FP4 hardware/software overview |

### llama.cpp Integration
| ID | Reference | URL | Key Relevance |
|----|-----------|-----|---------------|
| L1 | Flash Attention implementation | https://deepwiki.com/ggml-org/llama.cpp/7.4-flash-attention-and-optimizations | FA kernel architecture, quant dequant paths |
| L2 | KV cache quantization in Ollama | https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/ | K vs V sensitivity, empirical quality data |
| L3 | GPT-OSS MXFP4 guide | https://github.com/ggml-org/llama.cpp/discussions/15396 | MXFP4 model support, current limitations |
| L4 | Speculative decoding discussion | https://github.com/ggml-org/llama.cpp/discussions/10466 | Spec decoding + quantized KV interactions |
| L5 | Server README (spec config) | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md | Full server parameter reference |

