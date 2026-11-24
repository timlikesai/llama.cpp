#!/usr/bin/env bash
# Brief U (Phase 5): PPL + KL of the lossless NATIVE file vs the dequantized reference.
# Existing binaries only - no builds. Never pin -ngl (auto-fit only).
set -u
R=/home/tim/code/services/repos/llama.cpp
B=$R/build-cuda/bin
M=/home/tim/models/timlikesai
G=/home/tim/models/ggml-org/Inkling-2-Mini-GGUF
C=$R/logs/corpora/wiki.test.slice64k.raw
L=$R/logs
KLD=$L/corpora/u-kld-deq-q80.bin
export LLAMA_LOG_COLORS=0

# -b must equal -c: perplexity.cpp:2012 sets n_parallel = n_batch / n_ctx, and a
# 4-slot parallel context with -c 512 overflows the KV slot (init_batch: failed to
# prepare attention base ubatches). --ppl-stride also destabilised n_ctx, so it is
# left at the default (n_ctx/2) as upstream does.
COMMON=(-f "$C" -c 2048 -b 2048 -ub 512 -s 42)

echo "########## U1 base: Inkling-DEQ-Q80 (logits + base PPL) ##########"
date
"$B/llama-perplexity" -m "$M/Inkling-DEQ-Q80.gguf" "${COMMON[@]}" \
  --kl-divergence-base "$KLD" 2>&1
echo "U1_DONE exit=$?"

for pair in "NATIVE:$M/Inkling-2-Mini-NATIVE.gguf" \
            "SCALE1:$M/Inkling-2-Mini-NATIVE-scale1.gguf" \
            "IQ4NL:$G/Inkling-2-Mini-IQ4_NL.gguf"; do
  tag=${pair%%:*}; model=${pair#*:}
  echo "########## U2 $tag: KL vs DEQ-Q80 ##########"
  date
  "$B/llama-perplexity" -m "$model" "${COMMON[@]}" \
    --kl-divergence --kl-divergence-base "$KLD" 2>&1
  echo "U2_${tag}_DONE exit=$?"
done

echo "ALL_U_DONE"
