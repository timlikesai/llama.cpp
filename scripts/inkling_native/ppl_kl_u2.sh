#!/usr/bin/env bash
# Brief U follow-up: the DEQ-Q80 KL base came out nan, so rebuild the base from the
# NATIVE file and measure the other files against it. Also confirm the DEQ nan in the
# plain (v1) perplexity path, independent of the KL code path.
set -u
R=/home/tim/code/services/repos/llama.cpp
B=$R/build-cuda/bin
M=/home/tim/models/timlikesai
G=/home/tim/models/ggml-org/Inkling-2-Mini-GGUF
C=$R/logs/corpora/wiki.test.slice64k.raw
L=$R/logs
KLD=$L/corpora/u-kld-native.bin
export LLAMA_LOG_COLORS=0
COMMON=(-f "$C" -c 2048 -b 2048 -ub 512 -s 42)

echo "########## V0: DEQ-Q80 plain PPL (no KL path) ##########"
"$B/llama-perplexity" -m "$M/Inkling-DEQ-Q80.gguf" "${COMMON[@]}" 2>&1
echo "V0_DONE"

echo "########## V1 base: NATIVE logits ##########"
"$B/llama-perplexity" -m "$M/Inkling-2-Mini-NATIVE.gguf" "${COMMON[@]}" \
  --kl-divergence-base "$KLD" 2>&1
echo "V1_DONE"

for pair in "NATIVE-SELF:$M/Inkling-2-Mini-NATIVE.gguf" \
            "IQ4NL:$G/Inkling-2-Mini-IQ4_NL.gguf" \
            "SCALE1:$M/Inkling-2-Mini-NATIVE-scale1.gguf" \
            "DEQ-Q80:$M/Inkling-DEQ-Q80.gguf"; do
  tag=${pair%%:*}; model=${pair#*:}
  echo "########## V2 $tag vs NATIVE base ##########"
  "$B/llama-perplexity" -m "$model" "${COMMON[@]}" \
    --kl-divergence --kl-divergence-base "$KLD" 2>&1
  echo "V2_${tag}_DONE"
done
echo "ALL_V_DONE"
