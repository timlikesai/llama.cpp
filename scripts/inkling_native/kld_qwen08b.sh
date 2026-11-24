#!/usr/bin/env bash
# Brief AA job 2: isolated MXFP8 PPL/KLD table on Qwen3.5-0.8B, base = the bf16 source.
# Flag pattern from ppl_kl_u2.sh: -b must equal -c (perplexity.cpp:2012-2014 derives
# n_parallel = n_batch/n_ctx and multiplies back), corpus must hold >= 2x ctx tokens.
set -u
R=/home/tim/code/services/repos/llama.cpp
B=$R/build-cuda/bin
M=/home/tim/models/timlikesai
SRC=/home/tim/.cache/huggingface/hub/models--ggml-org--Qwen3.5-0.8B-GGUF/snapshots/004fd086eccea9f7cf867aa07ba328773beef81d/Qwen3.5-0.8B-BF16.gguf
C=$R/logs/corpora/wiki.test.slice64k.raw
L=$R/logs
BASE=$L/corpora/aa-kld-bf16.bin
export LLAMA_LOG_COLORS=0
COMMON=(-f "$C" -c 2048 -b 2048 -ub 512 -s 42)

echo "########## A0 base: bf16 logits -> $BASE ##########"
"$B/llama-perplexity" -m "$SRC" "${COMMON[@]}" --kl-divergence-base "$BASE" 2>&1
echo "A0_DONE"

for pair in "BF16-SELF:$SRC" "Q8_0:$M/Qwen3.5-0.8B-Q8_0.gguf" \
            "MXFP8-CUDA:$M/Qwen3.5-0.8B-MXFP8-PY.gguf" "MXFP4:$M/Qwen3.5-0.8B-MXFP4.gguf"; do
  tag=${pair%%:*}; model=${pair#*:}
  echo "########## A1 $tag vs bf16 base ##########"
  "$B/llama-perplexity" -m "$model" "${COMMON[@]}" --kl-divergence --kl-divergence-base "$BASE" 2>&1
  echo "A1_${tag}_DONE"
done

echo "########## A2 MXFP8 on CPU (-ngl 0) vs the same bf16 base ##########"
"$B/llama-perplexity" -m "$M/Qwen3.5-0.8B-MXFP8-PY.gguf" "${COMMON[@]}" -ngl 0 \
  --kl-divergence --kl-divergence-base "$BASE" 2>&1
echo "A2_DONE"

echo "########## A3 q8_0 on CPU (-ngl 0), CPU/CPU control ##########"
"$B/llama-perplexity" -m "$M/Qwen3.5-0.8B-Q8_0.gguf" "${COMMON[@]}" -ngl 0 \
  --kl-divergence --kl-divergence-base "$BASE" 2>&1
echo "A3_DONE"
echo "ALL_A_DONE"
