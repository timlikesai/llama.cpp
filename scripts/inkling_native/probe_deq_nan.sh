#!/usr/bin/env bash
# Brief V: localise the DEQ-Q80 non-finite logits. The file's stored values are all finite
# (scan_gguf_finite.py), so the question is which runtime path turns them into nan.
# Short probe corpus, -b always equals -c (perplexity.cpp derives n_parallel = n_batch/n_ctx).
# Never pin a nonzero -ngl: it disables memory fitting on this 2-card box.
set -u
R=/home/tim/code/services/repos/llama.cpp
B=$R/build-cuda/bin
M=/home/tim/models/timlikesai
C=$R/logs/corpora/deq-probe.raw
L=$R/logs
export LLAMA_LOG_COLORS=0
FLAGS=(-f "$C" -c 1024 -b 1024 -ub 512 -s 42)

echo "## binary under test"
stat -c '%y %s %n' "$B/llama-perplexity" "$B"/libggml-cuda.so "$B"/libggml-cpu.so
sha256sum "$B/libggml-cuda.so" "$B/libggml-cpu.so"
echo "## corpus"; wc -c "$C"

run() { # tag model extra...
  local tag=$1 model=$2; shift 2
  echo "########## $tag ##########"
  "$B/llama-perplexity" -m "$model" "${FLAGS[@]}" "$@" 2>&1
  echo "${tag}_DONE exit=$?"
}

run P0-native-gpu    "$M/Inkling-2-Mini-NATIVE.gguf"
run P1-deq-gpu       "$M/Inkling-DEQ-Q80.gguf"
run P2-deq-cpu       "$M/Inkling-DEQ-Q80.gguf" -ngl 0
run P3-native-cpu    "$M/Inkling-2-Mini-NATIVE.gguf" -ngl 0
run P4-deq-exp-cpu   "$M/Inkling-DEQ-Q80.gguf" -ot 'exps\.weight=CPU'
echo ALL_PROBE_DONE
