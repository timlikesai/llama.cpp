#!/bin/bash
# Localize the CPU/CUDA divergence of the bit-exact NATIVE Inkling file by moving one
# tensor group at a time to the other backend and watching which move flips the answer.
# Every line of output is kept: tee to logs/m2-device-matrix.log.
set -u

BIN=$PWD/build-cuda/bin/llama-completion
NATIVE=$HOME/models/timlikesai/Inkling-2-Mini-NATIVE.gguf
IQ4=$HOME/models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf

# arithmetic, completion, list, code
PROMPTS=(
  "Q: What is 17 times 4? A:"
  "The capital of France is"
  "List three primary colors:"
  "def fibonacci(n):"
)

run() {
    local tag=$1 model=$2
    shift 2
    for p in "${PROMPTS[@]}"; do
        echo "### ${tag} | ${p}"
        "$BIN" -m "$model" "$@" -c 4096 -n 24 --temp 0 -no-cnv --seed 42 -p "$p" 2>&1
        echo "### ${tag} exit=$?"
    done
}

# MXFP8 expert layers are blk.20-23, NVFP4 expert layers are blk.0-19
MXFP8_ON_CPU='blk\.2[0-3]\.ffn_.*exps=CPU'
NVFP4_ON_CPU='blk\.1[0-9]\.ffn_.*exps=CPU'
NVFP4_ON_CPU2='blk\.[0-9]\.ffn_.*exps=CPU'

run native-cpu      "$NATIVE" -ngl 0
run native-gpu      "$NATIVE"
run native-gpu-moe-on-cpu "$NATIVE" -cmoe
run native-gpu-mxfp8-on-cpu "$NATIVE" -ot "$MXFP8_ON_CPU"
run native-gpu-nvfp4-on-cpu "$NATIVE" -ot "$NVFP4_ON_CPU" -ot "$NVFP4_ON_CPU2"
run iq4nl-cpu       "$IQ4" -ngl 0
run iq4nl-gpu       "$IQ4"
echo "MATRIX_DONE"
