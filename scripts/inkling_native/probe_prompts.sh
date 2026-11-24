#!/bin/bash
# Run the fixed 4-prompt suite against one model/flag combination and print only the answers.
# Companion of probe_device_matrix.sh; usage: probe_prompts.sh <tag> <model.gguf> [extra llama-completion flags...]
set -u

BIN=$PWD/build-cuda/bin/llama-completion
TAG=$1
MODEL=$2
shift 2

PROMPTS=(
  "Q: What is 17 times 4? A:"
  "The capital of France is"
  "List three primary colors:"
  "def fibonacci(n):"
)

for p in "${PROMPTS[@]}"; do
    echo "### ${TAG} | ${p}"
    "$BIN" -m "$MODEL" "$@" -c 4096 -n 24 --temp 0 -no-cnv --seed 42 -p "$p" 2>&1
    echo "### ${TAG} exit=$?"
done
echo "PROBE_DONE ${TAG}"
