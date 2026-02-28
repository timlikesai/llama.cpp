#!/bin/bash
#
# MXFP4 Flash Attention Test Runner
#
# Wraps docker compose build + perplexity + benchmark for the MXFP4 flash attention kernel.
# All tests run inside the llama-cpp-ultra container with CUDA 13.1 / sm_120a.
#
# Usage:
#   ./mxfp4-test.sh                  # Full suite: build + perplexity + bench
#   ./mxfp4-test.sh --skip-build     # Skip Docker build
#   ./mxfp4-test.sh --quick          # 2-chunk perplexity only (fast sanity check)
#   ./mxfp4-test.sh --bench-only     # Performance benchmarks only
#   ./mxfp4-test.sh --chunks 59      # Custom chunk count for perplexity
#   ./mxfp4-test.sh --config mxfp4   # Single config (mxfp4, f16, q4_0)
#   ./mxfp4-test.sh --determinism    # Run determinism check (3x identical runs)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASETS_DIR="$SCRIPT_DIR/datasets"
DATASET_FILE="wikitext-2-raw/wiki.test.raw"
MODEL_PATH="/models/spectralyst/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE-GGUF/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE.gguf"
RESULTS_DIR="$SCRIPT_DIR/test-results"

# Defaults
DO_BUILD=true
DO_PERPLEXITY=true
DO_BENCH=true
DO_DETERMINISM=false
CHUNKS=59
CONFIGS=("mxfp4" "f16" "q4_0")

# ── Argument Parsing ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            DO_BUILD=false
            shift
            ;;
        --quick)
            CHUNKS=2
            DO_BENCH=false
            shift
            ;;
        --bench-only)
            DO_PERPLEXITY=false
            DO_BUILD=false
            shift
            ;;
        --determinism)
            DO_DETERMINISM=true
            DO_PERPLEXITY=false
            DO_BENCH=false
            shift
            ;;
        --chunks)
            CHUNKS="$2"
            shift 2
            ;;
        --config)
            CONFIGS=("$2")
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-build     Skip Docker build step"
            echo "  --quick          2-chunk perplexity only (fast sanity check)"
            echo "  --bench-only     Performance benchmarks only (no build, no perplexity)"
            echo "  --determinism    Run determinism check (3x identical runs)"
            echo "  --chunks N       Set chunk count for perplexity (default: 59)"
            echo "  --config NAME    Run single config: mxfp4, f16, q4_0 (default: all)"
            echo "  --help           Show this help"
            echo ""
            echo "Configs:"
            echo "  mxfp4   MXFP4 K+V with flash attention and Hadamard rotation"
            echo "  f16     F16 K+V with flash attention (baseline)"
            echo "  q4_0    Q4_0 K+V with flash attention (comparison)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run $0 --help for usage."
            exit 1
            ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "  $(timestamp)"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
}

# Build the common docker compose run prefix.
# Usage: docker_run <entrypoint_args...>
docker_run() {
    cd "$SERVICES_DIR"
    docker compose run \
        --rm \
        --entrypoint "" \
        --volume "$DATASETS_DIR:/datasets:ro" \
        llama-cpp-ultra \
        "$@"
}

# Run llama-perplexity with given cache types and chunk count.
# Usage: run_perplexity <cache_type_k> <cache_type_v> <chunks>
run_perplexity() {
    local type_k="$1"
    local type_v="$2"
    local chunks="$3"

    local args=(
        /app/llama-perplexity
        --model "$MODEL_PATH"
        --file "/datasets/$DATASET_FILE"
        --chunks "$chunks"
        --flash-attn on
        --gpu-layers 99
    )

    # Only pass cache-type flags for non-default types
    if [[ "$type_k" != "f16" ]]; then
        args+=(--cache-type-k "$type_k")
    fi
    if [[ "$type_v" != "f16" ]]; then
        args+=(--cache-type-v "$type_v")
    fi

    docker_run "${args[@]}"
}

# Run llama-bench with given cache types.
# Usage: run_bench <cache_type_k> <cache_type_v> <test_spec>
run_bench() {
    local type_k="$1"
    local type_v="$2"
    local test_spec="$3"

    local args=(
        /app/llama-bench
        --model "$MODEL_PATH"
        --flash-attn 1
        --gpu-layers 999
        --test "$test_spec"
    )

    if [[ "$type_k" != "f16" ]]; then
        args+=(--cache-type-k "$type_k")
    fi
    if [[ "$type_v" != "f16" ]]; then
        args+=(--cache-type-v "$type_v")
    fi

    docker_run "${args[@]}"
}

# Map config name to K/V cache types.
# Usage: config_types <config_name>  →  sets TYPE_K, TYPE_V
config_types() {
    case "$1" in
        mxfp4)
            TYPE_K="mxfp4"
            TYPE_V="mxfp4"
            ;;
        f16)
            TYPE_K="f16"
            TYPE_V="f16"
            ;;
        q4_0)
            TYPE_K="q4_0"
            TYPE_V="q4_0"
            ;;
        *)
            echo "Unknown config: $1"
            exit 1
            ;;
    esac
}

# ── Preflight ────────────────────────────────────────────────────────────────

if [[ ! -d "$DATASETS_DIR" ]]; then
    echo "ERROR: Datasets directory not found: $DATASETS_DIR"
    echo "Download wikitext-2-raw to $DATASETS_DIR/wikitext-2-raw/"
    exit 1
fi

if [[ ! -f "$DATASETS_DIR/$DATASET_FILE" ]]; then
    echo "ERROR: Dataset file not found: $DATASETS_DIR/$DATASET_FILE"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ── Build ────────────────────────────────────────────────────────────────────

if $DO_BUILD; then
    header "BUILD: docker compose build llama-cpp-ultra"
    cd "$SERVICES_DIR"
    docker compose build llama-cpp-ultra
    echo ""
    echo "Build completed successfully."
fi

# ── Determinism Check ────────────────────────────────────────────────────────

if $DO_DETERMINISM; then
    header "DETERMINISM: 3x identical runs (2 chunks, mxfp4)"

    results=()
    for i in 1 2 3; do
        echo "--- Run $i/3 ---"
        output=$(run_perplexity mxfp4 mxfp4 2 2>&1)
        ppl=$(echo "$output" | grep "^Final estimate: PPL" | grep --only-matching --perl-regexp '[\d.]+' | head --lines 1)
        echo "  PPL = $ppl"
        results+=("$ppl")
    done

    echo ""
    if [[ "${results[0]}" == "${results[1]}" && "${results[1]}" == "${results[2]}" ]]; then
        echo "PASS: All 3 runs identical (PPL = ${results[0]})"
    else
        echo "FAIL: Results differ: ${results[0]}, ${results[1]}, ${results[2]}"
        exit 1
    fi
fi

# ── Perplexity ───────────────────────────────────────────────────────────────

if $DO_PERPLEXITY; then
    header "PERPLEXITY: wikitext-2-raw, $CHUNKS chunks"

    results_file="$RESULTS_DIR/perplexity-$(date +%Y%m%d-%H%M%S).txt"
    echo "# MXFP4 Flash Attention Perplexity Results" > "$results_file"
    echo "# Date: $(timestamp)" >> "$results_file"
    echo "# Chunks: $CHUNKS" >> "$results_file"
    echo "# Model: $MODEL_PATH" >> "$results_file"
    echo "" >> "$results_file"

    for config in "${CONFIGS[@]}"; do
        config_types "$config"
        echo "--- Config: $config (K=$TYPE_K, V=$TYPE_V) ---"

        output=$(run_perplexity "$TYPE_K" "$TYPE_V" "$CHUNKS" 2>&1)
        ppl=$(echo "$output" | grep "^Final estimate: PPL" | grep --only-matching --perl-regexp '[\d.]+' | head --lines 1)

        echo "  Final PPL ($CHUNKS chunks): $ppl"
        echo "$config: PPL = $ppl ($CHUNKS chunks, K=$TYPE_K, V=$TYPE_V)" >> "$results_file"
        echo ""
    done

    echo "Results saved to: $results_file"
    cat "$results_file"
fi

# ── Benchmarks ───────────────────────────────────────────────────────────────

if $DO_BENCH; then
    header "BENCHMARKS: pp512 + tg128"

    results_file="$RESULTS_DIR/bench-$(date +%Y%m%d-%H%M%S).txt"
    echo "# MXFP4 Flash Attention Benchmark Results" > "$results_file"
    echo "# Date: $(timestamp)" >> "$results_file"
    echo "" >> "$results_file"

    for config in "${CONFIGS[@]}"; do
        config_types "$config"
        echo "--- Config: $config (K=$TYPE_K, V=$TYPE_V) ---"

        output=$(run_bench "$TYPE_K" "$TYPE_V" "pp512,tg128" 2>&1)
        echo "$output"
        echo "" >> "$results_file"
        echo "=== $config (K=$TYPE_K, V=$TYPE_V) ===" >> "$results_file"
        echo "$output" >> "$results_file"
        echo ""
    done

    echo "Results saved to: $results_file"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

header "COMPLETE"
echo "Results directory: $results_file"
echo "Configs tested: ${CONFIGS[*]}"
