#!/bin/bash
#
# KV Cache Quantization Benchmark Comparison
#
# Runs perplexity, throughput (pp512 + tg128), and KV cache memory analysis
# across multiple quantization types for side-by-side comparison.
#
# Usage:
#   ./kv-bench.sh                     # Full suite: build + perplexity + bench (all configs)
#   ./kv-bench.sh --skip-build        # Skip Docker build
#   ./kv-bench.sh --skip-perplexity   # Skip perplexity (no PPL or memory data)
#   ./kv-bench.sh --skip-bench        # Skip throughput benchmarks
#   ./kv-bench.sh --chunks 2          # Quick perplexity (2 chunks)
#   ./kv-bench.sh --config mxfp4      # Single config only
#   ./kv-bench.sh --help              # Show usage

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASETS_DIR="$SCRIPT_DIR/datasets"
DATASET_FILE="wikitext-2-raw/wiki.test.raw"
MODEL_PATH="/models/spectralyst/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE-GGUF/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE.gguf"

# Defaults
DO_BUILD=true
DO_PERPLEXITY=true
DO_BENCH=true
CHUNKS=59
CONFIGS=("f16" "q8_0" "q4_0" "mxfp4")

# ── Argument Parsing ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build)
            DO_BUILD=false
            shift
            ;;
        --skip-perplexity)
            DO_PERPLEXITY=false
            shift
            ;;
        --skip-bench)
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
            echo "KV Cache Quantization Benchmark Comparison"
            echo ""
            echo "Options:"
            echo "  --skip-build        Skip Docker build step"
            echo "  --skip-perplexity   Skip perplexity runs (no PPL or memory data)"
            echo "  --skip-bench        Skip throughput benchmarks"
            echo "  --chunks N          Chunk count for perplexity (default: 59)"
            echo "  --config NAME       Single config: f16, q8_0, q4_0, mxfp4 (default: all)"
            echo "  --help              Show this help"
            echo ""
            echo "Configs:"
            echo "  f16     F16 K+V with flash attention (baseline)"
            echo "  q8_0    Q8_0 K+V with flash attention"
            echo "  q4_0    Q4_0 K+V with flash attention"
            echo "  mxfp4   MXFP4 K+V with flash attention and Hadamard rotation"
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
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "  $(timestamp)"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
}

# Run a command inside the llama-cpp-ultra container.
docker_run() {
    cd "$SERVICES_DIR"
    docker compose run \
        --rm \
        --entrypoint "" \
        --volume "$DATASETS_DIR:/datasets:ro" \
        llama-cpp-ultra \
        "$@"
}

# Map config name to K/V cache types → sets TYPE_K, TYPE_V.
config_types() {
    case "$1" in
        f16)   TYPE_K="f16";   TYPE_V="f16"   ;;
        q8_0)  TYPE_K="q8_0";  TYPE_V="q8_0"  ;;
        q4_0)  TYPE_K="q4_0";  TYPE_V="q4_0"  ;;
        mxfp4) TYPE_K="mxfp4"; TYPE_V="mxfp4" ;;
        *)
            echo "Unknown config: $1"
            exit 1
            ;;
    esac
}

# Build cache-type flags for llama-perplexity (--cache-type-k/v, skip if f16).
cache_flags_perplexity() {
    local flags=()
    if [[ "$TYPE_K" != "f16" ]]; then
        flags+=(--cache-type-k "$TYPE_K")
    fi
    if [[ "$TYPE_V" != "f16" ]]; then
        flags+=(--cache-type-v "$TYPE_V")
    fi
    echo "${flags[@]+"${flags[@]}"}"
}

# Build cache-type flags for llama-bench (-ctk/-ctv, skip if f16).
cache_flags_bench() {
    local flags=()
    if [[ "$TYPE_K" != "f16" ]]; then
        flags+=(-ctk "$TYPE_K")
    fi
    if [[ "$TYPE_V" != "f16" ]]; then
        flags+=(-ctv "$TYPE_V")
    fi
    echo "${flags[@]+"${flags[@]}"}"
}

# ── Preflight ────────────────────────────────────────────────────────────────

if $DO_PERPLEXITY; then
    if [[ ! -d "$DATASETS_DIR" ]]; then
        echo "ERROR: Datasets directory not found: $DATASETS_DIR"
        echo "Download wikitext-2-raw to $DATASETS_DIR/wikitext-2-raw/"
        exit 1
    fi
    if [[ ! -f "$DATASETS_DIR/$DATASET_FILE" ]]; then
        echo "ERROR: Dataset file not found: $DATASETS_DIR/$DATASET_FILE"
        exit 1
    fi
fi

RESULTS_DIR="$SCRIPT_DIR/test-results/kv-bench-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ── Build ────────────────────────────────────────────────────────────────────

if $DO_BUILD; then
    header "BUILD: docker compose build llama-cpp-ultra"
    cd "$SERVICES_DIR"
    docker compose build llama-cpp-ultra
    echo ""
    echo "Build completed successfully."
fi

# ── Result Storage ───────────────────────────────────────────────────────────

# Associative arrays for parsed results.
declare -A RESULT_PPL
declare -A RESULT_K_MIB
declare -A RESULT_V_MIB
declare -A RESULT_CELLS
declare -A RESULT_PP512
declare -A RESULT_TG128

# ── Run Tests ────────────────────────────────────────────────────────────────

for config in "${CONFIGS[@]}"; do
    config_types "$config"

    # ── Perplexity ───────────────────────────────────────────────────────
    if $DO_PERPLEXITY; then
        header "PERPLEXITY: $config (K=$TYPE_K, V=$TYPE_V) — $CHUNKS chunks"

        local_log="$RESULTS_DIR/${config}-perplexity.log"

        # Run perplexity, capture all output (stdout + stderr).
        docker_run \
            /app/llama-perplexity \
            --model "$MODEL_PATH" \
            --file "/datasets/$DATASET_FILE" \
            --chunks "$CHUNKS" \
            --flash-attn on \
            --gpu-layers 99 \
            $(cache_flags_perplexity) \
            2>&1 | tee "$local_log"

        # Parse PPL from "Final estimate: PPL = X.XXXX".
        ppl=$(grep --perl-regexp --only-matching 'Final estimate: PPL = \K[\d.]+' "$local_log" || echo "N/A")
        RESULT_PPL[$config]="$ppl"

        # Parse KV cache memory from llama_kv_cache log line.
        # Format: "llama_kv_cache: size =  192.00 MiB (   512 cells,  48 layers, ...), K (f16):   96.00 MiB, V (f16):   96.00 MiB"
        kv_line=$(grep 'llama_kv_cache: size' "$local_log" | tail --lines 1 || echo "")
        if [[ -n "$kv_line" ]]; then
            cells_per_seq=$(echo "$kv_line" | grep --perl-regexp --only-matching '\(\s*\K\d+(?=\s+cells)')
            n_seq=$(echo "$kv_line" | grep --perl-regexp --only-matching '\d+(?=/\d+ seqs)')
            RESULT_CELLS[$config]=$(( cells_per_seq * n_seq ))
            RESULT_K_MIB[$config]=$(echo "$kv_line" | grep --perl-regexp --only-matching 'K \([^)]+\):\s*\K[\d.]+')
            RESULT_V_MIB[$config]=$(echo "$kv_line" | grep --perl-regexp --only-matching 'V \([^)]+\):\s*\K[\d.]+')
        fi

        echo ""
        echo "  PPL ($CHUNKS chunks): $ppl"
        echo "  KV cache: K=${RESULT_K_MIB[$config]:-?} MiB, V=${RESULT_V_MIB[$config]:-?} MiB, cells=${RESULT_CELLS[$config]:-?}"
        echo ""
    fi

    # ── Throughput ───────────────────────────────────────────────────────
    if $DO_BENCH; then
        header "BENCH: $config (K=$TYPE_K, V=$TYPE_V) — pp512 + tg128"

        local_log="$RESULTS_DIR/${config}-bench.jsonl"

        # Run llama-bench with JSONL output for reliable parsing.
        # Note: llama-bench uses short flags (-ngl, -fa, -p, -n, -o)
        # not --gpu-layers/--flash-attn/--test/--output.
        docker_run \
            /app/llama-bench \
            -m "$MODEL_PATH" \
            -fa 1 \
            -ngl 999 \
            -p 512 \
            -n 128 \
            -o jsonl \
            $(cache_flags_bench) \
            > "$local_log" 2>&1

        # Show the raw output.
        cat "$local_log"

        # Parse JSONL: lines with "n_prompt": 512 are pp, "n_gen": 128 are tg.
        # Field: "avg_ts" is average tokens/second.
        pp_ts=$(grep --perl-regexp --only-matching '"n_prompt": 512.*?"avg_ts": \K[\d.]+' "$local_log" || echo "N/A")
        tg_ts=$(grep --perl-regexp --only-matching '"n_gen": 128.*?"avg_ts": \K[\d.]+' "$local_log" || echo "N/A")

        RESULT_PP512[$config]="$pp_ts"
        RESULT_TG128[$config]="$tg_ts"

        echo ""
        echo "  pp512: ${pp_ts} t/s"
        echo "  tg128: ${tg_ts} t/s"
        echo ""
    fi
done

# ── Comparison Tables ────────────────────────────────────────────────────────

header "RESULTS SUMMARY"

# Get model name from path for display.
MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")" | sed 's/-GGUF$//' | sed 's/-MXFP4_MOE//')

echo "  Model: $MODEL_NAME"
echo "  Date:  $(timestamp)"
echo ""

# ── Memory Table ─────────────────────────────────────────────────────────────

if $DO_PERPLEXITY && [[ -n "${RESULT_CELLS[f16]:-}" ]]; then
    echo "Memory per 100K tokens:"
    echo ""
    printf "  %-8s  %10s  %10s  %10s  %8s\n" "Type" "K cache" "V cache" "Total" "vs F16"
    printf "  %-8s  %10s  %10s  %10s  %8s\n" "────────" "──────────" "──────────" "──────────" "────────"

    # Compute F16 total for ratio calculation (in GiB).
    f16_k_gib=$(awk "BEGIN { printf \"%.2f\", ${RESULT_K_MIB[f16]} / ${RESULT_CELLS[f16]} * 100000 / 1024 }")
    f16_v_gib=$(awk "BEGIN { printf \"%.2f\", ${RESULT_V_MIB[f16]} / ${RESULT_CELLS[f16]} * 100000 / 1024 }")
    f16_total=$(awk "BEGIN { printf \"%.2f\", $f16_k_gib + $f16_v_gib }")

    for config in "${CONFIGS[@]}"; do
        if [[ -z "${RESULT_CELLS[$config]:-}" ]]; then
            printf "  %-8s  %10s  %10s  %10s  %8s\n" "$config" "—" "—" "—" "—"
            continue
        fi

        k_gib=$(awk "BEGIN { printf \"%.2f\", ${RESULT_K_MIB[$config]} / ${RESULT_CELLS[$config]} * 100000 / 1024 }")
        v_gib=$(awk "BEGIN { printf \"%.2f\", ${RESULT_V_MIB[$config]} / ${RESULT_CELLS[$config]} * 100000 / 1024 }")
        total=$(awk "BEGIN { printf \"%.2f\", $k_gib + $v_gib }")
        ratio=$(awk "BEGIN { printf \"%.2fx\", $total / $f16_total }")

        printf "  %-8s  %7s GiB  %7s GiB  %7s GiB  %8s\n" \
            "$config" "$k_gib" "$v_gib" "$total" "$ratio"
    done
    echo ""
fi

# ── Perplexity Table ─────────────────────────────────────────────────────────

if $DO_PERPLEXITY && [[ -n "${RESULT_PPL[f16]:-}" ]]; then
    echo "Perplexity (wikitext-2-raw, $CHUNKS chunks):"
    echo ""
    printf "  %-8s  %12s  %10s\n" "Type" "PPL" "vs F16"
    printf "  %-8s  %12s  %10s\n" "────────" "────────────" "──────────"

    f16_ppl="${RESULT_PPL[f16]}"

    for config in "${CONFIGS[@]}"; do
        ppl="${RESULT_PPL[$config]:-N/A}"
        if [[ "$ppl" == "N/A" ]]; then
            printf "  %-8s  %12s  %10s\n" "$config" "N/A" "—"
        elif [[ "$config" == "f16" ]]; then
            printf "  %-8s  %12s  %10s\n" "$config" "$ppl" "—"
        else
            delta=$(awk "BEGIN { d = $ppl - $f16_ppl; printf \"%+.4f\", d }")
            printf "  %-8s  %12s  %10s\n" "$config" "$ppl" "$delta"
        fi
    done
    echo ""
fi

# ── Throughput Table ─────────────────────────────────────────────────────────

if $DO_BENCH; then
    echo "Throughput (tokens/sec):"
    echo ""
    printf "  %-8s  %12s  %12s\n" "Type" "pp512" "tg128"
    printf "  %-8s  %12s  %12s\n" "────────" "────────────" "────────────"

    for config in "${CONFIGS[@]}"; do
        pp="${RESULT_PP512[$config]:-N/A}"
        tg="${RESULT_TG128[$config]:-N/A}"

        # Format with comma separators if numeric.
        if [[ "$pp" != "N/A" ]]; then
            pp=$(printf "%'.0f" "${pp%.*}" 2>/dev/null || echo "$pp")
        fi
        if [[ "$tg" != "N/A" ]]; then
            tg=$(printf "%'.1f" "$tg" 2>/dev/null || echo "$tg")
        fi

        printf "  %-8s  %12s  %12s\n" "$config" "$pp" "$tg"
    done
    echo ""
fi

# ── Footer ───────────────────────────────────────────────────────────────────

echo "Raw results saved to: $RESULTS_DIR/"
echo ""
