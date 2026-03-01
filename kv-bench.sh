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

# Model presets: "name|/path/to/model.gguf"
MODELS=(
    "qwen3-coder|/models/spectralyst/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE-GGUF/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE.gguf"
    "gpt-oss-20b|/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
    "qwen35|/models/noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF/Qwen3.5-35B-A3B-MXFP4_MOE_F16.gguf"
    "glm-4.7-flash|/models/noctrex/GLM-4.7-Flash-MXFP4_MOE-GGUF/GLM-4.7-Flash-MXFP4_MOE.gguf"
    "nemotron-nano|/models/noctrex/Nemotron-3-Nano-30B-A3B-MXFP4_MOE-GGUF/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP4_MOE.gguf"
)
DEFAULT_MODEL="qwen3-coder"

# Resolve model name/path → sets MODEL_PATH and MODEL_NAME.
resolve_model() {
    local input="$1"
    # Check if it's a preset name.
    for entry in "${MODELS[@]}"; do
        local name="${entry%%|*}"
        local path="${entry#*|}"
        if [[ "$input" == "$name" ]]; then
            MODEL_NAME="$name"
            MODEL_PATH="$path"
            return
        fi
    done
    # Not a preset — treat as a raw path.
    MODEL_PATH="$input"
    MODEL_NAME="$(basename "$(dirname "$input")" | sed 's/-GGUF$//' | sed 's/-MXFP4_MOE//')"
}

# Defaults
DO_BUILD=true
DO_PERPLEXITY=true
DO_BENCH=true
CHUNKS_LIST=(100)
CONFIGS=("f16" "q8_0" "q4_0" "mxfp4" "mxfp8" "mxfp8_mxfp8")
MODEL_INPUT="$DEFAULT_MODEL"

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
            IFS=',' read -ra CHUNKS_LIST <<< "$2"
            shift 2
            ;;
        --config)
            CONFIGS=("$2")
            shift 2
            ;;
        --model)
            MODEL_INPUT="$2"
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
            echo "  --chunks N[,M]      Chunk counts for perplexity, comma-separated (default: 100)"
            echo "  --config NAME       Single config: f16, q8_0, q4_0, mxfp4, mxfp8, mxfp8_mxfp8 (default: all)"
            echo "  --model NAME|PATH   Model preset or path (default: $DEFAULT_MODEL)"
            echo "  --help              Show this help"
            echo ""
            echo "Models:"
            for entry in "${MODELS[@]}"; do
                echo "  ${entry%%|*}  → ${entry#*|}"
            done
            echo "  (or pass a raw GGUF path)"
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

# ── Resolve Model ────────────────────────────────────────────────────────────

resolve_model "$MODEL_INPUT"
echo "Model: $MODEL_NAME ($MODEL_PATH)"

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
        mxfp8) TYPE_K="mxfp8"; TYPE_V="mxfp4" ;;
        mxfp8_mxfp8) TYPE_K="mxfp8"; TYPE_V="mxfp8" ;;
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

RESULTS_DIR="$SCRIPT_DIR/test-results/kv-bench-${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
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

# Models with D_K > 256 cannot use MXFP4 (residual K needs too much shared memory).
# MXFP8 supports arbitrary D via computed config fallback.
MXFP4_UNSUPPORTED_MODELS=("glm-4.7-flash")

is_config_supported() {
    local config="$1"
    case "$config" in
        mxfp4)
            for m in "${MXFP4_UNSUPPORTED_MODELS[@]}"; do
                if [[ "$MODEL_NAME" == "$m" ]]; then
                    return 1
                fi
            done
            ;;
    esac
    return 0
}

for config in "${CONFIGS[@]}"; do
    if ! is_config_supported "$config"; then
        echo ""
        echo "  ⚠ Skipping $config — not supported for $MODEL_NAME (D_K > 256)"
        echo ""
        continue
    fi
    config_types "$config"

    # ── Perplexity ───────────────────────────────────────────────────────
    if $DO_PERPLEXITY; then
        for chunks in "${CHUNKS_LIST[@]}"; do
            header "PERPLEXITY: $config (K=$TYPE_K, V=$TYPE_V) — $chunks chunks"

            local_log="$RESULTS_DIR/${config}-perplexity-${chunks}ch.log"

            # Run perplexity, capture all output (stdout + stderr).
            docker_run \
                /app/llama-perplexity \
                --model "$MODEL_PATH" \
                --file "/datasets/$DATASET_FILE" \
                --chunks "$chunks" \
                --flash-attn on \
                --gpu-layers 99 \
                $(cache_flags_perplexity) \
                2>&1 | tee "$local_log"

            # Parse PPL from "Final estimate: PPL = X.XXXX".
            ppl=$(grep --perl-regexp --only-matching 'Final estimate: PPL = \K[\d.]+' "$local_log" || echo "N/A")
            RESULT_PPL["${config}:${chunks}"]="$ppl"

            # Parse KV cache memory (only need once per config — same for all chunk counts).
            if [[ -z "${RESULT_CELLS[$config]:-}" ]]; then
                # Format: "llama_kv_cache: size =  192.00 MiB (   512 cells,  48 layers, ...), K (f16):   96.00 MiB, V (f16):   96.00 MiB"
                kv_line=$(grep 'llama_kv_cache: size' "$local_log" | tail --lines 1 || echo "")
                if [[ -n "$kv_line" ]]; then
                    cells_per_seq=$(echo "$kv_line" | grep --perl-regexp --only-matching '\(\s*\K\d+(?=\s+cells)')
                    n_seq=$(echo "$kv_line" | grep --perl-regexp --only-matching '\d+(?=/\d+ seqs)')
                    RESULT_CELLS[$config]=$(( cells_per_seq * n_seq ))
                    RESULT_K_MIB[$config]=$(echo "$kv_line" | grep --perl-regexp --only-matching 'K \([^)]+\):\s*\K[\d.]+')
                    RESULT_V_MIB[$config]=$(echo "$kv_line" | grep --perl-regexp --only-matching 'V \([^)]+\):\s*\K[\d.]+')
                fi
            fi

            echo ""
            echo "  PPL ($chunks chunks): $ppl"
            echo "  KV cache: K=${RESULT_K_MIB[$config]:-?} MiB, V=${RESULT_V_MIB[$config]:-?} MiB, cells=${RESULT_CELLS[$config]:-?}"
            echo ""
        done
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

echo "  Model:   $MODEL_NAME"
echo "  Date:    $(timestamp)"
echo "  Chunks:  ${CHUNKS_LIST[*]}"
echo ""

# ── Memory Table ─────────────────────────────────────────────────────────────

if $DO_PERPLEXITY && [[ -n "${RESULT_CELLS[f16]:-}" ]]; then
    echo "Memory per 100K tokens:"
    echo ""
    printf "  %-8s  %10s  %10s  %10s\n" "Type" "K cache" "V cache" "Total"
    printf "  %-8s  %10s  %10s  %10s\n" "────────" "──────────" "──────────" "──────────"

    for config in "${CONFIGS[@]}"; do
        if [[ -z "${RESULT_CELLS[$config]:-}" ]]; then
            printf "  %-8s  %10s  %10s  %10s\n" "$config" "—" "—" "—"
            continue
        fi

        k_gib=$(awk "BEGIN { printf \"%.2f\", ${RESULT_K_MIB[$config]} / ${RESULT_CELLS[$config]} * 100000 / 1024 }")
        v_gib=$(awk "BEGIN { printf \"%.2f\", ${RESULT_V_MIB[$config]} / ${RESULT_CELLS[$config]} * 100000 / 1024 }")
        total=$(awk "BEGIN { printf \"%.2f\", $k_gib + $v_gib }")

        printf "  %-8s  %7s GiB  %7s GiB  %7s GiB\n" \
            "$config" "$k_gib" "$v_gib" "$total"
    done
    echo ""
fi

# ── Perplexity Table ─────────────────────────────────────────────────────────

if $DO_PERPLEXITY; then
    # Check if we have any PPL results at all.
    has_ppl=false
    for chunks in "${CHUNKS_LIST[@]}"; do
        [[ -n "${RESULT_PPL["f16:${chunks}"]:-}" ]] && has_ppl=true
    done

    if $has_ppl; then
        echo "Perplexity (wikitext-2-raw):"
        echo ""

        # Build header: Type | PPL@ch1 | delta | PPL@ch2 | delta | ...
        printf "  %-12s" "Type"
        for chunks in "${CHUNKS_LIST[@]}"; do
            printf "  %12s  %10s" "${chunks}ch PPL" "vs F16"
        done
        printf "\n"
        printf "  %-12s" "────────────"
        for chunks in "${CHUNKS_LIST[@]}"; do
            printf "  %12s  %10s" "────────────" "──────────"
        done
        printf "\n"

        for config in "${CONFIGS[@]}"; do
            printf "  %-12s" "$config"
            for chunks in "${CHUNKS_LIST[@]}"; do
                ppl="${RESULT_PPL["${config}:${chunks}"]:-N/A}"
                f16_ppl="${RESULT_PPL["f16:${chunks}"]:-N/A}"
                if [[ "$ppl" == "N/A" ]]; then
                    printf "  %12s  %10s" "—" "—"
                elif [[ "$config" == "f16" ]]; then
                    printf "  %12s  %10s" "$ppl" "—"
                elif [[ "$f16_ppl" == "N/A" ]]; then
                    printf "  %12s  %10s" "$ppl" "—"
                else
                    delta=$(awk "BEGIN { d = $ppl - $f16_ppl; printf \"%+.4f\", d }")
                    printf "  %12s  %10s" "$ppl" "$delta"
                fi
            done
            printf "\n"
        done
        echo ""
    fi
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
