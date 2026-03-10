#!/bin/bash
#
# KV Cache Quantization Benchmark — Vulkan Backend
#
# Adapted from kv-bench.sh for local Vulkan builds (no Docker).
# Uses build-vulkan/bin/ binaries and local model paths.
#
# Usage:
#   ./kv-bench-vulkan.sh                     # Full suite: build + perplexity + bench (all configs)
#   ./kv-bench-vulkan.sh --skip-build        # Skip build
#   ./kv-bench-vulkan.sh --skip-perplexity   # Skip perplexity (no PPL or memory data)
#   ./kv-bench-vulkan.sh --skip-bench        # Skip throughput benchmarks
#   ./kv-bench-vulkan.sh --chunks 2          # Quick perplexity (2 chunks)
#   ./kv-bench-vulkan.sh --config mxfp4      # Single config only
#   ./kv-bench-vulkan.sh --help              # Show usage

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build-vulkan"
BIN_DIR="$BUILD_DIR/bin"
DATASETS_DIR="$SCRIPT_DIR/datasets"
DATASET_FILE="wikitext-2-raw/wiki.test.raw"

LLAMA_BENCH="$BIN_DIR/llama-bench"
LLAMA_PERPLEXITY="$BIN_DIR/llama-perplexity"

# Model presets: "name|/path/to/model.gguf"
# Local paths (not Docker /models/ mount paths)
MODELS_BASE="$HOME/.lmstudio/models"
MODELS=(
    "qwen3-coder|${MODELS_BASE}/spectralyst/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE-GGUF/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE.gguf"
    "glm-4.7-flash|${MODELS_BASE}/noctrex/GLM-4.7-Flash-MXFP4_MOE-GGUF/GLM-4.7-Flash-MXFP4_MOE.gguf"
    "gemma-3n|${MODELS_BASE}/unsloth/gemma-3n-E4B-it-GGUF/gemma-3n-E4B-it-Q8_0.gguf"
    "gemma-3n-text-q8|${MODELS_BASE}/lmstudio-community/gemma-3n-E4B-it-text-GGUF/gemma-3n-E4B-it-Q8_0.gguf"
)
DEFAULT_MODEL="qwen3-coder"

# Default to both NVIDIA GPUs (skip Intel iGPU at Vulkan0)
DEFAULT_DEVICE="Vulkan1/Vulkan2"

# Resolve model name/path → sets MODEL_PATH and MODEL_NAME.
resolve_model() {
    local input="$1"
    for entry in "${MODELS[@]}"; do
        local name="${entry%%|*}"
        local path="${entry#*|}"
        if [[ "$input" == "$name" ]]; then
            MODEL_NAME="$name"
            MODEL_PATH="$path"
            return
        fi
    done
    MODEL_PATH="$input"
    MODEL_NAME="$(basename "$(dirname "$input")" | sed 's/-GGUF$//' | sed 's/-MXFP4_MOE//')"
}

# Defaults
DO_BUILD=true
DO_PERPLEXITY=true
DO_BENCH=true
CHUNKS_LIST=(16)
# Match CUDA kv-bench.sh configs for like-for-like comparison.
CONFIGS=(
    "f16"
    "q8_0"
    "q4_0"
    "q8_0+q4_0"
    "mxfp8"
    "mxfp6"
    "mxfp4"
    "mxfp8+mxfp8"
    "mxfp6+mxfp6"
)
MODEL_INPUTS=()
_config_overridden=false
DEVICE="$DEFAULT_DEVICE"

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
        --skip-bench|--skip-performance)
            DO_BENCH=false
            shift
            ;;
        --chunks)
            IFS=',' read -ra CHUNKS_LIST <<< "$2"
            shift 2
            ;;
        --config)
            if [[ "$_config_overridden" != "true" ]]; then
                CONFIGS=()
                _config_overridden=true
            fi
            case "$2" in
                f16)   CONFIGS+=("f16") ;;
                mxfp)  CONFIGS+=("mxfp8" "mxfp8_e5m2" "mxfp6" "mxfp6_e3m2" "mxfp4") ;;
                mxfp8) CONFIGS+=("mxfp8" "mxfp8_e5m2") ;;
                mxfp6) CONFIGS+=("mxfp6" "mxfp6_e3m2") ;;
                *)     CONFIGS+=("$2") ;;
            esac
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --model)
            MODEL_INPUTS+=("$2")
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "KV Cache Quantization Benchmark — Vulkan Backend"
            echo ""
            echo "Options:"
            echo "  --skip-build        Skip cmake build step"
            echo "  --skip-perplexity   Skip perplexity runs"
            echo "  --skip-bench        Skip throughput benchmarks (pp512 + tg128)"
            echo "  --chunks N[,M]      Chunk counts for perplexity (default: 16)"
            echo "  --config NAME       Config to test (repeatable). Groups: mxfp, mxfp8, mxfp6"
            echo "  --device DEV        Vulkan device(s), /-separated (default: $DEFAULT_DEVICE)"
            echo "  --model NAME|PATH   Model preset or path (default: $DEFAULT_MODEL)"
            echo "  --help              Show this help"
            echo ""
            echo "Models:"
            for entry in "${MODELS[@]}"; do
                echo "  ${entry%%|*}  → ${entry#*|}"
            done
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run $0 --help for usage."
            exit 1
            ;;
    esac
done

# ── Resolve Model List ───────────────────────────────────────────────────────

if [[ ${#MODEL_INPUTS[@]} -eq 0 ]]; then
    MODEL_INPUTS=("$DEFAULT_MODEL")
fi

echo "Models to test: ${MODEL_INPUTS[*]}"
echo "Backend: Vulkan (local build)"

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

# MLA models: V shares K buffer, so V type = K type.
MLA_MODELS=("glm-4.7-flash")

is_mla_model() {
    for m in "${MLA_MODELS[@]}"; do
        [[ "$MODEL_NAME" == "$m" ]] && return 0
    done
    return 1
}

# Map config name → display names and CLI args.
config_types() {
    case "$1" in
        f16)          TYPE_K="f16";         TYPE_V="f16";    CLI_K="f16";       CLI_V="f16"       ;;
        q8_0)         TYPE_K="q8_0";        TYPE_V="q8_0";   CLI_K="q8_0";      CLI_V="q8_0"      ;;
        q4_0)         TYPE_K="q4_0";        TYPE_V="q4_0";   CLI_K="q4_0";      CLI_V="q4_0"      ;;
        q8_0+q4_0)    TYPE_K="q8_0";        TYPE_V="q4_0";   CLI_K="q8_0";      CLI_V="q4_0"      ;;
        mxfp8|mxfp8_e4m3)   TYPE_K="mxfp8_e4m3";  TYPE_V="mxfp4";  CLI_K="mxfp8";     CLI_V="mxfp4"     ;;
        mxfp8_e5m2)         TYPE_K="mxfp8_e5m2";  TYPE_V="mxfp4";  CLI_K="mxfp8_e5m2"; CLI_V="mxfp4"     ;;
        mxfp6|mxfp6_e2m3)   TYPE_K="mxfp6_e2m3";  TYPE_V="mxfp4";  CLI_K="mxfp6";      CLI_V="mxfp4"     ;;
        mxfp6_e3m2)         TYPE_K="mxfp6_e3m2";  TYPE_V="mxfp4";  CLI_K="mxfp6_e3m2"; CLI_V="mxfp4"     ;;
        mxfp4)        TYPE_K="mxfp4";        TYPE_V="mxfp4";  CLI_K="mxfp4";     CLI_V="mxfp4"     ;;
        mxfp8+mxfp8)  TYPE_K="mxfp8_e4m3";  TYPE_V="mxfp8_e4m3"; CLI_K="mxfp8";  CLI_V="mxfp8"     ;;
        mxfp6+mxfp6)  TYPE_K="mxfp6_e2m3";  TYPE_V="mxfp6_e2m3"; CLI_K="mxfp6";  CLI_V="mxfp6"     ;;
        *)
            echo "Unknown config: $1"
            exit 1
            ;;
    esac

    if is_mla_model; then
        TYPE_V="$TYPE_K"
        CLI_V="$CLI_K"
    fi
}

# Build cache-type flags for llama-perplexity.
cache_flags_perplexity() {
    local flags=()
    if [[ "$CLI_K" != "f16" ]]; then
        flags+=(--cache-type-k "$CLI_K")
    fi
    if [[ "$CLI_V" != "f16" ]]; then
        flags+=(--cache-type-v "$CLI_V")
    fi
    echo "${flags[@]+"${flags[@]}"}"
}

# Build cache-type flags for llama-bench.
cache_flags_bench() {
    local flags=()
    if [[ "$CLI_K" != "f16" ]]; then
        flags+=(-ctk "$CLI_K")
    fi
    if [[ "$CLI_V" != "f16" ]]; then
        flags+=(-ctv "$CLI_V")
    fi
    echo "${flags[@]+"${flags[@]}"}"
}

# ── Preflight ────────────────────────────────────────────────────────────────

if $DO_PERPLEXITY; then
    if [[ ! -f "$DATASETS_DIR/$DATASET_FILE" ]]; then
        echo "ERROR: Dataset file not found: $DATASETS_DIR/$DATASET_FILE"
        exit 1
    fi
fi

# ── Build ────────────────────────────────────────────────────────────────────

if $DO_BUILD; then
    header "BUILD: cmake Vulkan"
    cmake -B "$BUILD_DIR" \
        -DGGML_VULKAN=ON \
        -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_TESTS=ON \
        -DGGML_CUDA=OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --config Release -j "$(nproc)"
    echo ""
    echo "Build completed successfully."
fi

if [[ ! -x "$LLAMA_BENCH" ]]; then
    echo "ERROR: llama-bench not found at $LLAMA_BENCH"
    echo "Run without --skip-build or build manually first."
    exit 1
fi

# ── Table Format ─────────────────────────────────────────────────────────────

table_top()  { echo "  ┌────────────┬─────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐"; }
table_hdr()  { printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │\n" \
                      "K type" "V type" "K GiB" "V GiB" "Total" "PPL" " Δ F16" "pp512" "tg128"; }
table_sep()  { echo "  ├────────────┼─────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤"; }
table_row()  { printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │\n" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"; }
table_bot()  { echo "  └────────────┴─────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘"; }

# ── Per-Model Loop ──────────────────────────────────────────────────────────

model_num=0
total_models=${#MODEL_INPUTS[@]}

for MODEL_INPUT in "${MODEL_INPUTS[@]}"; do

(( model_num += 1 ))
resolve_model "$MODEL_INPUT"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Model ${model_num}/${total_models}: ${MODEL_NAME}"
echo "  ${MODEL_PATH}"
echo "  Backend: Vulkan | $(timestamp)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

DEVICE_TAG=""
if [[ -n "$DEVICE" ]]; then
    DEVICE_TAG="-$(echo "$DEVICE" | tr '[:upper:]' '[:lower:]')"
fi
RESULTS_DIR="$SCRIPT_DIR/test-results/kv-bench-vulkan${DEVICE_TAG}-${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ── Result Storage (reset per model) ────────────────────────────────────

declare -A RESULT_PPL=()
declare -A RESULT_K_MIB=()
declare -A RESULT_V_MIB=()
declare -A RESULT_CELLS=()
declare -A RESULT_PP512=()
declare -A RESULT_TG128=()

# ── Run Tests ───────────────────────────────────────────────────────────

total_configs=${#CONFIGS[@]}
config_num=0

for config in "${CONFIGS[@]}"; do
    (( config_num += 1 ))
    config_types "$config"
    progress="[${model_num}/${total_models}] [${config_num}/${total_configs}]"

    # ── Perplexity ──────────────────────────────────────────────────
    if $DO_PERPLEXITY; then
        for chunks in "${CHUNKS_LIST[@]}"; do
            echo "${progress} K=${TYPE_K} V=${TYPE_V} — perplexity (${chunks} chunks)..."

            local_log="$RESULTS_DIR/${config}-perplexity-${chunks}ch.log"

            # llama-perplexity uses --device with comma separator
            DEVICE_PERPLEXITY="${DEVICE//\//,}"

            if ! "$LLAMA_PERPLEXITY" \
                --model "$MODEL_PATH" \
                --file "$DATASETS_DIR/$DATASET_FILE" \
                --chunks "$chunks" \
                --flash-attn 1 \
                --gpu-layers 99 \
                ${DEVICE_PERPLEXITY:+--device "$DEVICE_PERPLEXITY"} \
                $(cache_flags_perplexity) \
                > "$local_log" 2>&1; then
                echo "ERROR: Perplexity failed for ${config}. Log:"
                cat "$local_log"
                exit 1
            fi

            ppl=$(grep --perl-regexp --only-matching 'Final estimate: PPL = \K[\d.]+' "$local_log" || echo "N/A")
            RESULT_PPL["${config}:${chunks}"]="$ppl"

            if [[ -z "${RESULT_CELLS[$config]:-}" ]]; then
                kv_line=$(grep 'llama_kv_cache: size' "$local_log" | tail --lines 1 || echo "")
                if [[ -n "$kv_line" ]]; then
                    cells_per_seq=$(echo "$kv_line" | grep --perl-regexp --only-matching '\(\s*\K\d+(?=\s+cells)')
                    n_seq=$(echo "$kv_line" | grep --perl-regexp --only-matching '\d+(?=/\d+ seqs)')
                    RESULT_CELLS[$config]=$(( cells_per_seq * n_seq ))
                    RESULT_K_MIB[$config]=$(echo "$kv_line" | grep --perl-regexp --only-matching 'K \([^)]+\):\s*\K[\d.]+')
                    RESULT_V_MIB[$config]=$(echo "$kv_line" | grep --perl-regexp --only-matching 'V \([^)]+\):\s*\K[\d.]+')
                fi
            fi

            echo "        PPL: ${ppl} | K: ${RESULT_K_MIB[$config]:-?} MiB | V: ${RESULT_V_MIB[$config]:-?} MiB"
        done
    fi

    # ── Throughput ──────────────────────────────────────────────────
    if $DO_BENCH; then
        echo "${progress} K=${TYPE_K} V=${TYPE_V} — bench (pp512 + tg128)..."

        local_log="$RESULTS_DIR/${config}-bench.jsonl"

        if ! "$LLAMA_BENCH" \
            -m "$MODEL_PATH" \
            -fa 1 \
            -ngl 999 \
            ${DEVICE:+-dev "$DEVICE"} \
            -p 512 \
            -n 128 \
            -o jsonl \
            $(cache_flags_bench) \
            > "$local_log" 2>&1; then
            echo "ERROR: Bench failed for ${config}. Log:"
            cat "$local_log"
            exit 1
        fi

        pp_ts=$(grep --perl-regexp --only-matching '"n_prompt": 512.*?"avg_ts": \K[\d.]+' "$local_log" || echo "N/A")
        tg_ts=$(grep --perl-regexp --only-matching '"n_gen": 128.*?"avg_ts": \K[\d.]+' "$local_log" || echo "N/A")

        RESULT_PP512[$config]="$pp_ts"
        RESULT_TG128[$config]="$tg_ts"

        echo "        pp512: ${pp_ts} t/s | tg128: ${tg_ts} t/s"
    fi
done

# ── Results Table ───────────────────────────────────────────────────────

echo ""
echo "  ${MODEL_NAME} — Vulkan — ${MODEL_PATH}"
echo "  ${CHUNKS_LIST[*]} chunks — $(timestamp)"
echo ""

declare -a TABLE_ROWS=()

for config in "${CONFIGS[@]}"; do
    last_chunks="${CHUNKS_LIST[${#CHUNKS_LIST[@]}-1]}"
    has_ppl="${RESULT_PPL["${config}:${last_chunks}"]:-}"
    has_bench="${RESULT_PP512[$config]:-}"
    if [[ -z "$has_ppl" && -z "$has_bench" ]]; then
        continue
    fi
    config_types "$config"

    # Memory per 100K tokens.
    k_gib_fmt="-"
    v_gib_fmt="-"
    total_fmt="-"
    if [[ -n "${RESULT_CELLS[$config]:-}" ]]; then
        k_gib=$(awk "BEGIN { printf \"%.2f\", ${RESULT_K_MIB[$config]} / ${RESULT_CELLS[$config]} * 100000 / 1024 }")
        v_gib=$(awk "BEGIN { printf \"%.2f\", ${RESULT_V_MIB[$config]} / ${RESULT_CELLS[$config]} * 100000 / 1024 }")
        total=$(awk "BEGIN { printf \"%.2f\", $k_gib + $v_gib }")
        k_gib_fmt="$k_gib"
        v_gib_fmt="$v_gib"
        total_fmt="$total"
    fi

    # PPL.
    ppl_fmt="-"
    delta_fmt="-"
    sort_key="9999.9999"
    ppl_raw="${RESULT_PPL["${config}:${last_chunks}"]:-}"
    f16_raw="${RESULT_PPL["f16:${last_chunks}"]:-}"
    if [[ -n "$ppl_raw" && "$ppl_raw" != "N/A" ]]; then
        ppl_fmt=$(awk "BEGIN { printf \"%.2f\", $ppl_raw }")
        sort_key=$(printf "%010.4f" "$ppl_raw")
        if [[ "$config" != "f16" && -n "$f16_raw" && "$f16_raw" != "N/A" ]]; then
            delta_fmt=$(awk "BEGIN { printf \"%+.2f\", $ppl_raw - $f16_raw }")
        fi
    fi

    # Throughput.
    pp_fmt="-"
    tg_fmt="-"
    pp_raw="${RESULT_PP512[$config]:-}"
    tg_raw="${RESULT_TG128[$config]:-}"
    if [[ -n "$pp_raw" && "$pp_raw" != "N/A" ]]; then
        pp_fmt=$(printf "%'.0f" "${pp_raw%.*}")
    fi
    if [[ -n "$tg_raw" && "$tg_raw" != "N/A" ]]; then
        tg_fmt=$(printf "%'.1f" "$tg_raw")
    fi

    TABLE_ROWS+=("${sort_key}|${TYPE_K}|${TYPE_V}|${k_gib_fmt}|${v_gib_fmt}|${total_fmt}|${ppl_fmt}|${delta_fmt}|${pp_fmt}|${tg_fmt}")
done

# Sort rows by PPL ascending.
IFS=$'\n' SORTED_ROWS=($(printf '%s\n' "${TABLE_ROWS[@]}" | sort -t'|' -k1,1n)); unset IFS

if [[ ${#SORTED_ROWS[@]} -gt 0 ]]; then
    table_top
    table_hdr
    table_sep
    first=true
    for entry in "${SORTED_ROWS[@]}"; do
        if $first; then first=false; else table_sep; fi
        IFS='|' read -r _sort_key tk tv k_gib v_gib total ppl delta pp tg <<< "$entry"
        table_row "$tk" "$tv" "$k_gib" "$v_gib" "$total" "$ppl" "$delta" "$pp" "$tg"
    done
    table_bot
fi

echo ""
echo "  Memory: GiB per 100K tokens (K + V cache)"
echo "  Raw logs: $RESULTS_DIR/"
echo ""

done  # end per-model loop
