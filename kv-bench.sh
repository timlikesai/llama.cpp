#!/bin/bash
#
# KV Cache Quantization Benchmark — Unified (Metal local + CUDA Docker)
#
# Auto-detects platform: macOS → local Metal build, Linux → Docker CUDA.
# Override with --local or --docker.
#
# Usage:
#   ./kv-bench.sh                        # Auto-detect, default model
#   ./kv-bench.sh --model qwen3-coder    # Specific model
#   ./kv-bench.sh --model all            # Every model
#   ./kv-bench.sh --local                # Force local mode (Metal)
#   ./kv-bench.sh --docker               # Force Docker mode (CUDA)
#   ./kv-bench.sh --chunks 2             # Quick perplexity
#   ./kv-bench.sh --config mxfp          # All MXFP configs
#   ./kv-bench.sh --help                 # Show usage

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASETS_DIR="$SCRIPT_DIR/datasets"
DATASET_FILE="wikitext-2-raw/wiki.test.raw"

# Auto-detect mode
if [[ "$(uname)" == "Darwin" ]]; then
    MODE="local"
else
    MODE="docker"
fi

# Model paths are relative to MODELS_BASE, set per mode.
# Docker: /models (container mount)   Local: ~/.lmstudio/models
MODELS=(
    # ── MXFP4 MOE models ──
    "qwen3-coder|spectralyst/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE-GGUF/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE.gguf"
    "qwen3-coder-next|noctrex/Qwen3-Coder-Next-MXFP4_MOE-GGUF/Qwen3-Coder-Next-MXFP4_MOE.gguf"
    "qwen35-mxfp4|noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF/Qwen3.5-35B-A3B-MXFP4_MOE_F16.gguf"
    "gpt-oss-20b|lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
    "gpt-oss-120b|lmstudio-community/gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4-00001-of-00002.gguf"
    "nemotron-nano|noctrex/Nemotron-3-Nano-30B-A3B-MXFP4_MOE-GGUF/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP4_MOE.gguf"
    "glm-4.7-flash|noctrex/GLM-4.7-Flash-MXFP4_MOE-GGUF/GLM-4.7-Flash-MXFP4_MOE.gguf"
    "glm-4.7-flash-i1-xl|noctrex/GLM-4.7-Flash-i1-MXFP4_MOE_XL-exp-GGUF/GLM-4.7-Flash-i1-MXFP4_MOE_XL-exp.gguf"
    # ── Standard quant models ──
    "qwen35-q8|lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q8_0.gguf"
    "nemotron-nano-q4|unsloth/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-Q4_1.gguf"
    "glm-4.7-flash-q8|lmstudio-community/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q8_0.gguf"
    "glm-4.7-flash-q4|unsloth/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_1.gguf"
    "devstral-24b|unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-Q4_1.gguf"
    "granite-tiny|lmstudio-community/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q8_0.gguf"
    "rnj-1|lmstudio-community/rnj-1-instruct-GGUF/rnj-1-instruct-Q8_0.gguf"
    "gemma-3n|unsloth/gemma-3n-E4B-it-GGUF/gemma-3n-E4B-it-Q8_0.gguf"
    "gemma-3n-text-q8|lmstudio-community/gemma-3n-E4B-it-text-GGUF/gemma-3n-E4B-it-Q8_0.gguf"
    "gemma-3n-text-q6|lmstudio-community/gemma-3n-E4B-it-text-GGUF/gemma-3n-E4B-it-Q6_K.gguf"
    "gemma-3n-text-q4|lmstudio-community/gemma-3n-E4B-it-text-GGUF/gemma-3n-E4B-it-Q4_K_M.gguf"
    "gemma-3-27b-qat|lmstudio-community/gemma-3-27B-it-qat-GGUF/gemma-3-27B-it-QAT-Q4_0.gguf"
    "qwen3-4b-math|mradermacher/Qwen3-4B-math-GGUF/Qwen3-4B-math.Q8_0.gguf"
    "lfm2.5-thinking|unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q8_0.gguf"
    "lfm2.5-instruct|unsloth/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf"
    "embeddinggemma|unsloth/embeddinggemma-300m-GGUF/embeddinggemma-300M-Q8_0.gguf"
    "qwen3.5-27b-q4|unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_1.gguf"
)
DEFAULT_MODEL="qwen3-coder"

# Defaults
DO_BUILD=true
DO_PERPLEXITY=true
DO_BENCH=true
DO_GPU=true
DO_CPU=false
CHUNKS_LIST=(16)
REPEATS=3
PP=512
TG=128
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

# ── Argument Parsing ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)            MODE="local"; shift ;;
        --docker)           MODE="docker"; shift ;;
        --skip-build)       DO_BUILD=false; shift ;;
        --skip-perplexity)  DO_PERPLEXITY=false; shift ;;
        --skip-bench|--skip-performance) DO_BENCH=false; shift ;;
        --skip-cpu)         DO_CPU=false; shift ;;
        --skip-gpu)         DO_GPU=false; shift ;;
        --chunks)           IFS=',' read -ra CHUNKS_LIST <<< "$2"; shift 2 ;;
        --repeats|-r)       REPEATS="$2"; shift 2 ;;
        --pp)               PP="$2"; shift 2 ;;
        --tg)               TG="$2"; shift 2 ;;
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
        --model)
            case "$2" in
                gemma)
                    for entry in "${MODELS[@]}"; do
                        local_name="${entry%%|*}"
                        [[ "$local_name" == gemma* ]] && MODEL_INPUTS+=("$local_name")
                    done ;;
                glm)
                    for entry in "${MODELS[@]}"; do
                        local_name="${entry%%|*}"
                        [[ "$local_name" == glm* ]] && MODEL_INPUTS+=("$local_name")
                    done ;;
                qwen)
                    for entry in "${MODELS[@]}"; do
                        local_name="${entry%%|*}"
                        [[ "$local_name" == qwen* ]] && MODEL_INPUTS+=("$local_name")
                    done ;;
                all)
                    for entry in "${MODELS[@]}"; do
                        MODEL_INPUTS+=("${entry%%|*}")
                    done ;;
                *)  MODEL_INPUTS+=("$2") ;;
            esac
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "KV Cache Quantization Benchmark (Metal + CUDA)"
            echo ""
            echo "Mode (auto-detected from OS, override with flag):"
            echo "  --local             Run with local build (Metal on macOS)"
            echo "  --docker            Run with Docker (CUDA on Linux)"
            echo ""
            echo "Options:"
            echo "  --skip-build        Skip build step"
            echo "  --skip-perplexity   Skip perplexity runs"
            echo "  --skip-bench        Skip throughput benchmarks"
            echo "  --skip-cpu          Skip CPU benchmarks"
            echo "  --skip-gpu          Skip GPU benchmarks"
            echo "  --chunks N[,M]      Chunk counts for perplexity (default: 16)"
            echo "  --repeats N         Bench repeats (default: 3)"
            echo "  --pp N              Prompt size (default: 512)"
            echo "  --tg N              Token gen count (default: 128)"
            echo "  --config NAME       Config to test (repeatable). Available:"
            echo "                        f16, q8_0, q4_0, q8_0+q4_0"
            echo "                        mxfp8, mxfp8_e5m2, mxfp6, mxfp6_e3m2, mxfp4"
            echo "                        mxfp8+mxfp8, mxfp6+mxfp6"
            echo "                      Groups: mxfp (all 5), mxfp8 (both), mxfp6 (both)"
            echo "  --model NAME|PATH   Model preset, path, or group (repeatable)"
            echo "                      Groups: gemma, glm, qwen, all"
            echo ""
            echo "Models:"
            for entry in "${MODELS[@]}"; do
                echo "  ${entry%%|*}"
            done
            exit 0
            ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ${#MODEL_INPUTS[@]} -eq 0 ]]; then
    MODEL_INPUTS=("$DEFAULT_MODEL")
fi

# ── Mode-Specific Setup ─────────────────────────────────────────────────────

if [[ "$MODE" == "local" ]]; then
    MODELS_BASE="$HOME/.lmstudio/models"
    BUILD_DIR="$SCRIPT_DIR/build/bin"
    LABEL="Metal"

    if [[ ! -x "$BUILD_DIR/llama-bench" ]]; then
        echo "ERROR: $BUILD_DIR/llama-bench not found. Run: cmake --build build -j"
        exit 1
    fi

    run_cmd() { "$@"; }
    BIN_PERPLEXITY="$BUILD_DIR/llama-perplexity"
    BIN_BENCH="$BUILD_DIR/llama-bench"
    DATASET_PATH="$DATASETS_DIR/$DATASET_FILE"
    GPU_LAYERS=999

elif [[ "$MODE" == "docker" ]]; then
    MODELS_BASE=""  # paths stored with leading /models/ for docker
    SERVICES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    LABEL="CUDA"

    run_cmd() {
        cd "$SERVICES_DIR"
        docker compose run \
            --rm \
            --entrypoint "" \
            --volume "$DATASETS_DIR:/datasets:ro" \
            llama-cpp-ultra \
            "$@"
    }
    BIN_PERPLEXITY="/app/llama-perplexity"
    BIN_BENCH="/app/llama-bench"
    DATASET_PATH="/datasets/$DATASET_FILE"
    GPU_LAYERS=99
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

resolve_model() {
    local input="$1"
    for entry in "${MODELS[@]}"; do
        local name="${entry%%|*}"
        local relpath="${entry#*|}"
        if [[ "$input" == "$name" ]]; then
            MODEL_NAME="$name"
            if [[ "$MODE" == "docker" ]]; then
                MODEL_PATH="/models/${relpath}"
            else
                MODEL_PATH="${MODELS_BASE}/${relpath}"
            fi
            return
        fi
    done
    MODEL_PATH="$input"
    MODEL_NAME="$(basename "$(dirname "$input")" | sed 's/-GGUF$//' | sed 's/-MXFP4_MOE//')"
}

# MLA models: V shares K buffer, so V type = K type.
MLA_MODELS=("glm-4.7-flash" "glm-4.7-flash-i1-xl" "glm-4.7-flash-q8" "glm-4.7-flash-q4")

is_mla_model() {
    for m in "${MLA_MODELS[@]}"; do
        [[ "$MODEL_NAME" == "$m" ]] && return 0
    done
    return 1
}

config_types() {
    case "$1" in
        f16)          TYPE_K="f16";         TYPE_V="f16";    CLI_K="f16";       CLI_V="f16"       ;;
        q8_0)         TYPE_K="q8_0";        TYPE_V="q8_0";   CLI_K="q8_0";      CLI_V="q8_0"      ;;
        q4_0)         TYPE_K="q4_0";        TYPE_V="q4_0";   CLI_K="q4_0";      CLI_V="q4_0"      ;;
        q8_0+q4_0)    TYPE_K="q8_0";        TYPE_V="q4_0";   CLI_K="q8_0";      CLI_V="q4_0"      ;;
        mxfp8|mxfp8_e4m3)   TYPE_K="mxfp8";       TYPE_V="mxfp4";  CLI_K="mxfp8";     CLI_V="mxfp4"     ;;
        mxfp8_e5m2)         TYPE_K="mxfp8_e5m2"; TYPE_V="mxfp4";  CLI_K="mxfp8_e5m2"; CLI_V="mxfp4"     ;;
        mxfp6|mxfp6_e2m3)   TYPE_K="mxfp6";       TYPE_V="mxfp4";  CLI_K="mxfp6";      CLI_V="mxfp4"     ;;
        mxfp6_e3m2)         TYPE_K="mxfp6_e3m2"; TYPE_V="mxfp4";  CLI_K="mxfp6_e3m2"; CLI_V="mxfp4"     ;;
        mxfp4)              TYPE_K="mxfp4";       TYPE_V="mxfp4";  CLI_K="mxfp4";     CLI_V="mxfp4"     ;;
        mxfp8+mxfp8|mxfp8_e4m3+mxfp8_e4m3)   TYPE_K="mxfp8";       TYPE_V="mxfp8";       CLI_K="mxfp8";      CLI_V="mxfp8"      ;;
        mxfp8_e5m2+mxfp8_e5m2)               TYPE_K="mxfp8_e5m2"; TYPE_V="mxfp8_e5m2"; CLI_K="mxfp8_e5m2"; CLI_V="mxfp8_e5m2"  ;;
        mxfp6+mxfp6|mxfp6_e2m3+mxfp6_e2m3)   TYPE_K="mxfp6";       TYPE_V="mxfp6";       CLI_K="mxfp6";      CLI_V="mxfp6"       ;;
        mxfp6_e3m2+mxfp6_e3m2)               TYPE_K="mxfp6_e3m2"; TYPE_V="mxfp6_e3m2"; CLI_K="mxfp6_e3m2"; CLI_V="mxfp6_e3m2"  ;;
        *)  echo "Unknown config: $1"; exit 1 ;;
    esac

    if is_mla_model; then
        TYPE_V="$TYPE_K"
        CLI_V="$CLI_K"
    fi
}

cache_flags_perplexity() {
    local flags=()
    [[ "$CLI_K" != "f16" ]] && flags+=(--cache-type-k "$CLI_K")
    [[ "$CLI_V" != "f16" ]] && flags+=(--cache-type-v "$CLI_V")
    echo "${flags[@]+"${flags[@]}"}"
}

cache_flags_bench() {
    local flags=()
    [[ "$CLI_K" != "f16" ]] && flags+=(-ctk "$CLI_K")
    [[ "$CLI_V" != "f16" ]] && flags+=(-ctv "$CLI_V")
    echo "${flags[@]+"${flags[@]}"}"
}

# ── Preflight ────────────────────────────────────────────────────────────────

if $DO_PERPLEXITY; then
    if [[ ! -f "$DATASETS_DIR/$DATASET_FILE" ]]; then
        echo "ERROR: Dataset not found at $DATASETS_DIR/$DATASET_FILE"
        exit 1
    fi
fi

# ── Build ────────────────────────────────────────────────────────────────────

if $DO_BUILD && [[ "$MODE" == "docker" ]]; then
    SERVICES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
    echo "Building Docker image..."
    cd "$SERVICES_DIR"
    docker compose build llama-cpp-ultra
    echo "Build complete."
fi

# ── Table Format ─────────────────────────────────────────────────────────────

SHOW_GPU_BENCH=$($DO_BENCH && $DO_GPU && echo true || echo false)
SHOW_CPU_BENCH=$($DO_BENCH && $DO_CPU && echo true || echo false)
SHOW_GPU_PPL=$($DO_PERPLEXITY && $DO_GPU && echo true || echo false)
SHOW_CPU_PPL=$($DO_PERPLEXITY && $DO_CPU && echo true || echo false)

if ($SHOW_GPU_PPL || $SHOW_GPU_BENCH) && ($SHOW_CPU_PPL || $SHOW_CPU_BENCH); then
    table_top()  { echo "  ┌────────────┬─────────┬────────┬────────┬────────┬─────────┬─────────┬────────┬──────────────────┬──────────────────┐"; }
    table_hdr()  { printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %7s │ %7s │ %6s │ %8s %7s │ %8s %7s │\n" \
                          "K type" "V type" "K GiB" "V GiB" "Total" "GPU PPL" "CPU PPL" " Δ F16" "GPU pp" "GPU tg" "CPU pp" "CPU tg"; }
    table_sep()  { echo "  ├────────────┼─────────┼────────┼────────┼────────┼─────────┼─────────┼────────┼──────────────────┼──────────────────┤"; }
    table_row()  { printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %7s │ %7s │ %6s │ %8s %7s │ %8s %7s │\n" \
                          "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}"; }
    table_bot()  { echo "  └────────────┴─────────┴────────┴────────┴────────┴─────────┴─────────┴────────┴──────────────────┴──────────────────┘"; }
    TABLE_MODE="full"
elif $SHOW_GPU_BENCH || $SHOW_GPU_PPL; then
    table_top()  { echo "  ┌────────────┬─────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐"; }
    table_hdr()  { printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │\n" \
                          "K type" "V type" "K GiB" "V GiB" "Total" "PPL" " Δ F16" "pp${PP}" "tg${TG}"; }
    table_sep()  { echo "  ├────────────┼─────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤"; }
    table_row()  { printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │\n" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"; }
    table_bot()  { echo "  └────────────┴─────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘"; }
    TABLE_MODE="gpu"
elif $SHOW_CPU_BENCH || $SHOW_CPU_PPL; then
    table_top()  { echo "  ┌────────────┬─────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐"; }
    table_hdr()  { printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │\n" \
                          "K type" "V type" "K GiB" "V GiB" "Total" "PPL" " Δ F16" "pp${PP}" "tg${TG}"; }
    table_sep()  { echo "  ├────────────┼─────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤"; }
    table_row()  { printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │\n" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"; }
    table_bot()  { echo "  └────────────┴─────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘"; }
    TABLE_MODE="cpu"
else
    TABLE_MODE="none"
fi

# ── Per-Model Loop ───────────────────────────────────────────────────────────

echo "Mode: $MODE ($LABEL)"
echo "Models: ${MODEL_INPUTS[*]}"
echo ""

model_num=0
total_models=${#MODEL_INPUTS[@]}

for MODEL_INPUT in "${MODEL_INPUTS[@]}"; do

(( model_num += 1 ))
resolve_model "$MODEL_INPUT"

# In local mode, skip models that don't exist on disk
if [[ "$MODE" == "local" && ! -f "$MODEL_PATH" ]]; then
    echo "SKIP: ${MODEL_NAME} — not found at ${MODEL_PATH}"
    echo ""
    continue
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Model ${model_num}/${total_models}: ${MODEL_NAME}"
echo "  ${MODEL_PATH}"
echo "  $LABEL — $(timestamp)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

RESULTS_DIR="$SCRIPT_DIR/test-results/kv-bench-${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ── Result Storage ───────────────────────────────────────────────────────

declare -A RESULT_PPL=()
declare -A RESULT_K_MIB=()
declare -A RESULT_V_MIB=()
declare -A RESULT_CELLS=()
declare -A RESULT_PP=()
declare -A RESULT_TG=()
declare -A RESULT_PPL_CPU=()
declare -A RESULT_PP_CPU=()
declare -A RESULT_TG_CPU=()

# ── Run Tests ────────────────────────────────────────────────────────────

total_configs=${#CONFIGS[@]}
config_num=0

for config in "${CONFIGS[@]}"; do
    (( config_num += 1 ))
    config_types "$config"
    progress="[${model_num}/${total_models}] [${config_num}/${total_configs}]"

    # ── GPU Perplexity ──────────────────────────────────────────────
    if $DO_PERPLEXITY && $DO_GPU; then
        for chunks in "${CHUNKS_LIST[@]}"; do
            echo "${progress} K=${TYPE_K} V=${TYPE_V} — ${LABEL} perplexity (${chunks} chunks)..."

            local_log="$RESULTS_DIR/${config}-perplexity-gpu-${chunks}ch.log"

            if ! run_cmd \
                $BIN_PERPLEXITY \
                --model "$MODEL_PATH" \
                --file "$DATASET_PATH" \
                --chunks "$chunks" \
                --flash-attn on \
                --gpu-layers $GPU_LAYERS \
                $(cache_flags_perplexity) \
                > "$local_log" 2>&1; then
                echo "ERROR: perplexity failed for ${config}. See $local_log"
                RESULT_PPL["${config}:${chunks}"]="N/A"
                continue
            fi

            ppl=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$local_log" | grep -oE '[0-9.]+$' || echo "N/A")
            RESULT_PPL["${config}:${chunks}"]="$ppl"

            if [[ -z "${RESULT_CELLS[$config]:-}" ]]; then
                kv_line=$(grep 'llama_kv_cache: size' "$local_log" | tail -1 || echo "")
                if [[ -n "$kv_line" ]]; then
                    if [[ "$MODE" == "docker" ]]; then
                        cells_per_seq=$(echo "$kv_line" | grep -oP '\(\s*\K\d+(?=\s+cells)')
                        n_seq=$(echo "$kv_line" | grep -oP '\d+(?=/\d+ seqs)')
                        RESULT_CELLS[$config]=$(( cells_per_seq * n_seq ))
                        RESULT_K_MIB[$config]=$(echo "$kv_line" | grep -oP 'K \([^)]+\):\s*\K[\d.]+')
                        RESULT_V_MIB[$config]=$(echo "$kv_line" | grep -oP 'V \([^)]+\):\s*\K[\d.]+')
                    else
                        kv_total=$(echo "$kv_line" | grep -oE 'size = +[0-9.]+' | grep -oE '[0-9.]+')
                        kv_cells=$(echo "$kv_line" | grep -oE '[0-9]+ cells' | grep -oE '[0-9]+')
                        kv_seqs=$(echo "$kv_line" | grep -oE '[0-9]+/[0-9]+ seqs' | grep -oE '^[0-9]+')
                        RESULT_CELLS[$config]="$kv_cells"
                        # Approximate K/V split from total (local mode doesn't always show K/V separately)
                        RESULT_K_MIB[$config]=$(awk "BEGIN { printf \"%.2f\", $kv_total / 2 }")
                        RESULT_V_MIB[$config]=$(awk "BEGIN { printf \"%.2f\", $kv_total / 2 }")
                    fi
                fi
            fi

            echo "        PPL: ${ppl}"
        done
    fi

    # ── CPU Perplexity ──────────────────────────────────────────────
    if $DO_PERPLEXITY && $DO_CPU; then
        for chunks in "${CHUNKS_LIST[@]}"; do
            echo "${progress} K=${TYPE_K} V=${TYPE_V} — CPU perplexity (${chunks} chunks)..."

            local_log="$RESULTS_DIR/${config}-perplexity-cpu-${chunks}ch.log"

            if ! run_cmd \
                $BIN_PERPLEXITY \
                --model "$MODEL_PATH" \
                --file "$DATASET_PATH" \
                --chunks "$chunks" \
                --flash-attn on \
                --gpu-layers 0 \
                $(cache_flags_perplexity) \
                > "$local_log" 2>&1; then
                echo "ERROR: CPU perplexity failed for ${config}."
                RESULT_PPL_CPU["${config}:${chunks}"]="N/A"
                continue
            fi

            ppl_cpu=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$local_log" | grep -oE '[0-9.]+$' || echo "N/A")
            RESULT_PPL_CPU["${config}:${chunks}"]="$ppl_cpu"

            echo "        CPU PPL: ${ppl_cpu}"
        done
    fi

    # ── GPU Throughput ──────────────────────────────────────────────
    if $DO_BENCH && $DO_GPU; then
        echo "${progress} K=${TYPE_K} V=${TYPE_V} — ${LABEL} bench (pp${PP} + tg${TG}, ${REPEATS}x)..."

        local_log="$RESULTS_DIR/${config}-bench-gpu.jsonl"

        if ! run_cmd \
            $BIN_BENCH \
            -m "$MODEL_PATH" \
            -fa 1 \
            -ngl $GPU_LAYERS \
            -p "$PP" \
            -n "$TG" \
            -r "$REPEATS" \
            -o jsonl \
            $(cache_flags_bench) \
            > "$local_log" 2>&1; then
            echo "ERROR: bench failed for ${config}. See $local_log"
            RESULT_PP[$config]="N/A"
            RESULT_TG[$config]="N/A"
            continue
        fi

        pp_ts=$(grep -oE "\"n_prompt\": ${PP}[^}]*\"avg_ts\": [0-9.]+" "$local_log" | grep -oE '[0-9.]+$' || echo "N/A")
        tg_ts=$(grep -oE "\"n_gen\": ${TG}[^}]*\"avg_ts\": [0-9.]+" "$local_log" | grep -oE '[0-9.]+$' || echo "N/A")

        RESULT_PP[$config]="$pp_ts"
        RESULT_TG[$config]="$tg_ts"

        echo "        pp${PP}: ${pp_ts} t/s | tg${TG}: ${tg_ts} t/s"
    fi

    # ── CPU Throughput ──────────────────────────────────────────────
    if $DO_BENCH && $DO_CPU; then
        echo "${progress} K=${TYPE_K} V=${TYPE_V} — CPU bench (pp${PP} + tg${TG})..."

        local_log="$RESULTS_DIR/${config}-bench-cpu.jsonl"

        if ! run_cmd \
            $BIN_BENCH \
            -m "$MODEL_PATH" \
            -fa 1 \
            -ngl 0 \
            -p "$PP" \
            -n "$TG" \
            -r "$REPEATS" \
            -o jsonl \
            $(cache_flags_bench) \
            > "$local_log" 2>&1; then
            echo "ERROR: CPU bench failed for ${config}."
            RESULT_PP_CPU[$config]="N/A"
            RESULT_TG_CPU[$config]="N/A"
            continue
        fi

        pp_ts_cpu=$(grep -oE "\"n_prompt\": ${PP}[^}]*\"avg_ts\": [0-9.]+" "$local_log" | grep -oE '[0-9.]+$' || echo "N/A")
        tg_ts_cpu=$(grep -oE "\"n_gen\": ${TG}[^}]*\"avg_ts\": [0-9.]+" "$local_log" | grep -oE '[0-9.]+$' || echo "N/A")

        RESULT_PP_CPU[$config]="$pp_ts_cpu"
        RESULT_TG_CPU[$config]="$tg_ts_cpu"

        echo "        CPU: pp${PP}: ${pp_ts_cpu} t/s | tg${TG}: ${tg_ts_cpu} t/s"
    fi
done

# ── Results Table ────────────────────────────────────────────────────────

echo ""
echo "  <-- ${LABEL}   ${MODEL_NAME} — ${MODEL_PATH}"
echo "  ${CHUNKS_LIST[*]} chunks — $(timestamp)"
echo ""

declare -a TABLE_ROWS=()

for config in "${CONFIGS[@]}"; do
    last_chunks="${CHUNKS_LIST[${#CHUNKS_LIST[@]}-1]}"
    has_ppl="${RESULT_PPL["${config}:${last_chunks}"]:-}"
    has_ppl_cpu="${RESULT_PPL_CPU["${config}:${last_chunks}"]:-}"
    has_gpu="${RESULT_PP[$config]:-}"
    has_cpu="${RESULT_PP_CPU[$config]:-}"
    if [[ -z "$has_ppl" && -z "$has_ppl_cpu" && -z "$has_gpu" && -z "$has_cpu" ]]; then
        continue
    fi
    config_types "$config"

    k_gib_fmt="-"
    v_gib_fmt="-"
    total_fmt="-"
    if [[ -n "${RESULT_CELLS[$config]:-}" && -n "${RESULT_K_MIB[$config]:-}" ]]; then
        k_gib=$(awk "BEGIN { printf \"%.2f\", ${RESULT_K_MIB[$config]} / ${RESULT_CELLS[$config]} * 100000 / 1024 }")
        v_gib=$(awk "BEGIN { printf \"%.2f\", ${RESULT_V_MIB[$config]} / ${RESULT_CELLS[$config]} * 100000 / 1024 }")
        total=$(awk "BEGIN { printf \"%.2f\", $k_gib + $v_gib }")
        k_gib_fmt="$k_gib"
        v_gib_fmt="$v_gib"
        total_fmt="$total"
    fi

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

    ppl_cpu_fmt="-"
    ppl_cpu_raw="${RESULT_PPL_CPU["${config}:${last_chunks}"]:-}"
    if [[ -n "$ppl_cpu_raw" && "$ppl_cpu_raw" != "N/A" ]]; then
        ppl_cpu_fmt=$(awk "BEGIN { printf \"%.2f\", $ppl_cpu_raw }")
        if [[ "$sort_key" == "9999.9999" ]]; then
            sort_key=$(printf "%010.4f" "$ppl_cpu_raw")
        fi
        if [[ "$delta_fmt" == "-" && "$config" != "f16" ]]; then
            f16_cpu_raw="${RESULT_PPL_CPU["f16:${last_chunks}"]:-}"
            if [[ -n "$f16_cpu_raw" && "$f16_cpu_raw" != "N/A" ]]; then
                delta_fmt=$(awk "BEGIN { printf \"%+.2f\", $ppl_cpu_raw - $f16_cpu_raw }")
            fi
        fi
    fi

    pp_fmt="-"
    tg_fmt="-"
    pp_raw="${RESULT_PP[$config]:-}"
    tg_raw="${RESULT_TG[$config]:-}"
    if [[ -n "$pp_raw" && "$pp_raw" != "N/A" ]]; then
        pp_fmt=$(printf "%'.0f" "${pp_raw%.*}")
    fi
    if [[ -n "$tg_raw" && "$tg_raw" != "N/A" ]]; then
        tg_fmt=$(printf "%'.1f" "$tg_raw")
    fi

    pp_cpu_fmt="-"
    tg_cpu_fmt="-"
    pp_cpu_raw="${RESULT_PP_CPU[$config]:-}"
    tg_cpu_raw="${RESULT_TG_CPU[$config]:-}"
    if [[ -n "$pp_cpu_raw" && "$pp_cpu_raw" != "N/A" ]]; then
        pp_cpu_fmt=$(printf "%'.0f" "${pp_cpu_raw%.*}")
    fi
    if [[ -n "$tg_cpu_raw" && "$tg_cpu_raw" != "N/A" ]]; then
        tg_cpu_fmt=$(printf "%'.1f" "$tg_cpu_raw")
    fi

    TABLE_ROWS+=("${sort_key}|${TYPE_K}|${TYPE_V}|${k_gib_fmt}|${v_gib_fmt}|${total_fmt}|${ppl_fmt}|${ppl_cpu_fmt}|${delta_fmt}|${pp_fmt}|${tg_fmt}|${pp_cpu_fmt}|${tg_cpu_fmt}")
done

IFS=$'\n' SORTED_ROWS=($(printf '%s\n' "${TABLE_ROWS[@]}" | sort -t'|' -k1,1n)); unset IFS

if [[ "$TABLE_MODE" != "none" && ${#SORTED_ROWS[@]} -gt 0 ]]; then
    table_top
    table_hdr
    table_sep
    first=true
    for entry in "${SORTED_ROWS[@]}"; do
        if $first; then first=false; else table_sep; fi
        IFS='|' read -r _sort_key tk tv k_gib v_gib total ppl ppl_cpu delta pp tg pp_cpu tg_cpu <<< "$entry"
        case "$TABLE_MODE" in
            full) table_row "$tk" "$tv" "$k_gib" "$v_gib" "$total" "$ppl" "$ppl_cpu" "$delta" "$pp" "$tg" "$pp_cpu" "$tg_cpu" ;;
            gpu)  table_row "$tk" "$tv" "$k_gib" "$v_gib" "$total" "$ppl" "$delta" "$pp" "$tg" ;;
            cpu)  table_row "$tk" "$tv" "$k_gib" "$v_gib" "$total" "$ppl_cpu" "$delta" "$pp_cpu" "$tg_cpu" ;;
        esac
    done
    table_bot
fi

echo ""
echo "  Memory: GiB per 100K tokens (K + V cache)"
echo "  Raw logs: $RESULTS_DIR/"
echo ""

done  # end per-model loop
