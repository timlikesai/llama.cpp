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
    # ── MXFP4 MOE models (primary targets) ──
    "qwen3-coder|/models/spectralyst/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE-GGUF/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE.gguf"
    "qwen3-coder-next|/models/noctrex/Qwen3-Coder-Next-MXFP4_MOE-GGUF/Qwen3-Coder-Next-MXFP4_MOE.gguf"
    "qwen35|/models/noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF/Qwen3.5-35B-A3B-MXFP4_MOE_F16.gguf"
    "gpt-oss-20b|/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
    "nemotron-nano|/models/noctrex/Nemotron-3-Nano-30B-A3B-MXFP4_MOE-GGUF/NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP4_MOE.gguf"
    "glm-4.7-flash|/models/noctrex/GLM-4.7-Flash-MXFP4_MOE-GGUF/GLM-4.7-Flash-MXFP4_MOE.gguf"
    "glm-4.7-flash-i1-xl|/models/noctrex/GLM-4.7-Flash-i1-MXFP4_MOE_XL-exp-GGUF/GLM-4.7-Flash-i1-MXFP4_MOE_XL-exp.gguf"
    # ── Standard quant models ──
    "nemotron-nano-q4|/models/unsloth/Nemotron-3-Nano-30B-A3B-GGUF/Nemotron-3-Nano-30B-A3B-Q4_1.gguf"
    "glm-4.7-flash-q8|/models/lmstudio-community/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q8_0.gguf"
    "glm-4.7-flash-q4|/models/unsloth/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_1.gguf"
    "devstral-24b|/models/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-Q4_1.gguf"
    "granite-tiny|/models/lmstudio-community/granite-4.0-h-tiny-GGUF/granite-4.0-h-tiny-Q8_0.gguf"
    "gemma-3n|/models/unsloth/gemma-3n-E4B-it-GGUF/gemma-3n-E4B-it-Q8_0.gguf"
    "qwen3-4b-math|/models/mradermacher/Qwen3-4B-math-GGUF/Qwen3-4B-math.Q8_0.gguf"
    "lfm2.5-thinking|/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q8_0.gguf"
    "lfm2.5-instruct|/models/unsloth/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf"
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
CHUNKS_LIST=(16)
# All MXFP configs use V=mxfp4 (K dominates quality; avoids cartesian explosion).
# "mxfp8" = E4M3 (default), "mxfp6" = E2M3 (default). Full names also accepted.
CONFIGS=(
    "f16"
    "q8_0"
    "q4_0"
    "q8_0+q4_0"
    "mxfp8"
    "mxfp8_e5m2"
    "mxfp6"
    "mxfp6_e3m2"
    "mxfp4"
)
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
            echo "  --chunks N[,M]      Chunk counts for perplexity, comma-separated (default: 16)"
            echo "  --config NAME       Single config (default: all). Available:"
            echo "                        f16, q8_0, q4_0, q8_0+q4_0"
            echo "                        mxfp8, mxfp8_e5m2, mxfp6, mxfp6_e3m2, mxfp4"
            echo "  --model NAME|PATH   Model preset or path (default: $DEFAULT_MODEL)"
            echo "  --help              Show this help"
            echo ""
            echo "Models:"
            for entry in "${MODELS[@]}"; do
                echo "  ${entry%%|*}  → ${entry#*|}"
            done
            echo "  (or pass a raw GGUF path)"
            echo ""
            echo "Configs (baselines):"
            echo "  f16         F16 K+V (baseline)"
            echo "  q8_0        Q8_0 K+V"
            echo "  q4_0        Q4_0 K+V"
            echo "  q8_0+q4_0   Q8_0 K + Q4_0 V"
            echo ""
            echo "Configs (MXFP — all use V=mxfp4, K determines quality):"
            echo "  mxfp8       MXFP8 E4M3 K + MXFP4 V (8-bit, Hadamard) [default mxfp8]"
            echo "  mxfp8_e5m2  MXFP8 E5M2 K + MXFP4 V (8-bit, Hadamard)"
            echo "  mxfp6       MXFP6 E2M3 K + MXFP4 V (6-bit, Hadamard) [default mxfp6]"
            echo "  mxfp6_e3m2  MXFP6 E3M2 K + MXFP4 V (6-bit, Hadamard)"
            echo "  mxfp4       MXFP4 E2M1 K+V          (4-bit, Hadamard)"
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

# MLA models: V shares K buffer, so V type = K type (V arg is ignored by the model).
# For these models, MXFP configs use matched K/V instead of V=mxfp4.
MLA_MODELS=("glm-4.7-flash" "glm-4.7-flash-i1-xl" "glm-4.7-flash-q8" "glm-4.7-flash-q4")

is_mla_model() {
    for m in "${MLA_MODELS[@]}"; do
        [[ "$MODEL_NAME" == "$m" ]] && return 0
    done
    return 1
}

# Map config name → display names (TYPE_K/TYPE_V) and CLI args (CLI_K/CLI_V).
config_types() {
    case "$1" in
        f16)          TYPE_K="f16";         TYPE_V="f16";    CLI_K="f16";       CLI_V="f16"       ;;
        q8_0)         TYPE_K="q8_0";        TYPE_V="q8_0";   CLI_K="q8_0";      CLI_V="q8_0"      ;;
        q4_0)         TYPE_K="q4_0";        TYPE_V="q4_0";   CLI_K="q4_0";      CLI_V="q4_0"      ;;
        q8_0+q4_0)    TYPE_K="q8_0";        TYPE_V="q4_0";   CLI_K="q8_0";      CLI_V="q4_0"      ;;
        mxfp8|mxfp8_e4m3)   TYPE_K="mxfp8";       TYPE_V="mxfp4";  CLI_K="mxfp8";     CLI_V="mxfp4"     ;;
        mxfp8_e5m2)         TYPE_K="mxfp8_e5m2";  TYPE_V="mxfp4";  CLI_K="mxfp8_e5m2"; CLI_V="mxfp4"     ;;
        mxfp6|mxfp6_e2m3)   TYPE_K="mxfp6_e2m3";  TYPE_V="mxfp4";  CLI_K="mxfp6";      CLI_V="mxfp4"     ;;
        mxfp6_e3m2)         TYPE_K="mxfp6_e3m2";  TYPE_V="mxfp4";  CLI_K="mxfp6_e3m2"; CLI_V="mxfp4"     ;;
        mxfp4)        TYPE_K="mxfp4";        TYPE_V="mxfp4";  CLI_K="mxfp4";     CLI_V="mxfp4"     ;;
        *)
            echo "Unknown config: $1"
            exit 1
            ;;
    esac

    # MLA models: V shares K buffer, override V to match K.
    if is_mla_model; then
        TYPE_V="$TYPE_K"
        CLI_V="$CLI_K"
    fi
}

# Build cache-type flags for llama-perplexity (--cache-type-k/v, skip if f16).
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

# Build cache-type flags for llama-bench (-ctk/-ctv, skip if f16).
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

is_config_supported() {
    # All configs are supported for all models now (MLA uses matched K/V).
    return 0
}

# Count supported configs for progress display.
total_configs=0
for config in "${CONFIGS[@]}"; do
    is_config_supported "$config" && (( total_configs++ )) || true
done
config_num=0

for config in "${CONFIGS[@]}"; do
    if ! is_config_supported "$config"; then
        echo "  Skipping $config — not supported for $MODEL_NAME (D > 256, V=mxfp4 unsupported)"
        continue
    fi
    (( ++config_num ))
    config_types "$config"
    progress="[${config_num}/${total_configs}]"

    # ── Perplexity ───────────────────────────────────────────────────────
    if $DO_PERPLEXITY; then
        for chunks in "${CHUNKS_LIST[@]}"; do
            echo "${progress} K=${TYPE_K} V=${TYPE_V} — perplexity (${chunks} chunks)..."

            local_log="$RESULTS_DIR/${config}-perplexity-${chunks}ch.log"

            if ! docker_run \
                /app/llama-perplexity \
                --model "$MODEL_PATH" \
                --file "/datasets/$DATASET_FILE" \
                --chunks "$chunks" \
                --flash-attn on \
                --gpu-layers 99 \
                $(cache_flags_perplexity) \
                > "$local_log" 2>&1; then
                echo "ERROR: perplexity failed for ${config}. Log:"
                cat "$local_log"
                exit 1
            fi

            # Parse PPL from "Final estimate: PPL = X.XXXX".
            ppl=$(grep --perl-regexp --only-matching 'Final estimate: PPL = \K[\d.]+' "$local_log" || echo "N/A")
            RESULT_PPL["${config}:${chunks}"]="$ppl"

            # Parse KV cache memory (only need once per config).
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

    # ── Throughput ───────────────────────────────────────────────────────
    if $DO_BENCH; then
        echo "${progress} K=${TYPE_K} V=${TYPE_V} — bench (pp512 + tg128)..."

        local_log="$RESULTS_DIR/${config}-bench.jsonl"

        if ! docker_run \
            /app/llama-bench \
            -m "$MODEL_PATH" \
            -fa 1 \
            -ngl 999 \
            -p 512 \
            -n 128 \
            -o jsonl \
            $(cache_flags_bench) \
            > "$local_log" 2>&1; then
            echo "ERROR: bench failed for ${config}. Log:"
            cat "$local_log"
            exit 1
        fi

        # Parse JSONL: "n_prompt": 512 → pp, "n_gen": 128 → tg. Field: "avg_ts".
        pp_ts=$(grep --perl-regexp --only-matching '"n_prompt": 512.*?"avg_ts": \K[\d.]+' "$local_log" || echo "N/A")
        tg_ts=$(grep --perl-regexp --only-matching '"n_gen": 128.*?"avg_ts": \K[\d.]+' "$local_log" || echo "N/A")

        RESULT_PP512[$config]="$pp_ts"
        RESULT_TG128[$config]="$tg_ts"

        echo "        pp512: ${pp_ts} t/s | tg128: ${tg_ts} t/s"
    fi
done

# ── Results Table ────────────────────────────────────────────────────────────

echo ""
echo "  $MODEL_NAME — ${CHUNKS_LIST[*]} chunks — $(timestamp)"
echo ""

# Column widths: K(10) V(7) K_GiB(6) V_GiB(6) Total(6) PPL(6) delta(6) pp512(6) tg128(6)
table_top()  { echo "  ┌────────────┬─────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐"; }
table_hdr()  { printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │\n" \
                      "K type" "V type" "K GiB" "V GiB" "Total" "PPL" "Δ F16" "pp512" "tg128"; }
table_sep()  { echo "  ├────────────┼─────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤"; }
table_row()  { printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │ %6s │\n" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"; }
table_bot()  { echo "  └────────────┴─────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘"; }

# Build rows with sortable PPL prefix, then sort by PPL ascending.
declare -a TABLE_ROWS=()

for config in "${CONFIGS[@]}"; do
    # Only include configs that actually produced results (PPL or bench).
    last_chunks="${CHUNKS_LIST[${#CHUNKS_LIST[@]}-1]}"
    has_ppl="${RESULT_PPL["${config}:${last_chunks}"]:-}"
    has_bench="${RESULT_PP512[$config]:-}"
    if [[ -z "$has_ppl" && -z "$has_bench" ]]; then
        continue
    fi
    config_types "$config"

    # Memory per 100K tokens (K, V, total — all in GiB).
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

    # PPL (use last chunk count for the table).
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

    # Store with sort key prefix (pipe-separated).
    TABLE_ROWS+=("${sort_key}|${TYPE_K}|${TYPE_V}|${k_gib_fmt}|${v_gib_fmt}|${total_fmt}|${ppl_fmt}|${delta_fmt}|${pp_fmt}|${tg_fmt}")
done

# Sort rows by PPL (first field) ascending.
IFS=$'\n' SORTED_ROWS=($(printf '%s\n' "${TABLE_ROWS[@]}" | sort -t'|' -k1,1n)); unset IFS

table_top
table_hdr
table_sep
first=true
for entry in "${SORTED_ROWS[@]}"; do
    if $first; then first=false; else table_sep; fi
    # Strip sort key, split fields.
    IFS='|' read -r _sort_key tk tv k_gib v_gib total ppl delta pp tg <<< "$entry"
    table_row "$tk" "$tv" "$k_gib" "$v_gib" "$total" "$ppl" "$delta" "$pp" "$tg"
done
table_bot

echo ""
echo "  Memory: GiB per 100K tokens (K + V cache)"
echo "  Raw logs: $RESULTS_DIR/"
echo ""
