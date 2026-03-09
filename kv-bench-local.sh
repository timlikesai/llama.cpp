#!/bin/zsh
#
# KV Cache Quantization Benchmark — Local Mac (Metal) variant of kv-bench.sh
#
# Same configs, same table output, runs local build instead of Docker.
#
# Always runs ALL configs with BOTH perplexity AND throughput. No skipping.
#
# Usage:
#   ./kv-bench-local.sh                     # Full bench (all configs)
#   ./kv-bench-local.sh --chunks 2          # Quick perplexity (2 chunks)
#   ./kv-bench-local.sh --model qwen3-coder # Specific model
#   ./kv-bench-local.sh --repeats 5         # More repeats for bench
#   ./kv-bench-local.sh --help              # Show usage

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR="${0:A:h}"
BUILD_DIR="$SCRIPT_DIR/build/bin"
DATASETS_DIR="$SCRIPT_DIR/datasets"
DATASET_FILE="wikitext-2-raw/wiki.test.raw"

LMSTUDIO="$HOME/.lmstudio/models"
typeset -A MODEL_MAP
MODEL_MAP=(
    qwen3-coder "$LMSTUDIO/spectralyst/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE-GGUF/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE.gguf"
    qwen35      "$LMSTUDIO/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q8_0.gguf"
    gpt-oss-20b "$LMSTUDIO/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf"
    gemma-3n-text "$LMSTUDIO/lmstudio-community/gemma-3n-E4B-it-text-GGUF/gemma-3n-E4B-it-Q8_0.gguf"
)
DEFAULT_MODEL="qwen3-coder"

resolve_model() {
    if [[ -n "${MODEL_MAP[$1]:-}" ]]; then
        MODEL_NAME="$1"
        MODEL_PATH="${MODEL_MAP[$1]}"
    else
        MODEL_PATH="$1"
        MODEL_NAME="$(basename "$(dirname "$1")")"
    fi
}

CHUNKS=16
REPEATS=3
PP=512
TG=128
CONFIGS=(f16 q8_0 q8_0+q4_0 mxfp8_e4m3 mxfp8_e5m2 mxfp6_e2m3 mxfp6_e3m2 mxfp4)
MODEL_INPUTS=()

# ── Argument Parsing ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunks)           CHUNKS="$2"; shift 2 ;;
        --repeats|-r)       REPEATS="$2"; shift 2 ;;
        --pp)               PP="$2"; shift 2 ;;
        --tg)               TG="$2"; shift 2 ;;
        --model)            MODEL_INPUTS+=("$2"); shift 2 ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Local Mac (Metal) KV Cache Benchmark — mirrors kv-bench.sh"
            echo ""
            echo "Options:"
            echo "  --chunks N          Chunk count for perplexity (default: 16)"
            echo "  --repeats N         Bench repeats (default: 3)"
            echo "  --pp N              Prompt size for bench (default: 512)"
            echo "  --tg N              Token gen count for bench (default: 128)"
            echo "  --model NAME|PATH   Model preset or path (default: $DEFAULT_MODEL)"
            echo "  --help              Show this help"
            echo ""
            echo "Models:"
            for k in ${(k)MODEL_MAP}; do
                echo "  $k → ${MODEL_MAP[$k]}"
            done
            exit 0
            ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ${#MODEL_INPUTS[@]} -eq 0 ]]; then
    MODEL_INPUTS=("$DEFAULT_MODEL")
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

config_types() {
    case "$1" in
        f16)                TYPE_K="f16";        TYPE_V="f16";   CLI_K="f16";       CLI_V="f16"       ;;
        q8_0)               TYPE_K="q8_0";       TYPE_V="q8_0";  CLI_K="q8_0";      CLI_V="q8_0"      ;;
        q4_0)               TYPE_K="q4_0";       TYPE_V="q4_0";  CLI_K="q4_0";      CLI_V="q4_0"      ;;
        q8_0+q4_0)          TYPE_K="q8_0";       TYPE_V="q4_0";  CLI_K="q8_0";      CLI_V="q4_0"      ;;
        mxfp8|mxfp8_e4m3)   TYPE_K="mxfp8_e4m3"; TYPE_V="mxfp4"; CLI_K="mxfp8";     CLI_V="mxfp4"     ;;
        mxfp8_e5m2)          TYPE_K="mxfp8_e5m2"; TYPE_V="mxfp4"; CLI_K="mxfp8_e5m2"; CLI_V="mxfp4"   ;;
        mxfp6|mxfp6_e2m3)   TYPE_K="mxfp6_e2m3"; TYPE_V="mxfp4"; CLI_K="mxfp6";     CLI_V="mxfp4"     ;;
        mxfp6_e3m2)          TYPE_K="mxfp6_e3m2"; TYPE_V="mxfp4"; CLI_K="mxfp6_e3m2"; CLI_V="mxfp4"   ;;
        mxfp4)               TYPE_K="mxfp4";      TYPE_V="mxfp4"; CLI_K="mxfp4";     CLI_V="mxfp4"     ;;
        *)  echo "Unknown config: $1"; exit 1 ;;
    esac
}

# ── Preflight ────────────────────────────────────────────────────────────────

if [[ ! -x "$BUILD_DIR/llama-bench" ]]; then
    echo "ERROR: $BUILD_DIR/llama-bench not found. Run: cmake --build build -j"
    exit 1
fi

if [[ ! -f "$DATASETS_DIR/$DATASET_FILE" ]]; then
    echo "ERROR: Dataset not found at $DATASETS_DIR/$DATASET_FILE"
    echo "       Perplexity is mandatory. Download the dataset first."
    exit 1
fi

# ── Per-Model Loop ───────────────────────────────────────────────────────────

for MODEL_INPUT in "${MODEL_INPUTS[@]}"; do

resolve_model "$MODEL_INPUT"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  ${MODEL_NAME} — Local Metal Bench"
echo "  ${MODEL_PATH}"
echo "  $(timestamp)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

RESULTS_DIR="$SCRIPT_DIR/test-results/kv-bench-local-${MODEL_NAME}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Storage: parallel arrays (zsh-compatible, no bash 4+ needed)
typeset -A R_PP R_TG R_PPL R_KV100K
f16_tg=""

total=${#CONFIGS[@]}
num=0

for config in "${CONFIGS[@]}"; do
    (( num += 1 ))
    config_types "$config"

    # Build cache flags
    local bench_flags=()
    [[ "$CLI_K" != "f16" ]] && bench_flags+=(-ctk "$CLI_K")
    [[ "$CLI_V" != "f16" ]] && bench_flags+=(-ctv "$CLI_V")

    # ── Perplexity ─────────────────────────────────────────────────
    echo "[${num}/${total}] K=${TYPE_K} V=${TYPE_V} — perplexity (${CHUNKS} chunks)..."

    local ppl_log="$RESULTS_DIR/${config}-perplexity.log"
    local ppl_flags=()
    [[ "$CLI_K" != "f16" ]] && ppl_flags+=(--cache-type-k "$CLI_K")
    [[ "$CLI_V" != "f16" ]] && ppl_flags+=(--cache-type-v "$CLI_V")

    if "$BUILD_DIR/llama-perplexity" \
        --model "$MODEL_PATH" \
        --file "$DATASETS_DIR/$DATASET_FILE" \
        --chunks "$CHUNKS" \
        --flash-attn on \
        --gpu-layers 999 \
        "${ppl_flags[@]}" \
        > "$ppl_log" 2>&1; then
        ppl=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$ppl_log" | grep -oE '[0-9.]+$' || echo "N/A")
        R_PPL[$config]="$ppl"
        echo "        PPL: ${ppl}"

        # Extract KV cache size → scale to 100k tokens, single sequence
        local kv_line=$(grep 'llama_kv_cache: size =' "$ppl_log" || echo "")
        if [[ -n "$kv_line" ]]; then
            local kv_total=$(echo "$kv_line" | grep -oE 'size = +[0-9.]+' | grep -oE '[0-9.]+')
            local kv_cells=$(echo "$kv_line" | grep -oE '[0-9]+ cells' | grep -oE '[0-9]+')
            local kv_seqs=$(echo "$kv_line" | grep -oE '[0-9]+/[0-9]+ seqs' | grep -oE '^[0-9]+')
            R_KV100K[$config]=$(awk "BEGIN { printf \"%.1f\", $kv_total * (100000 / $kv_cells) / $kv_seqs / 1024 }")
        else
            R_KV100K[$config]="N/A"
        fi
    else
        echo "        ERROR (see $ppl_log)"
        R_PPL[$config]="N/A"
        R_KV100K[$config]="N/A"
    fi

    # ── Throughput ─────────────────────────────────────────────────
    echo "[${num}/${total}] K=${TYPE_K} V=${TYPE_V} — bench (pp${PP} + tg${TG}, ${REPEATS}x)..."

    local bench_jsonl="$RESULTS_DIR/${config}-bench.jsonl"
    local bench_log="$RESULTS_DIR/${config}-bench.log"

    "$BUILD_DIR/llama-bench" \
        -m "$MODEL_PATH" \
        -fa 1 \
        -ngl 999 \
        -p "$PP" \
        -n "$TG" \
        -r "$REPEATS" \
        "${bench_flags[@]}" \
        -o jsonl \
        > "$bench_jsonl" 2>"$bench_log"

    pp_ts=$(grep -oE "\"n_prompt\": ${PP}[^}]*\"avg_ts\": [0-9.]+" "$bench_jsonl" | grep -oE '[0-9.]+$' || echo "N/A")
    tg_ts=$(grep -oE "\"n_gen\": ${TG}[^}]*\"avg_ts\": [0-9.]+" "$bench_jsonl" | grep -oE '[0-9.]+$' || echo "N/A")

    R_PP[$config]="$pp_ts"
    R_TG[$config]="$tg_ts"

    [[ "$config" == "f16" && "$tg_ts" != "N/A" ]] && f16_tg="$tg_ts"

    echo "        pp512: ${pp_ts} t/s | tg128: ${tg_ts} t/s"
done

# ── Results Table ────────────────────────────────────────────────────────

echo ""
echo "  ${MODEL_NAME} — $(timestamp)"
echo ""

echo "  ┌────────────┬─────────┬────────┬────────┬────────┬────────┬────────┬─────────┐"
printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %6s │ %6s │ %7s │\n" "K type" "V type" "PPL" "Δ F16" "pp${PP}" "tg${TG}" "tg%f16" "KV@100k"
echo "  ├────────────┼─────────┼────────┼────────┼────────┼────────┼────────┼─────────┤"

first=true
for config in "${CONFIGS[@]}"; do
    config_types "$config"

    pp_raw="${R_PP[$config]:-N/A}"
    tg_raw="${R_TG[$config]:-N/A}"

    pp_fmt="-"
    tg_fmt="-"
    tg_pct="-"

    if [[ "$pp_raw" != "N/A" ]]; then
        pp_fmt=$(printf "%'.0f" "${pp_raw%.*}")
    fi
    if [[ "$tg_raw" != "N/A" ]]; then
        tg_fmt=$(printf "%.1f" "$tg_raw")
        if [[ -n "$f16_tg" ]]; then
            tg_pct=$(awk "BEGIN { printf \"%.1f\", $tg_raw / $f16_tg * 100 }")
        fi
    fi

    kv_raw="${R_KV100K[$config]:-N/A}"
    kv_fmt="-"
    if [[ "$kv_raw" != "N/A" ]]; then
        kv_fmt="${kv_raw} G"
    fi

    if $first; then first=false; else
        echo "  ├────────────┼─────────┼────────┼────────┼────────┼────────┼────────┼─────────┤"
    fi

    ppl_raw="${R_PPL[$config]:-N/A}"
    ppl_fmt="-"
    delta_fmt="-"
    if [[ "$ppl_raw" != "N/A" ]]; then
        ppl_fmt=$(awk "BEGIN { printf \"%.2f\", $ppl_raw }")
        f16_ppl="${R_PPL[f16]:-}"
        if [[ "$config" != "f16" && -n "$f16_ppl" && "$f16_ppl" != "N/A" ]]; then
            delta_fmt=$(awk "BEGIN { printf \"%+.2f\", $ppl_raw - $f16_ppl }")
        fi
    fi
    printf "  │ %-10s │ %-7s │ %6s │ %6s │ %6s │ %6s │ %5s%% │ %7s │\n" "$TYPE_K" "$TYPE_V" "$ppl_fmt" "$delta_fmt" "$pp_fmt" "$tg_fmt" "$tg_pct" "$kv_fmt"
done

echo "  └────────────┴─────────┴────────┴────────┴────────┴────────┴────────┴─────────┘"

echo ""
echo "  Raw logs: $RESULTS_DIR/"
echo ""

done  # end per-model loop
