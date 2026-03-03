#!/bin/bash
#
# VEC Kernel Iteration Tool
#
# One-command build → bench → SASS cycle for flash attention VEC optimization.
# Tracks iteration history so you can see progress across changes.
#
# Usage:
#   ./vec-iterate.sh --set-baseline     # Build, bench, SASS → save as baseline
#   ./vec-iterate.sh                    # Build, bench, SASS → compare to baseline
#   ./vec-iterate.sh --bench-only       # Skip SASS extraction (faster)
#   ./vec-iterate.sh --sass-only        # Skip bench (SASS comparison only)
#   ./vec-iterate.sh --config mxfp4     # Test specific config (default: mxfp6)
#   ./vec-iterate.sh --history          # Show all iteration results
#   ./vec-iterate.sh --help             # Show usage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/test-results/vec-iterations"
HISTORY_FILE="$RESULTS_DIR/history.tsv"

# Defaults.
SET_BASELINE=false
DO_BENCH=true
DO_SASS=true
CONFIG="mxfp6"
CONFIGS_TO_BENCH=("mxfp6" "mxfp4" "mxfp8" "f16")
SKIP_BUILD=false

# ── Argument Parsing ────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --set-baseline)
            SET_BASELINE=true
            shift
            ;;
        --bench-only)
            DO_SASS=false
            shift
            ;;
        --sass-only)
            DO_BENCH=false
            shift
            ;;
        --config)
            CONFIGS_TO_BENCH=("$2")
            CONFIG="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --history)
            if [[ -f "$HISTORY_FILE" ]]; then
                echo ""
                echo "═══ VEC Iteration History ═══"
                echo ""
                printf "  %-8s  %-12s  %-10s  %7s  %7s  %7s  %s\n" \
                    "Iter" "Config" "Commit" "pp512" "tg128" "SASS" "Notes"
                echo "  ────────────────────────────────────────────────────────────────────────────"
                while IFS=$'\t' read -r iter config commit pp512 tg128 sass notes; do
                    printf "  %-8s  %-12s  %-10s  %7s  %7s  %7s  %s\n" \
                        "$iter" "$config" "$commit" "$pp512" "$tg128" "$sass" "$notes"
                done < "$HISTORY_FILE"
            else
                echo "No history yet. Run ./vec-iterate.sh --set-baseline first."
            fi
            echo ""
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "VEC Kernel Iteration Tool — build, bench, SASS in one command"
            echo ""
            echo "Actions:"
            echo "  --set-baseline        Save current build as baseline (first run)"
            echo "  (no flag)             Build, bench, SASS, compare to baseline"
            echo ""
            echo "Options:"
            echo "  --bench-only          Skip SASS extraction (faster iteration)"
            echo "  --sass-only           Skip benchmarks (SASS comparison only)"
            echo "  --config NAME         Test specific config (default: all MXFP + f16)"
            echo "  --skip-build          Skip docker build (use existing image)"
            echo "  --history             Show all iteration results"
            echo "  --help                Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# ── Build ───────────────────────────────────────────────────────────────────

if ! $SKIP_BUILD; then
    echo ""
    echo "═══ Building... ═══"
    cd "$SERVICES_DIR"
    docker compose build llama-cpp-ultra
    echo "Build complete."
fi

# ── Get commit info ─────────────────────────────────────────────────────────

cd "$SCRIPT_DIR"
COMMIT=$(git rev-parse --short HEAD)
DIRTY=""
if ! git diff --quiet 2>/dev/null; then
    DIRTY="+dirty"
fi
COMMIT_LABEL="${COMMIT}${DIRTY}"

# ── Determine iteration number ─────────────────────────────────────────────

if $SET_BASELINE; then
    ITER="base"
else
    # Count existing non-baseline iterations.
    existing=0
    if [[ -f "$HISTORY_FILE" ]]; then
        existing=$(grep --count --invert-match '^base' "$HISTORY_FILE" 2>/dev/null || echo "0")
    fi
    ITER="iter-$(( existing + 1 ))"
fi

# ── Benchmark ───────────────────────────────────────────────────────────────

declare -A BENCH_PP512
declare -A BENCH_TG128

if $DO_BENCH; then
    echo ""
    echo "═══ Benchmarking ($ITER, commit $COMMIT_LABEL) ═══"

    # Stop the service to free GPU memory.
    cd "$SERVICES_DIR"
    docker compose stop llama-cpp-ultra 2>/dev/null || true

    for cfg in "${CONFIGS_TO_BENCH[@]}"; do
        echo ""
        echo "  Config: $cfg"
        result=$("$SCRIPT_DIR/kv-bench.sh" --config "$cfg" --skip-perplexity --skip-build --chunks 2 2>&1)

        pp512=$(echo "$result" | grep --perl-regexp --only-matching 'pp512: \K[\d.]+' || echo "N/A")
        tg128=$(echo "$result" | grep --perl-regexp --only-matching 'tg128: \K[\d.]+' || echo "N/A")

        BENCH_PP512[$cfg]="$pp512"
        BENCH_TG128[$cfg]="$tg128"

        echo "    pp512: $pp512 t/s | tg128: $tg128 t/s"
    done

    # Show comparison to baseline if we have one.
    if [[ -f "$HISTORY_FILE" ]] && ! $SET_BASELINE; then
        echo ""
        echo "  ── vs Baseline ──"
        for cfg in "${CONFIGS_TO_BENCH[@]}"; do
            base_pp=$(grep "^base" "$HISTORY_FILE" | grep "$cfg" | cut -f4 | head -1)
            base_tg=$(grep "^base" "$HISTORY_FILE" | grep "$cfg" | cut -f5 | head -1)
            curr_pp="${BENCH_PP512[$cfg]}"
            curr_tg="${BENCH_TG128[$cfg]}"

            if [[ -n "$base_pp" && "$base_pp" != "N/A" && "$curr_pp" != "N/A" ]]; then
                pp_delta=$(awk "BEGIN { printf \"%+.1f%%\", ($curr_pp - $base_pp) / $base_pp * 100 }")
                tg_delta=$(awk "BEGIN { printf \"%+.1f%%\", ($curr_tg - $base_tg) / $base_tg * 100 }")
                printf "    %-12s  pp512: %s (%s)  tg128: %s (%s)\n" \
                    "$cfg" "$curr_pp" "$pp_delta" "$curr_tg" "$tg_delta"
            fi
        done
    fi
fi

# ── SASS ────────────────────────────────────────────────────────────────────

SASS_COUNT="N/A"

if $DO_SASS; then
    echo ""
    echo "═══ SASS Extraction ($ITER) ═══"

    if $SET_BASELINE; then
        "$SCRIPT_DIR/sass-diff.sh" --baseline
        SASS_COUNT=$(grep --count --extended-regexp '^\s+/\*[0-9a-f]+\*/' \
            "$SCRIPT_DIR/test-results/sass-diff/baseline-filtered.sass" 2>/dev/null || echo "0")
    else
        "$SCRIPT_DIR/sass-diff.sh" --current
        SASS_COUNT=$(grep --count --extended-regexp '^\s+/\*[0-9a-f]+\*/' \
            "$SCRIPT_DIR/test-results/sass-diff/current-filtered.sass" 2>/dev/null || echo "0")

        echo ""
        echo "═══ SASS Comparison ═══"
        "$SCRIPT_DIR/sass-diff.sh" --compare
    fi
fi

# ── Record History ──────────────────────────────────────────────────────────

if $DO_BENCH; then
    for cfg in "${CONFIGS_TO_BENCH[@]}"; do
        pp="${BENCH_PP512[$cfg]:-N/A}"
        tg="${BENCH_TG128[$cfg]:-N/A}"
        sass_col="$SASS_COUNT"
        # Only show SASS count for the primary config.
        [[ "$cfg" != "${CONFIGS_TO_BENCH[0]}" ]] && sass_col="-"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$ITER" "$cfg" "$COMMIT_LABEL" "$pp" "$tg" "$sass_col" "" \
            >> "$HISTORY_FILE"
    done
fi

echo ""
echo "═══ Done ($ITER) ═══"
echo ""
