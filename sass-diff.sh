#!/bin/bash
#
# SASS Diff Tool for Flash Attention VEC Kernel Optimization
#
# Extracts SASS (assembly) from the CUDA shared library inside the docker image,
# filters to specific kernel functions, and compares instruction counts between
# a baseline and current build.
#
# Usage:
#   ./sass-diff.sh --baseline                   # Save current build as baseline
#   ./sass-diff.sh --current                    # Save current build as current
#   ./sass-diff.sh --compare                    # Compare baseline vs current
#   ./sass-diff.sh --stats [baseline|current]   # Show instruction stats for a snapshot
#   ./sass-diff.sh --list-kernels               # List all kernel function names
#   ./sass-diff.sh --kernel PATTERN             # Filter to kernels matching PATTERN
#   ./sass-diff.sh --help                       # Show usage
#
# Workflow:
#   1. Build baseline:    docker compose build llama-cpp-ultra
#   2. Save baseline:     ./sass-diff.sh --baseline
#   3. Edit code, rebuild
#   4. Save current:      ./sass-diff.sh --current
#   5. Compare:           ./sass-diff.sh --compare
#   6. Repeat 3-5
#
# The default kernel filter is "flash_attn_ext_vec" (the VEC kernel).
# Use --kernel to change, e.g. --kernel "flash_attn_ext_mma" for MMA kernels.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/test-results/sass-diff"
DOCKER_IMAGE="local/llama.cpp:full-cuda"
CUDA_DEVEL_IMAGE="nvidia/cuda:13.1.0-devel-ubuntu24.04"
SO_NAME="libggml-cuda.so"

# Default kernel filter (VEC kernel for tg128 optimization).
KERNEL_FILTER="flash_attn_ext_vec"

# ── Argument Parsing ────────────────────────────────────────────────────────

ACTION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)
            ACTION="baseline"
            shift
            ;;
        --current)
            ACTION="current"
            shift
            ;;
        --compare)
            ACTION="compare"
            shift
            ;;
        --stats)
            ACTION="stats"
            STATS_TARGET="${2:-baseline}"
            shift
            [[ $# -gt 0 ]] && shift
            ;;
        --list-kernels)
            ACTION="list-kernels"
            shift
            ;;
        --kernel)
            KERNEL_FILTER="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "SASS Diff Tool for Flash Attention VEC Kernel Optimization"
            echo ""
            echo "Actions:"
            echo "  --baseline              Save current docker build as baseline SASS"
            echo "  --current               Save current docker build as current SASS"
            echo "  --compare               Compare baseline vs current (instruction counts)"
            echo "  --stats [baseline|current]  Show detailed instruction stats"
            echo "  --list-kernels          List all CUDA kernel function names"
            echo ""
            echo "Options:"
            echo "  --kernel PATTERN        Filter to kernels matching PATTERN"
            echo "                          Default: flash_attn_ext_vec"
            echo "                          Examples: flash_attn_ext_mma, flash_attn"
            echo "  --help                  Show this help"
            echo ""
            echo "Workflow:"
            echo "  1. Build baseline:    docker compose build llama-cpp-ultra"
            echo "  2. Save baseline:     $0 --baseline"
            echo "  3. Edit code, rebuild"
            echo "  4. Save current:      $0 --current"
            echo "  5. Compare:           $0 --compare"
            echo "  6. Iterate steps 3-5"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run $0 --help for usage."
            exit 1
            ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    echo "No action specified. Run $0 --help for usage."
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ── Helpers ─────────────────────────────────────────────────────────────────

# Extract the .so from the docker image.
extract_so() {
    local target="$1"
    local so_path="$RESULTS_DIR/${target}.so"

    echo "Extracting $SO_NAME from $DOCKER_IMAGE..."
    local container_id
    container_id=$(docker create "$DOCKER_IMAGE")
    docker cp "$container_id:/app/$SO_NAME" "$so_path"
    docker rm "$container_id" >/dev/null
    echo "  Saved: $so_path ($(du --human-readable "$so_path" | cut -f1))"
}

# Dump full SASS for a .so file.
dump_sass() {
    local so_path="$1"
    local sass_path="$2"

    echo "Dumping SASS (this may take a moment)..."
    docker run --rm \
        --volume "$RESULTS_DIR:/work:rw" \
        --entrypoint /bin/bash \
        "$CUDA_DEVEL_IMAGE" \
        -c "cuobjdump --dump-sass /work/$(basename "$so_path")" \
        > "$sass_path" 2>/dev/null
    echo "  Saved: $sass_path ($(wc --lines < "$sass_path") lines)"
}

# Extract SASS for specific kernel(s) matching a pattern.
# Output: filtered SASS with function headers preserved.
filter_kernels() {
    local sass_path="$1"
    local pattern="$2"
    local output_path="$3"

    awk -v pat="$pattern" '
    /^[[:space:]]*\.text\./ {
        func_name = $0
        in_func = (func_name ~ pat)
        if (in_func) print ""
    }
    in_func { print }
    ' "$sass_path" > "$output_path"

    local kernel_count
    kernel_count=$(grep --count '\.text\.' "$output_path" 2>/dev/null || echo "0")
    echo "  Filtered: $kernel_count kernels matching '$pattern'"
}

# Print instruction statistics for a SASS file.
print_stats() {
    local sass_path="$1"
    local label="$2"

    if [[ ! -f "$sass_path" ]]; then
        echo "ERROR: $sass_path not found."
        exit 1
    fi

    local total
    total=$(grep --count --extended-regexp '^\s+/\*[0-9a-f]+\*/' "$sass_path" 2>/dev/null || echo "0")

    echo ""
    echo "═══ $label: $total total instructions ═══"
    echo ""

    # Count by instruction category.
    echo "  Category counts (sorted by frequency):"
    echo "  ────────────────────────────────────────"
    grep --extended-regexp --only-matching '^\s+/\*[0-9a-f]+\*/\s+\S+' "$sass_path" \
        | sed 's|.*/\*[0-9a-f]*\*/[[:space:]]*||' \
        | sed 's/\..*//' \
        | sort | uniq --count | sort --numeric-sort --reverse \
        | head -30 \
        | while read -r count instr; do
            printf "  %6d  %-20s" "$count" "$instr"
            # Annotate key instructions.
            case "$instr" in
                LDG)    echo "  (global load)" ;;
                STG)    echo "  (global store)" ;;
                LDS)    echo "  (shared memory load)" ;;
                STS)    echo "  (shared memory store)" ;;
                FFMA)   echo "  (FP32 fused multiply-add)" ;;
                HFMA2)  echo "  (FP16 fused multiply-add)" ;;
                HMMA)   echo "  (tensor core MMA)" ;;
                IMAD)   echo "  (integer multiply-add)" ;;
                MOV)    echo "  (register move)" ;;
                ISETP)  echo "  (integer compare)" ;;
                FSETP)  echo "  (FP32 compare)" ;;
                BSSY)   echo "  (barrier set)" ;;
                BSYNC)  echo "  (barrier sync)" ;;
                MUFU)   echo "  (special function unit)" ;;
                PRMT)   echo "  (permute bytes)" ;;
                SHF)    echo "  (shift/funnel)" ;;
                LOP3)   echo "  (3-input logic op)" ;;
                F2F)    echo "  (float convert)" ;;
                I2F)    echo "  (int to float)" ;;
                F2I)    echo "  (float to int)" ;;
                RED)    echo "  (atomic reduce)" ;;
                S2R)    echo "  (special reg read)" ;;
                CS2R)   echo "  (cycle counter)" ;;
                LEA)    echo "  (load effective addr)" ;;
                *)      echo "" ;;
            esac
        done

    echo ""

    # Per-kernel breakdown.
    echo "  Per-kernel instruction counts:"
    echo "  ────────────────────────────────────────"
    awk '
    /^[[:space:]]*\.text\./ {
        if (func_name != "" && count > 0) {
            printf "  %6d  %s\n", count, short_name
        }
        func_name = $0
        # Extract short name: last part after the mangled prefix.
        short_name = func_name
        gsub(/.*\.text\./, "", short_name)
        # Truncate at 80 chars for readability.
        if (length(short_name) > 80) short_name = substr(short_name, 1, 77) "..."
        count = 0
    }
    /^\s+\/\*[0-9a-f]+\*\// { count++ }
    END {
        if (func_name != "" && count > 0) {
            printf "  %6d  %s\n", count, short_name
        }
    }
    ' "$sass_path" | sort --numeric-sort --reverse
}

# Compare two SASS snapshots.
compare_sass() {
    local base_sass="$RESULTS_DIR/baseline-filtered.sass"
    local curr_sass="$RESULTS_DIR/current-filtered.sass"

    if [[ ! -f "$base_sass" ]]; then
        echo "ERROR: baseline not found. Run: $0 --baseline"
        exit 1
    fi
    if [[ ! -f "$curr_sass" ]]; then
        echo "ERROR: current not found. Run: $0 --current"
        exit 1
    fi

    local base_total curr_total
    base_total=$(grep --count --extended-regexp '^\s+/\*[0-9a-f]+\*/' "$base_sass" 2>/dev/null || echo "0")
    curr_total=$(grep --count --extended-regexp '^\s+/\*[0-9a-f]+\*/' "$curr_sass" 2>/dev/null || echo "0")
    local delta=$(( curr_total - base_total ))
    local sign="+"
    [[ $delta -lt 0 ]] && sign=""

    echo ""
    echo "═══ SASS Comparison: $KERNEL_FILTER ═══"
    echo ""
    echo "  Baseline:  $base_total instructions"
    echo "  Current:   $curr_total instructions"
    echo "  Delta:     ${sign}${delta} instructions"
    echo ""

    # Compare instruction categories side-by-side.
    echo "  Instruction category diff (baseline → current):"
    echo "  ─────────────────────────────────────────────────────────────"

    # Build category counts for both.
    local base_counts curr_counts
    base_counts=$(mktemp)
    curr_counts=$(mktemp)

    grep --extended-regexp --only-matching '^\s+/\*[0-9a-f]+\*/\s+\S+' "$base_sass" \
        | sed 's|.*/\*[0-9a-f]*\*/[[:space:]]*||' | sed 's/\..*//' \
        | sort | uniq --count | sort --numeric-sort --reverse > "$base_counts"

    grep --extended-regexp --only-matching '^\s+/\*[0-9a-f]+\*/\s+\S+' "$curr_sass" \
        | sed 's|.*/\*[0-9a-f]*\*/[[:space:]]*||' | sed 's/\..*//' \
        | sort | uniq --count | sort --numeric-sort --reverse > "$curr_counts"

    # Merge and compare.
    join -a1 -a2 -e0 -o '0 1.1 2.1' \
        <(awk '{print $2, $1}' "$base_counts" | sort) \
        <(awk '{print $2, $1}' "$curr_counts" | sort) \
        | while read -r instr base_n curr_n; do
            local d=$(( curr_n - base_n ))
            if [[ $d -ne 0 ]]; then
                local ds="+"
                [[ $d -lt 0 ]] && ds=""
                printf "  %-14s  %5d → %5d  (%s%d)\n" "$instr" "$base_n" "$curr_n" "$ds" "$d"
            fi
        done | sort -t'(' -k2 -n -r

    echo ""

    # Per-kernel comparison.
    echo "  Per-kernel instruction count diff:"
    echo "  ─────────────────────────────────────────────────────────────"

    local base_kernels curr_kernels
    base_kernels=$(mktemp)
    curr_kernels=$(mktemp)

    awk '
    /^[[:space:]]*\.text\./ {
        if (name != "" && count > 0) print count, name
        name = $0; gsub(/.*\.text\./, "", name)
        count = 0
    }
    /^\s+\/\*[0-9a-f]+\*\// { count++ }
    END { if (name != "" && count > 0) print count, name }
    ' "$base_sass" | sort -k2 > "$base_kernels"

    awk '
    /^[[:space:]]*\.text\./ {
        if (name != "" && count > 0) print count, name
        name = $0; gsub(/.*\.text\./, "", name)
        count = 0
    }
    /^\s+\/\*[0-9a-f]+\*\// { count++ }
    END { if (name != "" && count > 0) print count, name }
    ' "$curr_sass" | sort -k2 > "$curr_kernels"

    join -a1 -a2 -e0 -o '0 1.1 2.1' "$base_kernels" "$curr_kernels" \
        | while read -r name base_n curr_n; do
            local d=$(( curr_n - base_n ))
            local ds="+"
            [[ $d -lt 0 ]] && ds=""
            # Truncate name for display.
            local short="$name"
            [[ ${#short} -gt 60 ]] && short="${short:0:57}..."
            printf "  %5d → %5d  (%s%-4d)  %s\n" "$base_n" "$curr_n" "$ds" "$d" "$short"
        done | sort -t'(' -k2 -n

    rm -f "$base_counts" "$curr_counts" "$base_kernels" "$curr_kernels"
    echo ""
}

# ── Actions ─────────────────────────────────────────────────────────────────

case "$ACTION" in
    baseline|current)
        extract_so "$ACTION"
        dump_sass "$ACTION.so" "$RESULTS_DIR/${ACTION}-full.sass"
        filter_kernels "$RESULTS_DIR/${ACTION}-full.sass" "$KERNEL_FILTER" "$RESULTS_DIR/${ACTION}-filtered.sass"
        print_stats "$RESULTS_DIR/${ACTION}-filtered.sass" "${ACTION^^}"
        ;;

    compare)
        compare_sass
        ;;

    stats)
        print_stats "$RESULTS_DIR/${STATS_TARGET}-filtered.sass" "${STATS_TARGET^^}"
        ;;

    list-kernels)
        local so_path="$RESULTS_DIR/current.so"
        if [[ ! -f "$so_path" ]]; then
            so_path="$RESULTS_DIR/baseline.so"
        fi
        if [[ ! -f "$so_path" ]]; then
            echo "No .so found. Run --baseline or --current first."
            exit 1
        fi
        echo "Kernel functions in $(basename "$so_path"):"
        docker run --rm \
            --volume "$RESULTS_DIR:/work:rw" \
            --entrypoint /bin/bash \
            "$CUDA_DEVEL_IMAGE" \
            -c "cuobjdump --dump-sass /work/$(basename "$so_path")" 2>/dev/null \
            | grep '\.text\.' \
            | sed 's/.*\.text\.//' \
            | sort -u
        ;;
esac
