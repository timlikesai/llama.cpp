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
# The default kernel filter is "flash_attn_ext_vec" (the VEC kernel).
# Use --kernel to change, e.g. --kernel "flash_attn_ext_mma" for MMA kernels.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/test-results/sass-diff"
DOCKER_IMAGE="local/llama.cpp:full-cuda"
CUDA_DEVEL_IMAGE="nvidia/cuda:13.1.0-devel-ubuntu24.04"
SO_NAME="libggml-cuda.so"

# Default kernel filter: D=128, ncols=1 VEC kernels (the tg128 hot path).
# ILi128ELi1E = template args D=128, ncols=1.
KERNEL_FILTER="flash_attn_ext_vecILi128ELi1E"

# ── Argument Parsing ────────────────────────────────────────────────────────

ACTION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)  ACTION="baseline"; shift ;;
        --current)   ACTION="current"; shift ;;
        --compare)   ACTION="compare"; shift ;;
        --stats)
            ACTION="stats"
            STATS_TARGET="${2:-baseline}"
            shift; [[ $# -gt 0 ]] && shift
            ;;
        --list-kernels) ACTION="list-kernels"; shift ;;
        --kernel)    KERNEL_FILTER="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Actions:"
            echo "  --baseline              Save current docker build as baseline SASS"
            echo "  --current               Save current docker build as current SASS"
            echo "  --compare               Compare baseline vs current"
            echo "  --stats [baseline|current]  Show detailed instruction stats"
            echo "  --list-kernels          List all CUDA kernel function names"
            echo ""
            echo "Options:"
            echo "  --kernel PATTERN        Filter to kernels matching PATTERN"
            echo "                          Default: flash_attn_ext_vec"
            echo "  --help                  Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
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

# Dump SASS for matching kernels only. Uses nm (instant, host-side) to find
# kernel names, then cuobjdump --fun to disassemble only those.
dump_filtered_sass() {
    local so_file="$1"
    local pattern="$2"
    local output_path="$3"

    local so_path="$RESULTS_DIR/$so_file"

    # Step 1: Get matching kernel names via nm (instant, host-side).
    echo "Finding kernels matching '$pattern'..."
    local names
    names=$(nm --defined-only "$so_path" | awk '{print $3}' \
        | grep "^_Z" | grep "$pattern" | sort -u)

    local count
    count=$(echo "$names" | grep --count . 2>/dev/null || echo "0")
    echo "  Found $count matching kernels"

    if [[ $count -eq 0 ]]; then
        echo "" > "$output_path"
        return
    fi

    # Step 2: Build comma-separated function list for cuobjdump --function.
    local fun_list
    fun_list=$(echo "$names" | tr '\n' ',' | sed 's/,$//')

    # Step 3: Dump only those kernels (fast — skips everything else).
    echo "  Disassembling $count kernels..."
    docker run --rm \
        --volume "$RESULTS_DIR:/work:rw" \
        --entrypoint /bin/bash \
        "$CUDA_DEVEL_IMAGE" \
        -c "cuobjdump --dump-sass --function '$fun_list' /work/$so_file 2>/dev/null" \
        > "$output_path"

    local line_count
    line_count=$(wc --lines < "$output_path")
    echo "  Saved: $count kernels, $line_count lines"
}

# Print instruction statistics for a filtered SASS file.
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

    echo "  Category counts (sorted by frequency):"
    echo "  ────────────────────────────────────────"
    grep --extended-regexp --only-matching '^\s+/\*[0-9a-f]+\*/\s+\S+' "$sass_path" \
        | sed 's|.*/\*[0-9a-f]*\*/[[:space:]]*||' \
        | sed 's/\..*//' \
        | sort | uniq --count | sort --numeric-sort --reverse \
        | head -30 \
        | while read -r count instr; do
            printf "  %6d  %-20s" "$count" "$instr"
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

    echo "  Per-kernel instruction counts:"
    echo "  ────────────────────────────────────────"
    awk '
    /Function :/ {
        if (func_name != "" && count > 0) {
            printf "  %6d  %s\n", count, short_name
        }
        func_name = $0
        short_name = func_name
        gsub(/.*Function : /, "", short_name)
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
        echo "ERROR: baseline not found. Run: $0 --baseline"; exit 1
    fi
    if [[ ! -f "$curr_sass" ]]; then
        echo "ERROR: current not found. Run: $0 --current"; exit 1
    fi

    local base_total curr_total
    base_total=$(grep --count --extended-regexp '^\s+/\*[0-9a-f]+\*/' "$base_sass" 2>/dev/null || echo "0")
    curr_total=$(grep --count --extended-regexp '^\s+/\*[0-9a-f]+\*/' "$curr_sass" 2>/dev/null || echo "0")
    local delta=$(( curr_total - base_total ))
    local sign="+"; [[ $delta -lt 0 ]] && sign=""

    echo ""
    echo "═══ SASS Comparison: $KERNEL_FILTER ═══"
    echo ""
    echo "  Baseline:  $base_total instructions"
    echo "  Current:   $curr_total instructions"
    echo "  Delta:     ${sign}${delta} instructions"
    echo ""

    echo "  Instruction category diff (baseline → current):"
    echo "  ─────────────────────────────────────────────────────────────"

    local base_counts curr_counts
    base_counts=$(mktemp)
    curr_counts=$(mktemp)

    grep --extended-regexp --only-matching '^\s+/\*[0-9a-f]+\*/\s+\S+' "$base_sass" \
        | sed 's|.*/\*[0-9a-f]*\*/[[:space:]]*||' | sed 's/\..*//' \
        | sort | uniq --count | sort --numeric-sort --reverse > "$base_counts"

    grep --extended-regexp --only-matching '^\s+/\*[0-9a-f]+\*/\s+\S+' "$curr_sass" \
        | sed 's|.*/\*[0-9a-f]*\*/[[:space:]]*||' | sed 's/\..*//' \
        | sort | uniq --count | sort --numeric-sort --reverse > "$curr_counts"

    # Show all categories with changes.
    join -a1 -a2 -e0 -o '0 1.1 2.1' \
        <(awk '{print $2, $1}' "$base_counts" | sort) \
        <(awk '{print $2, $1}' "$curr_counts" | sort) \
        | while read -r instr base_n curr_n; do
            local d=$(( curr_n - base_n ))
            if [[ $d -ne 0 ]]; then
                local ds="+"; [[ $d -lt 0 ]] && ds=""
                printf "  %-14s  %5d → %5d  (%s%d)\n" "$instr" "$base_n" "$curr_n" "$ds" "$d"
            fi
        done | sort -t'(' -k2 -n -r

    echo ""

    echo "  Per-kernel instruction count diff:"
    echo "  ─────────────────────────────────────────────────────────────"

    local base_kernels curr_kernels
    base_kernels=$(mktemp)
    curr_kernels=$(mktemp)

    awk '
    /Function :/ {
        if (name != "" && count > 0) print count, name
        name = $0; gsub(/.*Function : /, "", name)
        count = 0
    }
    /^\s+\/\*[0-9a-f]+\*\// { count++ }
    END { if (name != "" && count > 0) print count, name }
    ' "$base_sass" | sort -k2 > "$base_kernels"

    awk '
    /Function :/ {
        if (name != "" && count > 0) print count, name
        name = $0; gsub(/.*Function : /, "", name)
        count = 0
    }
    /^\s+\/\*[0-9a-f]+\*\// { count++ }
    END { if (name != "" && count > 0) print count, name }
    ' "$curr_sass" | sort -k2 > "$curr_kernels"

    join -a1 -a2 -e0 -o '0 1.1 2.1' "$base_kernels" "$curr_kernels" \
        | while read -r name base_n curr_n; do
            local d=$(( curr_n - base_n ))
            local ds="+"; [[ $d -lt 0 ]] && ds=""
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
        dump_filtered_sass "${ACTION}.so" "$KERNEL_FILTER" "$RESULTS_DIR/${ACTION}-filtered.sass"
        print_stats "$RESULTS_DIR/${ACTION}-filtered.sass" "${ACTION^^}"
        ;;

    compare)
        compare_sass
        ;;

    stats)
        print_stats "$RESULTS_DIR/${STATS_TARGET}-filtered.sass" "${STATS_TARGET^^}"
        ;;

    list-kernels)
        local_so="$RESULTS_DIR/current.so"
        [[ ! -f "$local_so" ]] && local_so="$RESULTS_DIR/baseline.so"
        if [[ ! -f "$local_so" ]]; then
            echo "No .so found. Run --baseline or --current first."
            exit 1
        fi
        echo "Kernel functions in $(basename "$local_so"):"
        nm --defined-only "$local_so" | awk '{print $3}' | sort -u
        ;;
esac
