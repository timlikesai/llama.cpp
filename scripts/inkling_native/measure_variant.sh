#!/usr/bin/env bash
#
# EVERY llama-cli / llama-perplexity run in this file must pass -v: without it this build writes only
# warnings to its log, so DEBUG-level facts (sched_reserve splits, KV buffer sizes, the
# GGML_CUDA_GRAPH_DEBUG lines) are silently absent and a 155-byte log reads like "no problem found".
# With -v the "[ Prompt | Generation ]" summary disappears in favour of print_timing lines - the two
# modes need different parsers (see parse_spec_run.py).
# measure_variant.sh - A/B the lossless Inkling file against the vocab-q8_0 variant, and run the
# long-context retrieval gate that the shipped config still lacks.
#
# It needs the GPUs to itself, so it stops the llama-cpp container and ALWAYS brings it back
# (trap on EXIT, including failure and Ctrl-C). Logs are never deleted: each mode writes its own
# file names under logs/, and re-running a mode overwrites only that mode's files.
#
#   ./measure_variant.sh quick       # ctx 8192: FULL vs variant, 2 reps, prefill/decode + text hash
#   ./measure_variant.sh long        # ctx 262144: same, plus the memory breakdown
#   ./measure_variant.sh retrieval   # 256K needle with q4_1 caches + drafting (TODO #11)
#   ./measure_variant.sh ppl         # WikiText-2 PPL for both files (accuracy gate)
#   ./measure_variant.sh all         # retrieval, quick, long, ppl
#
# Flags mirror the deployment in /home/tim/code/services/docker-compose.yml (see profile_shipped.sh,
# which parses them from the compose file); only -c / -p / -n vary per mode.
set -uo pipefail

REPO=${REPO:-/home/tim/code/services/repos/llama.cpp}
SVC=${SVC:-/home/tim/code/services}
BIN=$REPO/build-cuda/bin
MODELS=/home/tim/models/timlikesai
FULL=$MODELS/Inkling-2-Mini-FULL.gguf
VAR=$MODELS/Inkling-2-Mini-FULL-vocabq8.gguf
DRAFT=$MODELS/Inkling-2-Mini-DSpark-BF16.gguf
CORPUS=$REPO/logs/corpora/wiki.test.slice64k.raw
NEEDLE=$REPO/logs/needle-256k-deep.txt
MMPROJ_FILE=${MMPROJ_FILE:-/home/tim/models/timlikesai/Inkling-2-Mini-vision-mmproj.gguf}
# which file the retrieval/needle modes test (the A/B modes name both files explicitly)
MODEL=${MODEL:-$FULL}
LOGS=$REPO/logs
mkdir -p "$LOGS"

# the shipped flag vector, minus the model/draft/server-only parts.
# NOTE: -ngl is deliberately NOT set - the deployment leaves it unset so the fitter can balance
# layers across the two cards (pinning -ngl 99 both disables that and was the crash-loop's cause).
SHIPPED=(-sm layer -fa on -b 2048 -ub 256 -ctk q4_1 -ctv q4_1
         --spec-type draft-dspark --spec-draft-model "$DRAFT"
         --spec-draft-type-k q4_1 --spec-draft-type-v q4_1
         --spec-draft-n-max 2 --spec-draft-n-ubatch 64 --jinja --fit on)

restore() {
    (cd "$SVC" && docker compose up -d llama-cpp > "$LOGS/mv-restore.log" 2>&1)
    echo "[restore] llama-cpp container start requested; see $LOGS/mv-restore.log"
}
trap restore EXIT

precheck() {
    for f in "$FULL" "$DRAFT"; do
        [ -r "$f" ] || { echo "missing $f"; exit 1; }
    done
    if [ ! -r "$VAR" ]; then
        echo "variant $VAR not built; run scripts/inkling_native/make_vocabq_variant.py first"
        exit 1
    fi
    # server-driven modes measure the DEPLOYED process, so they need the container up, not down
    if [ "$MODE" = "srvdet" ]; then
        echo "=== server mode: ensuring llama-cpp is up ==="
        (cd "$SVC" && docker compose up -d llama-cpp > "$LOGS/mv-ensure.log" 2>&1)
        for _ in $(seq 1 40); do
            curl -s --max-time 5 http://127.0.0.1:9931/health | grep -q '"ok"' && { echo "server healthy"; return 0; }
            sleep 3
        done
        echo "ERROR: server did not become healthy; refusing to measure"; exit 3
    fi
    echo "=== stopping llama-cpp for exclusive GPU use ==="
    # A stop can race a start that an earlier run's restore trap left in flight, so poll until the
    # cards are actually free. Measuring against a foreign PID produced four bogus OOM "results"
    # (see docs/mxfp8/49) - refuse rather than report numbers from a contended GPU.
    for _ in $(seq 1 20); do
        (cd "$SVC" && docker compose stop llama-cpp > "$LOGS/mv-stop.log" 2>&1) || true
        apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)"
        [ -z "$apps" ] && break
        sleep 3
    done
    if [ -n "$apps" ]; then
        echo "ERROR: GPUs still busy, refusing to measure (this is why a run's numbers are junk):"
        echo "$apps"
        exit 3
    fi
    echo "GPUs free"
}

# run_one <tag> <model> <ctx> <n-gen> <prompt-file>
run_one() {
    local tag=$1 model=$2 ctx=$3 ngen=$4 pf=$5
    local out="$LOGS/mv-$tag.log"
    echo "--- $tag (ctx $ctx, -n $ngen)"
    "$BIN/llama-cli" -m "$model" "${SHIPPED[@]}" -c "$ctx" \
        -f "$pf" -n "$ngen" --temp 0 -s 42 --no-display-prompt -v < /dev/null \
        > "$out" 2>&1
    echo "    exit=$? -> $out"
}

mode_quick() {
    for rep in 1 2; do
        run_one "quick-full-$rep"  "$FULL" 8192 128 "$REPO/logs/prompt-code.txt"
        run_one "quick-var-$rep"   "$VAR"  8192 128 "$REPO/logs/prompt-code.txt"
    done
}

mode_long() {
    run_one "long-full" "$FULL" 262144 96 "$REPO/logs/prompt-code.txt"
    run_one "long-var"  "$VAR"  262144 96 "$REPO/logs/prompt-code.txt"
}

mode_retrieval() {
    [ -r "$NEEDLE" ] || { echo "missing $NEEDLE"; return 1; }
    local stem=${MODEL##*/}; stem=${stem%.gguf}
    echo "needle prompt: $(wc -c < "$NEEDLE") bytes; the planted code is:"
    grep -a -o -m 3 '[0-9]-[0-9]-[0-9]-[0-9]-[0-9]' "$NEEDLE" | head -3
    echo "model under test: $MODEL"
    # drafting on, q4_1 caches - the exact shipped configuration, at the deployed context
    run_one "retrieval-$stem-draft" "$MODEL" 262144 64 "$NEEDLE"
    # and without the drafter, to separate cache quality from speculation
    local tag=retrieval-$stem-nodraft
    local out="$LOGS/mv-$tag.log"
    echo "--- $tag"
    "$BIN/llama-cli" -m "$MODEL" -sm layer -fa on -b 2048 -ub 256 -ctk q4_1 -ctv q4_1 \
        -c 262144 --temp 0 -s 42 -f "$NEEDLE" -n 64 --no-display-prompt -v --jinja --fit on \
        < /dev/null > "$out" 2>&1
    echo "    exit=$? -> $out"
}

mode_ppl() {
    [ -r "$CORPUS" ] || { echo "missing $CORPUS"; return 1; }
    for tag in ppl-full ppl-var; do
        local model=$FULL
        [ "$tag" = ppl-var ] && model=$VAR
        echo "--- $tag"
        "$BIN/llama-perplexity" -m "$model" -fa on -sm layer -c 2048 -b 2048 -ub 512 \
            -ctk q4_1 -ctv q4_1 -s 42 -f "$CORPUS" < /dev/null \
            > "$LOGS/mv-$tag.log" 2>&1
        echo "    exit=$? -> $LOGS/mv-$tag.log"
    done
}

# The historical "CPU batched-eval nan" probes (docs/mxfp8/2x, TODO #10) used `-ngl 0` WITHOUT
# A pure-CPU run (-dev none) is finite. This mode re-runs the old shape while the GPUs are free, for
# both our lossless file and the public IQ4_NL file, so the claim is either reproduced or retracted
# on evidence.
mode_cpuprobe() {
    local IQ4=/home/tim/models/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-IQ4_NL.gguf
    [ -r "$IQ4" ] || IQ4=$(find /home/tim/models -name "*IQ4_NL*.gguf" -path "*nking*" | head -1)
    local TINY=$LOGS/corpora/wiki.tiny.raw
    for tag in ngl0-full cpudev-full ngl0-iq4; do
        local extra=()
        local model=$FULL
        case "$tag" in
            cpudev-full) extra=(-dev none) ;;
            ngl0-iq4)    model=$IQ4 ;;
        esac
        echo "--- $tag (${model##*/} ${extra[*]-})"
        # tiny corpus + ctx 512: CPU prefill is ~7 t/s on this MoE, so the 64k corpus would take
        # ~40 min per config. Two chunks is enough to see whether the first chunk is nan.
        "$BIN/llama-perplexity" -m "$model" -ngl 0 "${extra[@]}" -c 512 -b 512 -ub 512 \
            -f "$TINY" -s 42 < /dev/null > "$LOGS/mv-cpu-$tag.log" 2>&1
        echo "    exit=$? first chunks: $(grep -a -o '\[[0-9]*\][0-9a-z.]*' "$LOGS/mv-cpu-$tag.log" | head -2 | tr '\n' ' ')"
    done
}

# The deployed config uses q4_1 on both sides, but this build's FA vector (decode) kernels are only
# compiled for GGML_CUDA_FA_QUANTS = q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;f8_e4m3-f8_e4m3, so q4_1
# logs "converting K and V to f16 instead (slow)" on every decode step. This mode measures the cost
# and the alternatives, all of which use compiled kernels.
mode_cache() {
    local P=$REPO/logs/prompt-code.txt
    for ctx in 8192 262144; do
        for pair in "q41 q4_1 q4_1" "q40 q4_0 q4_0" "f8q40 f8_e4m3 q4_0" "f8q80 f8_e4m3 q8_0"; do
            set -- $pair
            local tag=$1 tk=$2 td=$3
            local out="$LOGS/mv-cache-$tag-c$ctx.log"
            echo "--- cache $tag: target $tk, draft $td, ctx $ctx"
            "$BIN/llama-cli" -m "$FULL" -sm layer -fa on -b 2048 -ub 256 -c "$ctx" \
                -ctk "$tk" -ctv "$tk" --spec-type draft-dspark --spec-draft-model "$DRAFT" \
                --spec-draft-type-k "$td" --spec-draft-type-v "$td" --spec-draft-n-max 2 \
                --spec-draft-n-ubatch 64 --jinja --fit on --temp 0 -s 42 \
                -f "$P" -n 128 --no-display-prompt -v < /dev/null > "$out" 2>&1
            echo "    exit=$? -> $out"
        done
    done
}

# The deep needle file is 367,053 tokens - larger than the 262,144 deployment, so the server rejects
# the whole request. Build a needle prompt sized to fit with room to answer.
mode_retrieval_sized() {
    local src=$NEEDLE out=$LOGS/needle-fit.txt
    python3 - "$src" "$out" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
# ~2.55 chars/token measured from the rejected request (937,390 B -> 367,053 tokens)
want = int(230_000 * (len(open(src, encoding='utf-8', errors='ignore').read()) / 367_053))
txt = open(src, encoding='utf-8', errors='ignore').read()
assert len(txt) > want, f"needle file is only {len(txt)} chars"
open(dst, 'w', encoding='utf-8').write(txt[:want])
print("wrote %s: %d chars (~%.0fk tokens at 2.55 ch/tok)" % (dst, want, want / 2550))
PY
    local saved_needle=$NEEDLE
    NEEDLE=$out
    mode_retrieval
    NEEDLE=$saved_needle
}

# Long-context cache probe: does an f8_e4m3 *target* cache (the fastest measured) fit the deployed
# 262,144 context when the vocab is q8_0?  -ub is swept because the fit is marginal.
mode_f8probe() {
    local m="${MODEL:-$FULL}"
    local ctx="${CTX:-262144}"
    for ub in 256 192 128; do
        local tag=f8probe-${m##*/}-c${ctx}-ub$ub
        local out="$LOGS/mv-$tag.log"
        echo "--- $tag"
        "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c "$ctx" -ub $ub -ctk f8_e4m3 -ctv f8_e4m3 \
            -p "Write a function that reverses a linked list in C." -n 128 --temp 0 -s 42 \
            --no-display-prompt -v < /dev/null > "$out" 2>&1
        echo "    exit=$? -> $out"
    done
}

# The f8 probe died with a 14-20 GiB CUDA0 compute ask: as the fitter moves layers off CUDA0 the
# graph split count climbs (2 -> 1719) and every split adds a copy buffer. Total demand at 262144 with
# an f8 cache is only ~28 GiB of the 32 GiB available, so ask for no replanning at all.
mode_f8fitoff() {
    local m="${MODEL:-$FULL}"
    for cfg in "fitoff -ngl 99" "fitoff-ub128 -ngl 99 -ub 128" "fiton-default"; do
        set -- $cfg
        local tag=f8fit-$m-$1
        tag=f8fit-${m##*/}-$1
        local out="$LOGS/mv-$tag.log"
        echo "--- $tag (extra args: $2 $3)"
        if [ "$1" = "fiton-default" ]; then
            "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c 262144 -ctk f8_e4m3 -ctv f8_e4m3 \
                -p "Write a function that reverses a linked list in C." -n 128 --temp 0 -s 42 \
                --no-display-prompt -v < /dev/null > "$out" 2>&1
        else
            "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c 262144 -ctk f8_e4m3 -ctv f8_e4m3 --fit off $2 $3 \
                -p "Write a function that reverses a linked list in C." -n 128 --temp 0 -s 42 \
                --no-display-prompt -v < /dev/null > "$out" 2>&1
        fi
        echo "    exit=$? -> $out"
    done
}

# Name the small kernels. GGML_SCHED_DEBUG=1 prints every node with its assigned backend, so op
# counts per graph x decode steps explains a profile's tiny-kernel call counts (scale_f32,
# concat_cont<unsigned int>, cpy_scalar_transpose<__half>) instead of guessing at them.
mode_scheddump() {
    local m="${MODEL:-$FULL}"
    local tag=scheddump-${m##*/}
    local out="$LOGS/mv-$tag.log"
    echo "--- $tag"
    GGML_SCHED_DEBUG="${SCHEDDBG:-2}" "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c 8192 \
        -p "Write a function that reverses a linked list in C." -n 4 --temp 0 -s 42 \
        --no-display-prompt -v < /dev/null > "$out" 2>&1
    echo "    exit=$? -> $out"
}

# Does the shipped config get CUDA graph capture at all, and which node disqualifies it?
# ggml-cuda.cu:3313 disables graphs for a whole graph if ANY MUL_MAT_ID needs a stream sync.
mode_graphdbg() {
    local m="${MODEL:-$FULL}"
    local d="${NDEBUG_TOKENS:-32}"
    local tag=graphdbg-${m##*/}-n$d
    local out="$LOGS/mv-$tag.log"
    echo "--- $tag"
    GGML_CUDA_GRAPH_DEBUG=1 "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c 8192 \
        -p "Write a function that reverses a linked list in C." -n "$d" --temp 0 -s 42 \
        --no-display-prompt -v < /dev/null > "$out" 2>&1
    echo "    exit=$? -> $out"
}

# Quantify the multi-device penalty. With -sm layer the graph ping-pongs CUDA0<->CUDA1 16-17 times
# per pass (docs/mxfp8/54). A single-device run cannot split at all, so comparing decode t/s and the
# sched_reserve "splits =" field on a file small enough for one card measures what the split costs.
# Needs a model that fits one 16 GB card, hence the public IQ4_NL export, not the 19.6 GB lossless one.
mode_singlecard() {
    local m="${SMALL:-/home/tim/models/ggml-org/Inkling-2-Mini-GGUF-IQ4_NL/Inkling-2-Mini-IQ4_NL.gguf}"
    [ -r "$m" ] || { echo "missing single-card model $m (set SMALL=)"; return 1; }
    for cfg in "none -sm none -ngl 99" "layer -sm layer -ngl 99"; do
        set -- $cfg
        local tag=singlecard-$1
        local out="$LOGS/mv-$tag.log"
        echo "--- $tag ($2 $3)"
        "$BIN/llama-cli" -m "$m" -c 8192 -b 2048 -ub 256 -fa on -ctk q4_1 -ctv q4_1 \
            $2 $3 -p "Write a function that reverses a linked list in C." -n 128 --temp 0 -s 42 \
            --no-display-prompt -v < /dev/null > "$out" 2>&1
        echo "    exit=$? -> $out"
    done
}

# Sweep the two speculation knobs that were never varied together, on the deployed file and the
# deployed flag vector (which includes the ngram chain - it measured +14% decode at 262,144).
mode_shiptune() {
    local m="${MODEL:-$FULL}"
    for nm in 2 3; do
        for dub in 64 32; do
            local tag=shiptune-nm${nm}-dub${dub}
            local out="$LOGS/mv-$tag.log"
            echo "--- $tag"
            "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c 262144 \
                --spec-draft-n-max "$nm" --spec-draft-n-ubatch "$dub" \
                -p "Write a function that reverses a linked list in C." -n 128 --temp 0 -s 42 \
                --no-display-prompt -v < /dev/null > "$out" 2>&1
            echo "    exit=$? -> $out"
        done
    done
}

# Does any placement reduce the 16-17 CUDA0<->CUDA1 crossings per pass? (docs/mxfp8/54)
# sched_reserve prints "splits = N / N", so each config is judged on the count itself, not on a
# feeling about placement. llama-cli rejects -ot/-ts, so this uses llama-perplexity, which takes them.
# Is the run-to-run acceptance variance caused by the server's prompt-cache/checkpoint flags? (#8)
# Observed: three identical requests through profile_shipped.sh (which uses the compose flag vector,
# including --ctx-checkpoints/--checkpoint-min-step/--cache-idle-slots/--cache-reuse) gave acceptance
# 0.609 / 0.429 / 0.618, while the SHIPPED vector without them reproduces 0.65455 across runs and
# days. Mechanism under test: a cache hit skips the target forward, so the tap ring is never filled
# and the markov head drafts without target features ("no target taps recorded").
mode_cachedet() {
    local m="${MODEL:-$FULL}"
    local CACHE=(--ctx-checkpoints 256 --checkpoint-min-step 32768 --cache-idle-slots --cache-reuse 4096)
    for rep in 1 2 3; do
        for cfg in plain cache; do
            local extra=()
            [ "$cfg" = cache ] && extra=("${CACHE[@]}")
            local tag=cachedet-$cfg-$rep
            local out="$LOGS/mv-$tag.log"
            echo "--- $tag"
            "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c 32768 "${extra[@]}" \
                -f "$LOGS/prompt-code.txt" -n 384 --temp 0 -s 42 --no-display-prompt -v \
                < /dev/null > "$out" 2>&1
            echo "    exit=$? -> $out"
        done
    done
}

# Does the DEPLOYED server give different acceptance for identical requests? (#8)
# The CLI path is deterministic (cachedet: 0.63018 / 213-338 three times), while three identical
# requests through llama-server under nsys gave 0.609 / 0.429 / 0.618. llama-cli cannot even take
# --cache-reuse / --checkpoint-min-step / --cache-idle-slots, so this can only be tested against the
# server. Reads the slot report out of the container log, which is where acceptance is printed.
mode_srvdet() {
    local url="${SERVER_URL:-http://127.0.0.1:9931}"
    local n="${NREQ:-4}"
    command -v curl >/dev/null || { echo "curl missing"; return 1; }
    local mark
    mark=$(docker ps --filter name=llama-cpp --format '{{.Status}}')
    echo "container before: $mark"
    for i in $(seq 1 "$n"); do
        curl -s --max-time 600 "$url/v1/chat/completions" -H "Content-Type: application/json" \
            -d "{\"model\":\"inkling\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a python function that deduplicates a list while preserving order, with type hints and a docstring.\"}],\"max_tokens\":256,\"temperature\":0}" \
            > "$LOGS/srvdet-req-$i.json" 2>&1
        echo "  req $i -> $(wc -c < "$LOGS/srvdet-req-$i.json") bytes"
    done
    docker logs --tail 400 llama-cpp > "$LOGS/srvdet-log.txt" 2>&1
    echo "=== acceptance per slot report (in log order) ==="
    grep -oE "draft acceptance = [0-9.]+ \( *[0-9]+ accepted / *[0-9]+ generated\), mean len = *[0-9.]+" "$LOGS/srvdet-log.txt" | tail -8
    echo "=== tap warnings ==="
    grep -c "no target taps recorded" "$LOGS/srvdet-log.txt"
}

# Exercise the draft-context length guard (#47): generate right up to the context limit so the last
# draft blocks would be numbered past llama_n_ctx_seq(ctx_dft). Expect "no room for a N-token block".
mode_ctxend() {
    local m="${MODEL:-$FULL}"
    local arm=guard
    [ -n "${LLAMA_NO_CTXGUARD:-}" ] && arm=noguard
    local out="$LOGS/mv-ctxend-$arm-$(date +%H%M%S).log"
    echo "--- ctxend: -c 128, generating to the edge"
    # --fit off: SHIPPED carries --fit on, which auto-inflates n_ctx and made the limit unreachable.
    # -c 128 still rounds up to n_ctx_seq 256, so the generation length (250) is what puts pos0
    # past 253 and makes a 3-token block overflow.
    # a short inline prompt: -f prompt-code.txt is 2527 tokens and would be rejected outright by a
    # small context, which is how the first version of this test produced a vacuous pass
    # n_max 8 makes the block 9 wide, which is where the guard's window is actually reachable: at
    # n_max 2 the last anchor lands at pos0 = n_ctx - 3 and 3 tokens fit exactly, so the condition
    # never holds (measured: 27 + 229 = 256, edge reached, zero hits - correct, not a miss).
    "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" --spec-draft-n-max 8 --fit off -c 128 \
        -p "Write a haiku about two graphics cards sharing one tensor." \
        -n 250 --temp 0 -s 42 --no-display-prompt -v \
        < /dev/null > "$out" 2>&1
    echo "    exit=$? -> $out"
    echo "=== guard hits ==="
    grep -c "no room for a" "$out"
    grep -m2 "no room for a" "$out"
    echo "=== acceptance / crash check ==="
    grep -a "draft acceptance" "$out" | tail -1
    grep -acE "GGML ASSERT|buffer not set|Segmentation|exiting due to" "$out"
    echo "LOG=$out"
}

# Which batch shape maximizes draft acceptance? (#33, and the reframe in doc 58)
# Measured confounds: -ub alone moved acceptance 0.495 -> 0.619, and a prompt-cache hit moves it
# 0.658 -> 0.47. Acceptance is a property of (model, prompt, shape), not of the drafter alone - so
# before hunting an implementation gap vs the other PR, find the shape that maximizes it.
# Each arm is a FRESH process (cold cache) so the numbers are comparable; -c is held below the
# deployed context so every -ub fits and memory is not a variable.
mode_shapeacc() {
    local m="${MODEL:-$FULL}"
    local ctx="${CTX:-131072}"
    local prompt="${PROMPT:-$LOGS/prompt-code.txt}"
    for ub in 64 128 256 512; do
        local out="$LOGS/mv-shape-ub$ub-c$ctx.log"
        echo "--- shape: -ub $ub -c $ctx"
        "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" --fit off -ngl 99 -c "$ctx" -ub "$ub" \
            -f "$prompt" -n 384 --temp 0 -s 42 --no-display-prompt -v \
            < /dev/null > "$out" 2>&1
        echo "    exit=$? -> $out"
    done
}

# Does -ub 512 fit at the DEPLOYED 262,144 now that the indexed-bias merge removed the n_kv-wide
# bias tensors? (shapeacc: at 131,072, -ub 512 beat -ub 256 on acceptance 0.722 vs 0.630, prefill
# 1157 vs 1034 and decode 51.6 vs 48.0 - i.e. the deployed value was the worst of four.) Before the
# merge, 256K at -ub 512 could not reserve; the merge made 256K/-ub 512 work on the PR build.
# Runs the real deployed flag vector (with the ngram chain) and reports memory + retries.
mode_ub256k() {
    local m="${MODEL:-$FULL}"
    local CHAIN=(--spec-type draft-dspark,ngram-mod,ngram-map-k)
    for ub in 256 512; do
        for rep in 1 2; do
            local out="$LOGS/mv-ub256k-ub${ub}-$rep.log"
            echo "--- deployed ctx 262144, -ub $ub, rep $rep"
            "$BIN/llama-cli" -m "$m" -sm layer -fa on -b 2048 -ub "$ub" -c 262144 \
                -ctk q4_1 -ctv q4_1 --jinja --fit on "${CHAIN[@]}" \
                --spec-draft-model "$DRAFT" --spec-draft-type-k q4_1 --spec-draft-type-v q4_1 \
                --spec-draft-n-max 2 --spec-draft-n-ubatch 64 \
                -f "$LOGS/prompt-code.txt" -n 384 --temp 0 -s 42 --no-display-prompt -v \
                < /dev/null > "$out" 2>&1
            echo "    exit=$? retries=$(grep -ac 'failed to allocate' "$out") -> $out"
        done
    done
}

# Context x micro-batch frontier, with the DEPLOYED spec chain. -ub 512 beat -ub 256 on acceptance,
# prefill and decode at 131,072, but cannot reserve at 262,144. Find the frontier: for each context,
# the largest -ub that runs clean (zero fitter retries), and what it is worth.
mode_ubfrontier() {
    local m="${MODEL:-$FULL}"
    local CHAIN=(--spec-type draft-dspark,ngram-mod,ngram-map-k)
    for cfg in "262144 256" "262144 320" "196608 512" "196608 384" "163840 512" "131072 512"; do
        set -- $cfg
        local ctx=$1 ub=$2
        local out="$LOGS/mv-front-c${ctx}-ub${ub}.log"
        echo "--- ctx $ctx, -ub $ub"
        "$BIN/llama-cli" -m "$m" -sm layer -fa on -b 2048 -ub "$ub" -c "$ctx" \
            -ctk q4_1 -ctv q4_1 --jinja --fit on "${CHAIN[@]}" \
            --spec-draft-model "$DRAFT" --spec-draft-type-k q4_1 --spec-draft-type-v q4_1 \
            --spec-draft-n-max 2 --spec-draft-n-ubatch 64 \
            -f "$LOGS/prompt-code.txt" -n 384 --temp 0 -s 42 --no-display-prompt -v \
            < /dev/null > "$out" 2>&1
        echo "    exit=$? retries=$(grep -ac 'failed to allocate' "$out") -> $out"
    done
}

# Is the FITTER (not the ngram chain) what blocks -ub 512? shapeacc ran 131072/-ub 512 clean with
# --fit off -ngl 99 and dspark-only; ubfrontier failed the same shape with --fit on and the chain.
# Cross the two variables: {fit on, fit off} x {dspark, chain} at the shape that should work.
mode_fitab() {
    local m="${MODEL:-$FULL}"
    local CHAIN=(--spec-type draft-dspark,ngram-mod,ngram-map-k)
    local DS=(--spec-type draft-dspark)
    for fit in "on" "off"; do
        for spec in chain dspark; do
            local extra=(); [ "$spec" = chain ] && extra=("${CHAIN[@]}") || extra=("${DS[@]}")
            local pin=(); [ "$fit" = off ] && pin=(-ngl 99)
            local out="$LOGS/mv-fitab-fit${fit}-${spec}.log"
            echo "--- fit $fit, spec $spec"
            "$BIN/llama-cli" -m "$m" -sm layer -fa on -b 2048 -ub 512 -c 131072 \
                -ctk q4_1 -ctv q4_1 --jinja --fit "$fit" "${pin[@]}" "${extra[@]}" \
                --spec-draft-model "$DRAFT" --spec-draft-type-k q4_1 --spec-draft-type-v q4_1 \
                --spec-draft-n-max 2 --spec-draft-n-ubatch 64 \
                -f "$LOGS/prompt-code.txt" -n 384 --temp 0 -s 42 --no-display-prompt -v \
                < /dev/null > "$out" 2>&1
            echo "    exit=$? retries=$(grep -ac 'failed to allocate' "$out")"
        done
    done
}

# Can --fit off unlock -ub 512 at the DEPLOYED 262,144? fitab proved the fitter (not the chain) is
# what blocks -ub 512 at 131,072: --fit on failed with either spec config, --fit off -ngl 99 ran
# clean. If 256K/-ub 512 fits with the fitter off, that is a strict win over the deployed
# 256K/-ub 256 (1024.7 prefill, 50.35 decode, 0.44423 acceptance with the chain).
mode_256k512() {
    local m="${MODEL:-$FULL}"
    for spec in dspark chain; do
        local extra=()
        [ "$spec" = chain ] && extra=(--spec-type draft-dspark,ngram-mod,ngram-map-k) || extra=(--spec-type draft-dspark)
        for rep in 1 2; do
            local out="$LOGS/mv-256k512-$spec-$rep.log"
            echo "--- 262144, -ub 512, fit off, $spec, rep $rep"
            "$BIN/llama-cli" -m "$m" -sm layer -fa on -b 2048 -ub 512 -c 262144 \
                -ctk q4_1 -ctv q4_1 --jinja --fit off -ngl 99 "${extra[@]}" \
                --spec-draft-model "$DRAFT" --spec-draft-type-k q4_1 --spec-draft-type-v q4_1 \
                --spec-draft-n-max 2 --spec-draft-n-ubatch 64 \
                -f "$LOGS/prompt-code.txt" -n 384 --temp 0 -s 42 --no-display-prompt -v \
                < /dev/null > "$out" 2>&1
            echo "    exit=$? retries=$(grep -ac 'failed to allocate' "$out") host=$(grep -aoE 'Host +[0-9]+ MiB' "$out" | tail -1)"
        done
    done
}

# The missing cell of the 2x2: deployed context, deployed -ub 256, deployed --fit on, but dspark-only
# (no ngram chain). Without it the -ub 512 result cannot be read as a win or a loss.
mode_256k256() {
    local m="${MODEL:-$FULL}"
    for spec in dspark chain; do
        local extra=()
        [ "$spec" = chain ] && extra=(--spec-type draft-dspark,ngram-mod,ngram-map-k) || extra=(--spec-type draft-dspark)
        for rep in 1 2; do
            local out="$LOGS/mv-256k256-$spec-$rep.log"
            echo "--- 262144, -ub 256, fit on, $spec, rep $rep"
            "$BIN/llama-cli" -m "$m" -sm layer -fa on -b 2048 -ub 256 -c 262144 \
                -ctk q4_1 -ctv q4_1 --jinja --fit on "${extra[@]}" \
                --spec-draft-model "$DRAFT" --spec-draft-type-k q4_1 --spec-draft-type-v q4_1 \
                --spec-draft-n-max 2 --spec-draft-n-ubatch 64 \
                -f "$LOGS/prompt-code.txt" -n 384 --temp 0 -s 42 --no-display-prompt -v \
                < /dev/null > "$out" 2>&1
            echo "    exit=$? retries=$(grep -ac 'failed to allocate' "$out")"
        done
    done
}

# --fit off + -ub 512 freed 3.4 GiB of compute buffer at 262,144 (4,894 vs 8,302 MiB). f8_e4m3 target
# KV measured +14% decode but never fit at 262,144. Spend the freed space: f8 caches, and -ub 1024
# (if 512 beat 256, the trend may continue - and there is now room for it).
mode_freespace() {
    local m="${MODEL:-$FULL}"
    for cfg in "512 q4_1" "1024 q4_1" "512 f8_e4m3" "1024 f8_e4m3" "256 f8_e4m3"; do
        set -- $cfg
        local ub=$1 ct=$2
        local out="$LOGS/mv-free-ub${ub}-${ct}.log"
        echo "--- -ub $ub, target cache $ct, ctx 262144, fit off, dspark only"
        "$BIN/llama-cli" -m "$m" -sm layer -fa on -b 2048 -ub "$ub" -c 262144 \
            -ctk "$ct" -ctv "$ct" --jinja --fit off -ngl 99 --spec-type draft-dspark \
            --spec-draft-model "$DRAFT" --spec-draft-type-k q4_1 --spec-draft-type-v q4_1 \
            --spec-draft-n-max 2 --spec-draft-n-ubatch 64 \
            -f "$LOGS/prompt-code.txt" -n 384 --temp 0 -s 42 --no-display-prompt -v \
            < /dev/null > "$out" 2>&1
        echo "    exit=$? retries=$(grep -ac 'failed to allocate' "$out")"
    done
}

# Validate the candidate config against the REAL deployment: vision on (the container runs -mm with
# the CPU tower, whose buffers were never in the CLI runs above), across all three prompt domains,
# with the memory breakdown. The candidate came from code-prompt-only, no-vision, single-sequence
# runs - each of those is a confound this closes or exposes.
mode_candidate() {
    local m="${MODEL:-$FULL}"
    local MM=()
    [ "${VISION:-0}" = 1 ] && MM=(-mm "$MMPROJ_FILE" --no-mmproj-offload)
    # g0-prompt-long.txt is the repetitive technical text (2,817 tokens of near-duplicate
    # lines) that the acceptance matrix used as its third domain.
    for dom in code prose repetitive; do
        local pf="$LOGS/prompt-$dom.txt"
        [ "$dom" = repetitive ] && pf="$LOGS/g0-prompt-long.txt"
        [ -r "$pf" ] || { echo "skip $dom (no $pf)"; continue; }
        for cfg in ${CFGS:-deployed candidate}; do
            local flags
            local shape=(); local spec=()
            case "$cfg" in
              deployed)     shape=(-ub 256 -ctk q4_1 -ctv q4_1 --fit on);        spec=(--spec-type draft-dspark,ngram-mod,ngram-map-k) ;;
              candidate)    shape=(-ub 512 -ctk f8_e4m3 -ctv f8_e4m3 --fit off -ngl 99); spec=(--spec-type draft-dspark) ;;
              cand-chain)   shape=(-ub 512 -ctk f8_e4m3 -ctv f8_e4m3 --fit off -ngl 99); spec=(--spec-type draft-dspark,ngram-mod,ngram-map-k) ;;
              dep-dspark)   shape=(-ub 256 -ctk q4_1 -ctv q4_1 --fit on);        spec=(--spec-type draft-dspark) ;;
            esac
            flags=("${shape[@]}" "${spec[@]}")
            local out="$LOGS/mv-cand-$dom-$cfg${VISION:+-vision}.log"
            echo "--- $dom / $cfg (vision=${VISION:-0})"
            "$BIN/llama-cli" -m "$m" -sm layer -fa on -b 2048 -c 262144 --jinja \
                "${MM[@]}" "${flags[@]}" \
                --spec-draft-model "$DRAFT" --spec-draft-type-k q4_1 --spec-draft-type-v q4_1 \
                --spec-draft-n-max 2 --spec-draft-n-ubatch 64 \
                -f "$pf" -n 384 --temp 0 -s 42 --no-display-prompt -v \
                < /dev/null > "$out" 2>&1
            echo "    exit=$? retries=$(grep -ac 'failed to allocate' "$out")"
        done
    done
}

mode_splitsprobe() {
    local m="${MODEL:-$FULL}"
    local c="${CORPUS:-$LOGS/corpora/wiki.tiny.raw}"
    [ -r "$c" ] || { echo "missing corpus $c"; return 1; }
    for cfg in \
        "layer" \
        "row -sm row" \
        "vocab-split -ot token_embd.weight=CUDA0,output.weight=CUDA1" \
        "experts-c1 -ot .ffn_gate_exps.weight=CUDA1,.ffn_down_exps.weight=CUDA1,.ffn_up_exps.weight=CUDA1" \
        "trunk-c0 -ot blk.0.attn_norm.weight=CUDA0,token_embd.weight=CUDA0,output.weight=CUDA0"; do
        set -- $cfg
        local tag=splitsprobe-$1
        local out="$LOGS/mv-$tag.log"
        if [ "$#" -gt 1 ]; then
        echo "--- $tag: $*"
    else
        echo "--- $tag: default placement"
    fi
        "$BIN/llama-perplexity" -m "$m" -c 8192 -b 8192 -ub 256 -fa on -sm layer \
            -ctk q4_1 -ctv q4_1 --temp 0 -s 42 -f "$c" "${@:2}" -v < /dev/null > "$out" 2>&1
        echo "    exit=$? -> $out"
    done
}

# The standing gate set. Run after every rebase/merge, before believing any number, and before
# committing. Each gate is named so a failure points at itself; all output is teed, nothing filtered.
#   ops-flash  upstream test_flash_attn_ext_indexed_bias - in this tree it really hits the CUDA kernel
#   ops-mxfp8  our type 44 on CPU + both cards
#   archs      every architecture builds
#   dflare     the drafter fusion oracle, plus its python re-derivation via ctest
#   quantize   round-trip fns (catches QI_*/block-size convention errors)
#   cpu-cuda   greedy output must agree between CPU and CUDA
#   ppl        pinned corpus, expect 5.5787 +/- 0.14423 on the lossless file
mode_gates() {
    local m="${MODEL:-$FULL}"
    local ok=0 bad=0
    g() { # g <name> <cmd...>
        local name=$1; shift
        if "$@" > "$LOGS/gate-$name.log" 2>&1; then
            echo "  ok    $name"; ok=$((ok+1))
        else
            echo "  FAIL  $name  (see logs/gate-$name.log)"; bad=$((bad+1))
        fi
    }
    precheck
    echo "--- gates on $m"
    # test-backend-ops notes, learned the hard way:
    #   -d <backend> is NOT a flag: it prints usage and exits 0 (a silent no-op that reads as success)
    #   -dev takes a NAME (CUDA0/CUDA1/CPU/none), not an index - `--list-devices` is authoritative
    #   -b <backend> prints "N/N backends passed" even though it skipped the others
    #   so the gate greps for the line proving the named backend actually executed.
    ops_dev() { # ops_dev <name> <backend> <op-filter args...>
        local name=$1 dev=$2; shift 2
        "$BIN/test-backend-ops" "$@" -b "$dev" > "$LOGS/gate-$name.log" 2>&1
        local rc=$?
        if [ "$rc" -ne 0 ]; then
            echo "  FAIL  $name (exit $rc)"; bad=$((bad+1)); return
        fi
        if grep -q "Backend .*: $dev\$" "$LOGS/gate-$name.log" && ! grep -q "Skipping $dev" "$LOGS/gate-$name.log"; then
            echo "  ok    $name ($dev executed)"; ok=$((ok+1))
        else
            echo "  FAIL  $name - $dev did not actually run (vacuous pass)"; bad=$((bad+1))
        fi
    }
    g flash-cuda  "$BIN/test-flash-attn-indexed-bias"   # CPU reference vs the CUDA kernel, incl. kv_rows
    g ops-flash   "$BIN/test-backend-ops" -o FLASH_ATTN_EXT
    ops_dev ops-mxfp8-cuda0 CUDA0 -p mxfp8
    ops_dev ops-mxfp8-cuda1 CUDA1 -p mxfp8
    g ops-mxfp8-all "$BIN/test-backend-ops" -p mxfp8   # no -b: iterates every backend incl. CPU reference
    g archs       "$BIN/test-llama-archs"
    g dflare      "$BIN/test-dflare-mix"
    g dflare-py   ctest --test-dir "$REPO/build-cuda" -R dflare --output-on-failure
    g quantize    "$BIN/test-quantize-fns"
    g cpu-greedy  "$BIN/llama-cli" -m "$m" -dev none -c 512 -p "Q: What is 17 times 4? A:" -n 8 \
                    --temp 0 --single-turn --no-warmup -v
    g cuda0-greedy "$BIN/llama-cli" -m "$m" -dev CUDA0 -c 512 -p "Q: What is 17 times 4? A:" -n 8 \
                    --temp 0 --single-turn --no-warmup -v
    g cuda1-greedy "$BIN/llama-cli" -m "$m" -dev CUDA1 -c 512 -p "Q: What is 17 times 4? A:" -n 8 \
                    --temp 0 --single-turn --no-warmup -v
    # The three greedy probes above only prove each backend ran. Inkling spends a short budget inside
    # its thinking block, so compare the reasoning stream - that is the cross-backend numerics gate.
    if [ "$bad" -eq 0 ]; then
        python3 - "$LOGS" <<'PYEOF'
import re, sys, pathlib
d = pathlib.Path(sys.argv[1])
def field(name, key):
    t = (d / f"gate-{name}.log").read_text(errors="ignore")
    t = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", t)
    t = re.sub(r"\r[^\n]*", "", t)
    m = re.search(r'"%s":"(.*?)"(?:,"|\})' % key, t, re.S)
    return m.group(1) if m else None
vals = {n: field(n, "reasoning_content") for n in ("cpu-greedy", "cuda0-greedy", "cuda1-greedy")}
print("   reasoning per backend:", {k: (v or "")[:24] for k, v in vals.items()})
if len(set(vals.values())) != 1 or None in vals.values():
    sys.exit("FAIL  cpu-vs-cuda: backends disagree")
print("  ok    cpu-vs-cuda (identical reasoning stream)")
PYEOF
    fi
    if [ -r "$CORPUS" ]; then
        g ppl "$BIN/llama-perplexity" -m "$m" -c 2048 -b 2048 -ub 512 -f "$CORPUS" --temp 0 -v
    else
        echo "  skip  ppl (no corpus at $CORPUS)"
    fi
    echo "GATES: ok=$ok fail=$bad"
}

# A/B the adaptive draft pausing (LLAMA_SPEC_ADAPT) across prompt domains and block widths.
# The case this exists for: repetitive technical prose measured -17% decode with drafting at n_max 4.
# Two states only, so the verification batch size never changes and no graph rebuild is triggered.
mode_adapt() {
    local m="${MODEL:-$FULL}"
    for adapt in 0 0.33; do
        for nm in 2 4; do
            # three domains, because acceptance is domain-dependent (code 0.6-0.8, prose 0.2,
            # repetitive technical was the measured regression case)
            for pr in code prose repetitive; do
                case "$pr" in
                    code)       f="$LOGS/prompt-code.txt" ;;
                    prose)      f="$LOGS/prompt-prose.txt" ;;
                    repetitive) f="$LOGS/g0-prompt-long.txt" ;;
                esac
                local N="${ADAPT_N:-192}"
                [ -r "$f" ] || { echo "  skip  missing $f"; continue; }
                local tag=adapt-$adapt-nm$nm-$pr-n$N
                local out="$LOGS/mv-$tag.log"
                echo "--- $tag"
                LLAMA_SPEC_ADAPT=$adapt "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c 32768 \
                    --spec-draft-n-max "$nm" -f "$f" -n "$N" --temp 0 -s 42 --no-display-prompt -v \
                    < /dev/null > "$out" 2>&1
                echo "    exit=$? -> $out"
            done
        done
    done
}

# Does --spec-draft-p-min actually gate proposals? The code path exists (dspark draft() reads
# llama_get_embeddings_nextn and breaks below p_min) but has never been measured. If nothing moves,
# the confidence head is not reaching the host and the check is vacuous.
mode_pminsweep() {
    local m="${MODEL:-$FULL}"
    local n="${PMIN_N:-384}"
    for pm in 0 0.3 0.6 0.9; do
        for pr in code repetitive; do
            local f
            case "$pr" in
                code)       f="$LOGS/prompt-code.txt" ;;
                repetitive) f="$LOGS/g0-prompt-long.txt" ;;
            esac
            local tag=pmin-$pm-$pr-n$n
            local out="$LOGS/mv-$tag.log"
            echo "--- $tag"
            "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c 32768 --spec-draft-p-min "$pm" \
                -f "$f" -n "$n" --temp 0 -s 42 --no-display-prompt -v < /dev/null > "$out" 2>&1
            echo "    exit=$? -> $out"
        done
    done
}

# Paired control for the "refused spec init is slower" claim. Same target flags, three reps each:
#   nodraft  - no --spec-type at all
#   refused  - --spec-type + a p_min the draft rejects, so init throws and the server continues
# If both land in the same band, the earlier 4.9% gap was a single-run artifact, not a penalty.
mode_pminctl() {
    local m="${MODEL:-$FULL}"
    for rep in 1 2 3; do
        for cfg in nodraft refused; do
            local extra=()
            [ "$cfg" = refused ] && extra=(--spec-type draft-dspark --spec-draft-model "$DRAFT" \
                --spec-draft-type-k q4_1 --spec-draft-type-v q4_1 --spec-draft-n-max 2 \
                --spec-draft-n-ubatch 64 --spec-draft-p-min 0.6)
            local tag=pminctl-$cfg-$rep
            local out="$LOGS/mv-$tag.log"
            echo "--- $tag"
            "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c 32768 "${extra[@]}" \
                -f "$LOGS/prompt-code.txt" -n 384 --temp 0 -s 42 --no-display-prompt -v \
                < /dev/null > "$out" 2>&1
            echo "    exit=$? -> $out"
        done
    done
}

# A/B the reference bf16 rounding chain in the markov head. Acceptance is the metric (throughput
# follows it); the chain adds ~10 tiny CPY/FILL nodes per block, so decode t/s is checked too.
mode_chainab() {
    local m="${MODEL:-$FULL}"
    local n="${CHAIN_N:-384}"
    for on in 0 1; do
        for pr in code prose repetitive; do
            local f
            case "$pr" in
                code)       f="$LOGS/prompt-code.txt" ;;
                prose)      f="$LOGS/prompt-prose.txt" ;;
                repetitive) f="$LOGS/g0-prompt-long.txt" ;;
            esac
            local tag=chain-${ABENV:-env}-$on-$pr-n$n
            local out="$LOGS/mv-$tag.log"
            echo "--- $tag"
            # ABENV names the gate under test (e.g. LLAMA_ZERO_FRNN, LLAMA_DSPARK_BF16_CHAIN)
            if [ "$on" = 1 ] && [ -n "${ABENV:-}" ]; then export "$ABENV"=1; else unset "${ABENV:-LLAMA_UNUSED}"; fi
            "$BIN/llama-cli" -m "$m" "${SHIPPED[@]}" -c 32768 --spec-draft-n-max 2 \
                -f "$f" -n "$n" --temp 0 -s 42 --no-display-prompt -v < /dev/null > "$out" 2>&1
            echo "    exit=$? -> $out"
        done
    done
    unset LLAMA_DSPARK_BF16_CHAIN
}

MODE=${1:-all}
echo "=== measure_variant.sh mode=$MODE ==="
precheck
case "$MODE" in
    quick)     mode_quick ;;
    long)      mode_long ;;
    retrieval) mode_retrieval ;;
    ppl)       mode_ppl ;;
    all)       mode_retrieval_sized; mode_cache; mode_quick; mode_long; mode_ppl; mode_cpuprobe ;;
    cache)     mode_cache ;;
    retrieval-sized) mode_retrieval_sized ;;
    cpuprobe)  mode_cpuprobe ;;
    f8probe)   mode_f8probe ;;
    f8fitoff)  mode_f8fitoff ;;
    scheddump) mode_scheddump ;;
    graphdbg)  mode_graphdbg ;;
    singlecard) mode_singlecard ;;
    shiptune)  mode_shiptune ;;
    splitsprobe) mode_splitsprobe ;;
    cachedet)  mode_cachedet ;;
    srvdet)    mode_srvdet ;;
    ctxend)    mode_ctxend ;;
    shapeacc)  mode_shapeacc ;;
    ub256k)    mode_ub256k ;;
    ubfrontier) mode_ubfrontier ;;
    fitab)     mode_fitab ;;
    k512)      mode_256k512 ;;
    k256)      mode_256k256 ;;
    freespace) mode_freespace ;;
    candidate) mode_candidate ;;
    gates)     mode_gates ;;
    adapt)     mode_adapt ;;
    pminsweep) mode_pminsweep ;;
    pminctl)   mode_pminctl ;;
    chainab)   mode_chainab ;;
    *) echo "unknown mode: $MODE"; exit 1 ;;
esac
echo "MV_DONE mode=$MODE"
