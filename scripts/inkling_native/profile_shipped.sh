#!/usr/bin/env bash
# Profile the SHIPPED Inkling configuration, not a hand-written approximation of it.
#
# Why this exists: the earlier profile scripts (ad_reprofile.sh and friends) pointed at
# Inkling-2-Mini-NATIVE.gguf, which no longer exists, and hand-listed flags that had since
# drifted from the deployment. Every perf conclusion in docs/mxfp8/2x-3x was measured against
# a config nobody runs. So: read the flag vector straight out of docker-compose.yml, launch
# llama-server under nsys with exactly those args, drive it with real HTTP requests, and stop
# it. If the deployment changes, this follows automatically.
#
# Requires the GPUs to be free: `docker compose down llama-cpp` first (the model holds ~30 GiB
# across both cards, so a second load cannot coexist with the container).
#
# Everything is tee'd to logs/ and nothing is filtered or deleted.
#
# usage: scripts/inkling_native/profile_shipped.sh [prefill|decode|both|parse]
set -u

REPO=/home/tim/code/services/repos/llama.cpp
COMPOSE=/home/tim/code/services/docker-compose.yml
PORT=8099
HOST=127.0.0.1
MODE=${1:-both}
cd "$REPO"

# The container's ENTRYPOINT is /app/tools.sh, whose FIRST argument is a selector, not a model
# flag: '--server' execs ./llama-server, '--run' execs ./llama-cli, '--bench' ./llama-bench, etc.
# '--server' is not defined in common/arg.cpp at all, so passing it to either binary fails with
# "error: invalid argument: --server". Anyone copying the compose command: to a native run must
# strip that first token (it is dropped by DROP below).
SERVER_BIN=./build-cuda/bin/llama-server
[ -x "$SERVER_BIN" ] || { echo "FATAL: $SERVER_BIN not built"; exit 1; }

# ---------------------------------------------------------------- 1. flag vector
# The compose `command:` is a list of separate items, so a token starting with '-' is a flag and
# the next token, if it does not start with '-', is its value. Booleans are followed by another
# flag, so they drop cleanly. Container-only flags are removed by name; host/port are re-added.
python3 - "$COMPOSE" > logs/ps-args.txt <<'PY' || { echo "FATAL: could not read compose command"; exit 1; }
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
cmd = list(d['services']['llama-cpp']['command'])

# split into (flag, value_or_None) pairs
items, i = [], 0
while i < len(cmd):
    tok = cmd[i]
    if tok.startswith('-'):
        val = cmd[i + 1] if i + 1 < len(cmd) and not cmd[i + 1].startswith('-') else None
        items.append((tok, val)); i += 2 if val is not None else 1
    else:
        items.append((tok, None)); i += 1

# what only makes sense inside the container, or would fight the profiler
DROP = {
    '--server',                                     # tools.sh entrypoint selector, not a model flag
    '--host', '--port', '--ui', '--ui-mcp-proxy', '--allow-unauthenticated-tools',
    '--cors-origins', '--slot-save-path',           # container-local paths / auth shims
    '--metrics',                                    # adds timing overhead to the trace
}
# multimodal is a CPU cost and adds ~40 s of projector load per profile; keep it out of the
# GPU attribution by default (set PROFILE_VISION=1 to include it)
if not __import__('os').environ.get('PROFILE_VISION'):
    DROP |= {'--mmproj', '--no-mmproj-offload'}
out = []
for flag, val in items:
    if flag in DROP:
        continue
    out.append(flag)
    if val is not None:
        out.append(val)
print('\n'.join(out))
PY
mapfile -t ARGS < logs/ps-args.txt
[ "${#ARGS[@]}" -gt 8 ] || { echo "FATAL: parsed only ${#ARGS[@]} args from $COMPOSE"; exit 1; }
echo "flag vector: ${#ARGS[@]} args, taken from $COMPOSE"
printf '  %s\n' "${ARGS[@]}"
# container paths -> host paths
for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in /models/*) ARGS[$i]="/home/tim${ARGS[$i]}" ;; esac
done
for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in /*) [ -e "${ARGS[$i]}" ] || { echo "FATAL: path in flag vector does not exist: ${ARGS[$i]}"; exit 1; } ;; esac
done

if [ "$MODE" = "parse" ]; then
    echo "PARSE_ONLY ok: ${#ARGS[@]} args, all paths exist"
    exit 0
fi

# ---------------------------------------------------------------- 1b. knob sweep
# Every config here is measured on the shipped flags with only the named knob changed, at -c 262144
# and -v (without -v this build logs nothing but warnings, which reads as "no data" -- see docs/mxfp8/46).
# NOTE on GGML_CUDA_MMQ_PREC: it picks how src1 (the ACTIVATION) is quantized for the block-scaled MMA
# (mmq.cu:100-122). The default already gives the FP4 weight types native 4-bit activations (W4A4) and
# gives MXFP8 e4m3 activations (W8A8) - so this knob is NOT questioning whether our MXFP8 layers use fp8
# activations; they do. 'q8' is a dial on the 20 NVFP4 layers: more accurate activations, fewer ops/byte.
# Flags that were hypothesized but DO NOT exist in this tree and must not be re-added blind:
#   --moe-fused-sum, --moe-fused-up, --custom-moe, --nobd, --unified
# Those were checked against common/arg.cpp on 2026-11-23; the MoE route choice is internal (mmid.cu).
if [ "$MODE" = "sweep" ]; then
    set +e   # this function probes a server that is deliberately allowed to be not-ready
    run_one() { # tag, env assignments, args to append, args to drop (name + its value)
        local tag=$1; shift
        local envs=$1; shift
        local extra=$1; shift
        local drops=$1; shift
        local log="logs/sweep-${ROUND}-${tag}.log"
        # NOTE: never delete a previous sweep's logs (they are measurements). Each round writes its
        # own file names via ROUND, and one port per config, so a leftover server from an earlier run
        # cannot answer the health check and make a failed run look healthy (that produced a whole
        # bogus sweep on 2026-11-23 - see docs/mxfp8/46 and TODO #23).
        local port=$(( 8199 + (RANDOM % 300) ))
        echo "--- $tag: env[$envs] append[$extra] drop[$drops] (port $port)"
        local -a a=("${ARGS[@]}")
        if [ -n "$drops" ]; then
            a=()
            local skip=0
            for ((k=0; k<${#ARGS[@]}; k++)); do
                local cur="${ARGS[$k]}"
                if [ $skip = 1 ]; then skip=0; continue; fi
                local hit=0
                for d in $drops; do [ "$cur" = "$d" ] && hit=1; done
                if [ $hit = 1 ]; then
                    # also drop the following token if this flag takes a value (heuristic: next token
                    # does not start with --)
                    if [ $((k+1)) -lt ${#ARGS[@]} ] && [[ "${ARGS[$((k+1))]}" != --* ]]; then skip=1; fi
                    continue
                fi
                a+=("$cur")
            done
        fi
        # shellcheck disable=SC2086
        env $envs "$SERVER_BIN" "${a[@]}" $extra --host "$HOST" --port "$port" -v < /dev/null > "$log" 2>&1 &
        local pid=$! ok=0 hc=""
        for i in $(seq 1 90); do
            if ! kill -0 $pid 2>/dev/null; then echo "    server exited early (see $log)"; break; fi
            hc=$(curl -s -m 3 -o "$log.health" -w '%{http_code}' "http://$HOST:$port/health" || true)
            if [ "$hc" = "200" ] && grep -q '"ok"' "$log.health" 2>/dev/null; then ok=1; echo "    healthy after ${i} polls"; break; fi
            sleep 2
        done
        if [ "$ok" = "1" ]; then
            curl -s --max-time 900 -H 'Content-Type: application/json' -o "$log.json" \
                -w 'HTTP_CODE=%{http_code} time_total=%{time_total}\n' \
                "http://$HOST:$port/v1/chat/completions" < /dev/null > "$log.req" 2>&1 --data-binary @logs/sweep-request.json
            echo "    request: $(cat "$log.req" 2>/dev/null) resp=$(wc -c < "$log.json" 2>/dev/null || echo 0)B"
        else
            echo "    NOT healthy: last http_code=$hc (see $log)"
        fi
        curl -s -X POST "http://$HOST:$port/shutdown" > /dev/null 2>&1 < /dev/null
        kill $pid 2>/dev/null; wait $pid 2>/dev/null
    }
    python3 -c "import json; json.dump({'messages':[{'role':'user','content':open('logs/prompt-code.txt').read()}],'max_tokens':192,'temperature':0,'seed':42,'cache_prompt':False,'timings':True}, open('logs/sweep-request.json','w'))" || { echo 'request JSON build failed'; exit 1; }
    echo "request JSON: $(wc -c < logs/sweep-request.json) bytes"
    ROUND=${ROUND:-r2}
    # round 2: the SERVER-ONLY flags that no CLI measurement ever included, plus thread count.
    run_one baseline      ""                                  ""                                    ""
    run_one no-ckpt       ""                                  "" "--ctx-checkpoints --checkpoint-min-step --cache-idle-slots --cache-reuse"
    run_one ngram-off     ""                                  "--spec-type draft-dspark"              "--spec-type"
    run_one threads-16    ""                                  "--threads 16"                          "--threads"
    run_one threads-24    ""                                  "--threads 24"                          "--threads"
    echo "SWEEP_DONE"
    exit 0
fi

# ---------------------------------------------------------------- 2. profile run
echo
echo "=== starting llama-server under nsys, args mirrored from compose (mode=$MODE) ==="
# '--' keeps nsys from interpreting the target's flags as its own.
nsys profile --trace=cuda,nvtx --force-overwrite=true --export=sqlite -o logs/ps-$MODE -- \
    "$SERVER_BIN" "${ARGS[@]}" --host "$HOST" --port "$PORT" \
    > logs/ps-$MODE-server.log 2>&1 &
NSYS_PID=$!

echo "waiting for /health ..."
for i in $(seq 1 90); do
    if curl --compressed -s -m 3 "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
        echo "healthy after ~$((i*2))s"; break
    fi
    sleep 2
    if ! kill -0 $NSYS_PID 2>/dev/null; then
        echo "FATAL: server died during startup; see logs/ps-$MODE-server.log"
        exit 1
    fi
done
curl --compressed -s "http://$HOST:$PORT/health" || { echo "FATAL: never became healthy"; exit 1; }

# ---------------------------------------------------------------- 3. workload
workload() {
    local tag=$1 body=$2
    echo "--- request: $tag"
    curl --compressed -s "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$body" > "logs/ps-req-$tag.json" 2>&1
    echo "    rc=$? (response saved to logs/ps-req-$tag.json)"
}

CODE=$(python3 -c "import json,sys; print(json.dumps(open('logs/prompt-code.txt').read()))")
LONG=$(python3 -c "import json; print(json.dumps(open('logs/corpora/wiki.test.slice64k.raw').read()))")

if [ "$MODE" = "prefill" ] || [ "$MODE" = "both" ]; then
    # cold cache, one large prompt: this is the tensor-core / memory-movement path
    workload prefill-64k "{\"messages\":[{\"role\":\"user\",\"content\":$LONG}],\"max_tokens\":1,\"temperature\":0,\"cache_prompt\":false}"
fi
if [ "$MODE" = "decode" ] || [ "$MODE" = "both" ]; then
    # cold cache each time so the drafter, not the prompt cache, dominates the sample
    for r in 1 2 3; do
        workload "decode-r$r" "{\"messages\":[{\"role\":\"user\",\"content\":$CODE}],\"max_tokens\":256,\"temperature\":0,\"seed\":42,\"cache_prompt\":false}"
    done
fi

# ---------------------------------------------------------------- 4. stop + analyze
echo
echo "=== stopping the server so nsys finalizes the trace ==="
pkill -INT -f "llama-server.*--port $PORT" 2>/dev/null
for i in $(seq 1 60); do
    kill -0 $NSYS_PID 2>/dev/null || break
    sleep 2
done
wait $NSYS_PID 2>/dev/null
echo "nsys finished (rc=$?)"
ls -la logs/ps-$MODE.nsys-rep logs/ps-$MODE.sqlite 2>&1

echo
echo "=== kernel + API attribution ==="
python3 scripts/inkling_native/ad_nsys_report.py "logs/ps-$MODE.sqlite" --window whole > "logs/ps-$MODE-kernels.txt" 2>&1
echo "kernels -> logs/ps-$MODE-kernels.txt (rc=$?)"
python3 scripts/inkling_native/z_sync_attrib.py "logs/ps-$MODE.sqlite" > "logs/ps-$MODE-sync.txt" 2>&1
echo "syncs   -> logs/ps-$MODE-sync.txt (rc=$?)"
python3 scripts/inkling_native/ab_peer_traffic.py "logs/ps-$MODE.sqlite" > "logs/ps-$MODE-peer.txt" 2>&1
echo "peer    -> logs/ps-$MODE-peer.txt (rc=$?)"
python3 scripts/inkling_native/w_nsys_gaps.py "logs/ps-$MODE.sqlite" > "logs/ps-$MODE-gaps.txt" 2>&1
echo "gaps    -> logs/ps-$MODE-gaps.txt (rc=$?)"

echo
echo "=== server-side timings as recorded (last lines) ==="
tail -40 "logs/ps-$MODE-server.log"
echo "PROFILE_DONE mode=$MODE"
