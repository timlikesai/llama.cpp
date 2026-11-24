#!/bin/bash
# Gates for todo #38 (DFlare per-stage target-feature mixing), docs/mxfp8/35-drafter-step2-fusion.md.
# One-shot binaries only; the llama-server gates are driven separately (tmux window per server).
# Every run is teed to logs/, nothing is deleted. usage: scripts/inkling_native/check-dflare-gates.sh
set -u
cd /home/tim/code/services/repos/llama.cpp

NATIVE=/home/tim/models/timlikesai/Inkling-2-Mini-NATIVE.gguf
INKLING_DRAFT=/home/tim/models/unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf

gate () { # name, then command
    local name=$1; shift
    echo "=== $name ==="
    "$@" > "logs/dflare-gate-$name.log" 2>&1
    echo "$name exit=$?"
}

gate dflare-ctest    bash -c 'cd build-cuda && ctest -R dflare --output-on-failure'
gate test-archs      ./build-cuda/bin/test-llama-archs
gate ops-mxfp8-cpu   ./build-cuda/bin/test-backend-ops -p mxfp8 -b CPU
gate ops-mxfp8-cuda0 ./build-cuda/bin/test-backend-ops -p mxfp8 -b CUDA0

# the standing Inkling correctness probe: must print A: 68 B: 64 C: 72 D: 70 E: 66 F: 69 G:
gate inkling-cuda bash -c "build-cuda/bin/llama-completion -m '$NATIVE' -c 4096 --temp 0 -s 42 -n 24 -no-cnv \
    -p 'Q: What is 17 times 4? A:' < /dev/null"
gate inkling-cpu bash -c "build-cuda/bin/llama-completion -m '$NATIVE' -c 4096 -ngl 0 --temp 0 -s 42 -n 24 -no-cnv \
    -p 'Q: What is 17 times 4? A:' < /dev/null"

# the Inkling draft must still refuse at graph build: exit != 0 and the named reason
gate inkling-refuse bash -c "build-cuda/bin/llama-server -m '$NATIVE' -md '$INKLING_DRAFT' \
    --spec-type draft-dspark --jinja -c 8192 --host 127.0.0.1 --port 8098 < /dev/null"

echo "=== summary ==="
grep -h "Summary:\|backends passed\|Backend CPU: OK\|Backend CUDA0: OK\|all DFlare\|tests passed\|drafts need" \
    logs/dflare-gate-dflare-ctest.log logs/dflare-gate-test-archs.log logs/dflare-gate-ops-mxfp8-cpu.log \
    logs/dflare-gate-ops-mxfp8-cuda0.log logs/dflare-gate-dflare-ctest.log 2>/dev/null
grep -h "Q: What is 17 times 4" logs/dflare-gate-inkling-cuda.log logs/dflare-gate-inkling-cpu.log
grep -h "markov head\|exiting due to" logs/dflare-gate-inkling-refuse.log
