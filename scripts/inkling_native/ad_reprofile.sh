#!/usr/bin/env bash
# Re-profile the Inkling-2-Mini hot path after e046294a7 (docs/mxfp8/28-post-fix-reprofile.md).
#
# Phases are profiled in separate runs so each trace window is unambiguous:
#   prefill-<fa> : 2,817-token prompt, -n 1        (same prompt as lane AB's logs/ab-after-*.log)
#   decode-<fa>  : 12-token prompt,   -n 64
#   long-prefill : ~16k-token prompt (wiki.test.slice64k.raw), -c 18432, fp8 KV
#
# -ub 512 is the build default (common/common.h:480) and the shape lane AB measured 901/40 t/s at;
# -ub 2048 with FA off cannot be profiled at all because the auto-fitter crashes (doc 29).
#
# Full stdout+stderr of every step goes to logs/ (tee'd by the caller), nothing is filtered.
set -u

M=/home/tim/models/timlikesai/Inkling-2-Mini-NATIVE.gguf
LONG=logs/g0-prompt-long.txt
cd "$(dirname "$0")/../.."

profile() { # tag, tokens, extra args...
    local tag=$1 tok=$2; shift 2
    echo "### $tag"
    nsys profile --trace=cuda --force-overwrite=true --export=sqlite -o "logs/ad-p2-$tag" \
        ./build-cuda/bin/llama-completion -m "$M" -b 2048 -ub 512 -sm layer \
        --temp 0 -s 42 -no-cnv "$@" \
        > "logs/ad-p2-$tag-run.log" 2>&1
    echo "  run rc=$?"
    python3 scripts/inkling_native/ad_nsys_report.py "logs/ad-p2-$tag.sqlite" --window whole \
        --tokens-prompt "$tok" > "logs/ad-p2-$tag-report.txt" 2>&1
    echo "  report rc=$? -> logs/ad-p2-$tag-report.txt"
    python3 scripts/inkling_native/ab_peer_traffic.py "logs/ad-p2-$tag.sqlite" \
        > "logs/ad-p2-$tag-peer.txt" 2>&1
    echo "  peer   -> logs/ad-p2-$tag-peer.txt"
}

profile prefill-faon  2817 -c 8192  -f "$LONG"                        -fa on  -ctk f16      -ctv f16      -n 1
profile prefill-faoff 2817 -c 8192  -f "$LONG"                        -fa off -ctk f16      -ctv f16      -n 1
profile decode-faon   64   -c 8192  -p "Q: What is 17 times 4? A:"     -fa on  -ctk f16      -ctv f16      -n 64
profile decode-faoff  64   -c 8192  -p "Q: What is 17 times 4? A:"     -fa off -ctk f16      -ctv f16      -n 64
profile long-faon     16000 -c 18432 -f logs/corpora/wiki.test.slice64k.raw -fa on -ctk f8_e4m3 -ctv f8_e4m3 -n 1
echo "ALL_DONE"
