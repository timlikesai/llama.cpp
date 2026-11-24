#!/bin/bash
# Why does the Inkling CUDA correctness probe print a different continuation than the standing gate string?
# A/B the same probe across (#38 | #38 reverted) x (FA auto | -fa off). Nothing is deleted; the restore is
# verified against docs/mxfp8/drafter-dflare-mix.patch, which must equal HEAD's src diff.
# usage: scripts/inkling_native/check-dflare-inkling-fa.sh    (from the repository root)
set -u
cd /home/tim/code/services/repos/llama.cpp

NATIVE=/home/tim/models/timlikesai/Inkling-2-Mini-NATIVE.gguf
patch=docs/mxfp8/drafter-dflare-mix.patch

probe () { # $1 = log suffix, $2 = extra flags
    build-cuda/bin/llama-completion -m "$NATIVE" -c 4096 --temp 0 -s 42 -n 24 -no-cnv $2 \
        -p "Q: What is 17 times 4? A:" < /dev/null > "logs/dflare-fa-$1.log" 2>&1
    echo "--- $1 (exit $?)"
    grep -h "Q: What is 17 times 4" "logs/dflare-fa-$1.log"
    grep -h "Flash Attention\|is assigned to device" "logs/dflare-fa-$1.log" | head -2
}

build () {
    cmake --build build-cuda --target llama-completion -j 16 > "logs/dflare-fa-$1-build.log" 2>&1
    echo "build $1 exit=$?"
}

revert () {
    git apply -R "$patch"
    echo "#38 reverted"
}

restore () {
    git checkout -- src/models/dflash.cpp src/models/models.h
    git apply --reverse --check "$patch" || { echo "RESTORE FAILED: src does not match $patch"; exit 1; }
    echo "restored, src verified against $patch"
}

echo "=== #38 (HEAD), FA auto ==="
build 38
probe 38-faon ""
probe 38-faoff "-fa off"

echo "=== #38 reverted, FA auto ==="
revert
build pre38
probe pre38-faon ""
probe pre38-faoff "-fa off"

echo "=== restore #38 ==="
restore
build restored
probe 38-faon-again ""

echo "=== final state ==="
git diff --stat src
