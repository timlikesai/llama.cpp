#!/bin/bash
# Is the DFlare oracle in tests/test-dflare-mix.cpp load-bearing? Break src/models/dflash.cpp two ways, show the
# test catches both, restore and show it passes again. Everything is teed to logs/, nothing is deleted.
#
# The change is committed, so HEAD holds it and docs/mxfp8/drafter-dflare-mix.patch must stay equal to HEAD's
# src diff - the restore checks that by reverse-applying. Regenerate it with: git diff HEAD~1 -- src > patch
#
# usage: scripts/inkling_native/check-dflare-oracle.sh   (from the repository root)
set -u
cd /home/tim/code/services/repos/llama.cpp

patch=docs/mxfp8/drafter-dflare-mix.patch
f=src/models/dflash.cpp

restore () {
    git checkout -- src/models/dflash.cpp src/models/models.h
    git apply --reverse --check "$patch" || { echo "RESTORE FAILED: src does not match $patch"; exit 1; }
    echo "restored, src verified against $patch"
}

run () { # $1 = log suffix
    cmake --build build-cuda --target test-dflare-mix -j 16 > "logs/dflare-mutation-$1-build.log" 2>&1
    echo "$1 build exit=$?"
    ./build-cuda/bin/test-dflare-mix > "logs/dflare-mutation-$1.log" 2>&1
    echo "$1 test exit=$?  (134 = GGML_ASSERT)"
    grep -E "max err|first mismatch|GGML_ASSERT" "logs/dflare-mutation-$1.log" | head -8
}

mutate () { # $1 = old line, $2 = new line
    python3 - "$1" "$2" <<'PY'
import pathlib, sys
p = pathlib.Path('src/models/dflash.cpp')
s = p.read_text()
assert s.count(sys.argv[1]) == 1, "old line not found exactly once"
p.write_text(s.replace(sys.argv[1], sys.argv[2]))
PY
}

echo "=== M1: hidden and taps swapped in the tap split (shape level) ==="
mutate "ggml_tensor * taps = ggml_reshape_3d(ctx0, feats, n_embd, n_taps, n_tokens);" \
       "ggml_tensor * taps = ggml_reshape_3d(ctx0, feats, n_taps, n_embd, n_tokens);"
run M1-shape

restore

echo "=== M2: wrong dspark_fuse column per stage (numeric only, shapes stay valid) ==="
mutate "    ggml_tensor * cur = ggml_mul(ctx0, taps, ggml_view_3d(ctx0, alpha, n_taps, 1, 1, alpha->nb[1], alpha->nb[2], i_mix*alpha->nb[1]));" \
       "    ggml_tensor * cur = ggml_mul(ctx0, taps, ggml_view_3d(ctx0, alpha, n_taps, 1, 1, alpha->nb[1], alpha->nb[2], (i_mix/2)*alpha->nb[1]));"
run M2-wrong-column

echo "=== final state ==="
restore
git diff --stat src
run restored
