# 28 — Re-profile the Inkling hot path after `e046294a7`

Status: **running.** `00-status.md` (14:40): "at 901/40 t/s the earlier '5-10% of peak' framing is
obsolete, so the new hot path needs re-profiling before more optimization is attempted."

Question: with the alias-straddling peer traffic gone (57,353 -> 8,867 MiB, 6,991 -> 972 syncs),
what actually limits prefill and decode now?

Harness: `scripts/inkling_native/ad_reprofile.sh` (nsys, CUDA trace only, one run per phase so the
window is unambiguous) + `scripts/inkling_native/ad_nsys_report.py` (busy/idle per device, top
kernels, op-class rollup, runtime API counts) + lane AB's `scripts/inkling_native/ab_peer_traffic.py`.
Artifacts: `logs/ad-p2-*.{nsys-rep,sqlite,txt,log}`, plus the first pass
`logs/ad-prof-faon.{nsys-rep,sqlite}` (prefill+decode in one trace, `-ub 2048`).

## Config note

`n_ubatch` defaults to 512 (`common/common.h:480`). Lane AB's 901.28 / 40.36 t/s (`logs/ac-verify-perf.log`)
ran with that default and `-b 2048`. Forcing `-ub 2048` on this model is not a neutral change: the
compute buffer roughly doubles (doc 29) and, with `-fa off`, the auto-fitter dies with SIGFPE, so the
matched FA-on/FA-off comparison below is done at the shipped `-ub 512`.

## Baseline (no profiler, 2,817-token prompt, `-c 8192 -b 2048 -sm layer`)

| | prefill t/s | decode t/s | source |
|---|---|---|---|
| TODO | | | |

## Where the time goes (nsys)

TODO - per phase: device busy %, idle attribution, top kernels, op-class rollup.

## Candidate optimizations, ranked by measured share

TODO

## Corrections to my brief

TODO
