#!/bin/bash
# GPT-J 6B leakage measurement.
# The 2026-09-27 run was OOM-killed: read README.md ("Measurement FAILED — GPT-J 6B")
# and apply both fixes to measure_gptj.py before trusting a rerun. The weights are no
# longer cached, so this re-downloads ~12 GB (fp16) before it measures anything.
#
# Runs from wherever the directory lives; every script here reads bare filenames
# (corpus.txt, measure.json, ...) out of the cwd, so the cd is load-bearing.
cd "$(dirname "$0")" || exit 1
PY=${PY:-./.venv/bin/python}
if [ ! -x "$PY" ]; then
  echo "no interpreter at $PY — recreate it, see README.md (Environment)" >&2
  exit 1
fi
"$PY" measure_gptj.py 2>&1 | grep -v -i -E "warn|urllib3|Token indices"
exit "${PIPESTATUS[0]}"
