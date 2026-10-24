#!/usr/bin/env bash
# One-command setup + run for macOS / Linux.
#   bash run.sh            install, build everything if missing, start the dashboard
#   bash run.sh --quick    ~4-minute smoke-test build (60 patients), then dashboard
#   bash run.sh --rebuild  force a full rebuild
set -euo pipefail
cd "$(dirname "$0")"
ARG="${1:-}"

# Bit-exact reproducibility: without this, HistGradientBoosting's thread scheduling moves the
# 4th decimal of a few metrics between runs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# --- pick a supported interpreter -------------------------------------------------------------
# macOS still ships Python 3.9 as `python3`. This project needs 3.10+, and on 3.9 pip resolves an
# older pandas whose pickle format cannot read artefacts written by a newer one, which surfaces
# much later as a confusing 500 from the dashboard. Fail here instead, with instructions.
pick_python() {
  for c in python3.13 python3.12 python3.11 python3.10; do
    command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }
  done
  if command -v python3 >/dev/null 2>&1 && \
     python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then
    echo python3; return
  fi
  echo ""
}

if [ ! -d .venv ]; then
  PY="$(pick_python)"
  if [ -z "$PY" ]; then
    echo "ERROR: this project needs Python 3.10 or newer; none was found." >&2
    echo "       macOS ships 3.9 as python3. Install a newer one, then re-run:" >&2
    echo "         brew install python@3.12   # or download from python.org" >&2
    exit 1
  fi
  echo "Creating .venv with $PY ($($PY --version 2>&1))"
  "$PY" -m venv .venv
fi

source .venv/bin/activate

if ! python -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)'; then
  echo "ERROR: .venv was built with $(python --version 2>&1), but this project needs 3.10+." >&2
  echo "       Delete it and re-run:   rm -rf .venv && bash run.sh" >&2
  exit 1
fi

pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# --- decide whether to rebuild ----------------------------------------------------------------
# Rebuild when the artefacts the dashboard needs are missing. reports/metrics.json is committed to
# the repo, so it must NOT be the thing we test for - otherwise a fresh clone skips the build and
# serve.py then exits with "Artifacts missing".
need_build=0
for f in artifacts/twin_A_full.npz artifacts/predictions.npz data/cohort/ehr.csv reports/metrics.json; do
  [ -f "$f" ] || need_build=1
done
# Also rebuild when the artefacts on disk were written by a different pandas/scikit-learn than the
# one installed now. Their pickle formats are not compatible across major versions.
if [ "$need_build" = "0" ] && ! python scripts/check_build_env.py --quiet; then
  echo "Artefacts on disk were built with a different Python/pandas/scikit-learn. Rebuilding."
  need_build=1
fi

if [ "$ARG" = "--quick" ]; then
  python scripts/run_pipeline.py --quick
elif [ "$need_build" = "1" ] || [ "$ARG" = "--rebuild" ]; then
  echo "Building the cohort, twins and models. This takes about 15 minutes on a laptop CPU."
  echo "In a hurry? Ctrl-C and run:  bash run.sh --quick   (60 patients, about 4 minutes)"
  python scripts/run_pipeline.py
fi

python docs/make_architecture.py
python serve.py
