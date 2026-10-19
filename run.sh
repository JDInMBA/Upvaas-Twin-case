#!/usr/bin/env bash
# One-command setup + run for macOS / Linux.
#   bash run.sh            install, build everything if missing, start the dashboard
#   bash run.sh --quick    ~5-minute smoke-test build (60 patients), then dashboard
#   bash run.sh --rebuild  force a full rebuild
set -euo pipefail
cd "$(dirname "$0")"
ARG="${1:-}"

# Bit-exact reproducibility: without this, HistGradientBoosting's thread scheduling moves the
# 4th decimal of a few metrics between runs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# Rebuild when the artefacts the dashboard actually needs are missing. reports/metrics.json is
# committed to the repo, so it must NOT be the thing we test for - otherwise a fresh clone skips
# the build and serve.py then exits with "Artifacts missing".
need_build=0
for f in artifacts/twin_A_full.npz artifacts/predictions.npz data/cohort/ehr.csv reports/metrics.json; do
  [ -f "$f" ] || need_build=1
done

if [ "$ARG" = "--quick" ]; then
  python scripts/run_pipeline.py --quick
elif [ "$need_build" = "1" ] || [ "$ARG" = "--rebuild" ]; then
  echo "Building the cohort, twins and models. This takes roughly 25 minutes on a laptop CPU."
  echo "In a hurry? Ctrl-C and run:  bash run.sh --quick   (60 patients, about 5 minutes)"
  python scripts/run_pipeline.py
fi

python docs/make_architecture.py
python serve.py
