PY ?= python3
# bit-exact reproducibility (see run.sh)
export OMP_NUM_THREADS = 1

.PHONY: setup pipeline quick serve test figures architecture presentation ablations all

setup:            ## create a virtual environment and install dependencies
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

pipeline:         ## full run: 200 patients x 28 days, 3 sensor tiers, 5-fold CV (~25 min)
	$(PY) scripts/run_pipeline.py

quick:            ## smoke test: 60 patients (~5 min)
	$(PY) scripts/run_pipeline.py --quick

ablations:        ## the two honesty ablations: what misspecification and messy eating cost us
	$(PY) scripts/run_pipeline.py --matched-model --tag _matched_model
	$(PY) scripts/run_pipeline.py --easy-meals    --tag _easy_meals

serve:            ## clinician dashboard on http://localhost:8000
	$(PY) serve.py

test:
	$(PY) -m pytest -q tests

figures:
	$(PY) scripts/make_figures.py

architecture:
	$(PY) docs/make_architecture.py

presentation:     ## build docs/presentation.pdf from reports/metrics.json
	$(PY) docs/make_presentation.py

all: pipeline architecture presentation test
