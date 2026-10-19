PY ?= python3
# bit-exact reproducibility (see run.sh)
export OMP_NUM_THREADS = 1

.PHONY: setup pipeline quick serve test figures architecture presentation ablations fhir all

setup:            ## create a virtual environment and install dependencies
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

pipeline:         ## full run: 200 patients x 28 days, 3 sensor tiers, 5-fold CV (~100 min)
	$(PY) scripts/run_pipeline.py

quick:            ## smoke test: 60 patients (~5 min)
	$(PY) scripts/run_pipeline.py --quick

ablations:        ## what an easier benchmark would have scored (-> reports/BENCHMARK_ABLATIONS.md)
	$(PY) scripts/run_pipeline.py --quick                              --tag _q_hard
	$(PY) scripts/run_pipeline.py --quick --matched-model              --tag _q_matched
	$(PY) scripts/run_pipeline.py --quick --easy-meals                 --tag _q_easy
	$(PY) scripts/run_pipeline.py --quick --matched-model --easy-meals --tag _q_both
	$(PY) scripts/compare_ablations.py

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

fhir:             ## validate the exported bundles against the real FHIR R4 schema (optional dep)
	$(PY) -m pip install -q "fhir.resources==7.1.0" && $(PY) scripts/validate_fhir.py

all: pipeline architecture presentation test
