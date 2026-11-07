# Uses the active Python (for example `uv run make verify` or an activated venv).
PYTHON ?= python

.PHONY: verify verify-quick test test-all benchmark figures

verify:            ## SBP residuals, inf-sup constants, heat-cavity orders (q = 4, 6, 8)
	$(PYTHON) code/verify.py

verify-quick:      ## same, without the q = 8 ladder
	$(PYTHON) code/verify.py --quick

test:              ## pytest numerical-validation suite
	$(PYTHON) -m pytest -q tests

test-all:          ## full regression suite: unit tests plus artifact verifiers
	$(PYTHON) code/run_all_tests.py

benchmark:         ## cost-accuracy benchmark of the heat-cavity march
	$(PYTHON) benchmarks/cost_accuracy.py

figures:           ## README result cards from code/results
	$(PYTHON) tools/make_readme_assets.py
