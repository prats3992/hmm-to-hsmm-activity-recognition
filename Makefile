PYTHON ?= python

.PHONY: all eda gaussian-hmm gmm-hmm hsmm compare test clean

all: eda gaussian-hmm gmm-hmm hsmm compare

eda:
	$(PYTHON) scripts/01_exploratory_analysis.py

gaussian-hmm:
	$(PYTHON) scripts/02_gaussian_hmm_assumptions.py
	$(PYTHON) scripts/03_gaussian_hmm.py

gmm-hmm:
	$(PYTHON) scripts/04_gmm_hmm_assumptions.py
	$(PYTHON) scripts/05_gmm_hmm.py

hsmm:
	$(PYTHON) scripts/06_hsmm_assumptions.py
	$(PYTHON) scripts/07_hsmm.py

compare:
	$(PYTHON) scripts/08_baselines_and_comparison.py

test:
	$(PYTHON) -m pytest

clean:
	rm -rf outputs/figures outputs/tables
