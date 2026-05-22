PYTHON = .venv/bin/python

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

.venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

install: .venv

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

pipeline: install
	$(PYTHON) scripts/01_data_exploration.py
	$(PYTHON) scripts/02_clean_data.py
	$(PYTHON) scripts/03_feature_engineering.py
	$(PYTHON) scripts/04_baseline_model.py
	$(PYTHON) scripts/05_model_comparison.py
	$(PYTHON) scripts/06_error_analysis.py

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	rm -rf data/processed reports/figures/*.png

.PHONY: install pipeline clean
