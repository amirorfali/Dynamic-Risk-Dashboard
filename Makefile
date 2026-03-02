.PHONY: help setup test lint fmt fmt-check run-api run-dashboard dev

VENV := ./venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
BLACK := $(VENV)/bin/black

help:
	@echo "Targets:"
	@echo "  setup         Create venv and install dependencies"
	@echo "  test          Run pytest"
	@echo "  lint          Run ruff lint"
	@echo "  fmt           Auto-format with black"
	@echo "  fmt-check     Verify formatting with black"
	@echo "  run-api       Start FastAPI server"
	@echo "  run-dashboard Start Streamlit dashboard"
	@echo "  dev           Start API + dashboard together"

setup:
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) -q

lint:
	$(RUFF) check .

fmt:
	$(BLACK) .

fmt-check:
	$(BLACK) --check .

run-api:
	$(PYTHON) -m uvicorn app.main:app --reload

run-dashboard:
	$(PYTHON) -m streamlit run dashboard/streamlit_app.py

dev:
	./scripts/dev.sh
