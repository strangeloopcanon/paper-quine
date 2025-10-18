PYTHON ?= python3.11
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: setup check test llm-live deps-audit all clean

setup:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -e ".[dev]"
	@if [ -d .git ]; then \
		$(VENV)/bin/pre-commit install --hook-type pre-commit --hook-type commit-msg; \
	else \
		echo "Skipping pre-commit install because .git directory is missing."; \
	fi

check:
	@$(VENV)/bin/black --check src tests
	@$(VENV)/bin/ruff check src tests
	@$(VENV)/bin/mypy src
	@$(VENV)/bin/bandit -q -r src
	@$(VENV)/bin/detect-secrets scan --baseline .secrets.baseline

test:
	@$(VENV)/bin/pytest

llm-live:
	@$(PY) -m paper_quine.verify --llm-live

deps-audit:
	@$(VENV)/bin/pip-audit --skip-editable --ignore-vuln GHSA-4xh5-x5gv-qwph

all: check test llm-live

clean:
	@rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache coverage.xml htmlcov
