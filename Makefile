.PHONY: help setup_dev lint lint_fix type complexity test test_coverage quick_validate validate clean secrets install_tools bandit

PY := uv run --active

# Tool paths. Gitleaks is installed via Makefile to ``~/.local/bin``
# (or ``/usr/local/bin`` if the user has sudo). We honour the user's
# PATH first, then fall back to common install locations.
GITLEAKS ?= $(shell command -v gitleaks 2>/dev/null || echo "$(HOME)/.local/bin/gitleaks")

help:
	@echo "setup_dev      Install dev dependencies via uv"
	@echo "lint           Ruff lint (read-only)"
	@echo "lint_fix       Ruff lint + format (writes changes)"
	@echo "type           Pyright strict"
	@echo "complexity     Complexipy (cognitive complexity ≤10)"
	@echo "test           pytest"
	@echo "test_coverage  pytest --cov (branch + term-missing)"
	@echo "secrets        Run gitleaks secret scan (fails on any leak)"
	@echo "bandit         Run bandit Python SAST scan (fails on any high issue)"
	@echo "install_tools  Install gitleaks + bandit CLI tools (gitleaks is required)"
	@echo "quick_validate lint + type + secrets + bandit (fast dev loop)"
	@echo "validate       lint + type + complexity + test_coverage + secrets + bandit (full gate)"
	@echo "clean          Remove caches"

setup_dev:
	uv sync --dev

lint:
	$(PY) ruff format --check src/ tests/
	$(PY) ruff check src/ tests/

lint_fix:
	$(PY) ruff format src/ tests/
	$(PY) ruff check --fix src/ tests/

type:
	$(PY) pyright src/

complexity:
	$(PY) complexipy src/

test:
	$(PY) pytest -q

test_coverage:
	$(PY) pytest -q --cov --cov-branch --cov-report=term-missing

# -----------------------------------------------------------------------------
# Secret scanning (gitleaks). The .gitleaks.toml config at the repo root
# is the canonical rule set.
# -----------------------------------------------------------------------------
# gitleaks 8.30.1's ``detect`` subcommand has a known bug: it returns
# exit code 0 even when leaks are found in --no-git mode. The workaround
# is to use ``--report-format json`` which DOES return exit code 1
# when leaks are present, and to count the JSON findings to surface
# the count in the CI log.
secrets:
	@if [ ! -x "$(GITLEAKS)" ]; then \
		echo "gitleaks not found at $(GITLEAKS) — run \`make install_tools\`"; \
		exit 2; \
	fi
	@if [ ! -f .gitleaks.toml ]; then \
		echo ".gitleaks.toml not found"; exit 2; \
	fi
	@report=$$(mktemp); \
	if $(GITLEAKS) detect \
		--config .gitleaks.toml \
		--source . \
		--no-git \
		--report-path $$report \
		--report-format json 2>/dev/null; then \
		rm -f $$report; \
		echo "✓ gitleaks: no secrets found"; \
	else \
		count=$$(python3 -c "import json;d=json.load(open('$$report'));print(len(d) if isinstance(d,list) else 1)" 2>/dev/null || echo "1+"); \
		echo "❌ gitleaks: $$count secret(s) found in working tree"; \
		cat $$report 2>/dev/null | head -50; \
		rm -f $$report; \
		exit 1; \
	fi

bandit:
	@if [ ! -x "$(shell command -v bandit 2>/dev/null || echo bandit)" ]; then \
		echo "bandit not installed — run \`make install_tools\`"; \
		exit 2; \
	fi
	@if [ -d src ]; then \
		$(PY) bandit -r src/ -ll -ii 2>/dev/null || exit 1; \
	else \
		echo "no src/ directory; skipping bandit"; \
	fi

# Install gitleaks + bandit. Idempotent. Both are required by
# `make secrets` and `make bandit` respectively. The script also
# tries to install gitleaks system-wide via sudo if available; if
# not, it falls back to ``~/.local/bin/gitleaks`` which is already
# in PATH.
install_tools:
	@echo "Installing gitleaks..."
	@if [ ! -x "$(GITLEAKS)" ]; then \
		mkdir -p $(HOME)/.local/bin; \
		curl -sL https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz \
			| tar -xz -C $(HOME)/.local/bin gitleaks; \
		chmod +x $(HOME)/.local/bin/gitleaks; \
		echo "  installed to $(HOME)/.local/bin/gitleaks"; \
	else \
		echo "  gitleaks already present at $(GITLEAKS)"; \
	fi
	@echo "Installing bandit..."
	@$(PY) pip install --quiet bandit 2>&1 | tail -3
	@echo "  bandit installed: $$($(PY) bandit --version 2>&1)"

quick_validate: lint type secrets bandit

validate: lint type complexity test_coverage secrets bandit

clean:
	rm -rf .pytest_cache .ruff_cache .pyright .complexipy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
