.PHONY: help setup_dev lint lint_fix type complexity test test_coverage quick_validate validate clean

PY := uv run --active

help:
	@echo "setup_dev      Install dev dependencies via uv"
	@echo "lint           Ruff lint (read-only)"
	@echo "lint_fix       Ruff lint + format (writes changes)"
	@echo "type           Pyright strict"
	@echo "complexity     Complexipy (cognitive complexity ≤10)"
	@echo "test           pytest"
	@echo "test_coverage  pytest --cov (branch + term-missing)"
	@echo "quick_validate lint + type (fast dev loop)"
	@echo "validate       lint + type + complexity + test_coverage (full gate)"
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

quick_validate: lint type

validate: lint type complexity test_coverage

clean:
	rm -rf .pytest_cache .ruff_cache .pyright .complexipy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
