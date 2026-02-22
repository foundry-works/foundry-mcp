.PHONY: lint fmt test ci

## Run all lint checks (matches GitHub Actions lint workflow)
lint:
	ruff format --check src/ tests/
	ruff check src/ tests/ --statistics

## Auto-fix formatting and lint issues
fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

## Run test suite
test:
	python -m pytest tests/ --tb=short -q

## Run full local CI (lint + tests)
ci: lint test
