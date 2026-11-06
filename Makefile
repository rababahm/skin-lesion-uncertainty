.PHONY: setup format lint test data train eval reproduce

setup:
	uv sync

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

data:
	@echo "TODO: download and prepare data"

train:
	@echo "TODO: train models"

eval:
	@echo "TODO: evaluate models"

reproduce: setup data train eval

.PHONY: baselines
baselines:
	uv run python -m skin_lesion_uncertainty.baselines

figures:
	uv run python -m skin_lesion_uncertainty.figures
