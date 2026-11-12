SHELL := /bin/bash
TRAIN := uv run python -m skin_lesion_uncertainty.train
RESNET := --arch resnet18 --size 64 --lr 3e-4
SEEDS := 0 1 2

.PHONY: setup format lint test data figures baselines train experiments final eval reproduce

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

data:  # download DermaMNIST at 28 and 64 pixels
	uv run python -c "from skin_lesion_uncertainty.data import make_loaders; make_loaders(size=28); make_loaders(size=64)"

figures:  # class counts and example images
	uv run python -m skin_lesion_uncertainty.figures

baselines:
	uv run python -m skin_lesion_uncertainty.baselines

train:  # one CNN run with the default settings
	$(TRAIN)

experiments:  # CNN with and without class weights, and ResNet-18; three seeds each
	for seed in $(SEEDS); do \
		$(TRAIN) --seed $$seed --save-logits artifacts/cw_on_seed$$seed && \
		$(TRAIN) --seed $$seed --no-class-weights --save-logits artifacts/cw_off_seed$$seed && \
		$(TRAIN) $(RESNET) --seed $$seed --save-logits artifacts/resnet_cw_on_seed$$seed \
		|| exit 1; \
	done
	uv run python -m skin_lesion_uncertainty.report

final:  # the frozen model (ResNet-18, seed 0) with test logits; run after experiments
	$(TRAIN) $(RESNET) --seed 0 --save-logits artifacts/resnet_final --save-test-logits

eval:  # calibration figure, final test evaluation, shift stress test
	uv run python -m skin_lesion_uncertainty.calibration
	set -o pipefail; uv run python -m skin_lesion_uncertainty.final_evaluation | tee reports/test_results.md
	set -o pipefail; uv run python -m skin_lesion_uncertainty.shift | tee reports/shift_results.md

reproduce: setup data figures baselines experiments final eval
