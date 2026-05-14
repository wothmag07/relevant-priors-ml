# Common dev tasks. On Windows without `make`, read this file and run
# the commands directly — it's a short list and each target is one command.

.PHONY: help install install-dev test lint typecheck check eval eval-heuristic \
        train train-save plots serve docker-build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production deps only (what ships in the Docker image)
	pip install -r requirements.txt

install-dev: ## Install production + dev deps (for local work)
	pip install -r requirements-dev.txt

test: ## Run unit + contract tests
	pytest tests/ -q

lint: ## Run ruff
	ruff check .

typecheck: ## Run mypy
	mypy app eval

check: lint typecheck test ## Lint + typecheck + tests (matches CI)

eval: ## Score the LightGBM classifier on the public split
	python -m eval.run_eval --predictor classifier

eval-heuristic: ## Score the heuristic baseline on the public split
	python -m eval.run_eval --predictor heuristic

train: ## 5-fold GroupKFold CV (does not save a model)
	python -m eval.train_classifier

train-save: ## Retrain on full data + save app/classifier_model.pkl
	python -m eval.train_classifier --save

plots: ## Regenerate docs/confusion.png + roc_pr.png + calibration.png
	python -m eval.plot_metrics

slices: ## Regenerate docs/slice_analysis.md (per-modality breakdown)
	python -m eval.slice_analysis

walkthrough: ## Run the worked-example walkthrough on one pair
	python docs/walkthrough.py

serve: ## Start the FastAPI dev server on :8000
	uvicorn app.main:app --reload

docker-build: ## Build the production image locally
	docker build -t relevant-priors-ml .

clean: ## Remove caches + build artefacts (keeps the trained model + data)
	rm -rf __pycache__ */__pycache__ */*/__pycache__ \
	       .pytest_cache .ruff_cache .mypy_cache .cache
