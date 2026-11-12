.PHONY: setup lint format test test-all data cache train bench export web-data web-dev web-build figures

UV := uv run

setup:
	uv sync --extra teachers
	$(UV) pre-commit install || true

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

format:
	$(UV) ruff check --fix .
	$(UV) ruff format .

test:
	$(UV) pytest -q

test-all:
	$(UV) pytest -q -m ""

data:
	$(UV) python scripts/download_data.py


cache:  # make cache T=t0-gamma C=configs/train/<recipe>.yaml
	$(UV) python scripts/cache_teachers.py --teacher $(T) --train-config $(C)

# make train C=configs/train/gt_only_m.yaml
train:
	$(UV) python scripts/train.py --config $(C)

# make bench S=shapes
bench:
	$(UV) python scripts/benchmark.py --suite $(S)

# make export R=runs/<run>
export:
	$(UV) python scripts/export.py --run $(R)

web-data:
	$(UV) python scripts/build_web_data.py

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

figures:
	$(UV) python scripts/make_figures.py
