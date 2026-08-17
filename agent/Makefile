.PHONY: dev test lint format docker-build docker-up clean env-setup

dev:
	./scripts/run_dev.sh

test:
	python -m pytest tests/test_testclient_hygiene.py -q -o addopts=""
	./scripts/run_tests.sh

lint:
	ruff check huginn tests
	black --check huginn tests
	mypy huginn

format:
	black huginn tests
	ruff check --fix huginn tests

docker-build:
	docker build -t huginn-agent:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf build dist .mypy_cache .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Rebuild the Python env in a fresh/restored sandbox (uv may hang on git deps;
# falls back to pip, which honours the sandbox proxy). --fix-uv 先试 uv；
# --all-extras 额外安装可选科学计算依赖（pymatgen/chromadb/rdkit 等）。
env-setup:
	./scripts/env-setup.sh --fix-uv --all-extras
