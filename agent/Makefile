.PHONY: dev test lint format docker-build docker-up clean

dev:
	./scripts/run_dev.sh

test:
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
