# Contributing to Huginn

Thank you for your interest in Huginn! This document outlines how to set up a
development environment, run tests, and submit changes.

## Development Setup

```bash
git clone <repo-url>
cd matsci-agent/agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,all]"
pre-commit install
```

## Running Tests

```bash
# Full suite with coverage
python -m pytest tests -q

# Specific module
python -m pytest tests/test_adapter_constraints.py -q --no-cov
```

## Code Quality

```bash
ruff check huginn tests
black --check huginn tests
mypy huginn
```

All CI checks must pass before merging.

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make focused, well-tested changes.
3. Update documentation if public behavior changes.
4. Ensure the full test suite passes.
5. Open a PR with a clear description and link any related issues.

## Commit Style

Use clear, descriptive commit messages. Prefer present tense and concise
summaries, e.g.:

```
Add container sandbox default for bash_tool
```

## Code of Conduct

Be respectful, constructive, and inclusive in all interactions.
