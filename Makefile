.PHONY: setup test lint

setup:
	uv sync --locked
	git submodule update --init

test:
	uv run --locked pytest $(PYTEST_ARGS)

lint:
	uv run --locked ruff check .