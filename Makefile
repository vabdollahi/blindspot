.PHONY: setup test lint view-scene view-control benchmark-scene

ifeq ($(shell uname -s),Darwin)
VIEWER_PYTHON := uv run mjpython
else
VIEWER_PYTHON := uv run python
endif

setup:
	uv sync --locked
	git submodule update --init

test:
	uv run --locked pytest $(PYTEST_ARGS)

lint:
	uv run --locked ruff check .

view-scene:
	$(VIEWER_PYTHON) scripts/view_scene.py

view-control:
	$(VIEWER_PYTHON) scripts/view_control.py $(VIEW_CONTROL_ARGS)

benchmark-scene:
	uv run --locked python scripts/benchmark_scene.py