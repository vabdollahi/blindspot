.PHONY: setup test lint view-scene view-control view-expert view-bc view-rl evaluate baseline evaluate-bc report benchmark-scene benchmark-rl-env demos train-bc train-rl evaluate-rl

ifeq ($(shell uname -s),Darwin)
VIEWER_PYTHON := uv run mjpython
BC_VIEWER_PYTHON := mjpython
else
VIEWER_PYTHON := uv run python
BC_VIEWER_PYTHON := python
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

view-expert:
	$(VIEWER_PYTHON) scripts/view_expert.py $(VIEW_EXPERT_ARGS)

view-bc:
	uv run --locked --extra imitation $(BC_VIEWER_PYTHON) scripts/view_bc.py $(BC_VIEW_ARGS)

view-rl:
	uv run --locked --extra rl $(BC_VIEWER_PYTHON) scripts/view_rl.py $(RL_VIEW_ARGS)

evaluate:
	uv run --locked python scripts/evaluate.py $(EVALUATE_ARGS)

evaluate-bc:
	uv run --locked --extra imitation python scripts/evaluate_bc.py $(BC_EVAL_ARGS)

baseline:
	uv run --locked python scripts/evaluate_baseline.py $(BASELINE_ARGS)

report:
	uv run --locked python -m eval.report $(REPORT_ARGS)

benchmark-scene:
	uv run --locked python scripts/benchmark_scene.py

benchmark-rl-env:
	uv run --locked --extra rl python scripts/benchmark_rl_env.py $(BENCHMARK_RL_ENV_ARGS)

demos:
	uv run --locked --extra imitation python -m data.generate $(DEMO_ARGS)

train-bc:
	uv run --locked --extra imitation python scripts/train_bc.py $(BC_ARGS)

train-rl:
	uv run --locked --extra rl python scripts/train_rl.py $(RL_ARGS)

evaluate-rl:
	uv run --locked --extra rl python scripts/evaluate_rl.py $(RL_EVAL_ARGS)