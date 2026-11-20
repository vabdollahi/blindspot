.PHONY: setup test lint view-scene view-control view-expert view-bc view-rl view-vla render-vla-trajectory evaluate baseline evaluate-bc report benchmark-scene benchmark-rl-env demos interaction interaction-coverage train-bc train-rl evaluate-rl train-vla evaluate-vla smoke-vla jepa-data train-jepa evaluate-jepa verify-training tensorboard

INTERACTION_COVERAGE_ARGS ?= data/datasets/interaction-l1/collection.json
TENSORBOARD_LOGDIR ?= checkpoints

ifeq ($(shell uname -s),Darwin)
VIEWER_PYTHON := uv run mjpython
BC_VIEWER_PYTHON := mjpython
VLA_VIEWER_PYTHON := uv run --locked --extra vla mjpython
else
VIEWER_PYTHON := uv run python
BC_VIEWER_PYTHON := python
VLA_VIEWER_PYTHON := uv run --locked --extra vla python
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

view-vla:
	$(VLA_VIEWER_PYTHON) scripts/view_vla.py $(VLA_VIEW_ARGS)

render-vla-trajectory:
	$(VLA_VIEWER_PYTHON) scripts/render_vla_trajectory.py $(VLA_RENDER_ARGS)

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
	uv run --locked --extra imitation --extra rl python -m data.generate $(DEMO_ARGS)

interaction:
	uv run --locked --extra imitation --extra rl python -m data.interaction $(INTERACTION_ARGS)

interaction-coverage:
	uv run --locked python -m data.interaction --plot-coverage $(INTERACTION_COVERAGE_ARGS)

train-bc:
	uv run --locked --extra imitation --extra training python scripts/train_bc.py $(BC_ARGS)

train-rl:
	uv run --locked --extra rl --extra training python scripts/train_rl.py $(RL_ARGS)

evaluate-rl:
	uv run --locked --extra rl python scripts/evaluate_rl.py $(RL_EVAL_ARGS)

train-vla:
	uv run --locked --extra vla python scripts/train_vla.py $(VLA_ARGS)

evaluate-vla:
	$(VLA_VIEWER_PYTHON) scripts/evaluate_vla.py $(VLA_EVAL_ARGS)

smoke-vla:
	$(VLA_VIEWER_PYTHON) scripts/smoke_vla.py $(VLA_SMOKE_ARGS)

jepa-data:
	[ -d data/datasets/t3-l1-probe-val ] || uv run --locked --extra jepa --extra imitation --extra rl python -m data.interaction --tasks T3 --seed-split validation --episodes 100 --dataset-name t3-l1-probe-val
	[ -d data/datasets/t3-l1-probe-test ] || uv run --locked --extra jepa --extra imitation --extra rl python -m data.interaction --tasks T3 --seed-split evaluation --episodes 200 --dataset-name t3-l1-probe-test
	uv run --locked --extra jepa python -m encoders.frames interaction-l1 t3-l1 t3-l1-probe-val t3-l1-probe-test

train-jepa:
	uv run --locked --extra jepa python scripts/train_jepa.py $(JEPA_ARGS)

evaluate-jepa:
	uv run --locked --extra jepa python scripts/evaluate_jepa.py $(JEPA_EVAL_ARGS)

verify-training:
	uv run --locked --extra training python scripts/verify_training.py

tensorboard:
	uv run --locked --extra training tensorboard --logdir $(TENSORBOARD_LOGDIR) $(TENSORBOARD_ARGS)
