# Blindspot

A simulated Franka Panda arm that follows natural-language pick-and-place instructions. The project builds from a MuJoCo scene and scripted expert through imitation learning, vision-language-action models, reinforcement learning, and an LLM/VLM agent. See [the project idea](docs/idea.md) and [the roadmap](docs/roadmap.md).

## Setup

Install `uv` and Python 3.12, then run:

```sh
make setup
make test
make lint
```

The test suite is configured with a `rendering` marker. CI excludes rendering tests because they require a graphics backend; they can be run locally when the required MuJoCo rendering dependencies are available.

## Evaluation

Run one scripted-expert task and level with 100 episodes by default, using evaluation seeds starting at 1,000,000:

```sh
make evaluate EVALUATE_ARGS="--task T1 --level L1"
```

Useful options include `--episodes`, `--seed-start`, `--workers`, and `--obs-mode state|image`. For example, to run a small smoke evaluation:

```sh
make evaluate EVALUATE_ARGS="--task T1 --level L0 --episodes 2 --workers 2"
```

Run the Phase 1.8 baseline matrix with 100 episodes for T1-T4 across L0-L2:

```sh
make baseline
```

The baseline command prints each cell's success rate, Wilson 95% interval, and failure breakdown. It exits unsuccessfully if an evaluated success threshold is missed. Results are written as JSON under `eval/results/<method>/<task>-<level>.json`; the default expert method directory is `scripted-expert/`.

Generate the Markdown summary and success-rate figure from the saved JSON files:

```sh
make report
```

By default, the report is written to `eval/results/report.md` and `eval/results/success-rates.png`. Use `REPORT_ARGS="--results-dir PATH --output-dir PATH"` to read results from or write reports to other directories. Both `evaluate` and `baseline` accept `--results-dir PATH` through their corresponding `*_ARGS` variable.

## Demonstrations

Install the optional pinned LeRobot dataset dependency and generate 400 successful T2-L1 and T3-L1 demonstrations (380 training episodes and 20 validation episodes per task):

```sh
make demos
```

Use `DEMO_ARGS="--task T2 --episodes 10 --workers 2"` for a smaller collection. Outputs are LeRobot datasets under `data/datasets/<task>-l1/{train,validation}/`; the collection manifest records episode seeds, discard rates, target-object counts, and the noise event rate. Existing default datasets are preserved; reruns automatically use names such as `t2-l1-run-02`. An explicit `--dataset-name` that already exists remains an error. Each policy step has a 1% chance of receiving a Gaussian perturbation (1 cm xyz, 3 degrees yaw); the clean expert action remains the label. This sparse schedule preserves successful collection while exposing recovery examples. To run the open-loop replay check, set both noise standard deviations to zero and pass `--verify-open-loop`.

Train a behavior-cloning baseline from a generated dataset (MPS is selected automatically when available):

```sh
make train-bc BC_ARGS="--dataset-name t2-l1-run-02 --policy state"
make train-bc BC_ARGS="--dataset-name t2-l1-run-02 --policy image"
```

Checkpoints and `run.json` records are written under `checkpoints/bc/<dataset>/<policy>/`. Default reruns use numbered directories such as `state-run-02`, preserving earlier checkpoints. An explicit `--output-dir` that already exists remains an error. State BC uses the 12-D proprioception, 28-D oracle object poses, and 4-D target pose. Image BC uses both 128 px cameras with ImageNet-pretrained, frozen ResNet-18 encoders and the 12-D proprioception. The camera embeddings are cached once per train/validation split; only the small fusion/action head is optimized. `train-bc` displays cache, batch, and epoch progress bars in interactive terminals.

The run record includes held-out action MAE by component and frame L1 grouped by demonstration episode. Add `--rollout-validation` to also run the checkpoint in MuJoCo on the collection's held-out validation seeds; the run record stores its success summary and the evaluator writes the per-episode outcomes under the checkpoint directory.

Evaluate a checkpoint on the fixed evaluation seed range and inspect its Wilson 95% interval:

```sh
make evaluate-bc BC_EVAL_ARGS="--checkpoint checkpoints/bc/t2-l1/state/checkpoint.pt --task T2 --level L1 --episodes 100 --workers 1"
```

Evaluation shows a per-episode progress bar. BC evaluation defaults to MPS, then CUDA, then CPU when available; accelerator and image evaluations use one worker by default to avoid duplicating large models. Image BC still takes more compute per step because it renders both cameras and runs both ResNet encoders. Override with `--device` or `--workers` when needed.

Watch one evaluation-seed episode in the passive MuJoCo viewer (`mjpython` is selected automatically on macOS):

```sh
make view-bc BC_VIEW_ARGS="--checkpoint checkpoints/bc/t2-l1/state/checkpoint.pt --task T2 --level L1 --seed 1000002"
```

Close the viewer after the episode result is printed. Use `--device cpu` to reduce accelerator memory use, or `--show-tcp` to display the TCP marker. `view_expert.py` remains the scripted-expert viewer; `view-bc` loads the learned checkpoint.

## Third-party licenses

| Component | License information |
| --- | --- |
| Project code | MIT; see [LICENSE](LICENSE). |
| MuJoCo Menagerie | Per-directory terms are retained in the [Menagerie license file](third_party/mujoco_menagerie/LICENSE); check the license for each model used. |
| Franka Emika Panda model | Apache-2.0; see the upstream [Panda license](third_party/mujoco_menagerie/franka_emika_panda/LICENSE). |
| Model weights | No model weights are bundled in this phase. Record each checkpoint's license and use terms before adding or downloading it. |

The [MuJoCo Menagerie submodule](third_party/mujoco_menagerie) is pinned by Git to a specific commit. Dependencies retain their respective upstream licenses. No third-party model weights are redistributed by this repository at this stage.