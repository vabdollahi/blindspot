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

## Third-party licenses

| Component | License information |
| --- | --- |
| Project code | MIT; see [LICENSE](LICENSE). |
| MuJoCo Menagerie | Per-directory terms are retained in the [Menagerie license file](third_party/mujoco_menagerie/LICENSE); check the license for each model used. |
| Franka Emika Panda model | Apache-2.0; see the upstream [Panda license](third_party/mujoco_menagerie/franka_emika_panda/LICENSE). |
| Model weights | No model weights are bundled in this phase. Record each checkpoint's license and use terms before adding or downloading it. |

The [MuJoCo Menagerie submodule](third_party/mujoco_menagerie) is pinned by Git to a specific commit. Dependencies retain their respective upstream licenses. No third-party model weights are redistributed by this repository at this stage.