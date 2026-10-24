# Blindspot

A simulated Franka Panda arm used as a testbed for robot foundation models. A MuJoCo scene, a scripted expert, and a reinforcement-learning expert provide data and baselines for a fine-tuned vision-language-action model (SmolVLA), JEPA representations, latent world models with calibrated uncertainty, and distilled policies, all benchmarked on the same tasks and seeds. See [the project idea](docs/idea.md) and [the roadmap](docs/roadmap.md).

## Setup

Install `uv` and Python 3.12, then run:

```sh
make setup
make test
make lint
```

The test suite is configured with a `rendering` marker. CI excludes rendering tests because they require a graphics backend; they can be run locally when the required MuJoCo rendering dependencies are available.

## Shared training infrastructure (Phase 2.5)

Install the optional training dependencies and run the tiny infrastructure check:

```sh
uv sync --locked --extra training
make verify-training
```

This is shared run infrastructure, not a replacement for each algorithm's training loop. It lives in `common/training.py`; BC, Stable-Baselines3 PPO, LeRobot, and research models keep their own data pipelines, validation rules, and native checkpoint formats, and connect through thin adapters.

The generic check compares a single-process run with a two-process CPU/Gloo run and verifies exact checkpoint resume. When a trainer starts a `TrainingRun`, it creates its run directory with `run.json` and a `tensorboard/` event directory; the method-specific adapter writes its native checkpoint there. The run record includes config, git commit and dirty-worktree status, seeds, dataset ID/version, hardware, and precision selection/fallback. TensorBoard receives scalar loss, gradient norm, learning rate, and throughput values only—no parameter histograms and no Weights & Biases.

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

## Reinforcement learning (Phase 1b)

Phase 1b trains a pure-PPO policy for each non-language task (T1-T4) from scratch, with no dependency on the scripted expert. Once a task's checkpoint qualifies, it replaces the scripted expert as the demonstrator for the learned models and serves as the privileged teacher for distillation (see [the roadmap](docs/roadmap.md)).

- **Action:** bounded deltas from the controller's TCP target and a binary gripper. A shared command tracker integrates them into the absolute `[x, y, z, yaw, g]` action, so a frozen checkpoint is a drop-in `Policy` like any other.
- **Reward:** staged and non-positive per step (reach → grasp → carry → place → open), plus a success bonus.
- **Curriculum:** by default, 30% of object-task training episodes start with the cube already grasped. `--destination-start-fraction` optionally starts some of those partway along the path to the destination, which long carries such as T3's need.

The [design decisions](docs/decisions.md#rl-environment-specification) explain these choices.

![Pure-PPO policy placing the red cube in the target zone (T2-L1, evaluation seed 1,000,030)](docs/media/t2-l1-pure-ppo.gif)

*Pure-PPO T2-L1 policy (seed 0, best validation checkpoint) on evaluation seed 1,000,030: reach, grasp, carry, place, release.*

![Pure-PPO policy carrying a cube into the bowl (T3-L1)](docs/media/t3-l1-pure-ppo.gif)

*Pure-PPO T3-L1 policy (seed 0, best validation checkpoint) on a held-out evaluation seed: the cube is carried 30-50 cm and released into the bowl. Trained with destination-side grasped starts (`--grasped-start-probability 0.5 --destination-start-fraction 0.6`), a reverse curriculum in which some episodes begin with the cube already held on the way to the bowl, so the policy learns to place and release before it can carry that far.*

Profile the vectorized environment's throughput and render random-policy rollouts to sanity-check the reward stages:

```sh
make benchmark-rl-env
```

Train one `(task, level, seed)` PPO expert (defaults match the roadmap: 8 parallel environments, 10M environment steps, seed 0):

```sh
make train-rl RL_ARGS="--task T1 --level L1"
```

T3's bowl sits 30-50 cm from the cube, so train it with destination-side starts:

```sh
make train-rl RL_ARGS="--task T3 --level L1 --grasped-start-probability 0.5 --destination-start-fraction 0.6"
```

For a quick smoke run instead of a full 10M-step training job, scale the budget down:

```sh
make train-rl RL_ARGS="--task T1 --level L1 --total-timesteps 20000 --eval-freq 5000 --eval-episodes 5"
```

Training prints a validation-seed success rate every `--eval-freq` steps. Every rollout also prints, and appends to `logs/progress.csv`:

- the training success rate;
- a stage funnel (`funnel/reached`, `grasped`, `lifted`, `carried`, `placed`): the fraction of normal-start episodes that reached each stage, which is the quickest way to see which stage limits a run;
- per-term reward sums (`reward/*`) and PPO diagnostics (`train/std`, `train/approx_kl`, `train/explained_variance`).

The final checkpoint, its `VecNormalize` observation statistics, and a `run.json` record (full config, git commit, hardware, and the validation learning curve) are written under `checkpoints/rl/<task>-<level>/seed-<seed>/`. The roadmap calls for 3 training seeds per task; run the command once per `--seed`.

While training runs, the same directory also accumulates resumable intermediate checkpoints: `checkpoints/step_<N>/` every `--checkpoint-freq` steps (defaults to `--eval-freq`; only the last `--keep-last`, default 2, are kept), and a `best/` snapshot that's overwritten whenever the validation success rate improves. Both use the same `model.zip` + `vecnormalize.pkl` layout as the final checkpoint, so either is a valid `--resume-from` target, for example to recover from an interrupted run or to continue from the best validation point.

To extend a run's step budget, pass `--resume-from`. The checkpoint's own hyperparameters (learning rate, network size, and so on) are reused; only `--n-envs`, `--total-timesteps`, `--eval-freq`, `--eval-episodes`, `--checkpoint-freq`, `--keep-last`, `--grasped-start-probability`, `--destination-start-fraction`, and `--device` apply, and `--total-timesteps` counts additional steps. A resumed run writes to a new, auto-versioned output directory and never overwrites its source checkpoint. A reward or observation change requires a fresh run instead, because the value function and normalization statistics depend on both:

```sh
make train-rl RL_ARGS="--task T1 --level L1 --resume-from checkpoints/rl/t1-l1/seed-0 --total-timesteps 5000000"
```

The learning curve in `run.json` is an in-training signal on a small, fixed set of validation seeds. For a reportable result (100 episodes on the evaluation-seed range, with a Wilson 95% interval, through the same harness as the scripted expert and BC), evaluate the checkpoint directly:

```sh
make evaluate-rl RL_EVAL_ARGS="--checkpoint-dir checkpoints/rl/t1-l1/seed-0 --task T1 --level L1 --episodes 100"
```

Results are written as JSON under `eval/results/pure-ppo/<task>-<level>.json`, matching the scripted expert's and BC's result format.

To calibrate the evaluation-only T2-H variant on 200 validation seeds with the frozen T2 checkpoint:

```sh
make evaluate-rl RL_EVAL_ARGS="--checkpoint-dir checkpoints/rl/t2-l1/seed-0/best --task T2-H --level L1 --seed-start 100000 --episodes 200 --results-dir eval/results/t2h-calibration-validation"
```

To watch a checkpoint run in the MuJoCo passive viewer:

```sh
make view-rl RL_VIEW_ARGS="--checkpoint-dir checkpoints/rl/t1-l1/seed-0 --task T1 --level L1 --seed 1000000"
```

Close the viewer after the episode result is printed. Use `--device cpu` (the default) to avoid accelerator overhead for single-episode inference, or `--show-tcp` to display the TCP marker.

To save the episode as a looping GIF instead of opening the viewer, add `--gif PATH`; the episode is rendered offscreen from the viewer's camera angle. `--gif-fps` (default 10, subsampled from the 20 Hz policy rate) and `--gif-width`/`--gif-height` (default 480×360, up to the scene's 512×512 offscreen buffer) control the output. The GIFs above were produced this way from each task's best checkpoint, for example:

```sh
make view-rl RL_VIEW_ARGS="--checkpoint-dir checkpoints/rl/t2-l1/seed-0/best --task T2 --level L1 --seed 1000030 --gif docs/media/t2-l1-pure-ppo.gif"
```

## Demonstrations

Generate 400 successful T2-L1 and T3-L1 demonstrations with the matching frozen PPO checkpoints (380 training episodes and 20 validation episodes per task):

```sh
make demos
```

The generator uses `checkpoints/rl/t2-l1/seed-0/best` and `checkpoints/rl/t3-l1/seed-0/best` by default. Use `DEMO_ARGS="--task T2 --episodes 10 --workers 2"` for a smaller collection, or `--policy-source scripted` to explicitly collect a scripted-expert baseline. New LeRobot datasets use 256×256 RGB, retain both the clean `action` and applied `action.executed`, and store simulator object poses in `observation.environment_state`; `collection.json` records seeds, discard rates, task-label counts, the policy checkpoint path, and its SHA-256 digest. Default names are `t2-l1`, `t3-l1`, and `t3l-l1` (T3-L). Existing datasets are preserved and reruns automatically use versioned names such as `t2-l1-run-02`.

Generate the language-conditioned T3-L dataset using the T3 PPO checkpoint and the training paraphrase set, then generate a separate held-out paraphrase dataset:

```sh
make demos DEMO_ARGS="--task T3-L"
make demos DEMO_ARGS="--task T3-L --instruction-template-split held-out"
```

Each policy step has a 1% chance of receiving a Gaussian perturbation (1 cm xyz, 3 degrees yaw); the clean expert action remains the label, and the applied action is stored separately. The optional open-loop diagnostic requires both noise standard deviations to be zero and `--verify-open-loop`. It records the same-seed clean-action replay rate in `collection.json`; a low rate warns but does not reject the dataset.

## World-model interaction corpus

Collect the 2,000-episode T1-T3 L1 corpus from frozen RL experts, the same experts with large action noise, intermediate PPO checkpoints, and scripted exploration:

```sh
make interaction
```

The collector keeps every episode, including failures, and saves the executed action, the next-state success predicate, simulator object poses, and a source ID for every transition. Scripted exploration targets are checked with Mink IK before they are used. The `interaction-l1/collection.json` manifest records per-source and per-task counts, checkpoint hashes, failures, and workspace/yaw/gripper action-coverage histograms. It also writes `action-coverage.png` with XY, XZ, and YZ occupancy projections plus yaw and gripper histograms. Defaults for the noisy source are 5 cm position noise and 15° yaw noise per step.

Regenerate the coverage plot from an existing manifest without collecting episodes again:

```sh
make interaction-coverage
```

For another corpus or output path, pass `INTERACTION_COVERAGE_ARGS="PATH/collection.json --plot-output PATH/action-coverage.png"`.

Train a behavior-cloning baseline from a generated dataset (MPS is selected automatically when available):

```sh
make train-bc BC_ARGS="--dataset-name t2-l1 --policy state"
make train-bc BC_ARGS="--dataset-name t2-l1 --policy image"
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