# Robot Foundation Models Testbed

A simulated robot arm used as a controlled testbed for robot foundation models: a fine-tuned vision-language-action (VLA) model, JEPA representations, latent world models with calibrated uncertainty, and distillation into small, fast policies. A scripted expert and a reinforcement-learning expert provide the data and the baselines, and every model is benchmarked on the same tasks and seeds.

---

## 1. Overview

**Motivation.** Robot foundation models are usually demonstrated one at a time, on different robots and tasks, with results that are hard to compare. This project builds them side by side in one simulator, where ground truth is available for everything: object poses for probing representations, the simulator itself as a physics-based model to benchmark learned world models against, and exact success predicates for every evaluation.

**Focus.** VLA fine-tuning, self-supervised visual representations (JEPA), action-conditioned latent world models (prediction, uncertainty, out-of-distribution detection, planning, and policy evaluation), and distillation with efficient inference. The project contains no kinematics, dynamics, or control code. An off-the-shelf IK library (`mink`) maps end-effector targets to joint targets, and MuJoCo's position servos track them.

**Reference hardware.** A single Apple Silicon laptop (M3 Max, 36 GB unified memory). No robot and no NVIDIA GPU are used; every model is sized to train on this machine, and the training code is distributed-ready for larger hardware.

**What the project covers.**

| Area | Where |
| --- | --- |
| Simulation environment | Phases 0-1: MuJoCo scene, Gymnasium environment, and a scripted expert built on off-the-shelf IK |
| Reinforcement learning | Phase 1b: PPO trained from scratch into a privileged-state expert, the demonstrator and distillation teacher |
| Data engine | Phase 2: versioned LeRobot datasets with language labels, an interaction corpus for world models, and shared training infrastructure |
| Vision-language-action models | Phase 3: SmolVLA fine-tuned on the RL expert's demonstrations and evaluated under distribution and dynamics shift |
| Representations and JEPA | Phase 4: ResNet-18, DINOv2, V-JEPA 2, and an in-domain JEPA compared by probing and as policy backbones |
| World models | Phase 5: latent dynamics ensembles with calibrated uncertainty and OOD signals, a validity map, physics-based baselines, CEM planning, and policy evaluation in imagination |
| Distillation and inference | Phase 6: privileged-teacher and VLA-teacher distillation with DAgger, flow-step reduction, and ONNX latency benchmarks |
| Language agent | Phase 7: an LLM planner with SmolVLA skills, a verifier, and world-model look-ahead |
| Reproducible evaluation | A cross-model benchmark with confidence intervals, one-command regeneration, and a test suite |

---

## 2. Goals and success criteria

**Done means all of the following exist:**

1. A public repo where `make setup && make smoke` finishes in under 30 minutes from a clean clone. This excludes model-weight downloads and training. The smoke target runs the unit tests and one short run per layer.
2. Unit tests for the environment: determinism, success predicates, and camera projection.
3. A policy benchmark comparing the scripted expert, the RL expert, the BC baselines, SmolVLA, encoder-backbone policies, distilled students, and world-model planning on shared tasks and seeds, with 95% confidence intervals, parameter counts, and latency.
4. A world-model report: prediction error against physics-based models, calibrated uncertainty, out-of-distribution detection, a validity map, planning results, and policy-evaluation correlation.
5. A representation study and a distillation study with a success-latency-size trade-off.
6. A demo video (about 2 minutes): a typed instruction, the plan, the world model's imagined rollout next to the real execution, and one real failure with its recovery.
7. A technical write-up (README plus report) with ablations and a limitations section.

**Non-goals.** Real hardware, sim-to-real transfer, novel algorithms, a production system, collision-aware motion planning beyond simple approach and retreat waypoints, implementing kinematics, dynamics, or controllers, training VLAs or large world models from scratch, and multi-GPU training at scale.

---

## 3. Key design decisions

Current design choices and their rationale are summarized in `docs/decisions.md`.

### 3.1 Robot: Franka Panda by default

Use the **Franka Emika Panda** from MuJoCo Menagerie (`franka_emika_panda/`). The folder contains `panda.xml` (arm plus hand), `panda_nohand.xml`, `hand.xml`, `scene.xml`, and an MJX single-cube scene.

- The gripper is included and already tuned, so there is no integration work.
- The Panda is the most common arm in manipulation-learning work (robosuite, ManiSkill, MuJoCo Playground pick-cube), which makes results easy to compare.

**Alternative:** a UR5e (6-DOF) with a Robotiq 2F-85, joined with `MjSpec.attach`. One robot is used for all reported experiments.

### 3.2 Motion: off-the-shelf IK

The project writes no kinematics or control code. **`mink`**, a differential-IK library built on MuJoCo, converts TCP targets to joint targets, and Menagerie's built-in **position servos** track them. Run IK on a separate robot-only model and copy the robot joint positions from the simulation each tick, so free-floating objects never enter the IK problem.

Also add a **TCP site** between the fingertips. That site is the `ee` frame.

### 3.3 Control hierarchy and rates

| Layer | Rate | Output |
| --- | --- | --- |
| Physics (`implicitfast`, the Panda model's default) | 500 Hz (dt = 2 ms) | state |
| Joint control: MuJoCo position servos | 500 Hz, every physics step | joint torques |
| IK (`mink`) and waypoint interpolation | 100 Hz (every 5 physics steps) | joint position targets |
| Policy or skill | 20 Hz (every 25 physics steps) | TCP target pose plus gripper command |
| Agent (planner and verifier) | event-driven, per skill | skill calls |

### 3.4 Action and observation spaces

- **Learned-policy action (5-D):** an absolute TCP target in the `base` frame, given as position (3) plus yaw (1) with the gripper pointing straight down, plus a gripper command (1). Top-down grasps are enough for cubes and a bowl, and they avoid rotation-representation problems. ACT predicts absolute targets, and the Diffusion Policy paper reports that position control beats velocity control. IK and the position servos (§3.3) track this target. The VLA and every distilled student use the same action space, so all policies compare directly.
- **Phase 1b RL policy action:** a bounded delta from the controller's current TCP target (xyz up to one 1.25 cm rate-limit step, yaw up to 0.05 rad) plus a binary gripper. The RL wrapper, and the frozen policy's `Policy.act`, integrate it into the absolute action above. Deltas make holding still and millimeter precision closed-loop and keep exploration local, whereas absolute targets leave both open-loop and dominated by exploration noise.
- **Image-policy observation:** front RGB plus wrist RGB at 256×256 for every dataset from Phase 2 on (the earlier BC baselines use 128×128), plus proprioception (joint positions, TCP pose, gripper width). Pretrained encoders (DINOv2, V-JEPA 2) and SmolVLA expect 224-256 px inputs, and the cubes span only a few pixels at 128 px in the front view.
- **State-policy observation:** proprioception, object poses (noisy under T2-H), target pose, and gripper state.
- The BC baselines and the students distilled from the RL expert are **not** language-conditioned, so object identities that define a task are fixed by color for them (T4: red on blue). SmolVLA and its distilled student (T3-L) and the agent (T5) handle language references.

### 3.5 Conventions (fix on day one, in `docs/conventions.md`)

- SI units and radians everywhere.
- Frames: `world`, `base`, `ee` (the TCP site), `cam_front`, `cam_wrist`.
- Quaternions are **w, x, y, z** (MuJoCo). SciPy's `Rotation` defaults to **x, y, z, w**, so always pass `scalar_first=True` or convert in one helper.
- `data.xmat` and `data.site_xmat` are row-major 9-vectors: use `reshape(3, 3)`.
- One seeding utility is used everywhere, with separate seed ranges for training, validation, and evaluation (§9).

---

## 4. System architecture

Five layers. Each one runs on its own and is evaluated on its own. The architecture diagram is in [roadmap.md §2](roadmap.md#2-architecture).

1. **Physics and robot.** A MuJoCo scene, built with `MjSpec`, containing the Panda, a table, colored cubes, a bowl, a front camera, and a wrist camera.
2. **Motion and skills.** Off-the-shelf IK (`mink`) and MuJoCo's position servos, wrapped as parameterized skills and a scripted expert.
3. **Experts and data.** A PPO expert trained on privileged state, and a data engine that turns it into versioned demonstration datasets and an interaction corpus.
4. **Learned models.** SmolVLA; frozen and in-domain JEPA encoders with probes; latent world models with uncertainty, planning, and policy evaluation; and distilled students. Every policy shares one action interface.
5. **Agent.** An LLM planner emits a JSON plan of skill calls, executed by scripted skills or SmolVLA. A verifier checks each result, and the world model predicts each skill's outcome before it runs.

---

## 5. Tasks

### 5.1 Task suite (difficulty ladder)

| ID | Task | Randomization | Example success predicate (freeze before results) | Time limit |
| --- | --- | --- | --- | --- |
| T1 | Reach a target point | target position | TCP within 5 mm of target | 5 s |
| T2 | Pick a cube and place it in a target zone | cube pose, zone pose | cube center within 3 cm of zone center (xy), at rest, released | 15 s |
| T3 | Pick a cube and place it in a bowl | cube and bowl poses, cube color | cube inside the bowl volume, at rest, released | 15 s |
| T3-L | "Put the {color} cube in the bowl", with 2-3 cubes of different colors | poses, colors, which cube is named, instruction wording | named cube inside the bowl, other cubes moved less than 2 cm | 15 s |
| T4 | Stack cube A (red) on cube B (blue) | both poses | A resting on B's top face within 1.5 cm (xy), stable for 1 s after release | 20 s |
| T5 | Multi-step language instructions combining T2-T4 | all of the above plus distractors | goal predicate for each instruction (§8, Phase 7) | 60 s |

### 5.2 Randomization levels

- **L0:** fixed scene.
- **L1:** object and target poses.
- **L2:** L1 plus lighting, textures, distractor objects, small camera pose jitter, and small friction and mass perturbations.

### 5.3 Dynamics-shift variant T2-H

T2 with a **tight placement tolerance** (1 cm and 10° yaw), **unmodeled dynamics** (cube mass ×[0.5, 3], friction ×[0.5, 1.5]), and **observation noise** on object pose (σ ≈ 5 mm in position, 3° in yaw), all hidden from every policy and model and used only for evaluation. It tests how policies degrade, and whether world models flag the shift through their uncertainty, when dynamics leave the training distribution. Calibrate the perturbation magnitudes so the RL expert succeeds 40-75% of the time, then freeze the definition.

---

## 6. Environment and stack

### 6.1 Software

- **Python 3.12.** Current LeRobot requires Python ≥ 3.12. Use `uv` with a lockfile.
- **`mujoco`** (pin a 3.x version that `mink` supports) and **`mink`** for IK.
- **MuJoCo Menagerie** as a git submodule pinned to a specific commit. It is not a pip package; `robot_descriptions` is an alternative that fetches it.
- **PyTorch ≥ 2.10** with the MPS backend. LeRobot itself accepts older versions, but ≥ 2.10 pairs with a `torchcodec` that supports Homebrew's current `ffmpeg`.
- **`lerobot`** (pinned), with the `training` and `smolvla` extras. Install `ffmpeg` via Homebrew.
- **`transformers`** (pinned) for pretrained encoders: DINOv2 and V-JEPA 2.
- **`onnx`** and **`onnxruntime`** for exported students and world models.
- **`gymnasium`** and **`stable-baselines3`**.
- **Local models:** `mlx-lm` and `mlx-vlm`, or Ollama or LM Studio. Each exposes an OpenAI-compatible server.
- **`pydantic`** for the plan schema, the **`openai`** client for OpenAI-compatible model servers, and `pytest`.
- Pydantic configs, and TensorBoard for experiment tracking (Weights & Biases optional).

**Environment split.** Run the simulation and training stack (`mujoco`, `torch`, `lerobot`, `sb3`) in one environment. Run model serving (MLX, Ollama, or LM Studio) in a separate process and environment, reached over HTTP. This avoids dependency conflicts. It also means switching between local and frontier models is only a config change.

### 6.2 macOS notes

- `mujoco.viewer.launch_passive` requires the `mjpython` launcher on macOS. For quick inspection, run `python -m mujoco.viewer --mjcf=path.xml`.
- Offscreen `mujoco.Renderer` works on macOS without setup. Do not set `MUJOCO_GL=egl` or `osmesa`; those backends are Linux-only. In multiprocess environments, create the renderer **inside** each worker.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported MPS ops fall back to the CPU.
- MJX (JAX) and MuJoCo Warp target accelerators and give no useful speedup here. Use CPU MuJoCo with multiprocess vectorized environments (start with 8-10 workers and benchmark).
- Run long jobs under `caffeinate -i` so the machine does not sleep.
- CI on a Linux runner covers the unit tests and state-only smoke tests. Mark rendering tests and run them locally.

### 6.3 Compute and memory plan

| Workload | Where | Typical cost (measured values are recorded in `decisions.md`) |
| --- | --- | --- |
| Physics, scripted evaluation, data generation | CPU, multiprocess | minutes to about an hour per batch |
| State-based PPO | CPU, 8-10 processes | hours per run, capped at 2 days |
| SmolVLA fine-tune | MPS (LoRA preset if full fine-tuning does not fit) | up to about a day per run |
| Encoder feature caching (DINOv2, V-JEPA 2) | MPS | hours per dataset version |
| In-domain JEPA and world-model ensembles | MPS | hours per model |
| Distillation (DAgger) | CPU rollouts, MPS training | hours per student |
| Local LLM for the agent | MLX on unified memory | seconds per call |

**36 GB:** keep only one large model in memory at a time, and never alongside a training job.

---

## 7. Repository layout

- `third_party/mujoco_menagerie/`: pinned git submodule
- `common/`: seeding, transforms, config, run records
- `envs/`: MuJoCo scene building (MjSpec), task definitions, randomization, observations, rewards, success predicates
- `motion/`: `mink` IK wrapper, control loop, waypoint interpolation
- `skills/`: parameterized skills and the scripted expert
- `data/`: demo generation and LeRobot dataset writing
- `policies/`: PPO expert, BC baselines, SmolVLA wrapper, and distilled students
- `encoders/`: pretrained and in-domain JEPA encoders, the feature cache, and probes
- `world_models/`: latent dynamics, readouts, uncertainty, planners, and policy evaluation in imagination
- `distill/`: DAgger and teacher-student training, ONNX export, and latency benchmarks
- `agent/`: planner, verifier, skill executors (scripted and VLA), world-model look-ahead, and model client
- `eval/`: benchmark harness, metrics, statistics, plots, and `results/` (JSON)
- `tests/`: unit tests (scene, determinism, predicates, projection) and smoke tests for each layer
- `scripts/`: entry points for training, evaluation, and video
- `docs/`: this file, `conventions.md`, `decisions.md`, figures, write-up, video script

---

## 8. Phase plan

The project is built in ten phases (0, 1, 1b, and 2-8). Each phase ends with exit criteria that must hold before the next one starts.

### Phase 0: Setup and scene

**Build.**

- Repo, environment, pinned dependencies, and the Menagerie submodule.
- Scene via `MjSpec`: Panda, table, cubes, bowl, target zone, front and wrist cameras, and TCP site.
- `docs/conventions.md`, `docs/decisions.md`, and one seeding utility.
- Measure physics steps per second and render frames per second, and record them.

**Exit criteria.** The scene compiles and both cameras render. `make setup` works in a clean environment.

### Phase 1: Environment and scripted expert

**Build.**

- **Motion:** `mink` with a frame task on the TCP site, a posture task, and joint limits, solved on a robot-only model at 100 Hz. MuJoCo's position servos track the resulting joint targets.
- **Waypoints:** linear interpolation between TCP targets with a maximum TCP speed.
- **Gripper and contacts:** a grasp counts as successful when both fingers touch the target object (from `data.contact`), the gripper is not fully closed, and a lift check passes. Tune contacts: cubes of about 4-5 cm and 50-100 g, `cone="elliptic"` with `impratio` > 1, and `condim` ≥ 4 on fingertips and cubes to reduce slip and spinning. Record the final values.
- **Skills:** `reach(pose)`, `grasp(obj)`, `place(pose)`, `pick_and_place(obj, destination)`, and `stack(a, b)`. Each has preconditions, a postcondition check, a timeout, and a structured result with a success flag and a failure reason. The agent's planner calls `pick_and_place` and `stack`, and its verifier reuses the postcondition checks.
- **Scripted expert:** a closed-loop state machine that composes skills from oracle object poses and solves T2-T4. It recomputes each action from the current state, which Phase 2's noise injection relies on.
- **Environment API (Gymnasium):** `reset(seed)`, `step(action)`, `observe()`, `success()`, and randomization configs for L0-L2. The visual parts of L2 can be finished in Phase 2.
- **Evaluation harness:** one `Policy` interface for every method, parallel evaluation, JSON results, Wilson and bootstrap confidence intervals, and a report script.

**Evaluate.** Run the scripted expert on T1-T4 × L0-L2, 100 episodes per cell on fixed evaluation seeds. Produce a failure breakdown: IK failure or unreachable target, grasp miss, slip, collision, placement error, timeout.

**Exit criteria.** T1 ≥ 99%. T2 and T3 ≥ 90% at L1. T4 ≥ 70% at L1. IK solves random workspace targets, the arm settles within 2 mm of a held target, and the same seed reproduces the same episode. A script writes the results to `eval/results/`.

The scripted expert remains in the benchmark as a comparison baseline and as a fallback demonstrator for any task whose RL checkpoint does not meet its target. Phase 1b trains a pure-PPO expert per non-language task; each qualifying checkpoint is the primary demonstrator for later datasets and the privileged teacher for distillation.

### Phase 1b: Reinforcement learning (training an RL expert)

**Build.**

- **Pure PPO, not a residual policy on the scripted expert,** trained from scratch per task, covering every non-language task (T1, T2, T3, T4), so the result does not inherit the scripted expert's failure modes. T3-L demonstrations reuse the T3 expert with the named cube as its target (Phase 2). The state-based environment provides proprioception, the target object's own pose (not hard-coded by color, so one policy per task generalizes to naming an arbitrary object among distractors), the destination pose, relative vectors, and the command state (commanded target, last gripper command), acting in bounded deltas. The reward is staged, dense and non-positive per step (reach, grasp, carry along a lifted path, place, open), plus a sparse success bonus. A training-only curriculum starts some episodes with the cube already grasped, optionally partway along the path to the destination. T1 has no object: its observation and reward collapse to TCP-to-target distance plus gripper openness.
- Train with `SubprocVecEnv`, 8-10 workers. Budget up to 10-20 M environment steps per task, capped at 2 days wall-clock.

**Evaluate.** Run the Phase 1 evaluation harness on each trained policy, on the same evaluation seeds as the scripted expert's Phase 1 baseline.

**Exit criteria.** Every non-language task (T1, T2, T3, T4) has a frozen RL-expert checkpoint that meets the Phase 1 exit criteria for that task and clearly exceeds the scripted expert's measured rate. Wrap each in the `Policy` interface so it is a drop-in demonstrator for the data engine (Phase 2) and the privileged teacher for distillation (Phase 6); T1's checkpoint has no downstream demo generation and exists only to confirm baseline parity. Any task that does not reach the target keeps the scripted expert as its baseline and demo source (§11).

### Phase 2: Data engine

**Build.**

- **Demonstrations:** run the frozen RL expert on training seeds, recording front and wrist RGB at 256×256, proprioception, the clean and the executed 5-D action, and a language `task` string, with sparse action noise labeled with the clean action so the data contains recoveries. Write with the LeRobot dataset API (pinned, since the on-disk format changes between releases). Target 400 successful episodes each for T2, T3, and T3-L, where T3-L instructions come from templates with a held-out paraphrase split.
- **Interaction corpus:** world models must predict the outcome of actions an expert would not take, so collect about 2,000 T1-T3 episodes from a mixture of sources (the expert, the expert with large noise, its failures, intermediate PPO checkpoints, and scripted exploratory motion), keeping failures and recording object poses.
- **T2-H:** implement and freeze the dynamics-shift variant (§5.3).
- **Training infrastructure:** config-driven entry points with `run.json` records, TensorBoard tracking, mixed precision where the backend supports it, `torchrun` and DDP verified with a 2-process CPU run, and checkpoint and resume.

**Exit criteria.** The datasets and corpus exist with manifests and versions, T2-H is frozen, and the training utilities train identically in 1- and 2-process runs and resume exactly.

### Phase 3: VLA fine-tuning

**Build.** Fine-tune SmolVLA (LeRobot, from the pretrained base checkpoint) on T2, T3, and T3-L with the two 256 px cameras and the 5-D action. Measure MPS throughput with short runs first and choose full fine-tuning or the LoRA preset so a run fits in about a day. Check that the saved checkpoint's feature config matches the dataset, and profile inference latency and flow-matching steps.

**Experiments.** SmolVLA vs the RL expert and the BC baseline on T2-L1 and T3-L1; train on L1 and test on L2 (lighting, textures, distractors); dynamics shift on T2-H; language grounding on T3-L (success and wrong-object rate) on train and held-out paraphrases.

**Exit criteria.** SmolVLA results on T2-L1, T3-L1, T2-L2, T2-H, and T3-L with confidence intervals and the wrong-object rate.

### Phase 4: Representations and JEPA

**Build.**

- One encoder interface with a versioned feature cache: ImageNet ResNet-18 (the existing BC backbone), DINOv2 ViT-S/14, and V-JEPA 2 ViT-L, frozen and loaded through `transformers`.
- An in-domain JEPA: a small ViT trained on the interaction corpus to predict the embeddings of masked regions, with collapse prevented by SIGReg (LeJEPA) or an EMA target encoder (I-JEPA), and collapse monitored through embedding spread and rank.

**Experiments.** Linear and MLP probes from each encoder to object pose, gripper width, and grasp state at L1 and L2; the same policy head trained on each encoder for T2 and T3.

**Exit criteria.** Probe and policy tables per encoder exist, and the best encoder is selected for the world models and the students.

### Phase 5: Latent world models

**Build.**

- An action-conditioned transformer predictor over frozen encoder tokens from both cameras plus proprioception, at 5 Hz, trained with one-step and multi-step rollout losses (the DINO-WM recipe), on DINOv2 and V-JEPA 2 features and end to end with the in-domain JEPA (LeWorldModel-style). Train ensembles of 5.
- Readouts from latents: object poses and gripper width (so predictions are interpretable in cm) and task success.
- Physics baselines on the same action sequences: MuJoCo with true parameters (exact, the upper bound), MuJoCo with nominal parameters under T2-H (a physics model with parameter error), and a kinematic model in which the cube moves rigidly with the gripper while grasped.

**Experiments.**

- Prediction error vs horizon for learned and classical models, with compute per step, stating where the classical models win.
- Uncertainty: ensemble disagreement calibrated on validation seeds, evaluated by interval coverage, Gaussian NLL, and the correlation between predicted uncertainty and actual error; expected calibration error for the success readout.
- Out-of-distribution detection: AUROC for separating L1 from L2, T2-H, and unseen colors, using ensemble disagreement and one-step surprise.
- A validity map: error and calibration across horizon, randomization level, dynamics shift, task phase, and training data, with a stated trusted region.
- Planning: CEM in latent space toward a goal observation, with and without an uncertainty penalty, against the same planner using MuJoCo as the model, on T1, T2, and T2-H.
- Policy evaluation in imagination: predicted vs real success for at least 8 policies (Pearson, Spearman, MMRV).

**Exit criteria.** The prediction, calibration, OOD, validity-map, physics-comparison, and planning results exist. No planning success threshold is set; results are reported as measured.

### Phase 6: Distillation and efficient inference

**Build.**

- Privileged teacher to vision student: the RL expert labels states visited by an image student (DAgger), compared with offline behavior cloning on the same number of labels.
- VLA to small student: a 1-20 M-parameter policy on the best encoder, trained on SmolVLA's action chunks and refined with DAgger.
- SmolVLA success and latency vs the number of flow-matching steps, optionally with consistency-style step distillation.
- ONNX export with ONNX Runtime (CPU and Core ML), `torch.compile`, and int8 quantization.

**Exit criteria.** A success vs latency vs parameter-count Pareto table covering the RL expert, SmolVLA, the students, and their exported variants.

### Phase 7: Agent with VLA skills and world-model look-ahead

**Build.**

- An LLM planner (a local model through MLX by default) that receives the instruction and the oracle object list and emits a Pydantic-validated JSON plan of `pick_and_place` and `stack` calls, or a clarification or refusal.
- Skill executors: scripted skills, and SmolVLA prompted with an instruction from its training distribution.
- A verifier that checks each step's goal predicate, retries once, then replans with the failure reason.
- World-model look-ahead: before executing a skill, roll the executor out in the world model and switch executor or replan when predicted success is low or uncertainty leaves the trusted region.

**Experiments.** 20 instructions × 3 scene seeds: planner + SmolVLA skills vs SmolVLA end to end on two-step instructions, and the agent with and without look-ahead, reporting prevented failures and false alarms.

**Exit criteria.** The agent results, the executor comparison, and the look-ahead rates exist.

### Phase 8: Benchmark, packaging, and release

- Freeze the benchmark: final seeds, final configs, and one command that regenerates every table and figure.
- Run the final evaluations with Wilson (success rates) or bootstrap (across seeds) 95% confidence intervals.
- Demo video, about 2 minutes: an instruction, the plan, the world model's imagined rollout next to the real execution, one real failure and recovery, and the results tables.
- README: architecture diagram, results, how to reproduce, limitations.
- Write-up: what was built, the key findings (including where learned models beat physics-based models and where they do not), and directions for future work.
- Repo hygiene: CI green, pinned requirements and lockfile, license, documented seeds.

---

## 9. Evaluation protocol

- **Seeds.** Use separate seed ranges for training, validation, and evaluation, defined in config. Evaluation seeds are never used for training or tuning.
- **Episodes per cell.**
  - Scripted and state-based methods: at least 100.
  - Image policies, including SmolVLA and distilled students: at least 50.
  - World-model planners: at least 50.
  - RL: 3 training seeds × 100 evaluation episodes.
  - Agent: 20 instructions × 3 scene seeds.
- **Precision.** The 95% Wilson half-width at 50% success is about ±13 percentage points with n = 50, ±10 with n = 100, and ±7 with n = 200. Do not claim differences smaller than this. For RL, variance across seeds usually dominates, so bootstrap over seeds.
- **Success.** Task-specific geometric predicates, written in code before any results are seen (§5.1).
- **Metrics.**
  - Success rate with a 95% CI, mean completion time, and collision count.
  - Policies: parameter count and p50/p95 latency at batch size 1, with backend and precision.
  - World models: prediction error vs horizon (cm, decoded state), interval coverage and Gaussian NLL, expected calibration error, OOD AUROC, and Pearson, Spearman, and MMRV for policy evaluation.
  - Agent: instruction success, replan rate, and look-ahead prevented-failure and false-alarm rates.
- **Main benchmark table.**
  - Rows: scripted expert, RL expert, state and image BC baselines (scripted-expert demonstrations at 128 px), SmolVLA, the best encoder policy, distilled students, and world-model MPC.
  - **Shared columns:** T2-L1 and T2-H.
  - Extra columns: T3-L1, T4-L1, and T2-L2 for the methods whose observations support them, and T3-L for SmolVLA, its student, and the agent. Mark unsupported cells N/A explicitly.
  - Further columns: observation modality, parameters, latency, training data (demos or environment steps), and training compute (CPU or MPS hours).
- **Fairness.** Every method gets the same scene, seeds, action limits, and episode time limit, and every planner gets the same goal observation.
- **Reporting.** Every table and figure is generated by a script from `eval/results/*.json`. No number is pasted by hand.

---

## 10. Development principles

1. **Tests guard conventions.** Silent errors dominate robotics code: wrong frame, degrees vs radians, quaternion order, transposed rotations, OpenGL vs OpenCV camera axes. Unit tests with known answers and round-trip tests catch these, and no module is built on until its tests pass.
2. **Every run is reproducible.** The config, git commit, seeds, and dataset version are stored with each run.
3. **Results come from scripts.** Every reported number is generated from `eval/results/*.json`.
4. **Design choices stay synchronized.** `docs/decisions.md` summarizes the current action space, observations, reward terms, backends, contact parameters, and their rationale.

---

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Grasping is unreliable (slip, penetration, spinning) | High | Elliptic cones with `impratio`, `condim`, and `solref` tuning; lift check and regrasp; larger or lighter cubes; simpler object geometry |
| Frame or convention bugs | High | Unit tests, projection round-trip tests, `conventions.md` |
| World-model planning fails on contact-rich pick-and-place | High | Intermediate goal frames, shorter horizons, uncertainty penalties; report T1 and partial-progress results as measured |
| SmolVLA fine-tuning slow or unsupported on MPS | Medium | Short timing runs first; LoRA preset; frozen vision encoder; fewer steps |
| Pretrained encoders slow on MPS | Medium | Cache features once per dataset version; subsample frames; use DINOv2 ViT-S for world-model training |
| JEPA representation collapse | Medium | Monitor embedding spread and rank; SIGReg weight; EMA target encoder |
| Imagined rollouts drift over long horizons | Medium | Multi-step rollout losses; report the horizon where predictions stop being trustworthy |
| LeRobot API or dataset-format churn | Medium | Pin the version; check checkpoint feature configs after every save |
| Dependency conflicts (LeRobot, `transformers`, MLX) | Medium | Separate environments; local models behind an HTTP API; lockfile |
| Phase 1b RL expert does not beat the scripted expert on some task | Medium | Keep the scripted expert as that task's demo source; proceed with the remaining tasks. If pure PPO falls well short, the next options are SAC/TQC with HER or BC-initialized PPO, which need a roadmap decision. |
| Python control loop too slow for RL and data generation | Medium | Profile; IK only at 100 Hz; more worker processes |
| IK library integration stalls | Low | Drive the hand with a mocap body welded to it, with the arm actuators removed |

---

## 12. Core and optional components

**Core:** scene and determinism tests, the evaluation harness, the scripted expert, the RL expert, the datasets and training infrastructure, the SmolVLA fine-tune, encoder probes, the latent world-model ensemble with physics baselines, uncertainty and OOD evaluation, the validity map, CEM planning on T1 and T2, one distilled student with measured latency, and the write-up.

**Optional, in decreasing priority:**

1. Policy evaluation in imagination
2. The in-domain JEPA (the pretrained encoders remain)
3. The agent layer
4. Flow-matching step distillation
5. A pixel decoder for visualizing imagined rollouts
6. T4 stacking for learned methods (it remains for the scripted and RL experts)

---

## 13. Future work

- **Larger VLAs.** Fine-tune π0.5 or GR00T N1.7 on a rented GPU and compare them with SmolVLA.
- **Multi-GPU training.** Run the world-model and JEPA training on a multi-GPU node with the existing DDP code and report scaling efficiency.
- **Deployment.** TensorRT engines for the students and the world-model predictor on an NVIDIA GPU or Jetson.
- **Pretrained world models.** V-JEPA 2-AC zero-shot planning and a generative video world model (for example, NVIDIA Cosmos) as policy evaluators, compared with the in-repo models.
- **Standard benchmarks.** LIBERO, for comparability with published results.
- **Isaac Lab port.** GPU-parallel RL and data generation.
- **Sim-to-real.** The distilled students on an SO-101-class arm through LeRobot.

---

## 14. References

- MuJoCo, MuJoCo Menagerie, `mink`, LeRobot, and Hugging Face `transformers`.
- Papers:
  - SmolVLA: Shukor et al., 2025
  - π0: Black et al., 2024
  - DINOv2: Oquab et al., 2023
  - I-JEPA: Assran et al., 2023
  - V-JEPA 2: Assran et al., 2025 (arXiv:2506.09985)
  - LeJEPA: Balestriero and LeCun, 2025 (arXiv:2511.08544)
  - DINO-WM: Zhou et al., 2024 (arXiv:2411.04983)
  - LeWorldModel: 2026 (arXiv:2603.19312)
  - Deep ensembles for predictive uncertainty: Lakshminarayanan et al., 2017
  - SIMPLER (policy-evaluation correlation and MMRV): Li et al., 2024 (arXiv:2405.05941)
  - WorldGym (world models as policy evaluators): 2025 (arXiv:2506.00613)
  - DAgger: Ross et al., 2011
  - Learning by Cheating (privileged teacher, vision student): Chen et al., 2019
  - Consistency Policy: Prasad et al., 2024 (arXiv:2405.07503)
  - DART: Laskey et al., 2017
  - ACT: Zhao et al., 2023; Diffusion Policy: Chi et al., 2023 (action-space rationale, §3.4)
- Stable-Baselines3; rliable (Agarwal et al., 2021) for statistically sound RL reporting.
