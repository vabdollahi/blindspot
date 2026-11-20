# Robot Foundation Models Testbed

A simulated robot arm for controlled evaluation of robot foundation models: a frozen-backbone VLA baseline built from a pretrained vision-language model, JEPA representations, calibrated latent world models with a generative frame decoder and scenario generation for stress testing, and distillation into efficient policies and samplers. Scripted and reinforcement-learning experts provide demonstrations and baselines; benchmarked methods share tasks and seeds.

---

## 1. Overview

**Motivation.** Robot foundation models are usually demonstrated one at a time, on different robots and tasks, with results that are hard to compare. This project builds them side by side in one simulator, where ground truth is available for everything: object poses for probing representations, the simulator itself as a physics-based model to benchmark learned world models against, and exact success predicates for every evaluation.

**Focus.** A frozen-backbone VLA baseline built from pretrained VLM features and a learned action head, self-supervised visual representations (JEPA), action-conditioned latent world models (prediction, uncertainty, out-of-distribution detection, planning, and policy evaluation), generative modeling (a flow-matching decoder that renders imagined rollouts, and step distillation of its sampler), scenario generation that searches for failure-inducing test scenes, and distillation with efficient inference. The project contains no kinematics, dynamics, or control code. An off-the-shelf IK library (`mink`) maps end-effector targets to joint targets, and MuJoCo's position servos track them.

**Reference hardware.** A single Apple Silicon laptop (M3 Max, 36 GB unified memory). No robot and no NVIDIA GPU are used; every model is sized to train on this machine, and the training code is distributed-ready for larger hardware. Frozen-backbone features are cached once and reused, and expensive encoders are trained once per configuration, so each training run takes about an hour (estimated; measured values are recorded in `decisions.md`).

**What the project covers.**

| Area | Where |
| --- | --- |
| Simulation environment | Phases 0-1: MuJoCo scene, Gymnasium environment, and a scripted expert built on off-the-shelf IK |
| Reinforcement learning | Phase 1b: PPO trained from scratch into a privileged-state expert, the demonstrator and distillation teacher |
| Data engine | Phase 2: versioned LeRobot datasets with language labels, an interaction corpus for world models, and shared training infrastructure |
| Vision-language-action models | Phase 3: a frozen SmolVLM2-500M with a learned action head, evaluated on language grounding and under distribution shift |
| In-domain JEPA | Phase 4: a LeWorldModel-style JEPA trained from scratch on T3-L1, probed for task state against a random encoder, a supervised encoder, and frozen DINOv2 ([`jepa.md`](jepa.md)) |
| World models | Phase 5: latent dynamics ensembles with calibrated uncertainty and OOD signals, a validity map, physics-based baselines, CEM planning, and policy evaluation in imagination |
| Generative models and scenario generation | Phase 5: a latent flow-matching decoder that renders imagined rollouts, scored by the physical state recovered from its frames, with a distilled few-step sampler; and a surrogate-guided search for scenes that make the VLA fail, verified in MuJoCo |
| Distillation and inference | Phase 6: privileged-teacher distillation with DAgger and ONNX latency benchmarks |
| Reproducible evaluation | A cross-model benchmark with confidence intervals, one-command regeneration, and a test suite |

---

## 2. Goals and success criteria

**Done means all of the following exist:**

1. A public repo where `make setup && make smoke` finishes in under 30 minutes from a clean clone. This excludes model-weight downloads and training. The smoke target runs the unit tests and one short run per layer.
2. Unit tests for the environment: determinism, success predicates, and camera projection.
3. A policy benchmark comparing the scripted expert, the RL expert, the BC baselines, the frozen-backbone VLA, distilled students, and world-model planning on shared tasks and seeds, with 95% confidence intervals, parameter counts, and latency.
4. A world-model report: prediction error against physics-based models, calibrated uncertainty, out-of-distribution detection, a validity map, planning results, and policy-evaluation correlation.
5. A generative-decoder report (physical consistency of decoded rollouts, and its quality-latency trade-off under step distillation) and a scenario-generation report (verified failures found per rollout against uniform sampling).
6. A representation study (an in-domain JEPA probed against frozen DINOv2) and a distillation study with a success-latency-size trade-off.
7. A demo video (about 2 minutes): the world model's imagined rollout, decoded to frames, next to the real execution, a CEM-planned episode, a generated failure scenario, and the results tables.
8. A technical write-up (README plus report) with ablations and a limitations section.

**Non-goals.** Real hardware, sim-to-real transfer, novel algorithms, a production system, collision-aware motion planning beyond simple approach and retreat waypoints, implementing kinematics, dynamics, or controllers, pretraining VLMs or training large VLAs, world models, or generative video models from scratch, and multi-GPU training at scale. The Phase 3 baseline freezes a pretrained small VLM and trains only a supervised action head.

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

### 3.4 Action and observation spaces

- **Learned-policy action (5-D):** an absolute TCP target in the `base` frame, given as position (3) plus yaw (1) with the gripper pointing straight down, plus a gripper command (1). Top-down grasps are enough for cubes and a bowl, and they avoid rotation-representation problems. ACT predicts absolute targets, and the Diffusion Policy paper reports that position control beats velocity control. IK and the position servos (§3.3) track this target. The VLA and every distilled student use the same action space, so all policies compare directly.
- **Phase 1b RL policy action:** a bounded delta from the controller's current TCP target (xyz up to one 1.25 cm rate-limit step, yaw up to 0.05 rad) plus a binary gripper. The RL wrapper, and the frozen policy's `Policy.act`, integrate it into the absolute action above. Deltas make holding still and millimeter precision closed-loop and keep exploration local, whereas absolute targets leave both open-loop and dominated by exploration noise.
- **Image-policy observation:** front RGB plus wrist RGB at 256×256 for every dataset from Phase 2 on (the earlier BC baselines use 128×128), plus proprioception (joint positions, TCP pose, gripper width). Pretrained encoders such as DINOv2 expect about 224 px inputs, and the cubes span only a few pixels at 128 px in the front view.
- **State-policy observation:** proprioception, object poses (noisy under T2-H), target pose, and gripper state.
- The BC baselines and the students distilled from the RL expert are **not** language-conditioned, so object identities that define a task are fixed by color for them (T4: red on blue). The frozen-backbone VLA baseline uses language on T3-L.

### 3.5 Conventions (fix on day one, in `docs/conventions.md`)

- SI units and radians everywhere.
- Frames: `world`, `base`, `ee` (the TCP site), `cam_front`, `cam_wrist`.
- Quaternions are **w, x, y, z** (MuJoCo). SciPy's `Rotation` defaults to **x, y, z, w**, so always pass `scalar_first=True` or convert in one helper.
- `data.xmat` and `data.site_xmat` are row-major 9-vectors: use `reshape(3, 3)`.
- One seeding utility is used everywhere, with separate seed ranges for training, validation, and evaluation (§9).

---

## 4. System architecture

Four layers. Each one runs on its own and is evaluated on its own. The architecture diagram is in [roadmap.md §2](roadmap.md#2-architecture).

1. **Physics and robot.** A MuJoCo scene, built with `MjSpec`, containing the Panda, a table, colored cubes, a bowl, a front camera, and a wrist camera.
2. **Motion and skills.** Off-the-shelf IK (`mink`) and MuJoCo's position servos, wrapped as parameterized skills and a scripted expert.
3. **Experts and data.** A PPO expert trained on privileged state, and a data engine that turns it into versioned demonstration datasets and an interaction corpus.
4. **Learned models.** A frozen-backbone VLA baseline; an in-domain JEPA probed against frozen DINOv2; latent world models with uncertainty, planning, policy evaluation, a generative frame decoder, and scenario generation; and distilled students. Every policy shares one action interface.

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
| T5 | Multi-step language instructions combining T2-T4 | all of the above plus distractors | goal predicate for each instruction | 60 s |

### 5.2 Randomization levels

- **L0:** fixed scene.
- **L1:** object and target poses.
- **L2:** L1 plus lighting, textures, distractor objects, small camera pose jitter, and small friction and mass perturbations.

### 5.3 Dynamics-shift variant T2-H

T2-H is the evaluation-only T2 L1 scene with a **tight placement tolerance** (1 cm and 10° yaw), **unmodeled dynamics** (target cube mass and inertia × Uniform[0.9, 1.1], friction × Uniform[0.9, 1.1]), and **state-observation noise** on the target cube pose (independent Gaussian σ = 0.15 mm per position axis and 0.15° in yaw per observation). The randomized factors and noise parameters are hidden from policies and models; a state policy sees the noisy pose, while simulator state and the success check use the true pose. The visual observation is not synthetically altered.

The perturbations are calibrated against the frozen T2 RL expert: 95/200 successes (47.5%; Wilson 95% CI 40.7-54.4%) on validation seeds 100,000-100,199 and 83/200 (41.5%; CI 34.9-48.4%) on evaluation seeds 1,000,000-1,000,199. This leaves headroom in both directions while retaining the fixed 1 cm / 10° predicate.

---

## 6. Environment and stack

### 6.1 Software

- **Python 3.12.** Current LeRobot requires Python ≥ 3.12. Use `uv` with a lockfile.
- **`mujoco`** (pin a 3.x version that `mink` supports) and **`mink`** for IK.
- **MuJoCo Menagerie** as a git submodule pinned to a specific commit. It is not a pip package; `robot_descriptions` is an alternative that fetches it.
- **PyTorch ≥ 2.10** with the MPS backend. LeRobot itself accepts older versions, but ≥ 2.10 pairs with a `torchcodec` that supports Homebrew's current `ffmpeg`.
- **`lerobot`** (pinned), with its `dataset` extra. Install `ffmpeg` via Homebrew.
- **`transformers`** (pinned) for pretrained models: SmolVLM2 and DINOv2.
- **`diffusers`** (pinned), only to load the frozen pretrained image autoencoder used by the frame decoder.
- **`onnx`** and **`onnxruntime`** for exported students and world models.
- **`gymnasium`** and **`stable-baselines3`**.
- **`pytest`**.
- Pydantic configs, and TensorBoard for experiment tracking (Weights & Biases optional).

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
| Frozen-backbone VLA baseline (frozen VLM, trained action head) | MPS | about 33 min to encode the limited T3-L profile once (3.8 examples/s, about 3 GB), then minutes per head run from cached features |
| Frozen DINOv2 features for the Phase 4 probes | MPS | minutes to tens of minutes (estimate) |
| In-domain JEPA and world-model ensembles | MPS | about 1 h per run (measured for the JEPA: 0.54 s per step at 160 px front and 112 px wrist, 6,000 steps; about 2.5 h for the whole JEPA phase); ensemble members after the first train on cached latents |
| Flow-matching frame decoder | MPS | about 1–2 h per world-model configuration at 160 px, plus a shorter distillation run (estimate) |
| Scenario generation | CPU rollouts, MPS for the VLA | about 40 s per VLA episode, about 13 h for three search methods × 400 rollouts (estimate) |
| Distillation (DAgger) | CPU rollouts, MPS training | hours per student |

**36 GB:** keep only one large model in memory at a time, and never alongside a training job.

---

## 7. Repository layout

- `third_party/mujoco_menagerie/`: pinned git submodule
- `common/`: seeding, transforms, config, run records
- `envs/`: MuJoCo scene building (MjSpec), task definitions, randomization, observations, rewards, success predicates
- `motion/`: `mink` IK wrapper, control loop, waypoint interpolation
- `skills/`: parameterized skills and the scripted expert
- `data/`: demo generation and LeRobot dataset writing
- `policies/`: PPO expert, BC baselines, frozen-backbone VLA, and distilled students
- `encoders/`: the in-domain JEPA, the frozen DINOv2 reference, the frame cache, and probes
- `world_models/`: latent dynamics, readouts, uncertainty, planners, policy evaluation in imagination, the flow-matching frame decoder, and scenario generation
- `distill/`: DAgger and teacher-student training, ONNX export, and latency benchmarks
- `eval/`: benchmark harness, metrics, statistics, plots, and `results/` (JSON)
- `tests/`: unit tests (scene, determinism, predicates, projection) and smoke tests for each layer
- `scripts/`: entry points for training, evaluation, and video
- `docs/`: this file, `conventions.md`, `decisions.md`, figures, write-up, video script

---

## 8. Phase plan

The project is built in nine phases (0, 1, 1b, and 2-7). Each phase ends with exit criteria that must hold before the next one starts.

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
- **Skills:** `reach(pose)`, `grasp(obj)`, `place(pose)`, `pick_and_place(obj, destination)`, and `stack(a, b)`. Each has preconditions, a postcondition check, a timeout, and a structured result with a success flag and a failure reason.
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

### Phase 3: VLA baseline

**Build.**

- **Frozen-backbone VLA baseline:** freeze the open SmolVLM2-500M checkpoint and train a small action head on T3-L demonstration data. The inputs are both camera views (read as the VLM's image-token grids), proprioception, and the instruction; the head predicts a short chunk of future TCP changes and gripper commands, which the policy blends into one shared 5-D action per policy step. This baseline measures direct VLM-feature-to-action mapping.

**Experiments.** Record the frozen-backbone baseline's T3-L success, wrong-object rate, and latency; compare it with image BC where task support overlaps; train on L1 and test on L2; and measure its language grounding on T3-L train and held-out paraphrases.

**Exit criteria.** The frozen-backbone baseline's T3-L results at L1 and L2 and on held-out paraphrases have confidence intervals and the wrong-object rate.

### Phase 4: In-domain JEPA

One experiment; the full design is in [`jepa.md`](jepa.md).

**Build.**

- Data: the 1,046 training-seed T3-L1 episodes that already exist (the interaction corpus T3 slice and the `t3-l1` demonstrations), plus two held-out collections from the interaction collector: 100 episodes on validation seeds and 200 on evaluation seeds. Front frames at 160 px and wrist frames at 112 px, at 5 Hz, which keeps each training run to about an hour (estimate).
- A LeWorldModel-style JEPA trained from scratch, end to end from pixels: a ViT-Tiny/16 encoder shared by both cameras, a 256-D latent, and a causal transformer predictor conditioned on the executed actions through AdaLN-zero. The loss is next-embedding MSE plus SIGReg, with no EMA target or stop-gradient; collapse is monitored through per-dimension spread and effective rank.
- References: a random encoder of the same architecture (floor), the same architecture trained on the probe labels (ceiling), and frozen DINOv2 ViT-S/14.

**Experiments.** Linear and MLP probes from each encoder to cube, bowl, and TCP position, cube yaw, gripper width, and held and in-bowl flags, with cube error split by resting and held; predictor rollouts of 0.2–1.0 s against persistence; optionally, one-step surprise under physically impossible and visual-only interventions.

**Exit criteria.** The probe table and rollout figure exist on evaluation-seed episodes; the JEPA's linear probe locates the cube better than the random encoder's; and the encoder for the T3-L1 student (JEPA or DINOv2) is chosen on validation episodes.

### Phase 5: Latent world models

**Build.**

- An action-conditioned transformer predictor over frozen encoder tokens from both cameras plus proprioception, at 5 Hz, trained with one-step and multi-step rollout losses (the DINO-WM recipe), on frozen DINOv2 features and end to end with the Phase 4 JEPA recipe retrained on the full interaction corpus (LeWorldModel-style). Train ensembles of 5 predictors; each configuration's encoder is trained or run once and its features cached, so ensemble members after the first are cheap.
- Readouts from latents: object poses and gripper width (so predictions are interpretable in cm) and task success.
- Physics baselines on the same action sequences: MuJoCo with true parameters (exact, the upper bound), MuJoCo with nominal parameters under T2-H (a physics model with parameter error), and a kinematic model in which the cube moves rigidly with the gripper while grasped.
- A generative frame decoder: latent flow matching in the space of a frozen pretrained image autoencoder, conditioned on the world-model latent, so predicted latents can be rendered as front-camera video. Its sampler is distilled from many steps to one or two.
- Scenario generation: a parameterized scene generator restricted to the existing randomization ranges, searched with CEM for scenes that make a target policy fail, ranked by a learned failure surrogate and verified in MuJoCo.

**Experiments.**

- Prediction error vs horizon for learned and classical models, with compute per step, stating where the classical models win.
- Uncertainty: ensemble disagreement calibrated on validation seeds, evaluated by interval coverage, Gaussian NLL, and the correlation between predicted uncertainty and actual error; expected calibration error for the success readout.
- Out-of-distribution detection: AUROC for separating L1 from L2, T2-H, and unseen colors, using ensemble disagreement and one-step surprise.
- A validity map: error and calibration across horizon, randomization level, dynamics shift, task phase, and training data, with a stated trusted region.
- Planning: CEM in latent space toward a goal observation, with and without an uncertainty penalty, against the same planner using MuJoCo as the model, on T1, T2, and T2-H.
- Policy evaluation in imagination: predicted vs real success for at least 8 policies (Pearson, Spearman, MMRV).
- Generative decoder: physical consistency, measured by re-encoding generated frames with frozen DINOv2 and probing the cube, bowl, TCP, and gripper state against the simulator, along rollouts and against the number of sampling steps for the teacher and the distilled sampler.
- Scenario generation: verified failures per 100 rollouts for the surrogate-guided search, unguided CEM, and uniform sampling at equal budget, with the failure reasons found. The resulting stress-test suite is frozen and reported separately from the benchmark.

**Exit criteria.** The prediction, calibration, OOD, validity-map, physics-comparison, planning, policy-evaluation, decoder, and scenario-generation results exist. No planning success threshold is set; results are reported as measured.

### Phase 6: Distillation and efficient inference

**Build.**

- Privileged teacher to vision student: the RL expert labels states visited by an image student on frozen DINOv2 features (DAgger), compared with offline behavior cloning on the same number of labels. DINOv2 is used for every task, including T3-L1, because it beat the Phase 4 JEPA on the validation probes.
- ONNX export with ONNX Runtime (CPU and Core ML), `torch.compile`, and int8 quantization, for the students, the world-model predictor, and the distilled frame-decoder sampler.

**Exit criteria.** A success vs latency vs parameter-count Pareto table covering the RL expert, the students, and their exported variants.

### Phase 7: Benchmark, packaging, and release

- Freeze the benchmark: final seeds, final configs, and one command that regenerates every table and figure.
- Run the final evaluations with Wilson (success rates) or bootstrap (across seeds) 95% confidence intervals.
- Demo video, about 2 minutes: the world model's imagined rollout, decoded to frames, next to the real execution, a CEM-planned episode, a generated failure scenario, and the results tables.
- README: architecture diagram, results, how to reproduce, limitations.
- Write-up: what was built, the key findings (including where learned models beat physics-based models and where they do not), and directions for future work.
- Repo hygiene: CI green, pinned requirements and lockfile, license, documented seeds.

---

## 9. Evaluation protocol

- **Seeds.** Use separate seed ranges for training, validation, and evaluation, defined in config. Evaluation seeds are never used for training or tuning.
- **Episodes per cell.**
  - Scripted and state-based methods: at least 100.
  - Image policies, including the frozen-backbone VLA and distilled students: at least 50.
  - World-model planners: at least 50.
  - RL: 3 training seeds × 100 evaluation episodes.
- **Precision.** The 95% Wilson half-width at 50% success is about ±13 percentage points with n = 50, ±10 with n = 100, and ±7 with n = 200. Do not claim differences smaller than this. For RL, variance across seeds usually dominates, so bootstrap over seeds.
- **Success.** Task-specific geometric predicates, written in code before any results are seen (§5.1).
- **Metrics.**
  - Success rate with a 95% CI, mean completion time, and collision count.
  - Policies: parameter count and p50/p95 latency at batch size 1, with backend and precision.
  - World models: prediction error vs horizon (cm, decoded state), interval coverage and Gaussian NLL, expected calibration error, OOD AUROC, and Pearson, Spearman, and MMRV for policy evaluation.
- **Main benchmark table.**
  - Rows: scripted expert, RL expert, state and image BC baselines (scripted-expert demonstrations at 128 px), the frozen-backbone VLA, distilled students, and world-model MPC.
  - **Shared columns:** T2-L1 and T2-H.
  - Extra columns: T3-L1, T4-L1, and T2-L2 for the methods whose observations support them, and T3-L for the frozen-backbone VLA. Mark unsupported cells N/A explicitly.
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
| In-domain JEPA training slow or memory-bound on MPS | Medium | Smaller batch, activation checkpointing, shorter step budget, or the front camera only; record the change in `run.json` |
| JEPA representation collapse | Medium | Monitor per-dimension spread and effective rank; raise the SIGReg weight; EMA target encoder as a last resort |
| JEPA encodes the arm but not the cube ("static collapse") | Medium | Report cube error separately for resting and held cubes; more object-interaction data; predict further ahead |
| Imagined rollouts drift over long horizons | Medium | Multi-step rollout losses; report the horizon where predictions stop being trustworthy |
| Frame decoder too slow on MPS, or its frames look plausible but misplace the cube | Medium | Smaller decoder or 112 px pixel-space decoding; report physical consistency against the autoencoder floor as measured |
| Scenario surrogate no better than chance | Medium | Report it; fall back to CEM verified directly in MuJoCo with a halved budget |
| LeRobot API or dataset-format churn | Medium | Pin the version |
| Dependency conflicts (LeRobot, `transformers`) | Medium | Pinned versions and a lockfile |
| Phase 1b RL expert does not beat the scripted expert on some task | Medium | Keep the scripted expert as that task's demo source; proceed with the remaining tasks. If pure PPO falls well short, the next options are SAC/TQC with HER or BC-initialized PPO, which need a roadmap decision. |
| Python control loop too slow for RL and data generation | Medium | Profile; IK only at 100 Hz; more worker processes |
| IK library integration stalls | Low | Drive the hand with a mocap body welded to it, with the arm actuators removed |

---

## 12. Core and optional components

**Core:** scene and determinism tests, the evaluation harness, the scripted expert, the RL expert, the datasets and training infrastructure, the frozen-backbone VLA baseline, the in-domain JEPA probe study, the latent world-model ensemble with physics baselines, uncertainty and OOD evaluation, the validity map, CEM planning on T1 and T2, policy evaluation in imagination, the generative frame decoder with its distilled sampler, scenario generation with the classifier surrogate, one distilled student with measured latency, and the write-up.

**Optional, in decreasing priority:**

1. The imagined-rollout surrogate for scenario generation
2. The JEPA surprise test
3. T4 stacking for learned methods (it remains for the scripted and RL experts)

---

## 13. Future work

- **Larger VLAs.** Fine-tune π0.5 or GR00T N1.7 on a rented GPU and compare them with the frozen-backbone VLA.
- **Multi-GPU training.** Run the world-model and JEPA training on a multi-GPU node with the existing DDP code and report scaling efficiency.
- **Generative video world model.** One autoregressive, action-conditioned latent flow-matching model over frame sequences, replacing the JEPA predictor and frame decoder pair, trained on a rented GPU.
- **Distributed data and evaluation.** Data collection, evaluation, and scenario search across machines with Ray.
- **Deployment.** TensorRT engines for the students and the world-model predictor on an NVIDIA GPU or Jetson.
- **Broader representation study.** Pretrained encoders (V-JEPA 2 or 2.1, ResNet-18, DINOv3) probed alongside the in-domain JEPA, an L2 test set, and closed-loop policies on each frozen encoder.
- **Pretrained world models.** V-JEPA 2-AC zero-shot planning and a generative video world model (for example, NVIDIA Cosmos) as policy evaluators, compared with the in-repo models.
- **Standard benchmarks.** LIBERO, for comparability with published results.
- **Isaac Lab port.** GPU-parallel RL and data generation.
- **Sim-to-real.** The distilled students on an SO-101-class arm through LeRobot.

---

## 14. References

- MuJoCo, MuJoCo Menagerie, `mink`, LeRobot, and Hugging Face `transformers`.
- Papers:
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
  - Flow matching: Lipman et al., 2023 (arXiv:2210.02747)
  - Rectified flow: Liu et al., 2023 (arXiv:2209.03003)
  - Latent diffusion: Rombach et al., 2022 (arXiv:2112.10752)
  - Scalable diffusion models with transformers (DiT): Peebles and Xie, 2023 (arXiv:2212.09748)
  - Knowledge distillation of iterative samplers: Luhman and Luhman, 2021 (arXiv:2101.02388)
  - Progressive distillation: Salimans and Ho, 2022 (arXiv:2202.00512)
  - Black-box safety validation (survey): Corso et al., 2021, "A Survey of Algorithms for Black-Box Safety Validation of Cyber-Physical Systems"
  - Learning by Cheating (privileged teacher, vision student): Chen et al., 2019
  - DART: Laskey et al., 2017
  - ACT: Zhao et al., 2023; Diffusion Policy: Chi et al., 2023 (action-space rationale, §3.4)
- Stable-Baselines3; rliable (Agarwal et al., 2021) for statistically sound RL reporting.
