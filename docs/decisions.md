# Design decisions

The choices below define the current implementation. The [roadmap](roadmap.md) is the source of truth for project scope, build phases, tasks, and evaluation rules; this document records the rationale for the current design.

| Topic | Decision | Rationale |
| --- | --- | --- |
| Robot | Franka Emika Panda from Menagerie, arm and hand, 7 DOF | Common manipulation-learning reference with an integrated gripper model. |
| Robot base | World origin on the table surface; `base` = `world` | Keeps the base frame and table reference aligned. |
| Physics | `implicitfast`, 2 ms timestep (500 Hz) | Uses the Panda model's integrator choice at a fixed simulation rate. |
| Joint control | Menagerie position servos; `data.ctrl` contains position targets | Reuses the provided controller instead of adding custom torque control. |
| IK and interpolation | `mink` differential IK on a robot-only model at 100 Hz; linear waypoint interpolation | Uses off-the-shelf IK and keeps free objects out of the IK model. |
| Policy rate | 20 Hz, one step per 25 physics steps | Separates policy decisions from the higher-rate physics loop. |
| Learned-policy action | Absolute base-frame TCP target `[x, y, z, yaw, g]`, top-down gripper, `g` in [0, 1] | One compact action representation is shared across learned methods. |
| Action limits | Clip xyz to the workspace and yaw to [−π/2, π/2] | Constrains commands to the intended operating region. |
| Phase 1b RL policy action | Deltas from the controller's TCP target (xyz ≤ 1.25 cm, yaw ≤ 0.05 rad), binary gripper, integrated into the absolute action | Makes holding still and fine positioning closed-loop and keeps exploration local; the frozen policy emits absolute actions through the same tracker. |
| Image observations | Front and wrist RGB images, uint8, 256 × 256 for every dataset from Phase 2 on (the existing BC baselines use 128 × 128) | Matches the input resolution of the pretrained encoders, and keeps the 4 cm cubes several pixels wide in the front view. |
| State observation | 7 joint positions, gripper width, 3-D TCP position, TCP yaw | Defines a compact, shared proprioceptive state. |
| Units | SI units and radians | Avoids unit conversions between simulation, policies, and evaluation. |
| Frames | `world`, `base`, `ee`, `cam_front`, `cam_wrist` | Gives transforms and observations explicit frame names. |
| Quaternions | MuJoCo order w, x, y, z; SciPy calls use `scalar_first=True` | Prevents the libraries' different component ordering from being confused. |
| Rotation matrices | Reshape MuJoCo `xmat` row-major 9-vectors to 3 × 3 | Preserves MuJoCo's documented matrix layout. |
| Camera axes | Convert MuJoCo OpenGL camera axes to OpenCV axes for pinhole projection | Keeps camera projection formulas consistent with their expected axes. |
| Seed ranges | Train 0-99,999; validation 100,000-109,999; evaluation 1,000,000-1,009,999 | Keeps evaluation seeds separate from training and tuning. |
| Randomness | Spawn named independent generators from one `SeedSequence` | Makes random streams reproducible and separates purposes such as scene, noise, and policy. |
| Arm position-servo stiffness | Multiply the Panda arm actuators' position gain and position-feedback bias by 5; preserve actuator type and velocity damping. | Reduces gravity-induced TCP tracking error while retaining the Menagerie position-target interface. |
| Grasp contact settings | Use 50 g cubes and friction `[2.0, 0.005, 0.0001]` on cubes and fingertip pads; keep `impratio=10` and `solref=[0.02, 1.0]`. | Provides stable contacts for grasping, lifting, and holding the cube. |

## Frozen-backbone VLA baseline

- **VLM backbone:** [`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct), an approximately 500M-parameter, Apache-2.0 multimodal model that accepts text and multiple images. Its native output is text, not robot actions.
- **Why this candidate:** it is the smallest of the candidates considered and fits the reference laptop with headroom. [SmolVLM2-2.2B](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) offers more capacity at higher compute cost; [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) is another option, but its model card expects a newer Transformers stack. Revisit these only if the 500M backbone cannot ground T3-L instructions.
- **Policy:** freeze the VLM, standardize its image-token and instruction features, combine them with normalized 12-D proprioception, and train a small head: a shared 960→8 linear projection of every image token and a 960→32 projection of the instruction feature, followed by a 2-layer, 256-unit MLP. The head predicts the next 10 actions (0.5 s) as changes from the TCP pose at prediction time, normalized per chunk position by training-split statistics, plus one binary gripper logit per step. As with the RL expert, a prediction error becomes a drift that the closed loop corrects rather than a fixed offset.
- **Action chunking:** a single-step head decides the gripper independently at every frame, so isolated misclassifications at the open-to-close transition release or miss the cube, and it must run the VLM at every 20 Hz step (about 0.27 s per query on MPS). With chunks, the `Policy` queries the VLM every `replan_interval` steps (default 2) and executes the weighted average of all stored plans that cover the current step; a plan made `a` steps ago has weight `exp(-ensemble_decay · a)` (default 0, an equal average). Gripper decisions average the plans' open probabilities. Chunk positions past the end of a demonstration are masked out of the loss. On 10 limited-profile validation scenes, re-planning every 1, 2, 5, or 10 steps cut gripper command changes from 11.6 per episode (single-step head) to 3.2, 4.0, 5.2, and 6.8; every 2 steps lifted the named cube most often (5 of 10, against 2 for the single-step head) with half the VLM queries of re-planning every step. Checkpoint format 4 stores the chunk size; format-3 single-step heads load with a chunk size of 1.
- **Adapter:** pin `transformers==5.20.2`, load the checkpoint with `AutoProcessor` and `AutoModelForImageTextToText`, and disable image splitting so each 256 px camera is one 512 px view of 64 image tokens (an 8 × 8 grid). Place the instruction before both images in the user prompt, then read decoder layer 16 at the 128 image-token positions and at the final prompt token. No text is generated and the vocabulary projection is skipped. Frozen features are encoded once per split and can be cached on disk, keyed by data selection, prompt layout, and encoder identity.

**Supporting measurements.** Probes are ridge regressions or classifiers on frozen features of limited-profile training episodes (first frames, or 6 frames per episode), cross-validated in five folds grouped by episode. Readout comparisons used the images-first prompt; the bowl-position target does not depend on the instruction.

| Choice | Alternative measured | Evidence |
| --- | --- | --- |
| Instruction before images | Images before instruction | The decoder is causal, so image tokens cannot attend to a later instruction. Encoding each of 150 scenes once per active cube's color, images-first grids were bit-identical across instructions and a probe chose the named cube in 50% of scenes; instruction-first, 90–94%. On 10 validation scenes, the images-first head reached the wrong cube first in 6 and the instruction-first head in 2. |
| Image-token grid readout | Final prompt token, color-word token, and mean-pooled image tokens, each at layers 8–32 | Bowl position relative to the TCP at layer 16 (RMSE, x / y): image-token grid 1.5 / 1.9 cm; mean-pooled image tokens 2.5 / 3.0 cm; final prompt token 4.3 / 5.3 cm; color-word token 4.0 / 4.7 cm; robot state alone 3.0 / 4.1 cm. At every layer tested, the grid was the most accurate readout, and the final-prompt and color-word tokens were less accurate than robot state alone. The layer-16 final token identifies the named color in 99% of held-out episodes, so it is kept as the instruction feature. |
| No image tiling | Processor default: frame upsampled to 2048 px and split into 16 tiles plus a global view (2,279 tokens per sample) | With default tiling, the final-token feature explained little of the target cube's position (R² ≈ 0.0 / 0.3 for x / y), and a head on it was less accurate offline than a state-only MLP. Without tiling, the same readout reached R² ≈ 0.4–0.5 at 10× the encoding throughput and a quarter of the memory. |

- **Model revision and hardware profile:** resolve `main` to an immutable Hugging Face commit before loading and store it with the action-head checkpoint and `run.json`. With revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` on MPS (fp16), the untiled encoder measured 3.77 encoded examples/s and 3,094 MiB of driver-allocated memory over the 7,536 limited-profile examples (about 33 minutes); default tiling measured 0.32 examples/s and 13,285 MiB. Head training from cached features takes a few minutes. Each run's full configuration and metrics are in its `run.json`.

## RL expert and dataset design

- Train one pure-PPO policy from scratch per non-language task (T1-T4) on privileged state. A task's best qualifying checkpoint is the primary demonstrator for Phase 2 datasets and the privileged teacher for distillation.
- Keep the scripted expert in the benchmark as a comparison baseline and as a fallback demonstrator for any task whose RL checkpoint does not meet its target.
- Generate T3-L demonstrations with the T3 expert targeting the instructed cube. T1 has no object and uses its checkpoint to establish parity with the scripted baseline.

## RL training, checkpointing, and evaluation

- **Checkpoint selection by success rate.** A training callback evaluates the deterministic policy on fixed validation seeds every `eval_freq` steps and keeps a `best/` snapshot. It scores the task's own success predicate, including its hold time, rather than the mean return that SB3's `EvalCallback` reports.
- **Reproducible checkpoints.** Each run directory holds `model.zip`, the `VecNormalize` statistics, and `run.json` (config, git commit, seeds, hardware, learning curve), mirroring the BC layout. Storing the normalization statistics with every checkpoint reproduces the exact observation normalization at evaluation time.
- **Resuming.** A resumed run keeps the checkpoint's hyperparameters, counts `total_timesteps` as additional steps, preserves its evaluation cadence, and writes to a new directory that records its source checkpoint.
- **Shared evaluation.** `RLPolicy` exposes a checkpoint through the `Policy` protocol, so RL experts are evaluated by the same harness and reported in the same format as the scripted expert and BC.

## RL environment specification

`PickPlaceRLEnv` wraps `PickPlaceEnv` for the non-language tasks and defines the action parameterization, observation, reward, and training curriculum shared by every RL policy.

| Area | Decision | Rationale |
| --- | --- | --- |
| Action | Normalized deltas from the controller's TCP target: xyz up to one rate-limit step (1.25 cm), yaw up to 0.05 rad, and a binary gripper. `DeltaCommandTracker` integrates them into the absolute §3 action using the same target update as `ControlLoop`, in both the training wrapper and `RLPolicy`. | Holding still is a zero output and precision is closed-loop, so small network errors produce a drift the policy corrects rather than a fixed offset. Exploration noise stays local to the current target, and the binary gripper closes decisively. The frozen policy emits absolute actions, so it is a drop-in `Policy`. |
| Observation | 36-D: proprioception; the object pose, with yaw as (sin 4θ, cos 4θ) and the grasp yaw error wrapped to [−π/4, π/4); the destination; object − TCP and destination − object; the commanded target and yaw relative to the TCP; the last gripper command; and the object's displacement since the previous step. | The 4θ encoding makes the cube's 90° symmetry explicit, and the command state makes the observation Markov with respect to the rate-limited controller. Every feature is computable from the dict observation, so training and evaluation see identical inputs. |
| Reward | Each step pays `(staged − max) / max ∈ [−1, 0]`, a 0.001 action-magnitude penalty, and +10 on success. T1: `proximity(TCP-to-target) + gripper openness`. T2–T4: `proximity(TCP-to-object) + grasped · (1 + proximity(object-to-carry-path))` until the object is within 1.5 cm of the destination, then `4 + gripper openness`. `proximity(d)` averages `1 − tanh(k·d)` for k = 2, 8, 30 m⁻¹, and the carry path runs 6 cm above the destination, blending down to it within 5 cm horizontally. | Each stage's best value exceeds every earlier stage's, so the reward always favors progress, and non-positive steps make finishing sooner strictly better. The reward depends only on the current state, with nothing latched or repeatable. The three length scales give a usable gradient from the ≈ 0.5 m starting distance down to millimeters. The 1.5 cm placement tolerance matches the tightest per-task success tolerance, and releasing at the destination never lowers the reward. |
| Curriculum | 30% of object-task training episodes start with the cube already in a closed grasp at a random 0–8 cm lift. An opt-in `destination_start_fraction` starts that share of grasped episodes at a random point along the path to the destination, 6–9 cm above it; T3 uses 50% grasped starts with 60% of them on the path. Both are `PickPlaceEnv` reset options with their own named seed stream, never used for validation or evaluation. | Carrying and placing are learned alongside grasping rather than only after it. Destination-side starts are a reverse curriculum: the policy experiences placing and success from the outset, so value propagates back along long carries such as T3's 30–50 cm. The separate seed stream, and drawing no extra randomness when the option is off, leave scene sampling and every other episode unchanged. |
| Seeds | Unseeded training resets draw from the training range [0, 100,000). | Keeps automatic resets out of the validation and evaluation ranges (§3). |
| PPO | `n_steps = 2048`, `log_std_init = −0.5`, `target_kl = 0.03`; an entropy bonus and gSDE are available but off by default. | About 55 T2 episodes per update, initial exploration noise sized for delta actions, and bounded policy updates. |
| Diagnostics | Each rollout logs training success, a stage funnel (reached, grasped, lifted, carried, placed), per-term reward sums, and PPO's `train/*` metrics to `logs/progress.csv`. | Shows which stage limits a run without tracing rollouts by hand. |

## T2-H calibration status

The evaluation-only T2-H implementation uses independent uniform per-episode target-cube mass and friction scales from `envs/configs/t2_hard.yaml`; mass and inertia share the same scale. State observations add independent Gaussian position noise (0.15 mm standard deviation per axis) and yaw noise (0.15° standard deviation) only to the target cube pose. The success predicate requires the cube to be at rest, released, within 1 cm of the target-zone center, and aligned within 10° modulo the cube's 90° symmetry. Neither simulator state nor success checks are perturbed.

Calibration used `checkpoints/rl/t2-l1/seed-0/best`. It achieved 95/200 successes (47.5%; Wilson 95% CI 40.7-54.4%) on validation seeds 100,000-100,199 and 83/200 (41.5%; Wilson 95% CI 34.9-48.4%) on evaluation seeds 1,000,000-1,000,199. The frozen parameters are recorded in `envs/configs/t2_hard.yaml`; the evaluation result is `eval/results/pure-ppo/T2-H-L1.json`.

## In-domain JEPA (Phase 4)

Complete; the full design is in [`jepa.md`](jepa.md) and the results are below. Phase 4 is one experiment on T3-L1 instead of a multi-encoder study, so it fits the reference laptop and reuses data that already exists.

| Topic | Decision | Rationale |
| --- | --- | --- |
| Scope | One JEPA trained from scratch on T3-L1, compared by probing with a random encoder, a supervised encoder of the same architecture, and frozen DINOv2 ViT-S/14. No V-JEPA 2, ResNet-18, L2 test set, or per-encoder policies. | The image-BC pipeline the old policy comparison assumed scores 0% on T2-L1, V-JEPA 2 ViT-L clip encoding is costly on MPS, and no L2 image data exists. T3-L1 has object interaction, two objects, and occlusion, and 1,046 training-seed episodes already exist. |
| Objective | LeWorldModel: next-embedding MSE plus λ · SIGReg, end to end from pixels, with no EMA target or stop-gradient | Two loss terms and one effective hyperparameter; it uses the trajectories and executed actions the interaction corpus provides; its predictor is already a latent world model for Phase 5. Crop-invariance (LeJEPA) and masked-image (I-JEPA) objectives ignore actions and time. |
| Encoder | ViT-Tiny/16 shared by both cameras, front at 160 px and wrist at 112 px, with a learned position-embedding table per camera; a BatchNorm projector to a 256-D latent | LeWM's encoder scale; the BatchNorm projector lets SIGReg shape the latent despite the ViT's final LayerNorm. 160 px keeps the cube about 4–6 px wide in the front view, and the wrist camera sees it up close, so 112 px suffices there. Together with the step budget below, this keeps each run to about 1 h, against an estimated 3–5 h at 224 px. |
| Rate and actions | 5 Hz (every 4th frame), with the 4 executed actions per step concatenated into a 20-D block | Matches the Phase 5.1 world-model rate and action encoding. |
| Data and seeds | Train on the existing training-seed T3-L1 episodes; collect `t3-l1-probe-val` (validation seeds) and `t3-l1-probe-test` (evaluation seeds) with the interaction collector | λ and the Phase 6 T3-L1 encoder are selected on validation episodes; evaluation seeds are used only for final numbers. |
| Downstream use | Phase 5 world models use frozen DINOv2 and the JEPA recipe retrained on the full interaction corpus; Phase 6 students use DINOv2 for every task; the JEPA would have replaced it for the T3-L1 student had it won the validation probes, but DINOv2 won (validation probe score 0.067 vs 0.089) | The Phase 4 model sees only T3 frames, so it cannot serve every task and level. |
| SIGReg implementation | Implemented from the published Epps–Pulley formula (17 knots on [0, 4], using the integrand's symmetry), in fp32 outside autocast, with known-answer tests | The LeJEPA reference code is CC BY-NC 4.0, incompatible with this repository's MIT license, so it is not ported. |
| Precision | fp32 | Measured on MPS at batch 64: fp32 0.44 s per step of compute, bf16 autocast 0.50 s, and fp16 slower end to end; fp32 also needs no loss scaling. |
| Step budget | 6,000 steps per run (warmup 600). The λ sweep runs the first 1,000 steps of each value's seed-0 run; the winner is resumed to 6,000 steps as final seed 0 | At 0.54 s per step, the planned 10,000 steps would take 1.5 h. The plan's rule for an over-budget run is to cut steps before batch size. |
| Final training seeds | One JEPA training seed (0), resumed from the selected sweep run | Fits the reduced compute budget (about 2.5 h for the phase instead of 4.5 h). Probe intervals then cover test-episode variation but not training-seed variation, so H1 is a single-run result; more seeds are a follow-up. The random encoder keeps 3 seeds because it needs no training. |
| Data order and resume | Epoch permutations derived from (seed, epoch); every DataLoader has its own generator; `--stop-after-steps` ends a session without changing the schedule | A resumed run reproduces an uninterrupted one exactly (tested): a DataLoader iterator otherwise draws a seed from the global RNG and shifts the SIGReg directions. |

**Measurements (M2–M4).**

- **M2, smoke run** (500 steps, batch 64, fp32, on all 1,046 training episodes; the probe score was validated on held-out training episodes because `t3-l1-probe-val` did not exist yet): 0.54 s per step including data loading (0.44 s compute), 8.7 GB peak driver-allocated MPS memory. From step 250 to 500, held-out prediction loss fell from 0.111 to 0.043 and the prediction-to-persistence ratio from 0.80 to 0.35, so the predictor uses the actions. The latent's mean per-dimension standard deviation stayed near 1.0 and its effective rank rose from 38 to 42 (of 256), so there is no collapse. The run's predictor already decoded TCP position to 6.4–7.5 cm at 0.2–1.0 s, against 8.0–20.1 cm for persistence.
- **Shared training utilities.** `common.training.optimizer_step` checked each gradient and parameter for finite values with a separate device synchronization (464 per step for this model), which added 0.15 s per step on MPS. The checks now reduce on the device and synchronize once, with the same errors; this brought the guarded step to within 0.03 s of an unguarded one.
- **M3, held-out sets:** `t3-l1-probe-val` (seeds 100,000–100,099; 21 RL expert, 4 RL-expert failures, 25 noisy expert, 25 intermediate PPO, 25 scripted exploration; 42 successes) and `t3-l1-probe-test` (seeds 1,000,000–1,000,199; 41, 9, 50, 50, 50; 85 successes).

- **M4, probe pipeline** (random ViT-Tiny/16, seeds 0–2, and DINOv2 ViT-S/14 at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`; probes fit on all 50,428 training timesteps, scored on the 200 `t3-l1-probe-test` episodes; 18 min end to end). Linear-probe errors with 95% episode-bootstrap intervals:

  | Encoder | Readout | Cube xyz (cm) | Cube yaw (°) | Bowl xy (cm) | TCP xyz (cm) | Gripper (mm) |
  | --- | --- | --- | --- | --- | --- | --- |
  | Random ViT-T/16 | global | 8.6 [8.3, 8.9] | 19.5 [19.0, 20.0] | 2.6 [2.5, 2.7] | 5.4 [5.2, 5.5] | 4.0 [4.0, 4.1] |
  | DINOv2 ViT-S/14 | global | 5.4 [5.1, 5.7] | 14.1 [13.4, 14.8] | 2.3 [2.2, 2.5] | 2.5 [2.4, 2.6] | 3.7 [3.6, 3.9] |
  | DINOv2 ViT-S/14 | grid | 2.9 [2.8, 3.1] | 12.2 [11.5, 12.9] | 0.9 [0.8, 0.9] | 1.2 [1.1, 1.3] | 1.7 [1.7, 1.8] |

  DINOv2 beats the random encoder on TCP position, so the M4 check passes. The random encoder is a strong floor (its held flag reaches 98.6% balanced accuracy), so H1 is a demanding test. The grid readout halves DINOv2's position errors, consistent with the VLA baseline's finding that pooling discards spatial information. The logistic probes were later capped at 20,000 training timesteps to shorten the wide-readout fits.

- **M5, λ sweep** (seed 0; each run on the full 6,000-step schedule, stopped at step 1,000; validation probe score on `t3-l1-probe-val`, lower is better; about 9 min per run at 0.50 s per step):

  | λ | Probe score | Cube xyz (cm) | TCP xyz (cm) | Effective rank (of 256) | Prediction / persistence |
  | --- | --- | --- | --- | --- | --- |
  | 0.025 | 0.1144 | 7.18 | 3.74 | 27 | 0.33 |
  | **0.05 (selected)** | 0.1160 | 7.06 | 4.03 | 35 | 0.38 |
  | 0.1 | 0.1379 | 7.79 | 4.69 | 50 | 0.49 |
  | 0.2 | 0.1337 | 7.78 | 4.42 | 66 | 0.79 |

  0.05 was best of the planned grid but sat at its lower edge, so 0.025 was added as the planned bisection step. 0.025 and 0.05 differ by 1.4% in probe score, less than the single-seed noise visible between 0.1 and 0.2 (which rank opposite to their order), so they are treated as tied. **λ = 0.05 is selected**: it has the lower cube error, which the H1 pass condition tests, and a higher effective rank; lowering λ steadily reduces rank, the early sign of collapse the plan monitors. Larger λ spreads the latent (higher rank) but weakens prediction relative to persistence. The λ = 0.05 run is resumed to 6,000 steps as final seed 0.

- **M6, final results** (`eval/results/jepa/`, rendered by `make report`; JEPA λ = 0.05, seed 0, 6,000 steps in 50 min; supervised ceiling 6,000 steps in 15 min; probes fit on all 50,428 training timesteps and scored on the 200 `t3-l1-probe-test` episodes; one JEPA seed, so intervals cover test episodes only). Linear probes on the global readout unless noted, with 95% episode-bootstrap intervals:

  | Encoder | Cube xyz (cm) | Cube resting / held (cm) | Bowl xy (cm) | TCP xyz (cm) | Gripper (mm) |
  | --- | --- | --- | --- | --- | --- |
  | Random ViT-T/16 (3 seeds) | 8.6 [8.3, 8.9] | 8.9 / 5.5 | 2.6 | 5.4 [5.2, 5.5] | 4.0 |
  | **JEPA ViT-T/16** | **6.7 [6.2, 7.1]** | 6.9 / 3.7 | 1.9 | 2.9 [2.7, 3.1] | 2.7 |
  | JEPA, grid readout | 5.4 [5.0, 5.7] | 5.5 / 3.3 | 0.7 | 1.8 | 1.3 |
  | Supervised ViT-T/16, linear probe | 5.5 [5.1, 5.8] | 5.6 / 4.1 | 0.9 | 3.2 [3.0, 3.4] | 2.7 |
  | Supervised ViT-T/16, own output (ceiling) | 4.4 [4.0, 4.8] | 4.5 / 2.0 | 0.6 | 2.5 [2.3, 2.6] | 2.2 |
  | DINOv2 ViT-S/14 | 5.4 [5.1, 5.7] | 5.5 / 3.8 | 2.3 | 2.5 [2.4, 2.6] | 3.7 |
  | DINOv2, grid readout | 2.9 [2.8, 3.1] | 3.0 / 2.3 | 0.9 | 1.2 | 1.7 |

  - **H1 passes.** The JEPA's cube-position interval [6.2, 7.1] cm lies entirely below the random encoder's [8.3, 8.9] cm.
  - **The ceiling is read through the supervised model's own output.** A linear probe on its features understates it (cube 5.5 vs 4.4 cm, TCP 3.2 vs 2.5 cm) because its trained head is a nonlinear network on those features; against the probe alone, the JEPA appeared to beat the ceiling on the TCP (2.9 vs 3.2 cm), which it does not. Both readings are reported.
  - **H2 is partly supported.** Measured as the share of the gap between the random encoder and the ceiling that the JEPA closes, it learns the robot's own state best (TCP 86%, gripper width 72%) and objects less well (carried cube 51%, resting cube 45%, bowl 35%): the gripper moves in every frame and its commanded targets are the predictor's action input. A carried cube is located far better than a resting one in absolute terms (3.7 vs 6.9 cm), but relative to the ceiling the two close a similar share of the gap.
  - **H3, specialist vs generalist:** frozen DINOv2 (22.1M parameters, web-pretrained) beats the 6.8M-parameter in-domain JEPA (encoder and projector) on the cube (5.4 vs 6.7 cm) and TCP (2.5 vs 2.9 cm); the JEPA is better on the bowl (1.9 vs 2.3 cm) and gripper (2.7 vs 3.7 mm). With grid readouts DINOv2 leads clearly on the cube (2.9 vs 5.4 cm). An MLP probe on the JEPA's global readout reaches 5.6 cm (cube) and 2.0 cm (TCP), so part of its information is not linearly readable.
  - **H4 passes.** Decoded through a linear probe on the latent, the predictor's error stays nearly flat with horizon while persistence grows: at 1.0 s, TCP 5.6 vs 13.5 cm and held cube 4.1 vs 18.3 cm, within about 1 cm of the probe floor (the same probe on the true future latent: 4.5 and 3.4 cm). Latent prediction error is 14% of persistence's at 0.2 s and 6.5% at 1.0 s. For a resting cube in free motion, persistence is already near the floor and the predictor does not improve on it.
  - **Not informative:** cube yaw is at chance for every ViT-Tiny encoder (about 19°; uniform guessing gives 22.5°) and only partly recovered by DINOv2 (12–14°); the held and in-bowl flags are near-perfect for every encoder, including the random one (balanced accuracy 95–99%, AUROC ≥ 0.99), so they do not separate the encoders.
  - **The 256-D world-model latent** keeps less than the encoder's global readout (cube 7.0 vs 6.7 cm, TCP 4.2 vs 2.9 cm, gripper 7.9 vs 2.7 mm), so the projector discards some state.
  - **Phase 6 T3-L1 student encoder: DINOv2.** Chosen on `t3-l1-probe-val` by the λ-selection criterion (validation probe score 0.067 for DINOv2 against 0.089 for the JEPA).

## Generative models and scenario generation (Phases 5.9 and 5.10)

Planned, not implemented. These components extend the world models with a generative frame decoder, sampler distillation, and test-scenario generation, within the reference laptop's budget: every new model trains on features or latents that are cached once.

| Topic | Decision | Rationale |
| --- | --- | --- |
| Compute | All training on the single reference laptop; multi-GPU training is future work | Keeps the project reproducible on the reference hardware. The DDP path is verified with 2-process CPU runs (Phase 2.5). |
| World-model ensembles (5.2) | Each configuration's encoder is trained or run once; the 5 ensemble members are predictors on its cached latents | Training 5 end-to-end encoders would cost about 5× the JEPA budget. The cost is that encoder uncertainty is not represented, which is stated with every uncertainty result. |
| Frame decoder (5.9) | Latent flow matching in a frozen pretrained image autoencoder's space, a small DiT or U-Net conditioned only on the world-model latent, front camera at 160 px | Decoding in a compressed latent space keeps training to about 1–2 h at the Phase 4 cache's 160 px; conditioning only on the world-model latent makes the decoder a test of what that latent encodes. The autoencoder's revision and license are checked before use. |
| Decoder metric (5.9) | Physical consistency: the Phase 4 DINOv2 probes applied to generated frames, against true simulator state, with real-frame and autoencoder-reconstruction floors | The simulator provides exact state, so generated frames can be scored on whether they place the cube correctly rather than on perceptual similarity alone. |
| Sampler distillation (5.9) | Regress the teacher's ODE endpoint from the same noise to obtain 1- and 2-step students | The simplest step-distillation method; it gives a quality-vs-steps-vs-latency curve to export alongside the policy students. |
| Scenario generation (5.10) | CEM over explicit scene parameters inside the existing level ranges, ranked by a failure classifier on frozen encoder features and refit on verified rollouts; every counted failure is a real MuJoCo rollout | The search cannot widen the task definitions (§4), and verification keeps every reported failure real. Comparing with uniform sampling and unguided CEM at equal budget measures what the learned surrogate contributes. |
| Stress-test suite (5.10) | Search randomness uses validation seeds; the frozen suite is reported separately and never used to train or tune the policy it tests | Keeps the evaluation seed rule (§3) intact and keeps the benchmark cells comparable across methods. |
