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
| Image observations | Front and wrist RGB images, uint8, 256 × 256 for every dataset from Phase 2 on (the existing BC baselines use 128 × 128) | Matches the input resolution of SmolVLA and the pretrained encoders, and keeps the 4 cm cubes several pixels wide in the front view. |
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
- **Policy:** freeze the VLM, standardize its image-token and instruction features, combine them with normalized 12-D proprioception, and train a 0.38 M-parameter head: a shared 960→8 linear projection of every image token and a 960→32 projection of the instruction feature, followed by a 2-layer, 256-unit MLP. The head predicts the change from the current TCP pose (normalized by training-split statistics) plus a binary gripper logit; the `Policy` adds the change to the TCP pose and emits the absolute 5-D action. As with the RL expert, a prediction error becomes a drift that the closed loop corrects rather than a fixed offset. SmolVLA remains the primary fine-tuned VLA and uses its own action-chunk interface.
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
