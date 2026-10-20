# Decision Log

The initial decisions below are fixed in [the roadmap](roadmap.md#3-fixed-design-decisions). Changes to them require updating the roadmap, this log, and [the project rationale](idea.md) together.

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
| Residual-RL action | Expert-relative deltas bounded to ±2 cm and ±10° | Limits the learned correction around the scripted expert. |
| Image observations | Front and wrist RGB images, 128 × 128 uint8 | Provides consistent camera keys and a compact image size. |
| State observation | 7 joint positions, gripper width, 3-D TCP position, TCP yaw | Defines a compact, shared proprioceptive state. |
| Units | SI units and radians | Avoids unit conversions between simulation, policies, and evaluation. |
| Frames | `world`, `base`, `ee`, `cam_front`, `cam_wrist` | Gives transforms and observations explicit frame names. |
| Quaternions | MuJoCo order w, x, y, z; SciPy calls use `scalar_first=True` | Prevents the libraries' different component ordering from being confused. |
| Rotation matrices | Reshape MuJoCo `xmat` row-major 9-vectors to 3 × 3 | Preserves MuJoCo's documented matrix layout. |
| Camera axes | Convert MuJoCo OpenGL camera axes to OpenCV axes for pinhole projection | Keeps camera projection formulas consistent with their expected axes. |
| Seed ranges | Train 0-99,999; validation 100,000-109,999; evaluation 1,000,000-1,009,999 | Keeps evaluation seeds separate from training and tuning. |
| Randomness | Spawn named independent generators from one `SeedSequence` | Makes random streams reproducible and separates purposes such as scene, noise, and policy. |
| Arm position-servo stiffness | Multiply the Panda arm actuators' position gain and position-feedback bias by 5; preserve actuator type and velocity damping. | Reduces gravity-induced TCP tracking error enough to meet the 2 mm held-target tolerance while retaining the Menagerie position-target interface. |
| Grasp contact settings | Use 50 g cubes and friction `[2.0, 0.005, 0.0001]` on cubes and fingertip pads; keep `impratio=10` and `solref=[0.02, 1.0]`. | Passed the Phase 1.3 grasp, 8 cm lift, and 3 s hold checks on 20 seeded cube poses with less than 5 mm of slip. |

## Throughput baseline

Measured on 2026-10-14 with `make benchmark-scene`: macOS 26.6.2, Apple M3 Max, 36 GB memory, Python 3.12.4, and MuJoCo 3.16.0. The benchmark timed 10,000 physics steps after 1,000 warm-up steps and 500 `Renderer` update/render frames per camera after 20 warm-up frames, at 128 × 128.

| Measurement | Throughput |
| --- | ---: |
| Physics | 33,916.4 steps/s |
| Front camera | 119.5 frames/s |
| Wrist camera | 108.7 frames/s |

These local throughput figures are a reference, not a performance gate; they vary with hardware, graphics drivers, and system load. Re-run `make benchmark-scene` to update them.

## Roadmap reordering: an RL expert before imitation learning (2026-10-16)

After Phase 1 (scripted expert) and Phase 2.2 (state and image BC), the scripted expert's measured success on T3 and T4 fell short of its Phase 1 targets, and BC trained on its demonstrations inherited that ceiling. Rather than keep hand-tuning the scripted expert's state machine, a new **Phase 1b** was inserted between Phase 1 and Phase 2a (not "Phase 1.5": that number was already taken by the existing "1.5 Scripted expert" subsection of Phase 1):

- **Pure PPO**, not residual PPO, trained from scratch on T2, T3, and T4, since it does not inherit the scripted expert's failure modes (a residual policy built on a weak base inherits that base's ceiling).
- The frozen, best-performing checkpoint per task becomes the demonstrator for Phase 2a (BC/ACT/Diffusion Policy) and Phase 2b (SmolVLA), replacing the scripted expert as the data source. The scripted expert stays in the code and the final benchmark as a baseline row, but is no longer used to generate new training data.
- The already-collected T2-L1 scripted-expert demos and BC checkpoints are kept as a historical baseline, not deleted or regenerated retroactively.
- Phase 3 (previously "Phase 3: Reinforcement learning") is now specifically about residual RL refining the Phase 1b RL expert under perturbed dynamics (T2-H). Its T2-H headroom calibration (3.1) and its `ResidualWrapper` base policy now both reference the Phase 1b RL expert instead of the scripted expert; expect to recalibrate T2-H's perturbation magnitudes, since the RL expert is stronger than the scripted expert was.
- `docs/roadmap.md` and `docs/idea.md` were updated together to reflect this order; see `docs/roadmap.md`'s Phase 1b section for the full build/evaluate/freeze steps.
- **Scope (2026-10-16):** Phase 1b was widened from T2/T3/T4 to every non-language task — T1, T2, T3, and T4 — so the RL expert matches or beats the scripted expert across the whole non-language suite. T3-L and T5 need language conditioning (naming an object from an instruction, multi-step combinations) and stay out of scope; they continue to use the scripted expert. T1 has no object and no grasp stage, so its observation and reward collapse to TCP-to-target distance only, and it has no downstream demo generation (Phase 2a never trains on T1) — its checkpoint exists solely to confirm the RL expert reaches baseline parity on every non-language task.

## Phase 1b reward design (`envs/rl_env.py`, 2026-10-16)

`PickPlaceRLEnv` wraps `PickPlaceEnv` with a flat 27-D observation and a staged dense reward, implementing roadmap 1b.1. Reward weights, not specified numerically in the roadmap:

| Constant | Value | Rationale |
| --- | --- | --- |
| `SUCCESS_BONUS` | 10.0 | Dominates the bounded per-step shaping below, so the value function's dominant signal is "did it actually succeed." |
| `GRASP_LIFT_BONUS` | 5.0 | A one-time bonus on the first step the object is grasped and lifted, not every step it stays held, so carrying it on to the destination is never less rewarding than holding it in place. |
| `ACTION_RATE_WEIGHT` | 0.01 | Penalizes the change between consecutive absolute pose targets (not their magnitude, since the action is an absolute workspace pose and penalizing magnitude would bias the policy toward the coordinate origin). |
| `TIME_STEP_PENALTY` | 0.01 | A flat per-step cost, added because potential-based shaping (below) is path-length invariant and does not by itself reward finishing sooner. |

**Distance shaping is potential-based (progress made this step), not absolute remaining distance.** The first implementation subtracted the raw remaining distance every step; that is paid once per step for as many steps as the episode lasts, so a slow or hovering policy accumulates a much larger total penalty than a fast one even when both eventually succeed, swamping the one-time `SUCCESS_BONUS`/`GRASP_LIFT_BONUS` over a 300-400 step episode. Rewarding `previous_distance - current_distance` instead telescopes to `initial_distance - final_distance` over any path (Ng, Harada & Russell 1999's potential-based shaping), so the total shaping reward per stage is bounded by the stage's starting distance regardless of path length or episode duration — tested in `tests/test_rl_env.py::test_shaping_reward_telescopes_regardless_of_path`. A stage switch (pre-grasp to post-grasp) resets the progress baseline rather than comparing two different potential functions. Because this shaping is now path-length invariant, `TIME_STEP_PENALTY` was added back in explicitly to keep a "finish sooner" incentive (the alternative, relying only on discounting pushing the terminal bonus earlier, works but is weaker and depends on `gamma`).