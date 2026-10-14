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

## Throughput baseline

Measured on 2026-10-14 with `make benchmark-scene`: macOS 26.6.2, Apple M3 Max, 36 GB memory, Python 3.12.4, and MuJoCo 3.16.0. The benchmark timed 10,000 physics steps after 1,000 warm-up steps and 500 `Renderer` update/render frames per camera after 20 warm-up frames, at 128 × 128.

| Measurement | Throughput |
| --- | ---: |
| Physics | 33,916.4 steps/s |
| Front camera | 119.5 frames/s |
| Wrist camera | 108.7 frames/s |

These local throughput figures are a reference, not a performance gate; they vary with hardware, graphics drivers, and system load. Re-run `make benchmark-scene` to update them.