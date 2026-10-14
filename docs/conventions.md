# Conventions

These conventions apply throughout the project. They mirror the fixed design decisions in [the roadmap](roadmap.md#3-fixed-design-decisions).

| Topic | Convention |
| --- | --- |
| Robot | Franka Emika Panda from MuJoCo Menagerie (`franka_emika_panda`), arm plus hand, 7 DOF |
| Robot base | At the world origin, on the table surface (table top at z = 0), so `base` = `world` |
| Physics | `implicitfast` integrator, dt = 2 ms (500 Hz) |
| Joint control | Menagerie's built-in position servos; `data.ctrl` holds joint position targets |
| IK and interpolation | `mink` differential IK on a robot-only model, run at 100 Hz (every 5 physics steps); linear waypoint interpolation |
| Policy rate | 20 Hz (one environment step = 25 physics steps) |
| Action (all learned policies) | 5-D `[x, y, z, yaw, g]`: absolute TCP target in the base frame, gripper pointing straight down, `g` in [0, 1] (≥ 0.5 means open) |
| Action limits | xyz clipped to a workspace box in front of the base (about 45 × 60 × 40 cm above the table); yaw clipped to [−π/2, π/2] |
| Residual-RL action | Bounded delta on the expert's xyz and yaw: ±2 cm, ±10° |
| Image observations | `observation.images.front` and `observation.images.wrist`, uint8 RGB, 128 × 128 |
| State observation | `observation.state` = joint positions (7), gripper width (1), TCP position (3), TCP yaw (1) |
| Units | SI and radians everywhere |
| Frames | `world`, `base`, `ee` (TCP site), `cam_front`, `cam_wrist` |
| Quaternions | w, x, y, z (MuJoCo order). SciPy's `Rotation` defaults to x, y, z, w: always pass `scalar_first=True` or convert in one helper. |
| Rotation matrices | MuJoCo `xmat` arrays are row-major 9-vectors: `reshape(3, 3)` |
| Camera axes | MuJoCo cameras look along local −z with +y up (OpenGL). Convert to OpenCV axes (x right, y down, z forward) before using pinhole formulas. |
| Seed ranges | Train 0-99,999; validation 100,000-109,999; evaluation 1,000,000-1,009,999. Evaluation seeds are never used for training or tuning. |
| Randomness | One seeding utility. `np.random.SeedSequence(seed).spawn(...)` gives an independent stream for each purpose (scene, noise, policy). |