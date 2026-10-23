# Copilot Instructions — Language-Driven Pick-and-Place Agent

> **Audience: AI agents.** Terse rules; every line carries signal.

## General

- No filler. Start with the answer; match length to complexity.
- Never fill knowledge gaps with plausible-sounding information. `mujoco`, `mink`, `lerobot`, and model-serving APIs change between versions: verify against the pinned version and flag uncertainty.
- Ask before changing anything in `docs/roadmap.md` §3 (fixed design decisions), §4 (tasks and success predicates), or §5 (evaluation rules). Offer options with trade-offs.
- Do not add scope beyond `docs/roadmap.md`. Note worthwhile follow-ups at the end of a response instead.

## Project

- Simulated Franka Panda (MuJoCo) as a testbed for robot foundation models. Layers: environment + scripted expert → RL expert (PPO) → data engine → VLA (SmolVLA) → representations (DINOv2, V-JEPA 2, in-domain JEPA) → latent world models (uncertainty, planning, policy evaluation) → distillation and efficient inference → compact LLM agent.
- Sources of truth: `docs/roadmap.md` (build steps, decisions, tasks, evaluation rules), `docs/idea.md` (design rationale), `docs/decisions.md` (design rationale), `docs/conventions.md` (frames and units).
- Out of scope: custom kinematics, dynamics, or controllers. Motion is `mink` IK plus MuJoCo position servos.

## Stack

- Python 3.12 with `uv` only (never `pip`); `uv.lock` is committed.
- `mujoco` 3.x, MuJoCo Menagerie as a pinned submodule in `third_party/`, `mink`, `gymnasium`, `stable-baselines3`, PyTorch (MPS), `lerobot` (pinned), `transformers` (pinned, pretrained encoders), `onnxruntime`, `pydantic`, `openai` client for OpenAI-compatible model servers.
- Model servers (Ollama, LM Studio, `mlx-lm`/`mlx-vlm`) run in a separate environment and are reached only over HTTP. Never import them into the simulation/training environment.

## Conventions (must hold everywhere)

- SI units and radians.
- Quaternions are w, x, y, z (MuJoCo). SciPy `Rotation` needs `scalar_first=True`.
- MuJoCo `xmat` arrays are row-major 9-vectors: `reshape(3, 3)`.
- MuJoCo cameras look along −z with +y up (OpenGL). Convert to OpenCV axes before pinhole math.
- Look up joints, actuators, sites, and cameras by name; never hard-code indices.
- Every policy (scripted, learned, VLA) uses the 5-D action `[x, y, z, yaw, g]` and the `Policy` interface (`reset`, `act`).
- All randomness flows through the single seeding utility. Train, validation, and evaluation seed ranges are disjoint; evaluation seeds are never used for training or tuning.
- Create MuJoCo models and renderers inside worker processes, never before forking.

## Code

- Type hints and a one-line docstring on every public function.
- Pydantic models for anything crossing a boundary: configs, LLM plan schemas, results JSON, `run.json`. NumPy arrays for numerical data.
- No `assert` for runtime checks; raise an exception. No `# type: ignore` or `# noqa`; fix the cause.
- Imports at module top level. Remove dead and commented-out code.
- Format and lint with `ruff` before committing.

## Tests

- `pytest`. Frame, projection, and convention code is tested with known answers or round trips.
- Rendering tests carry `@pytest.mark.rendering` and are skipped in CI.
- Same seed must reproduce the same episode; keep a determinism test for every environment change.

## Results and reproducibility

- Every training run writes `run.json`: config, git commit, seeds, dataset ID and version, hardware.
- Every table and figure is generated from `eval/results/*.json` by `eval/report.py`. Never hand-edit reported numbers.
- Success predicates are frozen. Changing one invalidates affected results; re-run them.

## Secrets

- API keys (frontier models, Hugging Face) come from environment variables or a gitignored `.env`; commit `.env.example` only.
- Never write secrets to code, configs, logs, run records, or results JSON.

## LLM/VLM agent

- Planner output is validated against the Pydantic `Plan` schema; reject unknown object IDs and retry with the validation error.
- Instructions and other external text go in user messages, clearly delimited; never interpolate them into system prompts.

## Maintenance

- When a decision, task definition, or evaluation rule changes, update `docs/roadmap.md`, `docs/idea.md`, and `docs/decisions.md` together.
- Skills in `.github/skills/` are living documents. Record non-obvious, recurring gotchas (for example grasp tuning or LeRobot/SmolVLA integration) in the relevant skill; create a new skill only for substantial, repeatable procedures.
