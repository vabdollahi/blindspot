"""Phase 1b.1 check: RL env throughput, and reward sanity on random rollouts.

Profiles one environment step per task with a vectorized `SubprocVecEnv`,
and renders a handful of random-policy rollouts to video so the staged
reward (pre-grasp distance, the one-time grasp/lift bonus, post-grasp
distance, and the success bonus) can be inspected by eye.
"""

import argparse
import platform
import time
from pathlib import Path

import imageio.v3 as imageio
import mujoco
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.rl_env import RL_TASKS, PickPlaceRLEnv, make_rl_env

PROFILE_WARMUP_STEPS = 10
DEFAULT_WORKERS = 8
DEFAULT_PROFILE_STEPS = 200
DEFAULT_ROLLOUTS = 5
DEFAULT_LEVEL = "L1"
DEFAULT_OUTPUT_DIR = Path("videos/rl_env_check")


def profile_vectorized_throughput(
    task: str, level: str, workers: int, steps: int
) -> float:
    """Measure aggregate environment steps per second across worker processes."""
    vec_env = SubprocVecEnv(
        [make_rl_env(task, level) for _ in range(workers)], start_method="spawn"
    )
    try:
        vec_env.reset()
        random_state = np.random.default_rng(0)
        warmup_actions = _random_actions(vec_env, random_state, PROFILE_WARMUP_STEPS)
        for actions in warmup_actions:
            vec_env.step(actions)

        timed_actions = _random_actions(vec_env, random_state, steps)
        started_at = time.perf_counter()
        for actions in timed_actions:
            vec_env.step(actions)
        elapsed = time.perf_counter() - started_at
    finally:
        vec_env.close()
    return (steps * workers) / elapsed


def _random_actions(
    vec_env: SubprocVecEnv, random_state: np.random.Generator, count: int
) -> list[np.ndarray]:
    """Sample one batch of random actions per step for every worker."""
    low = vec_env.action_space.low
    high = vec_env.action_space.high
    return [
        random_state.uniform(low, high, size=(vec_env.num_envs, *low.shape)).astype(
            np.float32
        )
        for _ in range(count)
    ]


def record_random_rollout(
    task: str, level: str, seed: int, output_path: Path
) -> dict[str, object]:
    """Roll out a random policy, save an mp4, and summarize the reward stages."""
    env = PickPlaceRLEnv(task, level, render_mode="rgb_array")
    random_state = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    stages_seen: set[str] = set()
    success = False
    try:
        env.reset(seed=seed)
        frames.append(env.render())
        terminated = truncated = False
        while not (terminated or truncated):
            action = random_state.uniform(
                env.action_space.low, env.action_space.high
            ).astype(np.float32)
            _, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            total_reward += reward
            stages_seen.add(str(info["rl_stage"]))
            success = bool(info["success"])
    finally:
        env.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, np.stack(frames), fps=20, codec="libx264")
    return {
        "steps": len(frames) - 1,
        "total_reward": total_reward,
        "stages_seen": sorted(stages_seen),
        "success": success,
    }


def main() -> None:
    """Run the Phase 1b.1 throughput and reward-sanity checks and print results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(RL_TASKS), choices=RL_TASKS)
    parser.add_argument("--level", default=DEFAULT_LEVEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--profile-steps", type=int, default=DEFAULT_PROFILE_STEPS)
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.workers <= 0 or args.profile_steps <= 0 or args.rollouts < 0:
        parser.error("--workers and --profile-steps must be positive")

    print(
        f"Platform: {platform.platform()} | Python {platform.python_version()} "
        f"| MuJoCo {mujoco.__version__}"
    )
    for task in args.tasks:
        steps_per_second = profile_vectorized_throughput(
            task, args.level, args.workers, args.profile_steps
        )
        print(
            f"{task}: {steps_per_second:,.1f} env-steps/s "
            f"across {args.workers} workers"
        )
        for rollout_index in range(args.rollouts):
            output_path = args.output_dir / f"{task}_{args.level}_{rollout_index}.mp4"
            summary = record_random_rollout(
                task, args.level, seed=rollout_index, output_path=output_path
            )
            print(f"  rollout {rollout_index}: {summary} -> {output_path}")


if __name__ == "__main__":
    main()
