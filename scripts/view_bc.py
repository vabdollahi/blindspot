"""Run one trained BC episode in the MuJoCo passive viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from envs.pick_place_env import PickPlaceEnv
from motion.control import ENVIRONMENT_TIMESTEP
from policies.bc import ImageBCPolicy, load_bc_policy


def _parse_args() -> argparse.Namespace:
    """Parse checkpoint, task, seed, and inference-device options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/bc/t2-l1/state/checkpoint.pt"),
    )
    parser.add_argument("--task", choices=("T2", "T3"), default="T2")
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show-tcp", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Display one BC rollout while advancing the same Gymnasium environment."""
    arguments = _parse_args()
    checkpoint = arguments.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    if arguments.seed < 0:
        raise SystemExit("--seed must be non-negative")

    policy = load_bc_policy(checkpoint, device=arguments.device)
    obs_mode = "image" if isinstance(policy, ImageBCPolicy) else "state"
    environment = PickPlaceEnv(arguments.task, arguments.level, obs_mode)
    observation, info = environment.reset(seed=arguments.seed)
    instruction = observation.get("instruction")
    policy.reset(instruction if isinstance(instruction, str) else None)
    episode_steps = 0
    terminated = False
    truncated = False
    print(
        f"Running {arguments.task}-{arguments.level} seed {arguments.seed} "
        f"with {obs_mode} BC; close the viewer to stop."
    )

    try:
        with mujoco.viewer.launch_passive(
            environment.model,
            environment.data,
        ) as viewer:
            viewer.cam.lookat[:] = [0.45, 0.0, 0.20]
            viewer.cam.distance = 1.4
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -25.0
            if not arguments.show_tcp:
                tcp_site_group = environment.model.site_group[
                    environment.model.site("tcp").id
                ]
                viewer.opt.sitegroup[tcp_site_group] = 0
            while viewer.is_running():
                tick_start = time.perf_counter()
                if not terminated and not truncated:
                    observation, _, terminated, truncated, info = environment.step(
                        policy.act(observation)
                    )
                    episode_steps += 1
                    if terminated or truncated:
                        print(
                            f"Episode finished: success={info['success']}, "
                            f"failure={info['failure_reason']}, "
                            f"steps={episode_steps}, "
                            f"sim_time={info['episode_time']:.2f}s. "
                            "Close the viewer to exit."
                        )
                viewer.sync()
                remaining = ENVIRONMENT_TIMESTEP - (time.perf_counter() - tick_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
