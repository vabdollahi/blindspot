"""Run one trained T3-L VLA episode in the MuJoCo passive viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from envs.pick_place_env import PickPlaceEnv
from motion.control import ENVIRONMENT_TIMESTEP
from policies.vla import load_vla_policy


def _parse_args() -> argparse.Namespace:
    """Parse checkpoint, level, test seed, and inference-device options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show-tcp", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Display one seeded T3-L VLA rollout and its terminal outcome."""
    arguments = _parse_args()
    checkpoint = arguments.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    if arguments.seed < 0:
        raise SystemExit("--seed must be non-negative")

    policy = load_vla_policy(checkpoint, device=arguments.device)
    environment = PickPlaceEnv(
        "T3-L",
        arguments.level,
        obs_mode="state",
        image_size=256,
    )
    observation, _ = environment.reset(seed=arguments.seed)
    instruction = observation.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        environment.close()
        raise ValueError("T3-L environment did not return its language instruction")
    policy.reset(instruction)
    episode_steps = 0
    terminated = False
    truncated = False
    print(
        f"Running T3-L-{arguments.level} seed {arguments.seed} "
        f"with {policy.training_profile} VLA; instruction={instruction!r}. "
        "Close the viewer to stop."
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
            observation.update(environment.observe_images())
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
                    else:
                        observation.update(environment.observe_images())
                viewer.sync()
                remaining = ENVIRONMENT_TIMESTEP - (time.perf_counter() - tick_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
