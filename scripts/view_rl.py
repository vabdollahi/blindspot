"""Run one trained Phase 1b RL checkpoint episode in the MuJoCo passive viewer.

With `--gif PATH`, the same episode is rendered headless from the viewer's
camera and saved as a looping GIF instead of opening a window.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import imageio.v3 as imageio
import mujoco
import mujoco.viewer
import numpy as np

from envs.pick_place_env import PickPlaceEnv
from envs.rl_env import RL_TASKS
from eval.policy import Policy
from motion.control import ENVIRONMENT_TIMESTEP
from policies.rl import load_rl_policy

CAMERA_LOOKAT = (0.45, 0.0, 0.20)
CAMERA_DISTANCE = 1.4
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -25.0
GIF_END_PAUSE_SECONDS = 1.0


def _parse_args() -> argparse.Namespace:
    """Parse checkpoint, episode, inference-device, and GIF options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/rl/t1-l1/seed-0"),
    )
    parser.add_argument("--task", choices=RL_TASKS, default="T1")
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L1")
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--show-tcp", action="store_true")
    parser.add_argument(
        "--gif",
        type=Path,
        help="Render the episode headless to this GIF instead of opening the viewer.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="GIF frame rate; frames are subsampled from the 20 Hz policy rate.",
    )
    parser.add_argument("--gif-width", type=int, default=480)
    parser.add_argument("--gif-height", type=int, default=360)
    return parser.parse_args()


def _configure_camera(camera: Any) -> None:
    """Point a free MuJoCo camera at the workspace from the viewer's default angle."""
    camera.lookat[:] = CAMERA_LOOKAT
    camera.distance = CAMERA_DISTANCE
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION


def _record_gif(
    policy: Policy,
    environment: PickPlaceEnv,
    observation: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Roll out one episode headless, save it as a looping GIF, and return the final info."""
    if not 1 <= arguments.gif_fps <= round(1.0 / ENVIRONMENT_TIMESTEP):
        raise SystemExit("--gif-fps must be between 1 and the 20 Hz policy rate")
    max_width = environment.model.vis.global_.offwidth
    max_height = environment.model.vis.global_.offheight
    if not (
        0 < arguments.gif_width <= max_width and 0 < arguments.gif_height <= max_height
    ):
        raise SystemExit(
            f"--gif-width/--gif-height must fit the {max_width}x{max_height} offscreen buffer"
        )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    _configure_camera(camera)
    scene_option = mujoco.MjvOption()
    if not arguments.show_tcp:
        scene_option.sitegroup[
            environment.model.site_group[environment.model.site("tcp").id]
        ] = 0
    stride = max(1, round(1.0 / (ENVIRONMENT_TIMESTEP * arguments.gif_fps)))
    renderer = mujoco.Renderer(
        environment.model, height=arguments.gif_height, width=arguments.gif_width
    )

    def render() -> np.ndarray:
        renderer.update_scene(
            environment.data, camera=camera, scene_option=scene_option
        )
        return renderer.render().copy()

    try:
        frames = [render()]
        steps = 0
        terminated = truncated = False
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            observation, _, terminated, truncated, info = environment.step(
                policy.act(observation)
            )
            steps += 1
            if steps % stride == 0 or terminated or truncated:
                frames.append(render())
    finally:
        renderer.close()

    frame_seconds = stride * ENVIRONMENT_TIMESTEP
    # Hold the final frame so the outcome is visible before the GIF loops.
    frames.extend([frames[-1]] * round(GIF_END_PAUSE_SECONDS / frame_seconds))
    arguments.gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(
        arguments.gif, np.stack(frames), duration=frame_seconds * 1000.0, loop=0
    )
    return {**info, "steps": steps}


def main() -> None:
    """Display one pure-PPO rollout in the viewer, or record it as a GIF."""
    arguments = _parse_args()
    checkpoint_dir = arguments.checkpoint_dir.expanduser().resolve()
    if not (checkpoint_dir / "model.zip").is_file():
        raise SystemExit(f"No model.zip found under {checkpoint_dir}")
    if arguments.seed < 0:
        raise SystemExit("--seed must be non-negative")

    policy = load_rl_policy(checkpoint_dir, device=arguments.device)
    environment = PickPlaceEnv(arguments.task, arguments.level, "state")
    observation, info = environment.reset(seed=arguments.seed)
    policy.reset(None)
    if arguments.gif is not None:
        try:
            info = _record_gif(policy, environment, observation, arguments)
        finally:
            environment.close()
        print(
            f"Saved {arguments.gif}: success={info['success']}, "
            f"failure={info['failure_reason']}, steps={info['steps']}, "
            f"sim_time={info['episode_time']:.2f}s"
        )
        return
    episode_steps = 0
    terminated = False
    truncated = False
    print(
        f"Running {arguments.task}-{arguments.level} seed {arguments.seed} "
        f"with pure-PPO checkpoint {checkpoint_dir}; close the viewer to stop."
    )

    try:
        with mujoco.viewer.launch_passive(
            environment.model,
            environment.data,
        ) as viewer:
            _configure_camera(viewer.cam)
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
