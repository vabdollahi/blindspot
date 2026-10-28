"""Replay a recorded VLA episode and save a normal-speed MuJoCo GIF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v3 as imageio
import mujoco
import numpy as np

from data.vla_subset import active_cube_colors_from_positions
from envs.pick_place_env import PickPlaceEnv
from eval.vla_trajectory import VLAEpisodeTrajectory
from motion.control import ENVIRONMENT_TIMESTEP

CAMERA_LOOKAT = (0.45, 0.0, 0.20)
CAMERA_DISTANCE = 1.4
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -25.0
GIF_END_PAUSE_SECONDS = 1.0
REPLAY_TOLERANCE = 1e-4


def _parse_args() -> argparse.Namespace:
    """Parse a saved trajectory, GIF output path, and render options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--gif", type=Path, required=True)
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=20,
        help="GIF frame rate; the simulator policy rate is 20 Hz.",
    )
    parser.add_argument("--gif-width", type=int, default=480)
    parser.add_argument("--gif-height", type=int, default=360)
    parser.add_argument("--show-tcp", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Replay stored actions without loading or running the VLA."""
    arguments = _parse_args()
    trajectory_path = arguments.trajectory.expanduser().resolve()
    if not trajectory_path.is_file():
        raise SystemExit(f"Trajectory does not exist: {trajectory_path}")
    trajectory = VLAEpisodeTrajectory.model_validate_json(
        trajectory_path.read_text(encoding="utf-8")
    )
    if not trajectory.steps:
        raise ValueError("Trajectory contains no replayable actions")
    if trajectory.result.episode_steps != len(trajectory.steps):
        raise ValueError("Trajectory step count does not match its episode result")
    if not 1 <= arguments.gif_fps <= round(1.0 / ENVIRONMENT_TIMESTEP):
        raise SystemExit("--gif-fps must be between 1 and the 20 Hz policy rate")

    environment = PickPlaceEnv(
        trajectory.task,
        trajectory.level,
        obs_mode="state",
        image_size=256,
    )
    try:
        _render_trajectory(environment, trajectory, arguments)
    finally:
        environment.close()
    print(
        f"Saved {arguments.gif}: success={trajectory.result.success}, "
        f"failure={trajectory.result.failure_reason}, "
        f"steps={trajectory.result.episode_steps}, "
        f"sim_time={trajectory.result.episode_time:.2f}s"
    )


def _render_trajectory(
    environment: PickPlaceEnv,
    trajectory: VLAEpisodeTrajectory,
    arguments: argparse.Namespace,
) -> None:
    """Replay saved actions, verify recorded states, and write a GIF."""
    max_width = environment.model.vis.global_.offwidth
    max_height = environment.model.vis.global_.offheight
    if not (
        0 < arguments.gif_width <= max_width and 0 < arguments.gif_height <= max_height
    ):
        raise SystemExit(
            f"--gif-width/--gif-height must fit the "
            f"{max_width}x{max_height} offscreen buffer"
        )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    _configure_camera(camera)
    scene_option = mujoco.MjvOption()
    if not arguments.show_tcp:
        tcp_site_group = environment.model.site_group[environment.model.site("tcp").id]
        scene_option.sitegroup[tcp_site_group] = 0
    stride = max(1, round(1.0 / (ENVIRONMENT_TIMESTEP * arguments.gif_fps)))
    renderer = mujoco.Renderer(
        environment.model,
        height=arguments.gif_height,
        width=arguments.gif_width,
    )

    def render() -> np.ndarray:
        renderer.update_scene(
            environment.data,
            camera=camera,
            scene_option=scene_option,
        )
        return renderer.render().copy()

    try:
        observation, _ = environment.reset(
            seed=trajectory.seed,
            options={"instruction_template": trajectory.instruction_template},
        )
        if observation.get("instruction") != trajectory.instruction:
            raise RuntimeError("Replay instruction differs from the recorded episode")
        actual_colors = active_cube_colors_from_positions(
            observation.get("object_poses", {})
        )
        if actual_colors != trajectory.active_cube_colors:
            raise RuntimeError("Replay scene does not match the recorded cube layout")
        _verify_state(
            observation,
            trajectory.steps[0].state,
            trajectory.steps[0].object_positions_before,
            step=0,
            phase="before action",
        )
        frames = [render()]
        for index, step in enumerate(trajectory.steps):
            if step.index != index:
                raise ValueError("Trajectory step indices must be consecutive")
            _verify_state(
                observation,
                step.state,
                step.object_positions_before,
                step=index,
                phase="before action",
            )
            observation, _, terminated, truncated, info = environment.step(
                np.asarray(step.action, dtype=np.float32)
            )
            _verify_state(
                observation,
                step.next_state,
                step.object_positions_after,
                step=index,
                phase="after action",
            )
            is_last = index + 1 == len(trajectory.steps)
            if (index + 1) % stride == 0 or is_last:
                frames.append(render())
            if terminated or truncated:
                if not is_last:
                    raise RuntimeError(
                        "Replay terminated before the recorded trajectory ended"
                    )
                if bool(info.get("success", False)) != trajectory.result.success:
                    raise RuntimeError(
                        "Replay outcome differs from the recorded result"
                    )
    finally:
        renderer.close()

    frame_seconds = stride * ENVIRONMENT_TIMESTEP
    frames.extend([frames[-1]] * round(GIF_END_PAUSE_SECONDS / frame_seconds))
    gif_path = arguments.gif.expanduser().resolve()
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(
        gif_path,
        np.stack(frames),
        duration=frame_seconds * 1000.0,
        loop=0,
    )


def _configure_camera(camera: Any) -> None:
    """Point the free camera at the manipulation workspace."""
    camera.lookat[:] = CAMERA_LOOKAT
    camera.distance = CAMERA_DISTANCE
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION


def _verify_state(
    observation: dict[str, Any],
    expected_state: list[float],
    expected_positions: dict[str, list[float]],
    *,
    step: int,
    phase: str,
) -> None:
    """Reject replay if robot or object state diverges from the recording."""
    actual_state = np.asarray(observation.get("observation.state"), dtype=np.float64)
    if actual_state.shape != (12,) or not np.allclose(
        actual_state,
        expected_state,
        rtol=0.0,
        atol=REPLAY_TOLERANCE,
    ):
        raise RuntimeError(f"Replay robot state diverged at step {step} ({phase})")
    poses = observation.get("object_poses")
    if not isinstance(poses, dict):
        raise TypeError("Replay observation does not contain object poses")
    for name, expected_position in expected_positions.items():
        if name not in poses:
            raise KeyError(f"Replay is missing object {name!r}")
        pose = poses[name]
        if not isinstance(pose, dict):
            raise TypeError(f"Replay object {name!r} has an invalid pose")
        actual_position = np.asarray(pose.get("position"), dtype=np.float64)
        if actual_position.shape != (3,) or not np.allclose(
            actual_position,
            expected_position,
            rtol=0.0,
            atol=REPLAY_TOLERANCE,
        ):
            raise RuntimeError(
                f"Replay object {name!r} diverged at step {step} ({phase})"
            )


if __name__ == "__main__":
    main()
