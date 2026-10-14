"""Measure MuJoCo simulation and scene-camera rendering throughput."""

import argparse
import platform
import time

import mujoco

from envs.scene import build_scene

IMAGE_SIZE = 128
PHYSICS_WARMUP_STEPS = 1_000
RENDER_WARMUP_FRAMES = 20
CAMERAS = ("front", "wrist")


def measure_physics_steps_per_second(
    model: mujoco.MjModel, data: mujoco.MjData, step_count: int
) -> float:
    """Measure MuJoCo physics steps per second after a short warm-up."""
    for _ in range(PHYSICS_WARMUP_STEPS):
        mujoco.mj_step(model, data)

    started_at = time.perf_counter()
    for _ in range(step_count):
        mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - started_at
    return step_count / elapsed


def measure_camera_fps(
    model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, frame_count: int
) -> float:
    """Measure update-and-render frames per second for one camera."""
    with mujoco.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE) as renderer:
        for _ in range(RENDER_WARMUP_FRAMES):
            renderer.update_scene(data, camera=camera_name)
            renderer.render()

        started_at = time.perf_counter()
        for _ in range(frame_count):
            renderer.update_scene(data, camera=camera_name)
            image = renderer.render()
            if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise RuntimeError(f"Unexpected {camera_name} image shape: {image.shape}")
        elapsed = time.perf_counter() - started_at
    return frame_count / elapsed


def main() -> None:
    """Run the local scene throughput benchmark and print its results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--frames", type=int, default=500)
    args = parser.parse_args()
    if args.steps <= 0 or args.frames <= 0:
        parser.error("--steps and --frames must be positive")

    model = build_scene()
    physics_data = mujoco.MjData(model)
    physics_steps_per_second = measure_physics_steps_per_second(
        model, physics_data, args.steps
    )

    render_data = mujoco.MjData(model)
    mujoco.mj_forward(model, render_data)
    camera_fps = {
        camera: measure_camera_fps(model, render_data, camera, args.frames)
        for camera in CAMERAS
    }

    print(
        f"Platform: {platform.platform()} | Python {platform.python_version()} "
        f"| MuJoCo {mujoco.__version__}"
    )
    print(
        f"Physics: {physics_steps_per_second:,.1f} steps/s "
        f"({args.steps:,} timed steps after {PHYSICS_WARMUP_STEPS:,} warm-up steps)"
    )
    for camera, frames_per_second in camera_fps.items():
        print(
            f"{camera}: {frames_per_second:,.1f} frames/s "
            f"({IMAGE_SIZE}x{IMAGE_SIZE}, {args.frames:,} timed frames, "
            f"{RENDER_WARMUP_FRAMES} warm-up frames)"
        )


if __name__ == "__main__":
    main()