"""Open the generated pick-and-place scene in the MuJoCo viewer."""

import time

import mujoco
import mujoco.viewer

from envs.scene import build_scene


def main() -> None:
    """Display the initial scene without advancing its physics state."""
    model = build_scene()
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()