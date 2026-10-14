"""Run the Phase 1.5 scripted expert in the MuJoCo passive viewer."""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from numpy.typing import NDArray

from envs.scene import build_scene
from motion.control import ENVIRONMENT_TIMESTEP, ControlLoop
from motion.ik import ARM_JOINT_NAMES, MinkIK
from skills import ScriptedExpert

_OBJECT_NAMES = ("cube_red", "cube_blue", "cube_green", "bowl")
_FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")


def _initialize_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik: MinkIK,
) -> None:
    """Set the robot arm and gripper to the Menagerie home posture."""
    for name, position in zip(ARM_JOINT_NAMES, ik.rest_q, strict=True):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = position
    for name in _FINGER_JOINT_NAMES:
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = model.jnt_range[joint_id, 1]
    mujoco.mj_forward(model, data)


def _observation(
    task: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    target_pose: NDArray[np.float64] | None,
    target_object: str | None,
    instruction: str | None,
) -> dict[str, object]:
    """Build the policy observation with proprioception and oracle object poses."""
    mujoco.mj_forward(model, data)
    arm_positions = np.asarray(
        [
            data.qpos[model.jnt_qposadr[model.joint(name).id]]
            for name in ARM_JOINT_NAMES
        ],
        dtype=np.float64,
    )
    gripper_width = float(
        sum(
            data.qpos[model.jnt_qposadr[model.joint(name).id]]
            for name in _FINGER_JOINT_NAMES
        )
    )
    tcp_site_id = model.site("tcp").id
    tcp_rotation = data.site_xmat[tcp_site_id].reshape(3, 3)
    tcp_yaw = float(np.arctan2(tcp_rotation[1, 0], tcp_rotation[0, 0]))
    state = np.array(
        [
            *arm_positions,
            gripper_width,
            *data.site_xpos[tcp_site_id],
            tcp_yaw,
        ],
        dtype=np.float64,
    )
    object_poses = {
        name: {
            "position": data.xpos[model.body(name).id].copy(),
            "quaternion_wxyz": data.xquat[model.body(name).id].copy(),
        }
        for name in _OBJECT_NAMES
    }
    observation: dict[str, object] = {
        "task": task,
        "observation.state": state,
        "object_poses": object_poses,
    }
    if target_pose is not None:
        observation["target_pose"] = target_pose
    if target_object is not None:
        observation["target_object"] = target_object
    if instruction is not None:
        observation["instruction"] = instruction
    return observation


def _parse_arguments() -> argparse.Namespace:
    """Parse task and target options for the expert viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("T1", "T2", "T3", "T3-L", "T4"),
        default="T2",
        help="Task for the scripted expert (default: T2).",
    )
    parser.add_argument(
        "--object",
        dest="target_object",
        help="Object name, such as cube_red; T4 always stacks red on blue.",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="T1 target point or T2 target-zone center.",
    )
    parser.add_argument(
        "--instruction",
        help="Instruction used by T3-L, for example 'Put the blue cube in the bowl'.",
    )
    parser.add_argument(
        "--show-tcp",
        action="store_true",
        help="Show the small TCP site marker between the gripper fingers.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one scripted-expert episode in the passive MuJoCo viewer."""
    arguments = _parse_arguments()
    model = build_scene()
    data = mujoco.MjData(model)
    ik = MinkIK()
    _initialize_home(model, data, ik)
    control = ControlLoop(model, data, ik)
    instruction = arguments.instruction
    if arguments.task == "T3-L" and instruction is None:
        instruction = "Put the red cube in the bowl."
    expert = ScriptedExpert(arguments.task, object_name=arguments.target_object)
    expert.reset(instruction)

    target_pose: NDArray[np.float64] | None = None
    if arguments.task == "T1":
        target = arguments.target or (0.50, 0.0, 0.35)
        target_pose = np.asarray(target, dtype=np.float64)
    elif arguments.task == "T2":
        target = arguments.target or (0.49, 0.02, 0.02)
        target_pose = np.asarray(target, dtype=np.float64)

    print(f"Running expert task {arguments.task}; close the viewer to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.20]
        viewer.cam.distance = 1.4
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        if not arguments.show_tcp:
            tcp_site_group = model.site_group[model.site("tcp").id]
            viewer.opt.sitegroup[tcp_site_group] = 0
        while viewer.is_running():
            tick_start = time.perf_counter()
            observation = _observation(
                arguments.task,
                model,
                data,
                target_pose=target_pose,
                target_object=arguments.target_object,
                instruction=instruction,
            )
            control.apply(expert.act(observation))
            viewer.sync()
            remaining = ENVIRONMENT_TIMESTEP - (time.perf_counter() - tick_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
