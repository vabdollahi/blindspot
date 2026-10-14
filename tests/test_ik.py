import mink
import mujoco
import numpy as np

from common.seeding import named_generators
from envs.scene import build_scene
from motion.ik import ARM_JOINT_NAMES, MinkIK


def _top_down_pose(position: np.ndarray, yaw: float) -> mink.SE3:
    """Build a top-down TCP pose with a world-frame yaw rotation."""
    rotation = mink.SO3.from_z_radians(yaw) @ mink.SO3.from_x_radians(np.pi)
    return mink.SE3.from_rotation_and_translation(rotation, position)


def _arm_limits(model: mujoco.MjModel) -> np.ndarray:
    """Return position limits for the named Panda arm joints."""
    return np.asarray([model.joint(name).range for name in ARM_JOINT_NAMES])


def test_ik_model_contains_only_the_panda_and_matches_scene_tcp() -> None:
    """Ensure the IK model excludes scene objects and uses the same TCP site."""
    ik = MinkIK()
    scene = build_scene()

    assert ik.model.njnt == 9
    assert not np.any(ik.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    assert ik.model.site("tcp").id >= 0
    np.testing.assert_array_equal(
        ik.model.site_pos[ik.model.site("tcp").id],
        scene.site_pos[scene.site("tcp").id],
    )


def test_step_accepts_full_robot_qpos_and_respects_arm_limits() -> None:
    """Check a single IK step accepts all Panda qpos and returns arm targets."""
    ik = MinkIK(velocity_limits={name: 2.0 for name in ARM_JOINT_NAMES})
    configuration = mink.Configuration(ik.model)
    configuration.update_from_keyframe("home")
    target = configuration.get_transform_frame_to_world("tcp", "site")
    full_qpos = configuration.q.copy()

    arm_q = ik.step(target, full_qpos)

    assert arm_q.shape == (7,)
    limits = _arm_limits(ik.model)
    assert np.all(arm_q >= limits[:, 0])
    assert np.all(arm_q <= limits[:, 1])


def test_solve_reaches_at_least_99_percent_of_workspace_targets() -> None:
    """Solve 200 seeded top-down workspace targets within pose and joint limits."""
    ik = MinkIK()
    random = named_generators(42, ("ik_targets",))["ik_targets"]
    lower = np.array([0.25, -0.30, 0.05])
    upper = np.array([0.70, 0.30, 0.45])
    limits = _arm_limits(ik.model)
    successes = 0

    for position, yaw in zip(
        random.uniform(lower, upper, size=(200, 3)),
        random.uniform(-np.pi / 2, np.pi / 2, size=200),
        strict=True,
    ):
        arm_q, success = ik.solve(_top_down_pose(position, float(yaw)), ik.rest_q)
        successes += success
        assert np.all(arm_q >= limits[:, 0])
        assert np.all(arm_q <= limits[:, 1])

    assert successes >= 198