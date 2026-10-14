import mink
import mujoco
import numpy as np
import pytest

from envs.scene import build_scene
from motion.control import ControlLoop
from motion.ik import ARM_JOINT_NAMES, MinkIK
from skills import PickPlaceSkills, pick_place


def _make_skills() -> tuple[PickPlaceSkills, mujoco.MjModel, mujoco.MjData]:
    """Build an L0 scene with the Panda at home and the gripper open."""
    model = build_scene()
    data = mujoco.MjData(model)
    ik = MinkIK()
    for name, position in zip(ARM_JOINT_NAMES, ik.rest_q, strict=True):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = position
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = model.jnt_range[joint_id, 1]
    mujoco.mj_forward(model, data)
    control = ControlLoop(model, data, ik)
    return PickPlaceSkills(control), model, data


def test_reach_moves_tcp_and_rejects_unreachable_target() -> None:
    """Reach a valid L0 pose and reject a pose outside the action workspace."""
    skills, _, _ = _make_skills()

    result = skills.reach([0.50, -0.10, 0.25, 0.0])

    assert result.success, result
    assert result.reason == "success"
    assert result.info["position_error"] <= 0.005
    unreachable = skills.reach([0.10, 0.0, 0.20, 0.0])
    assert not unreachable.success
    assert unreachable.reason == "unreachable"


def test_grasp_then_place_cube_in_l0_target_zone() -> None:
    """Grasp a scene cube, lift it, and place it at the target-zone center."""
    skills, model, data = _make_skills()

    grasp_result = skills.grasp("cube_red")
    assert grasp_result.success, grasp_result
    assert grasp_result.info["lifted_height"] >= 0.08

    place_result = skills.place([0.49, 0.02, 0.02, 0.0])

    assert place_result.success, place_result
    cube_position = data.xpos[model.body("cube_red").id]
    assert np.linalg.norm(cube_position[:2] - [0.49, 0.02]) <= 0.03
    assert cube_position[2] == pytest.approx(0.02, abs=0.01)


def test_pick_and_place_composes_grasp_and_place() -> None:
    """Pick and place a different L0 cube through the composed skill."""
    skills, model, data = _make_skills()

    result = skills.pick_and_place("cube_red", [0.49, 0.02, 0.02, 0.0])

    assert result.success, result
    assert result.info["grasp"]["skill"] == "grasp"
    assert result.info["place"]["skill"] == "place"
    assert np.linalg.norm(data.xpos[model.body("cube_red").id, :2] - [0.49, 0.02]) <= 0.03


def test_stack_places_one_l0_cube_on_another() -> None:
    """Stack the red cube on the blue cube using their measured geometry."""
    skills, model, data = _make_skills()

    result = skills.stack("cube_red", "cube_blue")

    assert result.success, result
    red_position = data.xpos[model.body("cube_red").id]
    blue_position = data.xpos[model.body("cube_blue").id]
    assert np.linalg.norm(red_position[:2] - blue_position[:2]) <= 0.03
    assert red_position[2] == pytest.approx(blue_position[2] + 0.04, abs=0.01)


def test_reach_reports_forced_failure_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report unreachable, timeout, IK, and collision failures distinctly."""
    skills, _, _ = _make_skills()
    target = [0.50, -0.10, 0.25, 0.0]

    timed_out = skills.reach(target, timeout=1e-6)
    assert timed_out.reason == "timeout"

    def fail_ik(_: np.ndarray) -> None:
        raise mink.exceptions.NoSolutionFound("forced IK failure")

    with monkeypatch.context() as patch:
        patch.setattr(skills.control, "apply", fail_ik)
        ik_failed = skills.reach(target)
    assert ik_failed.reason == "ik_failed"

    with monkeypatch.context() as patch:
        patch.setattr(pick_place, "count_collisions", lambda *_: 1)
        collision = skills.reach(target)
    assert collision.reason == "collision"


def test_place_without_held_object_reports_grasp_missed() -> None:
    """Reject placement when the required held-object precondition is absent."""
    skills, _, _ = _make_skills()

    result = skills.place([0.49, 0.02, 0.02, 0.0])

    assert not result.success
    assert result.reason == "grasp_missed"


def test_grasp_reports_slipped_when_lift_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map a failed post-lift check to the documented slipped reason."""
    skills, _, _ = _make_skills()
    monkeypatch.setattr(pick_place, "is_lifted", lambda *_: False)

    result = skills.grasp("cube_red")

    assert not result.success
    assert result.reason == "slipped"