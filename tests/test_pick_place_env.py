import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.pick_place_env import SUCCESS_HOLD_STEPS, PickPlaceEnv


def _assert_tree_equal(first: object, second: object) -> None:
    """Compare nested observation mappings and their arrays exactly."""
    if isinstance(first, dict):
        assert isinstance(second, dict)
        assert first.keys() == second.keys()
        for key in first:
            _assert_tree_equal(first[key], second[key])
    elif isinstance(first, np.ndarray):
        assert isinstance(second, np.ndarray)
        np.testing.assert_array_equal(first, second)
    else:
        assert first == second


def test_environment_passes_gymnasium_checker() -> None:
    """Satisfy Gymnasium's reset, step, space, and determinism checks."""
    environment = PickPlaceEnv("T2", "L0", "state")
    check_env(environment, skip_render_check=True)
    environment.close()


def test_same_seed_reproduces_scene_and_action_trajectory() -> None:
    """Reproduce the L1 initial observation and subsequent states bit for bit."""
    first = PickPlaceEnv("T2", "L1", "state")
    second = PickPlaceEnv("T2", "L1", "state")
    first_observation, _ = first.reset(seed=2026)
    second_observation, _ = second.reset(seed=2026)
    _assert_tree_equal(first_observation, second_observation)

    action = np.array([0.50, 0.0, 0.35, 0.0, 1.0], dtype=np.float32)
    for _ in range(8):
        first_step = first.step(action)
        second_step = second.step(action)
        _assert_tree_equal(first_step[0], second_step[0])
        assert first_step[1:] == second_step[1:]
    first.close()
    second.close()


def test_different_l1_seeds_sample_different_scene_poses() -> None:
    """Vary the L1 target and cube poses across independent scene seeds."""
    environment = PickPlaceEnv("T2", "L1", "state")
    first, _ = environment.reset(seed=10)
    second, _ = environment.reset(seed=11)

    assert not np.array_equal(first["target_pose"], second["target_pose"])
    assert not np.array_equal(
        first["object_poses"]["cube_red"]["position"],
        second["object_poses"]["cube_red"]["position"],
    )
    environment.close()


def test_l2_has_seeded_visual_physical_randomization_and_distractors() -> None:
    """Create one to three distractors and reproduce L2 model randomization."""
    first = PickPlaceEnv("T3-L", "L2", "state")
    second = PickPlaceEnv("T3-L", "L2", "state")
    first_observation, _ = first.reset(seed=77)
    second_observation, _ = second.reset(seed=77)

    assert 1 <= first._distractor_count <= 3
    assert first._distractor_count == second._distractor_count
    _assert_tree_equal(first_observation, second_observation)
    np.testing.assert_array_equal(first.model.geom_rgba, second.model.geom_rgba)
    np.testing.assert_array_equal(first.model.cam_pos, second.model.cam_pos)
    np.testing.assert_array_equal(first.model.body_mass, second.model.body_mass)
    np.testing.assert_array_equal(first.model.tex_data, second.model.tex_data)
    assert not np.array_equal(first.model.tex_data, first._base_texture_data)
    bowl_id = first.model.body("bowl").id
    bowl_mass_scale = first.model.body_mass[bowl_id] / first._base_body_mass[bowl_id]
    assert 0.8 <= bowl_mass_scale <= 1.2
    bowl_geom_id = first.model.geom("bowl_base").id
    bowl_friction_scale = (
        first.model.geom_friction[bowl_geom_id, 0]
        / first._base_geom_friction[bowl_geom_id, 0]
    )
    assert 0.8 <= bowl_friction_scale <= 1.2
    bowl_texture_id = mujoco.mj_name2id(
        first.model,
        mujoco.mjtObj.mjOBJ_TEXTURE,
        "bowl_texture",
    )
    bowl_texture_start = int(first.model.tex_adr[bowl_texture_id])
    bowl_texture_size = (
        int(first.model.tex_width[bowl_texture_id])
        * int(first.model.tex_height[bowl_texture_id])
        * int(first.model.tex_nchannel[bowl_texture_id])
    )
    assert not np.array_equal(
        first.model.tex_data[
            bowl_texture_start : bowl_texture_start + bowl_texture_size
        ],
        first._base_texture_data[
            bowl_texture_start : bowl_texture_start + bowl_texture_size
        ],
    )
    first.close()
    second.close()


@pytest.mark.parametrize(
    ("task", "instruction"),
    [
        ("T3-L", "Put the red cube in the bowl."),
        (
            "T5",
            "Put the red cube in the target zone, then put the blue cube in the bowl.",
        ),
    ],
)
def test_language_tasks_include_instruction_observation(
    task: str,
    instruction: str,
) -> None:
    """Expose a generated instruction for T3-L and the composed T5 scene."""
    environment = PickPlaceEnv(task, "L0", "state")
    observation, _ = environment.reset(seed=42)

    if task == "T3-L":
        assert observation["target_object"] == "cube_red"
        assert set(environment._active_cube_names) == {"cube_red", "cube_blue"}
        color = observation["target_object"].removeprefix("cube_")
        assert color in observation["instruction"].lower()
        assert observation["instruction"] == instruction
    else:
        assert observation["instruction"] == instruction
    assert environment.observation_space.contains(observation)
    environment.close()


def test_l1_t3l_instruction_matches_seeded_named_cube() -> None:
    """Match every generated L1 T3-L instruction to its selected cube color."""
    environment = PickPlaceEnv("T3-L", "L1", "state")
    selected_colors = set()

    for seed in range(16):
        observation, _ = environment.reset(seed=seed)
        target_color = observation["target_object"].removeprefix("cube_")
        selected_colors.add(target_color)
        assert target_color in observation["instruction"].lower()
        assert target_color in {"red", "blue", "green"}

    assert len(selected_colors) > 1
    environment.close()


def test_t3l_snapshots_the_sampled_non_target_cubes() -> None:
    """Record actual initial positions so T3-L can enforce its 2 cm predicate."""
    environment = PickPlaceEnv("T3-L", "L1", "state")
    environment.reset(seed=19)

    assert 2 <= len(environment._active_cube_names) <= 3
    for name in environment._active_cube_names:
        np.testing.assert_array_equal(
            environment._initial_cube_positions[name],
            environment.data.xpos[environment.model.body(name).id],
        )
    environment.close()


@pytest.mark.parametrize(
    ("task", "visible_count"),
    [("T2", 1), ("T3", 1), ("T3-L", 2), ("T4", 2)],
)
def test_task_scene_only_exposes_its_active_cubes(
    task: str, visible_count: int
) -> None:
    """Hide unused cube placeholders and disable their contact geoms."""
    environment = PickPlaceEnv(task, "L0", "state")
    environment.reset(seed=42)
    visible_names = []
    for color in ("red", "blue", "green"):
        name = f"cube_{color}"
        geom_id = environment.model.geom(f"{name}_geom").id
        if environment.model.geom_rgba[geom_id, 3] > 0:
            visible_names.append(name)
            assert environment.model.geom_contype[geom_id] != 0
            assert environment.model.geom_conaffinity[geom_id] != 0
    assert set(visible_names) == set(environment._active_cube_names)
    assert len(visible_names) == visible_count
    environment.close()


def test_t5_advances_to_its_second_configured_goal() -> None:
    """Advance from the configured T2 red-cube goal to the T3 blue-cube goal."""
    environment = PickPlaceEnv("T5", "L0", "state")
    _, _ = environment.reset(seed=42)
    red_body_id = environment.model.body("cube_red").id
    environment._set_free_body("cube_red", np.array([0.49, 0.02, 0.02]), 0.0)
    mujoco.mj_forward(environment.model, environment.data)
    environment.control.reset()
    tcp_position = environment.data.site_xpos[environment._tcp_site_id].copy()
    action = np.array([*tcp_position, 0.0, 1.0], dtype=np.float32)

    for _ in range(SUCCESS_HOLD_STEPS):
        _, _, terminated, _, _ = environment.step(action)

    assert not terminated
    assert environment._t5_goal_index == 1
    assert environment._current_target_object() == "cube_blue"
    assert environment._current_target_pose()[0] == pytest.approx(
        environment.data.xpos[environment.model.body("bowl").id, 0]
    )
    assert environment.data.xpos[red_body_id, 2] == pytest.approx(0.02, abs=0.002)
    environment.close()


def test_success_requires_one_second_of_continuous_goal_state() -> None:
    """Hold a T1 target for 20 consecutive policy steps before termination."""
    environment = PickPlaceEnv("T1", "L0", "state")
    environment.reset(seed=42)
    target = np.array([0.50, 0.0, 0.35, 0.0, 1.0], dtype=np.float32)

    terminated = False
    for _ in range(100):
        _, _, terminated, truncated, _ = environment.step(target)
        assert not truncated
        if terminated:
            break

    assert terminated
    assert environment._success_hold_steps >= SUCCESS_HOLD_STEPS
    environment.close()


def test_t4_time_limit_truncates_after_exactly_four_hundred_policy_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the 20-second time limit by 20 Hz steps, not floating-point time."""
    environment = PickPlaceEnv("T4", "L0", "state")
    environment.reset(seed=42)

    def advance_one_policy_step(_: np.ndarray) -> None:
        mujoco.mj_step(environment.model, environment.data, nstep=25)

    monkeypatch.setattr(environment.control, "apply", advance_one_policy_step)
    action = np.array([0.5, 0.0, 0.3, 0.0, 1.0], dtype=np.float32)

    for _ in range(399):
        _, _, terminated, truncated, _ = environment.step(action)
        assert not terminated
        assert not truncated

    _, _, terminated, truncated, info = environment.step(action)

    assert not terminated
    assert truncated
    assert environment._episode_steps == 400
    assert info["episode_time"] <= environment._task_config.time_limit + 1e-9
    environment.close()


@pytest.mark.rendering
@pytest.mark.parametrize("level", ["L0", "L2"])
def test_image_observation_contains_both_128px_cameras(level: str) -> None:
    """Return the front and wrist camera images with uint8 RGB pixels."""
    environment = PickPlaceEnv("T2", level, "image")
    observation, _ = environment.reset(seed=42)

    for key in ("observation.images.front", "observation.images.wrist"):
        assert observation[key].shape == (128, 128, 3)
        assert observation[key].dtype == np.uint8
        assert np.ptp(observation[key]) > 10
    environment.close()
