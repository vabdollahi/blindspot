"""Gymnasium environment for the Panda pick-and-place task suite."""

import string
from pathlib import Path
from typing import Any, ClassVar, Literal

import gymnasium as gym
import mink
import mujoco
import numpy as np
import yaml
from gymnasium import spaces
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial.transform import Rotation

from common.seeding import TRAIN_SEED_START, TRAIN_SEED_STOP, named_generators
from common.transforms import (
    quat_wxyz_to_scipy,
    scipy_to_quat_wxyz,
    yaw_from_quat_wxyz,
)
from envs.scene import build_scene
from motion.control import (
    ENVIRONMENT_TIMESTEP,
    WORKSPACE_LOWER,
    WORKSPACE_UPPER,
    ControlLoop,
    tcp_pose,
)
from motion.ik import ARM_JOINT_NAMES, MinkIK
from skills.grasp_check import count_collisions, is_grasped

OBSERVATION_SIZE = 128
SUCCESS_HOLD_SECONDS = 1.0
SUCCESS_HOLD_STEPS = round(SUCCESS_HOLD_SECONDS / ENVIRONMENT_TIMESTEP)
# Grasped-start (RL curriculum) resets lift the held cube by up to this much.
GRASPED_START_MAX_LIFT = 0.08
# Destination-side grasped starts hold the cube this far above the destination,
# clearing a bowl rim or the support cube.
DESTINATION_START_HEIGHT = (0.06, 0.09)
CUBE_COLORS = ("red", "blue", "green")
BASE_CUBE_POSITIONS = {
    "cube_red": np.array([0.34, -0.20, 0.02], dtype=np.float64),
    "cube_blue": np.array([0.49, -0.20, 0.02], dtype=np.float64),
    "cube_green": np.array([0.64, -0.20, 0.02], dtype=np.float64),
}
STAGING_POSITIONS = (
    np.array([0.25, 0.45, 0.02], dtype=np.float64),
    np.array([0.80, 0.45, 0.02], dtype=np.float64),
    np.array([0.25, 0.49, 0.02], dtype=np.float64),
)
_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_TEXT_CHARSET = string.ascii_letters + string.digits + "_ -.,{}'!"


class GoalDefinition(BaseModel):
    """Validate one T5 subgoal using an already-defined task predicate."""

    model_config = ConfigDict(extra="forbid")

    task: Literal["T2", "T3", "T4"]
    object: str


class TaskDefinition(BaseModel):
    """Validate one task's time limit, defaults, and instruction templates."""

    model_config = ConfigDict(extra="forbid")

    time_limit: float
    default_target: tuple[float, float, float] | None = None
    cube_count: tuple[int, int] | None = None
    instruction_templates: list[str] = Field(default_factory=list)
    held_out_instruction_templates: list[str] = Field(default_factory=list)
    goal_sequence: list[GoalDefinition] = Field(default_factory=list)


class TaskConfigFile(BaseModel):
    """Validate the task configuration file."""

    model_config = ConfigDict(extra="forbid")

    tasks: dict[str, TaskDefinition]


class LevelDefinition(BaseModel):
    """Validate one randomization level's pose and perturbation ranges."""

    model_config = ConfigDict(extra="forbid")

    cube_x: tuple[float, float]
    cube_y: tuple[float, float]
    target_x: tuple[float, float]
    target_y: tuple[float, float]
    target_z: tuple[float, float]
    bowl_x: tuple[float, float]
    bowl_y: tuple[float, float]
    red_x: tuple[float, float]
    red_y: tuple[float, float]
    blue_x: tuple[float, float]
    blue_y: tuple[float, float]
    yaw: tuple[float, float]
    mass_scale: tuple[float, float]
    friction_scale: tuple[float, float]
    camera_position_jitter: float
    camera_rotation_jitter_degrees: float
    light_color_jitter: float
    material_color_jitter: float
    distractor_count: tuple[int, int]


class LevelConfigFile(BaseModel):
    """Validate the level configuration file."""

    model_config = ConfigDict(extra="forbid")

    levels: dict[str, LevelDefinition]


def _load_task_configs() -> tuple[
    dict[str, TaskDefinition], dict[str, LevelDefinition]
]:
    """Load and validate the task and randomization YAML files."""
    tasks_raw = _load_yaml(_CONFIG_DIR / "tasks.yaml")
    levels_raw = _load_yaml(_CONFIG_DIR / "levels.yaml")
    task_config = TaskConfigFile.model_validate(tasks_raw)
    level_config = LevelConfigFile.model_validate(levels_raw)
    return task_config.tasks, level_config.levels


def get_instruction_templates(
    task: str,
    split: Literal["train", "held-out", "all"] = "all",
) -> tuple[str, ...]:
    """Return the configured instruction templates for a task and data split."""
    normalized_task = task.strip().upper().replace("_", "-")
    if normalized_task == "T3L":
        normalized_task = "T3-L"
    if normalized_task not in _TASK_CONFIGS:
        raise ValueError(f"Unsupported task {task!r}; choose from {tuple(_TASK_CONFIGS)}")

    definition = _TASK_CONFIGS[normalized_task]
    if split == "train":
        return tuple(definition.instruction_templates)
    if split == "held-out":
        return tuple(definition.held_out_instruction_templates)
    if split == "all":
        return (
            *definition.instruction_templates,
            *definition.held_out_instruction_templates,
        )
    raise ValueError(f"Unsupported instruction template split {split!r}")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parse one YAML file and require a mapping at its root."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root in {path.name} must be a mapping")
    return value


_TASK_CONFIGS, _LEVEL_CONFIGS = _load_task_configs()


class PickPlaceEnv(gym.Env[dict[str, Any], np.ndarray]):
    """Simulate one pick-and-place task at L0, L1, or L2."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        task: str,
        level: str = "L0",
        obs_mode: str = "state",
        *,
        render_mode: str | None = None,
        image_size: int = OBSERVATION_SIZE,
    ) -> None:
        """Build the MuJoCo scene, controller, spaces, and validated config."""
        super().__init__()
        normalized_task = task.strip().upper().replace("_", "-")
        if normalized_task == "T3L":
            normalized_task = "T3-L"
        if normalized_task not in _TASK_CONFIGS:
            raise ValueError(
                f"Unsupported task {task!r}; choose from {tuple(_TASK_CONFIGS)}"
            )
        if level not in _LEVEL_CONFIGS:
            raise ValueError(f"Unsupported randomization level {level!r}")
        if obs_mode not in {"state", "image"}:
            raise ValueError("obs_mode must be 'state' or 'image'")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        if image_size < 1:
            raise ValueError("image_size must be positive")

        self.task = normalized_task
        self.level = level
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.image_size = image_size
        self._task_config = _TASK_CONFIGS[self.task]
        self._episode_time_limit_steps = round(
            self._task_config.time_limit / ENVIRONMENT_TIMESTEP
        )
        self._level_config = _LEVEL_CONFIGS[self.level]
        self.model = build_scene(
            include_distractors=self.level == "L2",
            include_visual_textures=self.level == "L2",
        )
        self.data = mujoco.MjData(self.model)
        self.ik = MinkIK()
        self.control = ControlLoop(self.model, self.data, self.ik)
        self._renderer: mujoco.Renderer | None = None
        self._distractor_names = (
            tuple(f"distractor_{index}" for index in range(3))
            if self.level == "L2"
            else ()
        )
        self._object_names = (
            tuple(f"cube_{color}" for color in CUBE_COLORS)
            + ("bowl",)
            + self._distractor_names
        )
        self._target_zone_geom_id = self.model.geom("target_zone").id
        self._tcp_site_id = self.model.site("tcp").id
        self._front_camera_id = self.model.camera("front").id
        self._wrist_camera_id = self.model.camera("wrist").id
        self._arm_qpos_indices = np.asarray(
            [
                self.model.jnt_qposadr[self.model.joint(name).id]
                for name in ARM_JOINT_NAMES
            ],
            dtype=np.intp,
        )
        self._arm_dof_indices = np.asarray(
            [
                self.model.jnt_dofadr[self.model.joint(name).id]
                for name in ARM_JOINT_NAMES
            ],
            dtype=np.intp,
        )
        self._finger_joint_ids = np.asarray(
            [self.model.joint(name).id for name in ("finger_joint1", "finger_joint2")],
            dtype=np.intp,
        )
        self._base_geom_rgba = self.model.geom_rgba.copy()
        self._base_mat_rgba = self.model.mat_rgba.copy()
        self._base_texture_data = self.model.tex_data.copy()
        self._base_geom_friction = self.model.geom_friction.copy()
        self._base_geom_contype = self.model.geom_contype.copy()
        self._base_geom_conaffinity = self.model.geom_conaffinity.copy()
        self._base_body_mass = self.model.body_mass.copy()
        self._base_body_inertia = self.model.body_inertia.copy()
        self._base_camera_position = self.model.cam_pos.copy()
        self._base_camera_quaternion = self.model.cam_quat.copy()
        self._base_light_position = self.model.light_pos.copy()
        self._base_light_diffuse = self.model.light_diffuse.copy()
        self._base_target_zone_position = self.model.geom_pos[
            self._target_zone_geom_id
        ].copy()
        self._rng_scene: np.random.Generator | None = None
        self._rng_visual: np.random.Generator | None = None
        self._episode_steps = 0
        self._collision_count = 0
        self._success_hold_steps = 0
        self._failure_reason: str | None = None
        self._episode_done = False
        self._start_grasped = False
        self._target_pose = np.array([0.50, 0.0, 0.35, 0.0], dtype=np.float64)
        self._target_object = ""
        self._active_cube_names: tuple[str, ...] = ()
        self._initial_cube_positions: dict[str, np.ndarray] = {}
        self._instruction = ""
        self._distractor_count = 0
        self._t5_goal_index = 0
        self._configure_spaces()

    def _configure_spaces(self) -> None:
        """Create action and observation spaces for the selected observation mode."""
        action_low = np.array([*WORKSPACE_LOWER, -np.pi / 2, 0.0], dtype=np.float32)
        action_high = np.array([*WORKSPACE_UPPER, np.pi / 2, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
        observation_spaces: dict[str, spaces.Space[Any]] = {
            "task": _text_space(8),
            "instruction": _text_space(256),
            "observation.state": spaces.Box(
                low=np.array(
                    [
                        *[self.model.joint(name).range[0] for name in ARM_JOINT_NAMES],
                        float(np.sum(self.model.jnt_range[self._finger_joint_ids, 0])),
                        -2.0,
                        -2.0,
                        -2.0,
                        -np.pi,
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        *[self.model.joint(name).range[1] for name in ARM_JOINT_NAMES],
                        float(np.sum(self.model.jnt_range[self._finger_joint_ids, 1])),
                        2.0,
                        2.0,
                        2.0,
                        np.pi,
                    ],
                    dtype=np.float32,
                ),
                shape=(12,),
                dtype=np.float32,
            ),
        }
        if self.obs_mode == "state":
            observation_spaces.update(
                {
                    "object_poses": spaces.Dict(
                        {
                            name: spaces.Dict(
                                {
                                    "position": spaces.Box(
                                        low=np.array(
                                            [-5.0, -5.0, -100_000.0], dtype=np.float32
                                        ),
                                        high=np.array(
                                            [5.0, 5.0, 5.0], dtype=np.float32
                                        ),
                                        shape=(3,),
                                        dtype=np.float32,
                                    ),
                                    "quaternion_wxyz": spaces.Box(
                                        low=-1.0,
                                        high=1.0,
                                        shape=(4,),
                                        dtype=np.float32,
                                    ),
                                }
                            )
                            for name in self._object_names
                        }
                    ),
                    "target_pose": spaces.Box(
                        low=np.array([*WORKSPACE_LOWER, -np.pi / 2], dtype=np.float32),
                        high=np.array([*WORKSPACE_UPPER, np.pi / 2], dtype=np.float32),
                        dtype=np.float32,
                    ),
                    "target_object": _text_space(32),
                }
            )
        else:
            image_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size, 3),
                dtype=np.uint8,
            )
            observation_spaces.update(
                {
                    "observation.images.front": image_space,
                    "observation.images.wrist": image_space,
                }
            )
        self.observation_space = spaces.Dict(observation_spaces)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the task from named reproducible random streams.

        RL training curriculum options (evaluation never sets them):
        `options["grasped_start_probability"]` (default 0) starts that fraction
        of episodes with the target cube already held in a closed grasp, and
        `options["destination_start_fraction"]` (default 0) starts that fraction
        of those grasped episodes partway along the path to the destination.
        """
        super().reset(seed=seed)
        grasped_start_probability = float(
            (options or {}).get("grasped_start_probability", 0.0)
        )
        destination_start_fraction = float(
            (options or {}).get("destination_start_fraction", 0.0)
        )
        if not 0.0 <= grasped_start_probability <= 1.0:
            raise ValueError("grasped_start_probability must be within [0, 1]")
        if not 0.0 <= destination_start_fraction <= 1.0:
            raise ValueError("destination_start_fraction must be within [0, 1]")
        if self.level == "L2":
            self.close()
        # Unseeded (auto-)resets stay inside the training seed range, so
        # training never lands on a validation or evaluation seed.
        actual_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(TRAIN_SEED_START, TRAIN_SEED_STOP))
        )
        generators = named_generators(actual_seed, ("scene", "visual", "curriculum"))
        self._rng_scene = generators["scene"]
        self._rng_visual = generators["visual"]
        self._restore_model()
        mujoco.mj_resetData(self.model, self.data)
        self._initialize_robot()
        instruction_template = (options or {}).get("instruction_template")
        if instruction_template is not None and not isinstance(
            instruction_template, str
        ):
            raise TypeError("instruction_template must be a string")
        self._sample_episode(instruction_template=instruction_template)
        self._apply_level_randomization()
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self.control.reset()
        self._start_grasped = False
        if grasped_start_probability > 0.0:
            if not self._target_object:
                raise ValueError(f"Task {self.task} has no object to start grasped")
            curriculum = generators["curriculum"]
            if curriculum.random() < grasped_start_probability:
                # Drawn only when enabled, so the default leaves the stream unchanged.
                toward_destination = (
                    destination_start_fraction > 0.0
                    and curriculum.random() < destination_start_fraction
                )
                self._start_grasped = self._start_with_target_grasped(
                    curriculum, toward_destination=toward_destination
                )
        self._episode_steps = 0
        self._collision_count = 0
        self._success_hold_steps = 0
        self._failure_reason = None
        self._episode_done = False
        self._t5_goal_index = 0
        return self._get_observation(), self._info(success=False)

    def _start_with_target_grasped(
        self,
        random: np.random.Generator,
        *,
        toward_destination: bool = False,
    ) -> bool:
        """Teleport the arm into a closed grasp around the target cube and settle it.

        By default the cube is held above its sampled table position; with
        `toward_destination`, it is held at a random point along the straight
        path to the destination, at `DESTINATION_START_HEIGHT` above it.
        """
        name = self._target_object
        body_id = self.model.body(name).id
        cube_position = self.data.xpos[body_id].copy()
        cube_yaw = yaw_from_quat_wxyz(self.data.xquat[body_id])
        # Cubes are 90° symmetric: grasp the face pair nearest the home yaw.
        grasp_yaw = float((cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        grasp_position = cube_position.copy()
        if toward_destination:
            destination = self._current_target_pose()[:3]
            progress = float(random.uniform(0.0, 1.0))
            grasp_position[:2] += progress * (destination[:2] - cube_position[:2])
            grasp_position[2] = destination[2] + float(
                random.uniform(*DESTINATION_START_HEIGHT)
            )
        else:
            grasp_position[2] += float(random.uniform(0.0, GRASPED_START_MAX_LIFT))
        arm_q, converged = self.ik.solve(
            tcp_pose(grasp_position, grasp_yaw), self.ik.rest_q
        )
        if not converged:
            return False

        self.data.qpos[self._arm_qpos_indices] = arm_q
        self.data.qvel[self._arm_dof_indices] = 0.0
        half_width = float(self.model.geom(f"{name}_geom").size[0])
        for joint_id in self._finger_joint_ids:
            low, high = self.model.jnt_range[joint_id]
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = np.clip(
                half_width, low, high
            )
        self._set_free_body(name, grasp_position, cube_yaw)
        mujoco.mj_forward(self.model, self.data)
        self.control.reset()
        try:
            self.control.apply(np.array([*grasp_position, grasp_yaw, 0.0]))
        except mink.exceptions.NoSolutionFound:
            return False
        return is_grasped(self.model, self.data, name)


    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one clipped action and update task success and timeout state."""
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (5,) or not np.all(np.isfinite(command)):
            raise ValueError("Action must be a finite [x, y, z, yaw, g] vector")
        clipped_action = np.clip(
            command,
            self.action_space.low.astype(np.float64),
            self.action_space.high.astype(np.float64),
        )
        if not self._episode_done:
            try:
                self.control.apply(clipped_action)
            except mink.exceptions.NoSolutionFound:
                self._failure_reason = self._failure_reason or "ik_failed"
                mujoco.mj_step(self.model, self.data, nstep=25)
            self._episode_steps += 1
            self._collision_count += count_collisions(self.model, self.data)
            if self._collision_count and self._failure_reason is None:
                self._failure_reason = "collision"

            if self._success_predicate():
                self._success_hold_steps += 1
            else:
                self._success_hold_steps = 0

            if self._success_hold_steps >= SUCCESS_HOLD_STEPS:
                if (
                    self.task == "T5"
                    and self._t5_goal_index < len(self._task_config.goal_sequence) - 1
                ):
                    self._t5_goal_index += 1
                    self._success_hold_steps = 0
                else:
                    self._episode_done = True

        terminated = self._episode_done
        truncated = (
            not terminated
            and self._episode_steps >= self._episode_time_limit_steps
        )
        if truncated and self._failure_reason is None:
            self._failure_reason = "timeout"
        success = terminated
        reward = 1.0 if success else 0.0
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._info(success=success),
        )

    def render(self) -> np.ndarray | None:
        """Render the front camera when rgb_array mode is enabled."""
        if self.render_mode is None:
            return None
        return self._render_camera("front")

    def observe_images(self) -> dict[str, np.ndarray]:
        """Render the synchronized front and wrist RGB observations."""
        return {
            "observation.images.front": self._render_camera("front"),
            "observation.images.wrist": self._render_camera("wrist"),
        }

    def close(self) -> None:
        """Release the lazy MuJoCo renderer, if one was created."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _restore_model(self) -> None:
        """Restore per-instance model arrays before seeded randomization."""
        self.model.geom_rgba[:] = self._base_geom_rgba
        self.model.mat_rgba[:] = self._base_mat_rgba
        if self.model.ntexdata:
            self.model.tex_data[:] = self._base_texture_data
        self.model.geom_friction[:] = self._base_geom_friction
        self.model.geom_contype[:] = self._base_geom_contype
        self.model.geom_conaffinity[:] = self._base_geom_conaffinity
        self.model.body_mass[:] = self._base_body_mass
        self.model.body_inertia[:] = self._base_body_inertia
        self.model.cam_pos[:] = self._base_camera_position
        self.model.cam_quat[:] = self._base_camera_quaternion
        self.model.light_pos[:] = self._base_light_position
        self.model.light_diffuse[:] = self._base_light_diffuse
        self.model.geom_pos[self._target_zone_geom_id] = self._base_target_zone_position

    def _initialize_robot(self) -> None:
        """Set the Panda arm to home and open both fingers."""
        for name, position in zip(ARM_JOINT_NAMES, self.ik.rest_q, strict=True):
            joint_id = self.model.joint(name).id
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = position
        for joint_id in self._finger_joint_ids:
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = self.model.jnt_range[
                joint_id, 1
            ]

    def _sample_episode(self, *, instruction_template: str | None = None) -> None:
        """Sample active objects, target pose, and task instruction."""
        if self._rng_scene is None:
            raise RuntimeError("reset must initialize the scene random generator")
        if instruction_template is not None and self.task != "T3-L":
            raise ValueError("instruction_template is only supported for T3-L")
        random = self._rng_scene
        l0 = self.level == "L0"
        self._instruction = ""
        self._active_cube_names = ()
        self._target_object = ""
        if self.task == "T1":
            if l0:
                target = np.asarray(self._task_config.default_target, dtype=np.float64)
            else:
                target = np.array(
                    [
                        random.uniform(*self._level_config.target_x),
                        random.uniform(*self._level_config.target_y),
                        random.uniform(*self._level_config.target_z),
                    ],
                    dtype=np.float64,
                )
            self._target_pose = np.array([*target, 0.0], dtype=np.float64)
        elif self.task == "T2":
            self._active_cube_names = ("cube_red",)
            self._target_object = "cube_red"
            target_xy = (
                self._task_config.default_target[:2]
                if l0
                else (
                    random.uniform(*self._level_config.target_x),
                    random.uniform(*self._level_config.target_y),
                )
            )
            self._set_target_zone(target_xy)
        elif self.task == "T3":
            color = "red" if l0 else str(random.choice(CUBE_COLORS))
            self._active_cube_names = (f"cube_{color}",)
            self._target_object = f"cube_{color}"
        elif self.task == "T3-L":
            if instruction_template is not None and instruction_template not in (
                get_instruction_templates("T3-L")
            ):
                raise ValueError("instruction_template is not configured for T3-L")
            if l0:
                colors = ["blue", "red"]
            else:
                minimum, maximum = self._task_config.cube_count or (2, 3)
                count = int(random.integers(minimum, maximum + 1))
                colors = sorted(
                    str(color)
                    for color in random.choice(CUBE_COLORS, size=count, replace=False)
                )
            self._active_cube_names = tuple(f"cube_{color}" for color in colors)
            named_color = "red" if l0 else str(random.choice(colors))
            self._target_object = f"cube_{named_color}"
            templates = get_instruction_templates("T3-L")
            template = instruction_template or (
                templates[0] if l0 else str(random.choice(templates))
            )
            self._instruction = template.format(color=named_color)
        elif self.task == "T4":
            self._active_cube_names = ("cube_red", "cube_blue")
            self._target_object = "cube_red"
        elif self.task == "T5":
            active = [
                goal.object
                for goal in self._task_config.goal_sequence
                if goal.task != "T4"
            ]
            if any(goal.task == "T4" for goal in self._task_config.goal_sequence):
                active.extend(("cube_red", "cube_blue"))
            self._active_cube_names = tuple(dict.fromkeys(active))
            if not self._task_config.goal_sequence:
                raise ValueError("T5 requires at least one configured goal")
            self._target_object = self._task_config.goal_sequence[0].object
            template = str(random.choice(self._task_config.instruction_templates))
            self._instruction = template
            if any(goal.task == "T2" for goal in self._task_config.goal_sequence):
                target_xy = (
                    (0.49, 0.02)
                    if l0
                    else (
                        random.uniform(*self._level_config.target_x),
                        random.uniform(*self._level_config.target_y),
                    )
                )
                self._set_target_zone(target_xy)
        self._place_cubes()
        needs_bowl = self.task in {"T3", "T3-L"} or (
            self.task == "T5"
            and any(goal.task == "T3" for goal in self._task_config.goal_sequence)
        )
        if needs_bowl:
            bowl_xy = (
                (0.66, 0.20)
                if l0
                else (
                    random.uniform(*self._level_config.bowl_x),
                    random.uniform(*self._level_config.bowl_y),
                )
            )
            self._set_free_body(
                "bowl",
                np.array([*bowl_xy, 0.0], dtype=np.float64),
                0.0 if l0 else float(random.uniform(*self._level_config.yaw)),
            )
        mujoco.mj_forward(self.model, self.data)
        self._initial_cube_positions = {
            name: self.data.xpos[self.model.body(name).id].copy()
            for name in self._active_cube_names
        }
        self._distractor_count = 0
        if self.level == "L2":
            lower, upper = self._level_config.distractor_count
            self._distractor_count = int(random.integers(lower, upper + 1))
            self._place_distractors()

    def _place_cubes(self) -> None:
        """Place active cubes in task-specific randomized regions."""
        if self._rng_scene is None:
            raise RuntimeError("reset must initialize the scene random generator")
        random = self._rng_scene
        active = set(self._active_cube_names)
        sampled_positions: list[np.ndarray] = []
        for color in CUBE_COLORS:
            name = f"cube_{color}"
            if name not in active:
                staging = STAGING_POSITIONS[CUBE_COLORS.index(color)]
                self._set_free_body(name, staging, 0.0)
                geom_id = self.model.geom(f"{name}_geom").id
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
                self.model.geom_rgba[geom_id, 3] = 0.0
                continue
            if self.level == "L0":
                position = BASE_CUBE_POSITIONS[name].copy()
                yaw = 0.0
            else:
                position = self._sample_cube_position(color, sampled_positions)
                yaw = float(random.uniform(*self._level_config.yaw))
            sampled_positions.append(position)
            self._set_free_body(name, position, yaw)

    def _sample_cube_position(
        self,
        color: str,
        existing: list[np.ndarray],
    ) -> np.ndarray:
        """Sample a table pose that does not overlap another active cube."""
        if self._rng_scene is None:
            raise RuntimeError("reset must initialize the scene random generator")
        random = self._rng_scene
        if self.task == "T4":
            x_range = (
                self._level_config.red_x
                if color == "red"
                else self._level_config.blue_x
            )
            y_range = (
                self._level_config.red_y
                if color == "red"
                else self._level_config.blue_y
            )
        else:
            x_range = self._level_config.cube_x
            y_range = self._level_config.cube_y
        for _ in range(100):
            candidate = np.array(
                [random.uniform(*x_range), random.uniform(*y_range), 0.02],
                dtype=np.float64,
            )
            if all(
                np.linalg.norm(candidate[:2] - value[:2]) >= 0.06 for value in existing
            ):
                return candidate
        raise RuntimeError("Could not sample non-overlapping cube poses")

    def _set_target_zone(self, xy: tuple[float, float]) -> None:
        """Move the non-colliding target marker to its episode pose."""
        position = self._base_target_zone_position.copy()
        position[:2] = xy
        self.model.geom_pos[self._target_zone_geom_id] = position
        self._target_pose = np.array([xy[0], xy[1], 0.02, 0.0], dtype=np.float64)

    def _set_free_body(
        self,
        name: str,
        position: np.ndarray,
        yaw: float,
    ) -> None:
        """Set the free-joint pose and zero its translational and angular velocity."""
        body_id = self.model.body(name).id
        joint_id = int(self.model.body_jntadr[body_id])
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        dof_address = int(self.model.jnt_dofadr[joint_id])
        self.data.qpos[qpos_address : qpos_address + 3] = position
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = [
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        ]
        self.data.qvel[dof_address : dof_address + 6] = 0.0

    def _place_distractors(self) -> None:
        """Place the sampled L2 distractors and disable unused placeholders."""
        if self._rng_scene is None:
            raise RuntimeError("reset must initialize the scene random generator")
        random = self._rng_scene
        avoid = [
            self.data.xpos[self.model.body(name).id][:2].copy()
            for name in self._active_cube_names
        ]
        avoid.append(self.data.xpos[self.model.body("bowl").id][:2].copy())
        for index, name in enumerate(self._distractor_names):
            geom_id = self.model.geom(f"{name}_geom").id
            if index < self._distractor_count:
                for _ in range(100):
                    xy = np.array(
                        [random.uniform(0.35, 0.76), random.uniform(-0.28, 0.28)]
                    )
                    if all(np.linalg.norm(xy - value) >= 0.08 for value in avoid):
                        break
                else:
                    raise RuntimeError("Could not sample a non-overlapping distractor")
                avoid.append(xy)
                self._set_free_body(
                    name, np.array([*xy, 0.02]), float(random.uniform(-1.5, 1.5))
                )
            else:
                self._set_free_body(
                    name, np.array([1.2 + 0.05 * index, 0.45, 0.02]), 0.0
                )
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
                self.model.geom_rgba[geom_id, 3] = 0.0

    def _apply_level_randomization(self) -> None:
        """Apply L2 physical, camera, lighting, and material appearance changes."""
        if self.level != "L2":
            return
        if self._rng_visual is None:
            raise RuntimeError("reset must initialize the visual random generator")
        random = self._rng_visual
        randomized_bodies = (
            [f"cube_{color}" for color in CUBE_COLORS]
            + ["bowl"]
            + list(self._distractor_names[: self._distractor_count])
        )
        for name in randomized_bodies:
            mass_scale = float(random.uniform(*self._level_config.mass_scale))
            friction_scale = float(random.uniform(*self._level_config.friction_scale))
            body_id = self.model.body(name).id
            self.model.body_mass[body_id] = self._base_body_mass[body_id] * mass_scale
            self.model.body_inertia[body_id] = (
                self._base_body_inertia[body_id] * mass_scale
            )
            geom_start = int(self.model.body_geomadr[body_id])
            geom_count = int(self.model.body_geomnum[body_id])
            geom_ids = slice(geom_start, geom_start + geom_count)
            self.model.geom_friction[geom_ids] = (
                self._base_geom_friction[geom_ids] * friction_scale
            )

        self._randomize_material_colors(random)
        self._randomize_textures(random)
        self._randomize_cameras(random)
        if self.model.nlight:
            self.model.light_pos[:] = self._base_light_position + random.uniform(
                -0.10, 0.10, size=self._base_light_position.shape
            )
            jitter = self._level_config.light_color_jitter
            self.model.light_diffuse[:] = np.clip(
                self._base_light_diffuse
                + random.uniform(-jitter, jitter, size=self._base_light_diffuse.shape),
                0.0,
                1.0,
            )

    def _randomize_material_colors(self, random: np.random.Generator) -> None:
        """Vary object and table material colors while preserving cube color identity."""
        jitter = self._level_config.material_color_jitter
        for name in [f"cube_{color}" for color in CUBE_COLORS] + ["bowl"]:
            geom_id = self.model.geom(
                f"{name}_geom" if name.startswith("cube_") else "bowl_base"
            ).id
            rgba = self._base_geom_rgba[geom_id].copy()
            rgba[:3] = np.clip(
                rgba[:3] * random.uniform(1.0 - jitter, 1.0 + jitter)
                + random.uniform(-jitter / 4, jitter / 4, size=3),
                0.0,
                1.0,
            )
            self.model.geom_rgba[geom_id] = rgba
            material_id = int(self.model.geom_matid[geom_id])
            if material_id >= 0:
                material_rgba = self._base_mat_rgba[material_id].copy()
                material_rgba[:3] = rgba[:3]
                self.model.mat_rgba[material_id] = material_rgba
        table_geom_id = self.model.geom("table").id
        table_rgba = self._base_geom_rgba[table_geom_id].copy()
        table_rgba[:3] = np.clip(
            table_rgba[:3] + random.uniform(-jitter, jitter, size=3),
            0.0,
            1.0,
        )
        self.model.geom_rgba[table_geom_id] = table_rgba
        for index in range(self._distractor_count):
            geom_id = self.model.geom(f"distractor_{index}_geom").id
            self.model.geom_rgba[geom_id, :3] = random.uniform(0.15, 0.85, size=3)

    def _randomize_textures(self, random: np.random.Generator) -> None:
        """Perturb each 4 by 4 material texture while preserving its base palette."""
        amount = self._level_config.material_color_jitter
        for name in (
            "table_texture",
            *(f"cube_{color}_texture" for color in CUBE_COLORS),
            "bowl_texture",
        ):
            texture_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_TEXTURE,
                name,
            )
            address = int(self.model.tex_adr[texture_id])
            width = int(self.model.tex_width[texture_id])
            height = int(self.model.tex_height[texture_id])
            channels = int(self.model.tex_nchannel[texture_id])
            size = width * height * channels
            base = self._base_texture_data[address : address + size].reshape(
                height, width, channels
            )
            variation = random.uniform(
                -amount * 255.0,
                amount * 255.0,
                size=base.shape,
            )
            pixels = np.clip(base.astype(np.float64) + variation, 0, 255).astype(
                np.uint8
            )
            self.model.tex_data[address : address + size] = pixels.reshape(-1)

    def _randomize_cameras(self, random: np.random.Generator) -> None:
        """Jitter camera positions by 1 cm and orientations by 2 degrees at L2."""
        position_jitter = self._level_config.camera_position_jitter
        angle_jitter = np.deg2rad(self._level_config.camera_rotation_jitter_degrees)
        for camera_id in (self._front_camera_id, self._wrist_camera_id):
            self.model.cam_pos[camera_id] = self._base_camera_position[
                camera_id
            ] + random.uniform(
                -position_jitter,
                position_jitter,
                size=3,
            )
            delta = Rotation.from_euler(
                "xyz",
                random.uniform(-angle_jitter, angle_jitter, size=3),
            )
            base_rotation = quat_wxyz_to_scipy(self._base_camera_quaternion[camera_id])
            self.model.cam_quat[camera_id] = scipy_to_quat_wxyz(delta * base_rotation)

    def _get_observation(self) -> dict[str, Any]:
        """Build a Gymnasium observation from the current simulation state."""
        tcp_rotation = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        tcp_yaw = float(np.arctan2(tcp_rotation[1, 0], tcp_rotation[0, 0]))
        arm_positions = self.data.qpos[self._arm_qpos_indices]
        finger_positions = np.array(
            [
                self.data.qpos[self.model.jnt_qposadr[joint_id]]
                for joint_id in self._finger_joint_ids
            ],
            dtype=np.float64,
        )
        state = np.asarray(
            [
                *arm_positions,
                np.sum(finger_positions),
                *self.data.site_xpos[self._tcp_site_id],
                tcp_yaw,
            ],
            dtype=np.float32,
        )
        observation: dict[str, Any] = {
            "task": self.task,
            "instruction": self._instruction,
            "observation.state": state,
        }
        if self.obs_mode == "state":
            observation["object_poses"] = {
                name: {
                    "position": np.asarray(
                        self.data.xpos[self.model.body(name).id], dtype=np.float32
                    ).copy(),
                    "quaternion_wxyz": np.asarray(
                        self.data.xquat[self.model.body(name).id], dtype=np.float32
                    ).copy(),
                }
                for name in self._object_names
            }
            observation["target_pose"] = self._current_target_pose().astype(np.float32)
            observation["target_object"] = self._current_target_object()
        else:
            observation.update(self.observe_images())
        return observation

    def _render_camera(self, camera_name: str) -> np.ndarray:
        """Render one named camera at this environment's configured resolution."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.image_size,
                width=self.image_size,
            )
        self._renderer.update_scene(self.data, camera=camera_name)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def _current_target_pose(self) -> np.ndarray:
        """Return the current task's goal pose in xyz+yaw form."""
        if self.task == "T5":
            goal = self._task_config.goal_sequence[self._t5_goal_index]
            if goal.task == "T2":
                return self._target_pose.copy()
            if goal.task == "T4":
                blue_position = self.data.xpos[self.model.body("cube_blue").id]
                return np.array(
                    [
                        blue_position[0],
                        blue_position[1],
                        blue_position[2] + 0.04,
                        0.0,
                    ],
                    dtype=np.float64,
                )
            bowl_position = self.data.xpos[self.model.body("bowl").id]
            return np.array(
                [bowl_position[0], bowl_position[1], bowl_position[2] + 0.026, 0.0],
                dtype=np.float64,
            )
        if self.task == "T1":
            return self._target_pose.copy()
        if self.task == "T2":
            return self._target_pose.copy()
        if self.task in {"T3", "T3-L"}:
            bowl_position = self.data.xpos[self.model.body("bowl").id]
            return np.array(
                [bowl_position[0], bowl_position[1], bowl_position[2] + 0.026, 0.0],
                dtype=np.float64,
            )
        blue_position = self.data.xpos[self.model.body("cube_blue").id]
        return np.array(
            [blue_position[0], blue_position[1], blue_position[2] + 0.04, 0.0],
            dtype=np.float64,
        )

    def _current_target_object(self) -> str:
        """Return the object associated with the current task goal."""
        if self.task == "T5":
            return self._task_config.goal_sequence[self._t5_goal_index].object
        return self._target_object

    def _success_predicate(self) -> bool:
        """Check the current frozen task predicate and open-gripper condition."""
        if not self._gripper_is_open():
            return False
        if self.task == "T1":
            tcp_position = self.data.site_xpos[self._tcp_site_id]
            return bool(np.linalg.norm(tcp_position - self._target_pose[:3]) <= 0.005)
        if self.task == "T2":
            return self._cube_in_target_zone(self._current_target_object())
        if self.task == "T5":
            goal = self._task_config.goal_sequence[self._t5_goal_index]
            return self._goal_predicate(goal)
        if self.task == "T3":
            return self._cube_in_bowl(self._current_target_object())
        if self.task == "T3-L":
            target_in_bowl = self._cube_in_bowl(self._target_object)
            others_unchanged = all(
                np.linalg.norm(
                    self.data.xpos[self.model.body(name).id]
                    - self._initial_cube_positions[name]
                )
                < 0.02
                for name in self._active_cube_names
                if name != self._target_object
            )
            return target_in_bowl and others_unchanged
        return self._cube_stacked_on_blue()

    def _goal_predicate(self, goal: GoalDefinition) -> bool:
        """Evaluate one configured T5 step using its frozen task predicate."""
        if goal.task == "T2":
            return self._cube_in_target_zone(goal.object)
        if goal.task == "T3":
            return self._cube_in_bowl(goal.object)
        return self._cube_stacked_on_blue()

    def _cube_in_target_zone(self, name: str) -> bool:
        """Check the T2 xy tolerance and that the cube is at rest."""
        position = self.data.xpos[self.model.body(name).id]
        target = self.model.geom_pos[self._target_zone_geom_id]
        return bool(
            np.linalg.norm(position[:2] - target[:2]) <= 0.03
            and self._body_is_at_rest(name)
        )

    def _cube_in_bowl(self, name: str) -> bool:
        """Check bowl radius, floor/rim height, and the cube's rest condition."""
        cube_position = self.data.xpos[self.model.body(name).id]
        bowl_position = self.data.xpos[self.model.body("bowl").id]
        radial_distance = np.linalg.norm(cube_position[:2] - bowl_position[:2])
        return bool(
            radial_distance < 0.07
            and bowl_position[2] + 0.006 <= cube_position[2] < bowl_position[2] + 0.056
            and self._body_is_at_rest(name)
        )

    def _cube_stacked_on_blue(self) -> bool:
        """Check red-on-blue horizontal alignment, support height, and rest."""
        red = self.data.xpos[self.model.body("cube_red").id]
        blue = self.data.xpos[self.model.body("cube_blue").id]
        return bool(
            np.linalg.norm(red[:2] - blue[:2]) <= 0.015
            and abs(red[2] - (blue[2] + 0.04)) <= 0.01
            and self._body_is_at_rest("cube_red")
        )

    def _body_is_at_rest(self, name: str) -> bool:
        """Check a free object's linear and angular velocities against tolerances."""
        body_id = self.model.body(name).id
        joint_id = int(self.model.body_jntadr[body_id])
        dof_address = int(self.model.jnt_dofadr[joint_id])
        velocity = self.data.qvel[dof_address : dof_address + 6]
        return bool(
            np.linalg.norm(velocity[:3]) <= 0.01 and np.linalg.norm(velocity[3:]) <= 0.1
        )

    def _gripper_is_open(self) -> bool:
        """Check the normalized average aperture of the two finger joints."""
        ranges = self.model.jnt_range[self._finger_joint_ids]
        positions = np.array(
            [
                self.data.qpos[self.model.jnt_qposadr[joint_id]]
                for joint_id in self._finger_joint_ids
            ]
        )
        widths = ranges[:, 1] - ranges[:, 0]
        if np.any(widths <= 0):
            return False
        fractions = (positions - ranges[:, 0]) / widths
        return bool(np.mean(fractions) >= 0.5)

    def _info(self, *, success: bool) -> dict[str, Any]:
        """Build the stable per-step diagnostic info mapping."""
        return {
            "success": bool(success),
            "failure_reason": None if success else self._failure_reason,
            "collision_count": int(self._collision_count),
            "episode_time": float(self.data.time),
            "instruction": self._instruction or None,
            "goal_index": self._t5_goal_index if self.task == "T5" else None,
            "start_grasped": bool(self._start_grasped),
        }


def _text_space(max_length: int) -> spaces.Text:
    """Create a Gymnasium text space that accepts task IDs and instructions."""
    return spaces.Text(
        min_length=0,
        max_length=max_length,
        charset=_TEXT_CHARSET,
    )
