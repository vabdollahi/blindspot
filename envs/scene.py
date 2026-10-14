"""MuJoCo scene construction for the pick-and-place tasks."""

import math
from pathlib import Path

import mujoco
import numpy as np

_MENAGERIE_SCENE = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

_CUBE_COLORS = {
    "red": (0.82, 0.12, 0.10, 1.0),
    "blue": (0.10, 0.28, 0.82, 1.0),
    "green": (0.12, 0.68, 0.24, 1.0),
}
_ARM_POSITION_SERVO_GAIN_SCALE = 5.0


def build_scene_spec() -> mujoco.MjSpec:
    """Build the pick-and-place scene specification from Menagerie."""
    spec = mujoco.MjSpec.from_file(str(_MENAGERIE_SCENE))
    for keyframe in tuple(spec.keys):
        spec.delete(keyframe)
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = 512
    spec.visual.global_.offheight = 512

    for side in ("left_finger", "right_finger"):
        pads = [
            geom
            for geom in spec.body(side).geoms
            if geom.type == mujoco.mjtGeom.mjGEOM_BOX
        ]
        if len(pads) != 5:
            raise RuntimeError(f"Expected five fingertip pads on {side}, found {len(pads)}.")
        for index, geom in enumerate(pads, start=1):
            geom.name = f"{side}_fingertip_pad_{index}"
            geom.condim = 4

    worldbody = spec.worldbody
    floor = spec.geom("floor")
    floor.contype = 0
    floor.conaffinity = 0
    worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.3, 0.0, -0.025],
        size=[0.6, 0.55, 0.025],
        rgba=[0.48, 0.52, 0.54, 1.0],
    )

    for color, rgba in _CUBE_COLORS.items():
        spec.add_material(name=f"cube_{color}_material", rgba=rgba)
    for color, position in (
        ("red", [0.34, -0.20, 0.02]),
        ("blue", [0.49, -0.20, 0.02]),
        ("green", [0.64, -0.20, 0.02]),
    ):
        cube = worldbody.add_body(name=f"cube_{color}", pos=position)
        cube.add_freejoint(name=f"cube_{color}_free")
        cube.add_geom(
            name=f"cube_{color}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],
            mass=0.075,
            material=f"cube_{color}_material",
            condim=4,
            friction=[1.0, 0.005, 0.0001],
        )

    bowl = worldbody.add_body(name="bowl", pos=[0.66, 0.20, 0.0])
    bowl.add_freejoint(name="bowl_free")
    bowl.add_geom(
        name="bowl_base",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.003],
        size=[0.076, 0.003],
        mass=0.03,
        rgba=[0.22, 0.25, 0.28, 1.0],
        condim=4,
        friction=[1.0, 0.005, 0.0001],
    )
    wall_count = 16
    inner_radius = 0.07
    wall_thickness = 0.006
    wall_center_radius = inner_radius + wall_thickness / 2
    wall_tangent_half_width = wall_center_radius * math.tan(math.pi / wall_count)
    for index in range(wall_count):
        angle = 2.0 * math.pi * index / wall_count
        bowl.add_geom(
            name=f"bowl_wall_{index:02d}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[
                wall_center_radius * math.cos(angle),
                wall_center_radius * math.sin(angle),
                0.031,
            ],
            euler=[0.0, 0.0, angle],
            size=[wall_thickness / 2, wall_tangent_half_width, 0.025],
            mass=0.005,
            rgba=[0.22, 0.25, 0.28, 1.0],
            condim=4,
            friction=[1.0, 0.005, 0.0001],
        )

    spec.add_material(name="target_zone_material", rgba=[0.94, 0.72, 0.12, 0.45])
    worldbody.add_geom(
        name="target_zone",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.49, 0.02, 0.001],
        size=[0.04, 0.04, 0.001],
        material="target_zone_material",
        contype=0,
        conaffinity=0,
    )

    worldbody.add_camera(
        name="front",
        pos=[0.9, -0.75, 1.0],
        xyaxes=[0.8725, 0.4886, 0.0, -0.3672, 0.6561, 0.6592],
        fovy=55.0,
    )
    spec.body("hand").add_site(
        name="tcp",
        pos=[0.0, 0.0, 0.10],
        size=[0.008],
        rgba=[1.0, 0.2, 0.1, 1.0],
    )
    spec.body("hand").add_camera(
        name="wrist",
        pos=[0.0, 0.025, 0.025],
        quat=[0.0, 1.0, 0.0, 0.0],
        fovy=70.0,
    )
    return spec


def build_scene() -> mujoco.MjModel:
    """Compile and return the pick-and-place MuJoCo model."""
    model = build_scene_spec().compile()
    arm_actuator_ids = np.asarray(
        [model.actuator(f"actuator{index}").id for index in range(1, 8)],
        dtype=np.intp,
    )
    model.actuator_gainprm[arm_actuator_ids, 0] *= _ARM_POSITION_SERVO_GAIN_SCALE
    model.actuator_biasprm[arm_actuator_ids, 1] *= _ARM_POSITION_SERVO_GAIN_SCALE
    return model