import mujoco
import numpy as np
import pytest

from envs.scene import build_scene


def test_scene_compiles_and_tcp_is_between_fingertips() -> None:
    """Compile the full scene and verify TCP placement between fingertip pads."""
    model = build_scene()
    assert model.nkey == 0
    assert model.geom("table").id >= 0
    assert model.geom("cube_red_geom").id >= 0
    assert model.geom("bowl_base").id >= 0
    assert model.geom("target_zone").id >= 0
    assert model.camera("front").id >= 0
    assert model.camera("wrist").id >= 0
    table_id = model.geom("table").id
    table_x, _, table_z = model.geom_pos[table_id]
    table_half_size = model.geom_size[table_id]
    assert table_x - table_half_size[0] < 0 < table_x + table_half_size[0]
    assert table_z + table_half_size[2] == pytest.approx(0.0)
    floor_id = model.geom("floor").id
    assert model.geom_contype[floor_id] == 0
    assert model.geom_conaffinity[floor_id] == 0
    assert model.geom_pos[floor_id, 2] < table_z + table_half_size[2]

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tcp = data.site_xpos[model.site("tcp").id]
    left = data.geom_xpos[model.geom("left_finger_fingertip_pad_1").id]
    right = data.geom_xpos[model.geom("right_finger_fingertip_pad_1").id]
    segment = right - left
    fraction = np.dot(tcp - left, segment) / np.dot(segment, segment)
    nearest = left + np.clip(fraction, 0.0, 1.0) * segment

    assert 0.0 <= fraction <= 1.0
    assert np.linalg.norm(tcp - nearest) < 0.01


@pytest.mark.rendering
def test_scene_cameras_render_nonblank_128px_images() -> None:
    """Render non-blank 128 by 128 RGB images from both scene cameras."""
    model = build_scene()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.Renderer(model, height=128, width=128) as renderer:
        for camera in ("front", "wrist"):
            renderer.update_scene(data, camera=camera)
            image = renderer.render()
            assert image.shape == (128, 128, 3)
            assert image.dtype == np.uint8
            assert np.ptp(image) > 10
