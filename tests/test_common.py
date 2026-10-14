import mujoco
import numpy as np
import pytest

from common.seeding import named_generators
from common.transforms import (
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    quat_wxyz_to_scipy,
    scipy_to_quat_wxyz,
    yaw_from_quat_wxyz,
)


def test_quaternion_matrix_matches_mujoco_for_1000_random_quaternions() -> None:
    """Check quaternion conversion against MuJoCo for normalized random inputs."""
    random = np.random.default_rng(17)
    quaternions = random.normal(size=(1000, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    for quaternion in quaternions:
        expected = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(expected, quaternion)
        np.testing.assert_allclose(
            quat_wxyz_to_matrix(quaternion), expected.reshape(3, 3), rtol=0, atol=1e-12
        )


def test_rotation_conversions_round_trip_to_1e_12() -> None:
    """Check quaternion, matrix, and SciPy conversions preserve rotations."""
    random = np.random.default_rng(29)
    quaternions = random.normal(size=(1000, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    for quaternion in quaternions:
        rotation = quat_wxyz_to_scipy(quaternion)
        scipy_round_trip = scipy_to_quat_wxyz(rotation)
        np.testing.assert_allclose(scipy_round_trip, quaternion, rtol=0, atol=1e-12)

        matrix = quat_wxyz_to_matrix(quaternion)
        round_trip_matrix = quat_wxyz_to_matrix(matrix_to_quat_wxyz(matrix))
        np.testing.assert_allclose(round_trip_matrix, matrix, rtol=0, atol=1e-12)


def test_yaw_from_quaternion_uses_z_axis_radians() -> None:
    """Extract the expected yaw for a pure Z-axis rotation."""
    yaw = 0.73
    quaternion = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

    assert yaw_from_quat_wxyz(quaternion) == pytest.approx(yaw, abs=1e-12)


def test_quaternion_conversion_rejects_invalid_inputs() -> None:
    """Reject malformed and zero-length quaternions and malformed matrices."""
    with pytest.raises(ValueError, match="shape"):
        quat_wxyz_to_matrix([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-zero"):
        quat_wxyz_to_matrix([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        matrix_to_quat_wxyz(np.eye(4))


def test_named_generators_are_reproducible_and_independent() -> None:
    """Recreate each named random stream from the same seed."""
    names = ("scene", "noise", "policy")
    first = named_generators(2026, names)
    second = named_generators(2026, names)
    first_draws = {
        name: generator.integers(0, 2**32, size=16, dtype=np.uint32)
        for name, generator in first.items()
    }
    second_draws = {
        name: generator.integers(0, 2**32, size=16, dtype=np.uint32)
        for name, generator in second.items()
    }

    for name in names:
        np.testing.assert_array_equal(first_draws[name], second_draws[name])
    assert not np.array_equal(first_draws["scene"], first_draws["noise"])


def test_named_generators_reject_duplicate_names() -> None:
    """Reject duplicate purpose names rather than silently replacing a stream."""
    with pytest.raises(ValueError, match="unique"):
        named_generators(5, ("scene", "scene"))
