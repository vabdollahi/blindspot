import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


def quat_wxyz_to_scipy(quaternion_wxyz: ArrayLike) -> Rotation:
    """Convert a MuJoCo-order quaternion to a SciPy rotation."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("A quaternion must have shape (4,)")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("A quaternion must contain only finite values")
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("A quaternion must have non-zero length")
    return Rotation.from_quat(quaternion / norm, scalar_first=True)


def scipy_to_quat_wxyz(rotation: Rotation) -> NDArray[np.float64]:
    """Convert a SciPy rotation to a MuJoCo-order quaternion."""
    return np.asarray(rotation.as_quat(scalar_first=True), dtype=np.float64)


def quat_wxyz_to_matrix(quaternion_wxyz: ArrayLike) -> NDArray[np.float64]:
    """Convert a MuJoCo-order quaternion to a 3x3 rotation matrix."""
    return quat_wxyz_to_scipy(quaternion_wxyz).as_matrix()


def matrix_to_quat_wxyz(matrix: ArrayLike) -> NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to a MuJoCo-order quaternion."""
    rotation_matrix = np.asarray(matrix, dtype=np.float64)
    if rotation_matrix.shape != (3, 3):
        raise ValueError("A rotation matrix must have shape (3, 3)")
    if not np.all(np.isfinite(rotation_matrix)):
        raise ValueError("A rotation matrix must contain only finite values")
    return scipy_to_quat_wxyz(Rotation.from_matrix(rotation_matrix))


def yaw_from_quat_wxyz(quaternion_wxyz: ArrayLike) -> float:
    """Extract the Z-axis yaw angle in radians from a MuJoCo-order quaternion."""
    rotation_matrix = quat_wxyz_to_matrix(quaternion_wxyz)
    return float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
