"""FK geometry, limit and rotation checks for both arms."""

import numpy as np
import pytest

from robot_model import load
from task2_fk import CONFIGURATIONS_DEG, FK
from task5_jacobian import chain_frames


def test_fk_examples_and_limits():
    for side in ("right", "left"):
        robot = load(side)
        for degrees in CONFIGURATIONS_DEG.values():
            q = np.deg2rad(degrees)
            T = FK(q, robot)
            assert np.linalg.norm(T[:3,:3].T @ T[:3,:3] - np.eye(3)) < 1e-12
            assert abs(np.linalg.det(T[:3,:3]) - 1) < 1e-12
            np.testing.assert_allclose(chain_frames(q, robot)[0], T, atol=1e-12)
        with pytest.raises(ValueError):
            FK(np.deg2rad([91, 0, 0, 0, 0, 0]), robot)
