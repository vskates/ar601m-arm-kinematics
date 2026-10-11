"""Check the starter's three-function contract with actual implementations."""

import inspect

import numpy as np

from kinematics import FK, IK, DifferentialKinematics
from robot_model import load


def test_public_contract():
    robot = load()
    q = np.deg2rad([10, 5, -30, -40, -30, 20])
    for function in (FK, IK, DifferentialKinematics):
        assert len(inspect.signature(function).parameters) == 2
    T = FK(q, robot)
    assert T.shape == (4, 4)
    assert np.linalg.norm(T[3] - [0, 0, 0, 1]) < 1e-12
    assert any(np.linalg.norm(FK(branch, robot) - T) < 1e-8 for branch in IK(T, robot))
    assert DifferentialKinematics(q, robot).shape == (6, 6)
