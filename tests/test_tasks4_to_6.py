"""Independent checks for the numerical solver, Jacobian and singularity claims."""

import numpy as np

from robot_model import load
from task2_fk import FK
from task3_ik import DUAL_DEG, solve_ik
from task4_numerical_ik import numerical_ik, nearest_branch
from task5_jacobian import DifferentialKinematics, jacobian_product, jacobian_fd


def test_two_numerical_branches_for_each_arm():
    perturb = np.deg2rad([2, -2, 1, 2, -1, 1])
    for side in ("right", "left"):
        robot = load(side)
        target = FK(np.deg2rad(DUAL_DEG), robot)
        branches = solve_ik(target, robot).feasible
        assert len(branches) == 2
        found = []
        for branch in branches:
            result = numerical_ik(target, robot, branch + perturb)
            assert result.status == "converged"
            assert max(result.position_error, result.orientation_error) < 1e-8
            found.append(nearest_branch(result.q, branches)[0])
        assert len(set(found)) == 2


def test_jacobians_and_singularity():
    for side in ("right", "left"):
        robot = load(side)
        for degrees in ([20, -35, 10, 25, 40, -15], [0, 0, 0, 0, 89.9, 0]):
            q = np.deg2rad(degrees)
            A, B, C = DifferentialKinematics(q, robot), jacobian_product(q, robot), jacobian_fd(q, robot)
            assert np.linalg.norm(A-B) < 1e-12
            assert np.linalg.norm(A-C) < 1e-8
        singular = DifferentialKinematics(np.deg2rad([0, 0, 0, 0, 90, 0]), robot)
        assert np.linalg.matrix_rank(singular, tol=1e-8) < 6
