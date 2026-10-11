"""Analytical IK branch, limit, and degeneracy checks for both arms."""

import unittest
from pathlib import Path

import numpy as np

from task2_fk import CONFIGURATIONS_DEG, FK, load_robot
from task3_ik import IK, InfiniteSolutions, evaluate_family, solve_ik, wrap


class TestAnalyticalIK(unittest.TestCase):
    def test_left_zero_pose_is_mirrored(self):
        right = load_robot(Path(__file__).resolve().parents[1] / "robot_model.json")
        left = load_robot(Path(__file__).resolve().parents[1] / "robot_model_left.json")
        mirror = np.diag([1, -1, 1, 1])
        np.testing.assert_allclose(FK(np.zeros(6), left),
                                   mirror @ FK(np.zeros(6), right) @ mirror,
                                   rtol=0, atol=1e-12)

    def test_generic_branches_for_both_arms(self):
        q = np.deg2rad(CONFIGURATIONS_DEG["generic"])
        for filename in ("robot_model.json", "robot_model_left.json"):
            with self.subTest(model=filename):
                robot = load_robot(Path(__file__).resolve().parents[1] / filename)
                target = FK(q, robot)
                result = solve_ik(target, robot)
                self.assertEqual(len(result.real), 8)
                self.assertTrue(any(max(abs(wrap(a - b)) for a, b in zip(q, found)) < 1e-8
                                    for found in IK(target, robot)))
                for found in result.real:
                    self.assertLess(np.linalg.norm(FK(found, robot, check_limits=False)
                                                   - target, ord="fro"), 1e-8)

    def test_unreachable_and_continuous(self):
        robot = load_robot()
        far = np.eye(4)
        far[0, 3] = 2
        self.assertEqual(IK(far, robot), [])
        target = FK(np.deg2rad([0, 0, 0, 0, 90, 0]), robot)
        with self.assertRaises(InfiniteSolutions) as caught:
            IK(target, robot)
        family = caught.exception.result.families[0]
        self.assertTrue(evaluate_family(family, 0, target, robot).feasible)

    def test_recovered_random_feasible_targets(self):
        rng = np.random.default_rng(601)
        for filename in ("robot_model.json", "robot_model_left.json"):
            robot = load_robot(Path(__file__).resolve().parents[1] / filename)
            limits = np.deg2rad([s["limits_deg"] for s in robot["chain"] if s["kind"] == "joint"])
            for _ in range(30):
                q = rng.uniform(limits[:, 0], limits[:, 1])
                target = FK(q, robot)
                result = solve_ik(target, robot)
                self.assertTrue(any(max(abs(wrap(a-b)) for a,b in zip(q, branch)) < 1e-7
                                    for branch in result.feasible))
                for branch in result.real:
                    self.assertLess(np.linalg.norm(FK(branch, robot, check_limits=False)
                                                   - target, ord="fro"), 1e-8)


if __name__ == "__main__":
    unittest.main()
