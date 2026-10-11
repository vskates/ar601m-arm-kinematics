"""Projected, damped least-squares pose IK, with backtracking."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from task2_fk import FK, load_robot, CONFIGURATIONS_DEG
from task3_ik import DUAL_DEG, solve_ik, wrap
from task5_jacobian import DifferentialKinematics


@dataclass
class NumericalResult:
    q: np.ndarray
    iterations: int
    status: str
    position_error: float
    orientation_error: float


def _error(target, current):
    return np.r_[target[:3, 3] - current[:3, 3],
                 Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()]


def numerical_ik(target, robot, initial, max_iterations=120, tolerance=1e-8):
    """Find one limit-feasible solution; orientation residual is rotation-vector norm."""
    q = np.asarray(initial, dtype=float).copy()
    limits = np.deg2rad([s["limits_deg"] for s in robot["chain"] if s["kind"] == "joint"])
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("Initial q must contain six finite angles")
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError("Initial q is outside joint limits")
    for iteration in range(max_iterations + 1):
        error = _error(target, FK(q, robot))
        ep, er = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
        if max(ep, er) <= tolerance:
            return NumericalResult(q, iteration, "converged", ep, er)
        if iteration == max_iterations:
            return NumericalResult(q, iteration, "max_iterations", ep, er)
        J = DifferentialKinematics(q, robot)
        step = np.linalg.solve(J.T @ J + 1e-8*np.eye(6), J.T @ error)
        step *= min(1, 0.35 / max(np.linalg.norm(step), 1e-15))
        cost = np.linalg.norm(error)
        for scale in (1, .5, .25, .125, .0625, .03125, .015625):
            candidate = np.clip(q + scale*step, limits[:, 0], limits[:, 1])
            if np.linalg.norm(_error(target, FK(candidate, robot))) < cost - 1e-13:
                q = candidate
                break
        else:
            return NumericalResult(q, iteration + 1, "no_descent_step", ep, er)


def nearest_branch(q, branches):
    if not branches:
        return "none", float("nan")
    distances = [np.linalg.norm([wrap(a-b) for a,b in zip(q, branch)]) for branch in branches]
    i = int(np.argmin(distances))
    return str(i + 1), distances[i]


def report():
    lines = ["## Task 4 — numerical IK", "",
             "Projected damped least squares with backtracking; pose error uses base-frame "
             "position and rotation vector log(R_target R_currentᵀ). Joint limits are closed. "
             "Convergence: both residual norms ≤ 1e−8; at most 120 iterations. "
             "Nearest branch is among feasible isolated branches from Task 3; "
             "a feasible continuous family is named instead. "
             "`max_iterations` means the tolerance was not met in 120 steps; "
             "`no_descent_step` means none of seven projected trial steps reduced the pose error. "
             "Source: `src/task4_numerical_ik.py`; test: `tests/test_tasks4_to_6.py`.", "",
             "| arm | target | initial q (deg) | status | iterations | final q (deg) | position (m) | orientation (rad) | nearest branch / distance (rad) |",
             "|---|---|---|---|---:|---|---:|---:|---|"]
    perturb = np.deg2rad([2, -2, 1, 2, -1, 1])
    for filename in ("robot_model.json", "robot_model_left.json"):
        robot = load_robot(filename)
        l2 = next(s["length_m"] for s in robot["chain"] if s.get("name") == "l2")
        l3 = next(s["length_m"] for s in robot["chain"] if s.get("name") == "l3")
        axial = np.array([0, 0, np.rad2deg(np.arcsin(robot["b"]*l2/l3)), 0, 0, 0])
        cases = {**CONFIGURATIONS_DEG, "two_branches": DUAL_DEG,
                 "wrist_singular": [0,0,0,0,90,0],
                 "axial": axial, "arm_degenerate": [0,0,90,0,0,0]}
        limits = np.deg2rad([s["limits_deg"] for s in robot["chain"] if s["kind"] == "joint"])
        for name, degrees in cases.items():
            target = FK(np.deg2rad(degrees), robot, check_limits=False)
            analytic = solve_ik(target, robot)
            starts = [np.deg2rad(degrees)]
            if name == "two_branches":
                starts = analytic.feasible
            for seed in starts:
                initial = np.clip(seed + perturb, limits[:, 0], limits[:, 1])
                result = numerical_ik(target, robot, initial)
                branch, distance = nearest_branch(result.q, analytic.feasible)
                families = [f["free"] for f in analytic.families if f["potentially_feasible"]]
                near = (f"{branch} / {distance:.2e}" if branch != "none" else
                        "family " + ",".join(families) if families else "none")
                lines.append(f"| {robot['arm_side']} | {name} | "
                             f"{np.rad2deg(initial).round(4).tolist()} | {result.status} | "
                             f"{result.iterations} | {np.rad2deg(result.q).round(4).tolist()} | "
                             f"{result.position_error:.2e} | {result.orientation_error:.2e} | {near} |")
        far = np.eye(4)
        far[0, 3] = 2
        result = numerical_ik(far, robot, np.zeros(6))
        lines.append(f"| {robot['arm_side']} | unreachable p=(2,0,0) m | [0,0,0,0,0,0] | "
                     f"{result.status} | {result.iterations} | {np.rad2deg(result.q).round(4).tolist()} | "
                     f"{result.position_error:.2e} | {result.orientation_error:.2e} | none |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(report())
