"""Task 3: analytical inverse kinematics for either arm (angles in radians)."""

from dataclasses import dataclass
from math import asin, atan2, ceil, floor, hypot, pi, sqrt
from pathlib import Path

import numpy as np

from task2_fk import CONFIGURATIONS_DEG, FK, load_robot, rotation


POS_TOL = 1e-10       # metres, for reachability and coincident positions
ANG_TOL = 1e-9        # radians, for joint-limit rounding
DUP_TOL = 1e-8        # radians modulo 2*pi
POSE_TOL = 1e-8       # Frobenius norm for poses and rotations
DUAL_DEG = (10, 5, -30, -40, -30, 20)


@dataclass
class IKResult:
    real: list
    feasible: list
    families: list
    status: str


class InfiniteSolutions(ValueError):
    """A finite list cannot contain every solution of a continuous IK family."""

    def __init__(self, result):
        self.result = result
        super().__init__("IK has continuous solution families; inspect result.families")


def wrap(angle):
    return (angle + pi) % (2 * pi) - pi


def offset(step):
    direction = step["direction"]
    vector = np.zeros(3)
    vector["xyz".index(direction[1])] = {"+": 1, "-": -1}[direction[0]] * step["length_m"]
    return vector


def feasible_copy(q, limits):
    if not np.all((limits[:, 0] - ANG_TOL <= q) & (q <= limits[:, 1] + ANG_TOL)):
        return None
    answer = q.copy()
    for i in range(6):
        for bound in limits[i]:
            if abs(answer[i] - bound) <= ANG_TOL:
                answer[i] = bound
    return answer


def wrist_intervals(family, limits):
    """Exact feasible q6 intervals for q5 = +/-90 degrees."""
    fixed = [(0, family["q1_rad"]), (1, family["q2_rad"]),
             (2, family["q3_rad"]), (4, family["q5_rad"])]
    if any(not limits[i, 0] - ANG_TOL <= value <= limits[i, 1] + ANG_TOL
           for i, value in fixed):
        return []
    s = 1 if family["q5_rad"] > 0 else -1
    delta = family["delta_rad"]
    lower4, upper4 = limits[3]
    lower6, upper6 = limits[5]
    minimum = min(delta + s * lower6, delta + s * upper6)
    maximum = max(delta + s * lower6, delta + s * upper6)
    first = ceil((lower4 - maximum) / (2 * pi) - ANG_TOL)
    last = floor((upper4 - minimum) / (2 * pi) + ANG_TOL)
    intervals = []
    for k in range(first, last + 1):
        a, b = sorted(((lower4 - delta - 2 * pi * k) / s,
                       (upper4 - delta - 2 * pi * k) / s))
        low, high = max(lower6, a), min(upper6, b)
        if low <= high + ANG_TOL:
            intervals.append([low, high])
    return intervals


def solve_ik(target, robot, *, fixed_q1=None, fixed_q2=None):
    """Return every isolated branch before/after limits and any continuous families."""
    target = np.asarray(target, dtype=float)
    if target.shape != (4, 4) or not np.all(np.isfinite(target)):
        raise ValueError("Target must be a finite 4x4 pose")
    R, p = target[:3, :3], target[:3, 3]
    if (np.linalg.norm(target[3] - [0, 0, 0, 1]) > POSE_TOL
            or np.linalg.norm(R.T @ R - np.eye(3), ord="fro") > POSE_TOL
            or abs(np.linalg.det(R) - 1) > POSE_TOL):
        raise ValueError("Target must belong to SE(3)")

    chain = robot["chain"]
    offsets = {s["name"]: offset(s) for s in chain if s["kind"] == "offset"}
    joints = [s for s in chain if s["kind"] == "joint"]
    limits = np.deg2rad([s["limits_deg"] for s in joints])
    alpha = np.deg2rad(next(s["angle_deg"] for s in chain if s["kind"] == "fixed_rotation"))
    C, side_offset = offsets["l1"], offsets["l2"]
    l2, l3 = np.linalg.norm(side_offset), np.linalg.norm(offsets["l3"])
    if (len(joints) != 6 or l2 <= 0 or l3 <= 0 or side_offset[0] != 0
            or side_offset[2] != 0 or not np.allclose(offsets["l3"], [0, 0, -l3])):
        raise ValueError("Model does not match the Task 1 arm geometry")

    W = p - R @ offsets["l4"]
    r = W - C
    x, y, _ = r
    rho = hypot(x, y)
    rhs = -(r @ r + l2 * l2 - l3 * l3) / (2 * side_offset[1])
    result = IKResult([], [], [], "ok")
    if rho < POS_TOL:
        if abs(rhs) > POS_TOL:
            result.status = "unreachable"
            return result
        if fixed_q1 is None:
            result.status = "continuous"
            uy, uz = -side_offset[1], r[2]
            root = sqrt(max(0.0, l3 * l3 - uy * uy))
            arm_possible = any(
                limits[1, 0] - ANG_TOL <= atan2(0.0, -sigma * uz) <= limits[1, 1] + ANG_TOL
                and limits[2, 0] - ANG_TOL <= atan2(uy, sigma * root) <= limits[2, 1] + ANG_TOL
                for sigma in (1, -1))
            result.families.append({"free": "q1", "limit_interval_rad": limits[0].tolist(),
                                    "potentially_feasible": arm_possible,
                                    "rule": "q1 is free; for each q1, solve q2,q3 and the wrist"})
            return result
    if abs(rhs) > rho + POS_TOL:
        result.status = "unreachable"
        return result

    if fixed_q1 is not None:
        theta_values = [fixed_q1 + alpha]
    elif rho < POS_TOL:
        theta_values = []  # fixed_q1 is required in the axial case
    else:
        phi = atan2(y, x)
        if abs(abs(rhs) - rho) <= POS_TOL:
            theta_values = [phi + (pi / 2 if rhs > 0 else -pi / 2)]
        else:
            a = asin(np.clip(rhs / rho, -1, 1))
            theta_values = [phi + a, phi + pi - a]

    for theta_raw in theta_values:
        q1 = wrap(theta_raw - alpha)
        theta = q1 + alpha
        u = rotation("z", -theta)[:3, :3] @ r - side_offset
        if hypot(u[0], u[2]) < POS_TOL:
            if fixed_q2 is None:
                result.families.append({"free": "q2", "limit_interval_rad": limits[1].tolist(),
                                        "q1_rad": q1,
                                        "q3_rad": pi / 2 if u[1] > 0 else -pi / 2,
                                        "feasible_intervals_rad": [],
                                        "potentially_feasible": False,
                                        "rule": "q2 is free; the wrist compensates; |q3|=90° violates its limit"})
                continue
            arm_branches = [(fixed_q2, pi / 2 if u[1] > 0 else -pi / 2)]
        else:
            root = sqrt(max(0.0, l3 * l3 - u[1] * u[1]))
            arm_branches = [(atan2(-sigma * u[0], -sigma * u[2]),
                             atan2(u[1], sigma * root)) for sigma in (1, -1)]
        for q2, q3 in arm_branches:
            A = (rotation("z", theta) @ rotation("y", q2) @ rotation("x", q3))[:3, :3]
            Q = A.T @ R
            h5 = hypot(Q[0, 0], Q[1, 0])
            if h5 < ANG_TOL:
                s = 1 if Q[2, 0] < 0 else -1
                delta = atan2(s * Q[1, 2], Q[1, 1])
                family = {"free": "q6", "limit_interval_rad": limits[5].tolist(),
                          "q1_rad": q1, "q2_rad": q2, "q3_rad": q3,
                          "q5_rad": s * pi / 2, "delta_rad": delta,
                          "rule": "q4 = delta + sin(q5)*q6 (mod 2π)"}
                family["feasible_intervals_rad"] = wrist_intervals(family, limits)
                family["potentially_feasible"] = bool(family["feasible_intervals_rad"])
                result.families.append(family)
                continue
            for tau in (1, -1):
                q5 = atan2(-Q[2, 0], tau * h5)
                q4 = atan2(tau * Q[1, 0], tau * Q[0, 0])
                q6 = atan2(tau * Q[2, 1], tau * Q[2, 2])
                q = np.array([wrap(v) for v in (q1, q2, q3, q4, q5, q6)])
                if np.linalg.norm(FK(q, robot, check_limits=False) - target, ord="fro") > POSE_TOL:
                    raise ArithmeticError("Analytical IK branch does not reproduce the target")
                if any(np.max(np.abs([wrap(a - b) for a, b in zip(q, old)])) < DUP_TOL
                       for old in result.real):
                    continue
                result.real.append(q)
                feasible = feasible_copy(q, limits)
                if feasible is not None:
                    result.feasible.append(feasible)
    if result.families:
        result.status = "continuous"
    elif not result.real:
        result.status = "unreachable"
    elif not result.feasible:
        result.status = "no_feasible_branch"
    return result


def evaluate_family(family, value, target, robot):
    """Resolve a free angle; returned feasible branches include joint-limit filtering."""
    if family["free"] == "q1":
        return solve_ik(target, robot, fixed_q1=value)
    if family["free"] == "q2":
        return solve_ik(target, robot, fixed_q1=family["q1_rad"], fixed_q2=value)
    if family["free"] == "q6":
        q6 = wrap(value)
        s = 1 if family["q5_rad"] > 0 else -1
        q4 = wrap(family["delta_rad"] + s * q6)
        q = np.array([family["q1_rad"], family["q2_rad"], family["q3_rad"],
                      q4, family["q5_rad"], q6])
        if np.linalg.norm(FK(q, robot, check_limits=False) - target, ord="fro") > POSE_TOL:
            raise ArithmeticError("Parametric IK branch does not reproduce the target")
        limits = np.deg2rad([s["limits_deg"] for s in robot["chain"] if s["kind"] == "joint"])
        feasible = feasible_copy(q, limits)
        return IKResult([q], [feasible] if feasible is not None else [], [],
                        "ok" if feasible is not None else "no_feasible_branch")
    raise ValueError("Unknown IK family")


def IK(target, robot):
    """Return feasible isolated joint vectors; signal continuous families explicitly."""
    result = solve_ik(target, robot)
    if any(family["potentially_feasible"] for family in result.families):
        raise InfiniteSolutions(result)
    return result.feasible


def main():
    print("# Task 3 — analytical inverse kinematics\n")
    print("Source: `src/task3_ik.py`; targets use Task 2's three q values plus the two-branch example.")
    print("Models: `robot_model.json` (right), `robot_model_left.json` (left).")
    print(f"Duplicate rule: max_i |wrap(q_i−q'_i)| < {DUP_TOL:.0e} rad. "
          f"Pose residual tolerance: {POSE_TOL:.0e}.")
    print("Residuals: e_p = ||p_target−p_FK||₂ (m), e_R = ||R_target−R_FK||_F, "
          "e_T = ||T_target−T_FK||_F.\n")
    print("Residuals use unrounded matrices; displayed numbers are rounded.\n")
    for filename in ("robot_model.json", "robot_model_left.json"):
        robot = load_robot(Path(__file__).resolve().parents[1] / filename)
        print(f"## {robot['arm_side']} arm, b = {robot['b']}\n")
        for name, degrees in {**CONFIGURATIONS_DEG, "two_branches": DUAL_DEG}.items():
            target = FK(np.deg2rad(degrees), robot)
            result = solve_ik(target, robot)
            print(f"### {name}: target from q = {list(degrees)}°\n")
            print("Target T =\n```text")
            print(np.array2string(target, formatter={"float_kind": lambda v: f"{v: .9f}"}))
            print("```")
            print(f"Status: {result.status}; real branches: {len(result.real)}; "
                  f"feasible branches: {len(result.feasible)}.")
            print("| # | q1, q2, q3, q4, q5, q6 (degrees) | feasible | e_p (m) | e_R | e_T |")
            print("|---:|---|:---:|---:|---:|---:|")
            for i, q in enumerate(result.real, 1):
                predicted = FK(q, robot, check_limits=False)
                ep = np.linalg.norm(target[:3, 3] - predicted[:3, 3])
                er = np.linalg.norm(target[:3, :3] - predicted[:3, :3], ord="fro")
                et = np.linalg.norm(target - predicted, ord="fro")
                feasible = any(np.max(np.abs([wrap(a - b) for a, b in zip(q, f)])) < DUP_TOL
                               for f in result.feasible)
                angles = ", ".join(f"{v:.6f}" for v in np.rad2deg(q))
                print(f"| {i} | {angles} | {'yes' if feasible else 'no'} | "
                      f"{ep:.2e} | {er:.2e} | {et:.2e} |")
            print()
        unreachable = np.eye(4)
        unreachable[0, 3] = 2.0
        print("### unreachable target: p = (2, 0, 0) m, R = I\n")
        print(f"Status: {solve_ik(unreachable, robot).status}; branches: 0.\n")
        singular = np.deg2rad([0, 0, 0, 0, 90, 0])
        singular_target = FK(singular, robot)
        singular_result = solve_ik(singular_target, robot)
        print("### wrist singularity: q5 = +90°\n")
        print(f"Status: {singular_result.status}; {len(singular_result.families)} real families:")
        for i, family in enumerate(singular_result.families, 1):
            fixed = np.rad2deg([family[f"q{j}_rad"] for j in (1, 2, 3, 5)])
            intervals = [np.rad2deg(pair).round(6).tolist()
                         for pair in family["feasible_intervals_rad"]]
            sample = evaluate_family(family, 0, singular_target, robot)
            sample_error = np.linalg.norm(FK(sample.real[0], robot, check_limits=False)
                                          - singular_target, ord="fro")
            print(f"{i}. (q1,q2,q3,q5) = {fixed.round(6).tolist()}°; "
                  f"q4 = {np.rad2deg(family['delta_rad']):.6f}° + "
                  f"sin(q5)·q6; feasible q6 intervals = {intervals}°; "
                  f"sample q6=0 residual = {sample_error:.2e}.")
        print()

        side = next(offset(s) for s in robot["chain"] if s["name"] == "l2")
        arm = next(offset(s) for s in robot["chain"] if s["name"] == "l3")
        axial_q = np.array([0, 0, np.arcsin(-side[1] / np.linalg.norm(arm)), 0, 0, 0])
        axial_target = FK(axial_q, robot)
        axial = solve_ik(axial_target, robot)
        axial_sample = evaluate_family(axial.families[0], 0, axial_target, robot)
        axial_error = np.linalg.norm(FK(axial_sample.real[0], robot, check_limits=False)
                                     - axial_target, ord="fro")
        print("### axial wrist-centre degeneracy: W lies on the base z axis\n")
        print(f"Status: {axial.status}; family: q1 free, joint interval [-90°,90°]. "
              f"At q1=0: {len(axial_sample.feasible)} feasible branch(es), "
              f"sample residual = {axial_error:.2e}.\n")

        arm_singular = np.deg2rad([0, 0, 90, 0, 0, 0])
        arm_target = FK(arm_singular, robot, check_limits=False)
        arm_result = solve_ik(arm_target, robot)
        print("### arm degeneracy before filtering: q3 = +90°\n")
        print(f"Status: {arm_result.status}; {len(arm_result.families)} real family/families: "
              f"q2 free; feasible branches: {len(arm_result.feasible)} "
              f"because |q3| > 45°.\n")


if __name__ == "__main__":
    main()
