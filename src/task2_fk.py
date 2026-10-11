"""Task 2: forward kinematics from robot_model.json; joint inputs are radians."""

import json
from math import atan2, cos, hypot, sin
from pathlib import Path

import numpy as np


MODEL_FILE = Path(__file__).resolve().parents[1] / "robot_model.json"
CONFIGURATIONS_DEG = {
    "zero": (0, 0, 0, 0, 0, 0),
    "at_limit": (90, 0, 0, 0, 0, 0),
    "generic": (20, -35, 10, 25, 40, -15),
}
TOL = 1e-12


def load_robot(path=MODEL_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rotation(axis, angle):
    """Homogeneous right-hand rotation around a local x, y or z axis."""
    i = ("xyz".index(axis) + 1) % 3
    j = (i + 1) % 3
    c, s = cos(angle), sin(angle)
    transform = np.eye(4)
    transform[i, i] = transform[j, j] = c
    transform[i, j], transform[j, i] = -s, s
    return transform


def step_transform(step, angle=0.0):
    if step["kind"] == "offset":
        part = np.eye(4)
        direction = step["direction"]
        part["xyz".index(direction[1]), 3] = {"+": 1, "-": -1}[direction[0]] * step["length_m"]
        return part
    if step["kind"] == "fixed_rotation":
        return rotation(step["axis"], np.deg2rad(step["angle_deg"]))
    if step["kind"] == "joint":
        return rotation(step["axis"], angle)
    raise ValueError(f"Unknown transform: {step['kind']}")


def FK(q, robot, check_limits=True):
    """Return the base-to-tool transform; q is ordered q1..q6 in radians."""
    joints = [step for step in robot["chain"] if step["kind"] == "joint"]
    q = np.asarray(q, dtype=float)
    if q.shape != (len(joints),) or not np.all(np.isfinite(q)):
        raise ValueError("q must contain one finite radian angle per joint")
    if check_limits:
        for angle, joint in zip(q, joints):
            low, high = np.deg2rad(joint["limits_deg"])
            if not low <= angle <= high:
                raise ValueError(f"{joint['name']} is outside its joint limits")

    transform = np.eye(4)
    joint_index = 0
    for step in robot["chain"]:
        transform = transform @ step_transform(step, q[joint_index] if step["kind"] == "joint" else 0)
        joint_index += step["kind"] == "joint"
    return transform


def rpy_zyx(rotation_matrix):
    """Return (roll, pitch, yaw) in radians, where R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r = rotation_matrix
    horizontal = hypot(r[0, 0], r[1, 0])
    if horizontal < TOL:
        raise ValueError("ZYX angles are not unique at this orientation")
    return np.array([atan2(r[2, 1], r[2, 2]),
                     atan2(-r[2, 0], horizontal),
                     atan2(r[1, 0], r[0, 0])])


def main():
    robot = load_robot()
    limits = np.array([step["limits_deg"] for step in robot["chain"] if step["kind"] == "joint"])
    print("# Task 2 — forward kinematics\n")
    print("Source: `src/task2_fk.py`; test: `tests/test_task2_fk.py`.")
    print(f"Model: `robot_model.json`; arm: {robot['arm_side']} (`b = {robot['b']}`).")
    print("Column vectors; ordered local offsets in metres; q in radians in FK.")
    print("ZYX convention: R = Rz(yaw) Ry(pitch) Rx(roll).")
    print(f"Checks on unrounded T: Frobenius norm ||RᵀR − I|| and |det(R) − 1|; "
          f"tolerance for each = {TOL:.0e}.\n")
    for name, degrees in CONFIGURATIONS_DEG.items():
        if name == "at_limit":
            assert np.any((degrees == limits[:, 0]) | (degrees == limits[:, 1]))
        if name == "generic":
            assert np.all((limits[:, 0] < degrees) & (degrees < limits[:, 1]))
        q = np.deg2rad(degrees)
        t = FK(q, robot)
        r = t[:3, :3]
        orthogonality = np.linalg.norm(r.T @ r - np.eye(3), ord="fro")
        determinant = abs(np.linalg.det(r) - 1)
        print(f"## {name}\n")
        print(f"q (degrees) = {list(degrees)}")
        if name == "at_limit":
            print("q1 is at its upper limit; all limits are closed intervals.")
        if name == "generic":
            print("All six angles are strictly inside their limits.")
        print("q (radians) = [" + ", ".join(f"{v:.9f}" for v in q) + "]")
        print("T =\n```text")
        print(np.array2string(t, formatter={"float_kind": lambda v: f"{v: .9f}"}))
        print("```")
        print(f"||RᵀR − I||_F = {orthogonality:.3e}; |det(R) − 1| = {determinant:.3e}; "
              f"check = {'PASS' if max(orthogonality, determinant) <= TOL else 'FAIL'}.")
        if max(orthogonality, determinant) > TOL:
            raise ArithmeticError(f"Invalid rotation for {name}")
        if name == "generic":
            roll, pitch, yaw = np.rad2deg(rpy_zyx(r))
            print(f"ZYX roll–pitch–yaw (degrees) = [{roll:.9f}, {pitch:.9f}, {yaw:.9f}].")
        print(f"![{name} arm drawing](figures/task2_{name}.png)")
        print()


if __name__ == "__main__":
    main()
