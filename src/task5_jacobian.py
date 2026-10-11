"""Spatial tool Jacobian: [linear velocity in m/s; angular velocity in rad/s]."""

import numpy as np

from task2_fk import FK, load_robot, step_transform


def _skew(axis):
    x, y, z = axis
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def chain_frames(q, robot):
    """Positions of offsets and joint axis lines, all in the base frame."""
    T = np.eye(4)
    origins, axes, offsets = [], [], []
    k = 0
    for step in robot["chain"]:
        if step["kind"] == "joint":
            origins.append(T[:3, 3].copy())
            axes.append(T[:3, :3] @ np.eye(3)["xyz".index(step["axis"])])
            T = T @ step_transform(step, q[k])
            k += 1
        elif step["kind"] == "offset":
            start = T[:3, 3].copy()
            T = T @ step_transform(step)
            offsets.append((step["name"], start, T[:3, 3].copy()))
        else:
            T = T @ step_transform(step)
    return T, np.array(origins), np.array(axes), offsets


def DifferentialKinematics(q, robot):
    """Geometric derivation: v_i = a_i × (p_tool-p_i), omega_i = a_i."""
    T, origins, axes, _ = chain_frames(q, robot)
    return np.vstack((np.cross(axes, T[:3, 3] - origins).T, axes.T))


def jacobian_product(q, robot):
    """Independent derivation by differentiating the complete matrix product."""
    parts = []
    k = 0
    for step in robot["chain"]:
        parts.append(step_transform(step, q[k] if step["kind"] == "joint" else 0))
        k += step["kind"] == "joint"
    prefix = [np.eye(4)]
    for part in parts:
        prefix.append(prefix[-1] @ part)
    suffix = [np.eye(4) for _ in range(len(parts) + 1)]
    for i in range(len(parts) - 1, -1, -1):
        suffix[i] = parts[i] @ suffix[i + 1]
    R = prefix[-1][:3, :3]
    J = np.zeros((6, 6))
    k = 0
    for i, step in enumerate(robot["chain"]):
        if step["kind"] != "joint":
            continue
        G = np.zeros((4, 4))
        G[:3, :3] = _skew(np.eye(3)["xyz".index(step["axis"])])
        dT = prefix[i] @ G @ parts[i] @ suffix[i + 1]
        W = dT[:3, :3] @ R.T
        J[:, k] = [*dT[:3, 3], W[2, 1], W[0, 2], W[1, 0]]
        k += 1
    return J


def jacobian_fd(q, robot, h=1e-6):
    """Centered FK difference, with omega = vee(dR/dq R.T)."""
    q = np.asarray(q, dtype=float)
    R = FK(q, robot, check_limits=False)[:3, :3]
    J = np.zeros((6, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = h
        Tp = FK(q + dq, robot, check_limits=False)
        Tm = FK(q - dq, robot, check_limits=False)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2*h)
        W = (Tp[:3, :3] - Tm[:3, :3]) @ R.T / (2*h)
        W = (W - W.T) / 2
        J[3:, i] = [W[2, 1], W[0, 2], W[1, 0]]
    return J


if __name__ == "__main__":
    robot = load_robot()
    for name, degrees in {"generic": [20, -35, 10, 25, 40, -15],
                          "near_wrist_singularity": [0, 0, 0, 0, 89.9, 0]}.items():
        q = np.deg2rad(degrees)
        A, B, C = DifferentialKinematics(q, robot), jacobian_product(q, robot), jacobian_fd(q, robot)
        print(name, degrees, "||A-B||F, ||A-C||F, ||B-C||F =",
              *(f"{np.linalg.norm(X-Y):.3e}" for X, Y in ((A,B),(A,C),(B,C))))
