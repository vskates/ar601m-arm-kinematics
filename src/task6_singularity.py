"""Feasible-joint singularity map and fixed-orientation Cartesian IK count slice."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from task2_fk import FK, load_robot
from task3_ik import DUAL_DEG, solve_ik
from task5_jacobian import DifferentialKinematics
from visualize import draw_arm


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
THRESHOLD = 1e-3


def map_joint_space(robot):
    limits = np.deg2rad([s["limits_deg"] for s in robot["chain"] if s["kind"] == "joint"])
    grids = [np.linspace(lo, hi, 5) for lo, hi in limits]
    length_scale = sum(s["length_m"] for s in robot["chain"]
                       if s["kind"] == "offset" and s["name"] != "l1")
    metric = np.empty((5,)*6)
    for index in product(range(5), repeat=6):
        q = np.array([grids[i][j] for i, j in enumerate(index)])
        J = DifferentialKinematics(q, robot)
        J[:3] /= length_scale
        metric[index] = np.linalg.svd(J, compute_uv=False)[-1]
    fraction = (metric <= THRESHOLD).mean(axis=(0,1,3,5))
    median = np.median(metric, axis=(0,1,3,5))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    extent = [-90, 90, -45, 45]
    for ax, data, label, cmap in ((axs[0], fraction, "fraction with σₘᵢₙ ≤ 10⁻³", "magma"),
                                  (axs[1], median, "median σₘᵢₙ", "viridis")):
        img = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap=cmap,
                        interpolation="nearest", vmin=0, vmax=1 if ax is axs[0] else None)
        ax.set(xlabel="q₅ (degrees)", ylabel="q₃ (degrees)", title=label)
        fig.colorbar(img, ax=ax, label=label)
    fig.savefig(FIGURES / "joint_singularity_map.png", dpi=160)
    plt.close(fig)
    return metric, length_scale


def map_workspace(robot):
    target = FK(np.deg2rad(DUAL_DEG), robot)
    xs = np.linspace(-.8, .8, 41)
    zs = np.linspace(-.15, .95, 29)
    y = target[1, 3]
    counts = np.zeros((len(zs), len(xs)), dtype=int)
    for iz, z in enumerate(zs):
        for ix, x in enumerate(xs):
            T = target.copy()
            T[:3, 3] = [x, y, z]
            try:
                result = solve_ik(T, robot)
                counts[iz, ix] = -1 if any(f["potentially_feasible"] for f in result.families) else len(result.feasible)
            except (ArithmeticError, ValueError, FloatingPointError):
                counts[iz, ix] = -2
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    maximum = max(1, int(counts.max()))
    image = ax.imshow(np.ma.masked_less(counts, 0), origin="lower", aspect="auto",
                      extent=[xs[0], xs[-1], zs[0], zs[-1]], cmap="viridis",
                      vmin=0, vmax=maximum, interpolation="nearest")
    cbar = fig.colorbar(image, ax=ax, label="distinct feasible IK configurations")
    cbar.set_ticks(range(maximum + 1))
    ax.scatter([target[0,3]], [target[2,3]], marker="*", s=110, color="red", label="known two-branch target")
    ax.scatter([xs[-1]], [zs[-1]], marker="x", color="white", label="unreachable sample (0)")
    for value, marker, color, label in ((-1, "o", "magenta", "continuous family"),
                                       (-2, "x", "red", "unresolved")):
        ij = np.argwhere(counts == value)
        ax.scatter(xs[ij[:,1]] if len(ij) else [], zs[ij[:,0]] if len(ij) else [],
                   marker=marker, color=color,
                   label=label if len(ij) else label + " (none sampled)")
    ax.set(xlabel="tool x (m)", ylabel="tool z (m)",
           title=f"{robot['arm_side']} arm; fixed y={y:.3f} m; fixed orientation from q={list(DUAL_DEG)}°")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(FIGURES / "workspace_ik_count.png", dpi=160)
    plt.close(fig)
    return counts, xs, zs, target


def report():
    FIGURES.mkdir(exist_ok=True)
    robot = load_robot(ROOT / "robot_model.json")
    metric, L = map_joint_space(robot)
    counts, xs, zs, target = map_workspace(robot)
    q = np.deg2rad([0, 0, 0, 0, 90, 0])
    J = DifferentialKinematics(q, robot)
    Js = J.copy()
    Js[:3] /= L
    _, singular_values, Vt = np.linalg.svd(Js)
    null_residual = np.linalg.norm(Js @ Vt[-1])
    draw_arm(q, robot, FIGURES / "singular_stick_figure.png", "Right arm: q=[0,0,0,0,90,0]°")
    plt.close("all")
    return ("## Task 6 — singularities and workspace\n\n"
            f"Right arm, b=+1; all six closed joint limits from `robot_model.json`. "
            f"Five equally spaced values per joint (including bounds), 5⁶={metric.size} samples. "
            f"Metric: σₘᵢₙ(diag(1/L,1/L,1/L,1,1,1)J), L=l₂+l₃+l₄={L:.2f} m; "
            f"near-singular threshold ≤{THRESHOLD:.0e} (dimensionless). "
            f"Near-singular samples: {np.count_nonzero(metric <= THRESHOLD)}; "
            f"minimum σₘᵢₙ={metric.min():.3e}.\n\n"
            "![Joint-space map](figures/joint_singularity_map.png)\n\n"
            "At q=[0,0,0,0,90,0]°: "
            f"rank(J_scaled, tol=1e−8)={np.linalg.matrix_rank(Js, tol=1e-8)}; "
            f"σₘᵢₙ={singular_values[-1]:.3e}; "
            f"||J_scaled v_null||₂={null_residual:.3e}. "
            "The stick figure labels all six axes and four fixed offsets.\n\n"
            "![Singular configuration](figures/singular_stick_figure.png)\n\n"
            f"Workspace slice: right arm; y={target[1,3]:.9f} m; "
            f"orientation R=FK({list(DUAL_DEG)}°)[:3,:3] (numerical matrix below); "
            f"x∈[{xs[0]:.2f},{xs[-1]:.2f}] m in {len(xs)} steps, "
            f"z∈[{zs[0]:.2f},{zs[-1]:.2f}] m in {len(zs)} steps, "
            f"{counts.size} targets. Closed limits as above; IK pose residual ≤1e−8 "
            f"and duplicate angle tolerance 1e−8 rad. Maximum finite count={counts.max()}, "
            f"zero count={np.count_nonzero(counts == 0)}, "
            f"continuous={np.count_nonzero(counts == -1)}, "
            f"unresolved={np.count_nonzero(counts == -2)}. "
            f"Far upper-right sampled point ({xs[-1]:.2f},{target[1,3]:.3f},{zs[-1]:.2f}) m "
            f"has count {counts[-1,-1]}.\n\n"
            "```text\nR =\n" + np.array2string(target[:3,:3], precision=9) + "\n```\n\n"
            "![Cartesian IK counts](figures/workspace_ik_count.png)\n\n"
            "Different joint-space samples may reach the same Cartesian point; a Cartesian "
            "plot can therefore overlay branches and is not a one-to-one representation of joint space. "
            "Source: `src/task6_singularity.py`; Jacobian verification: "
            "`src/task5_jacobian.py`, `tests/test_tasks4_to_6.py`.\n")


if __name__ == "__main__":
    print(report())
