"""Angle-to-pose and pose-to-all-IK-branches visualizer (angles in degrees on CLI)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from task2_fk import FK, load_robot, rotation
from task3_ik import solve_ik
from task5_jacobian import chain_frames


ROOT = Path(__file__).resolve().parents[1]


def draw_arm(q, robot, path, title=""):
    T, origins, axes, offsets = chain_frames(q, robot)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_axes([.02, .08, .59, .82], projection="3d")
    key = fig.add_axes([.63, .10, .34, .80])
    key.axis("off")
    for _, start, end in offsets:
        ax.plot(*np.array([start, end]).T, color="#303030", linewidth=3)
    for i, (origin, axis) in enumerate(zip(origins, axes), 1):
        color = {"x": "#c62828", "y": "#2e7d32", "z": "#1565c0"}["zyxzyx"[i-1]]
        ax.quiver(*origin, *axis, length=.09, normalize=True, color=color, linewidth=2)
    frame_points = [offsets[0][1], *[end for _, _, end in offsets]]
    for point, label in zip(frame_points, "BCSWT"):
        ax.text(*(point + np.array([.012, .012, .015])), label, fontsize=10, weight="bold")
    ax.scatter(*np.array([offsets[0][1], *[end for _, _, end in offsets]]).T,
               color="black", s=16)
    points = np.array([offsets[0][1], *[end for _, _, end in offsets],
                       *[origin + .11*axis for origin, axis in zip(origins, axes)]])
    centre = (points.min(axis=0) + points.max(axis=0))/2
    half = max((points.max(axis=0) - points.min(axis=0))/2) + .09
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
           xlim=(centre[0]-half, centre[0]+half),
           ylim=(centre[1]-half, centre[1]+half),
           zlim=(centre[2]-half, centre[2]+half))
    ax.set_box_aspect((1, 1, 1))
    fig.suptitle(title or f"{robot['arm_side']} arm", fontsize=15)
    entries = [
        ("FRAMES", "#222222", True),
        ("{base}, {1}: B     {2}: C     {3}: S", "#333333", False),
        ("{4}, {5}, {6}: W     {tool}: T", "#333333", False),
        ("JOINT AXES", "#222222", True),
    ]
    for i, (axis, point) in enumerate(zip("zyxzyx", "BCSWWW"), 1):
        entries.append((f"q{i}: {axis} at {point}",
                        {"x": "#c62828", "y": "#2e7d32", "z": "#1565c0"}[axis], False))
    entries.append(("FIXED OFFSETS (LOCAL)", "#222222", True))
    for step, points in zip((s for s in robot["chain"] if s["kind"] == "offset"),
                            ("B→C", "C→S", "S→W", "W→T")):
        entries.append((f"{step['name']}: {points}   {step['direction']}   {step['length_m']:.2f} m",
                        "#333333", False))
    alpha = next(s["angle_deg"] for s in robot["chain"] if s["kind"] == "fixed_rotation")
    entries.append((f"Fixed: Rz({alpha:+g}°) between q1 and q2", "#333333", False))
    for i, (label, color, heading) in enumerate(entries):
        key.text(0, .98 - i*.052, label, transform=key.transAxes,
                 fontsize=10 if heading else 9, color=color,
                 weight="bold" if heading else "normal", va="top")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    return fig, T


def pose_from_zyx(values):
    x, y, z, roll, pitch, yaw = values
    R = (rotation("z", np.deg2rad(yaw)) @ rotation("y", np.deg2rad(pitch))
         @ rotation("x", np.deg2rad(roll)))[:3, :3]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [x, y, z]
    return T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["right", "left"], default="right")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--q", nargs=6, type=float, metavar="DEG")
    mode.add_argument("--fk-target", nargs=6, type=float, metavar="DEG")
    mode.add_argument("--pose", nargs=6, type=float, metavar="X_Y_Z_R_P_Y")
    parser.add_argument("--output", default="figures/arm.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    robot = load_robot(ROOT / ("robot_model.json" if args.side == "right" else "robot_model_left.json"))
    if args.q:
        q = np.deg2rad(args.q)
        fig, T = draw_arm(q, robot, args.output, f"{args.side}: q = {args.q}°")
        print("T(base→tool) =\n", T)
    else:
        target = FK(np.deg2rad(args.fk_target), robot) if args.fk_target else pose_from_zyx(args.pose)
        result = solve_ik(target, robot)
        print("Target T =\n", target)
        print(f"{len(result.feasible)} feasible isolated branches; status={result.status}")
        for i, q in enumerate(result.feasible, 1):
            filename = str(Path(args.output).with_stem(Path(args.output).stem + f"_{i}"))
            fig, _ = draw_arm(q, robot, filename, f"{args.side}: IK branch {i}")
            print(f"{i}: {np.rad2deg(q)} deg; residual={np.linalg.norm(FK(q, robot)-target):.2e}; {filename}")
        for family in result.families:
            print("Continuous family:", family)
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
