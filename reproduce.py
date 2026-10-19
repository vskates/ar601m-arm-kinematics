"""Run tests and regenerate every number and figure in REPORT.md."""

import contextlib
import io
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["MPLBACKEND"] = "Agg"

from task2_fk import CONFIGURATIONS_DEG, load_robot, main as fk_report
from task3_ik import DUAL_DEG, main as ik_report, solve_ik
from task4_numerical_ik import report as numerical_report
from task5_jacobian import DifferentialKinematics, jacobian_product, jacobian_fd
from task6_singularity import report as singularity_report
from visualize import draw_arm
import matplotlib.pyplot as plt


def captured(function):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        function()
    return output.getvalue()


def task1_section():
    right, left = load_robot(ROOT / "robot_model.json"), load_robot(ROOT / "robot_model_left.json")
    joints = [s for s in right["chain"] if s["kind"] == "joint"]
    rows = [f"| {j['name']} | {j['type']} | {j['axis']} | {j['limits_deg'][0]}°…{j['limits_deg'][1]}° |"
            for j in joints]
    fig, _ = draw_arm(np.zeros(6), right, ROOT / "figures/task1_frames.png",
                      "Right arm, q=0: frames, joint axes and fixed offsets")
    plt.close(fig)
    return ("## Task 1 — kinematic model\n\n"
            "Inputs: `robot_model.json` (right, b=+1) and `robot_model_left.json` "
            "(left, b=−1). Right-handed frames, column vectors, local-frame translations; "
            "angles in code are radians. The chosen factor order matches the [course starter]"
            "(https://github.com/Lupasic/In2Ro_MIPT_2026/tree/main/slides/homework/W01/input_data).\n\n"
            "`T_base→tool = Rz(q₁) Dz(l₁) Rz(b·10°) Ry(q₂) Dy(−b·l₂) Rx(q₃) "
            "Dz(−l₃) Rz(q₄) Ry(q₅) Rx(q₆) Dx(l₄)`\n\n"
            "| joint | type | local axis | closed limit |\n|---|---|---|---|\n"
            + "\n".join(rows) + "\n\n"
            "| offset | value | right direction | left direction |\n|---|---:|---|---|\n"
            "| l₁ | 0.40 m | +z | +z |\n| l₂ | 0.18 m | −y | +y |\n"
            "| l₃ | 0.30 m | −z | −z |\n| l₄ | 0.12 m | +x | +x |\n\n"
            "Fixed rotation: +10° right, −10° left, about local z between q₁ and q₂. "
            "Frames in the factor chain: {base},{1} at B; {2} at C; {3} at S; "
            "{4},{5},{6} at W; {tool} at T. Axes q₁ and q₂ lie on their respective "
            "axial offset lines, so their origin can slide along l₁ or l₂ without "
            "changing FK; the factor order above fixes their displayed origins.\n\n"
            "![Frames, axes and offsets](figures/task1_frames.png)\n\n"
            "[Rotatable 3D diagram](task1_interactive_3d.html). Model-file contents: "
            "the table and chain above reproduce every geometric parameter; "
            "both JSON files additionally store the explicit ordered transform list. "
            "`tests/test_task1_model.py` loads both files and checks six revolute joints, "
            "their order, axes, limits and mirrored geometry; exact parameter check, "
            "numeric tolerance 0.\n\n")


def task2_section():
    result = re.sub(r"(?m)^(#+)(?= )", lambda m: "#" + m.group(1), captured(fk_report))
    robot = load_robot(ROOT / "robot_model.json")
    for name, degrees in CONFIGURATIONS_DEG.items():
        fig, _ = draw_arm(np.deg2rad(degrees), robot, ROOT / f"figures/task2_{name}.png",
                          f"Right arm: {name}; q={list(degrees)}°")
        plt.close(fig)
    return (result + "Visualizer: `python src/visualize.py --q Q1 Q2 Q3 Q4 Q5 Q6 --show` "
              "prints T and displays the arm; without `--show` saves a PNG.\n\n")


def task3_section():
    derivation = r"""## Task 3 — analytical IK

Standalone derivation: [PDF](TASK3_DERIVATION.pdf) · [LaTeX source](TASK3_DERIVATION.tex).

Both arms use the same limits and lengths; b=+1 (right), b=−1 (left). For target
\(T=\begin{bmatrix}R&p\\0&1\end{bmatrix}\), the wrist centre is \(W=p-l_4Re_x\). Put
\(r=W-l_1e_z=(x,y,z)\), \(\rho=\sqrt{x^2+y^2}\),
\(\phi=\operatorname{atan2}(y,x)\), and \(\theta=q_1+b10^\circ\).
Because \(R_y(q_2)e_y=e_y\), the shoulder offset is
\(S-l_1e_z=R_z(\theta)(-bl_2e_y)\). From \(\|W-S\|=l_3\):

\[\rho\sin(\theta-\phi)=bK,\quad K=(\|r\|^2+l_2^2-l_3^2)/(2l_2).\]

For \(\rho>0\) and \(|K|\le\rho\), the two q₁ candidates are
\(\theta=\phi+\arcsin(bK/\rho)\) and
\(\theta=\phi+\pi-\arcsin(bK/\rho)\), then \(q_1=\theta-b10^\circ\).
For each, transform the wrist displacement to the shoulder axes:

\[u=R_z(-\theta)r+bl_2e_y
=l_3(-\sin q_2\cos q_3,\sin q_3,-\cos q_2\cos q_3).\]

For \(\sigma=\pm1\), take
\(q_3=\operatorname{atan2}(u_y,\sigma\sqrt{l_3^2-u_y^2})\) and
\(q_2=\operatorname{atan2}(-\sigma u_x,-\sigma u_z)\).
Finally \(Q=(R_z(\theta)R_y(q_2)R_x(q_3))^TR=R_z(q_4)R_y(q_5)R_x(q_6)\).
With \(h=\sqrt{Q_{11}^2+Q_{21}^2}>0\), each \(\tau=\pm1\) yields
\(q_5=\operatorname{atan2}(-Q_{31},\tau h)\),
\(q_4=\operatorname{atan2}(\tau Q_{21},\tau Q_{11})\),
\(q_6=\operatorname{atan2}(\tau Q_{32},\tau Q_{33})\).

All real branches are listed before applying the closed limits, then feasible
ones are listed after. A duplicate has maximum wrapped joint-angle difference
<1e−8 rad. Each branch is checked against FK with pose Frobenius tolerance
1e−8. If \(|K|>\rho\), the target is unreachable. If \(\rho=0,K=0\),
q₁ is free; if \(|u_y|=l₃\), q₂ is free (but q₃=±90° violates the limits).
If \(h=0\), q₅=\(s90^\circ\), and the continuous wrist family is
\(q_4-sq_6=\delta\pmod{2\pi}\), where
\(s=-Q_{31}\), \(\delta=\operatorname{atan2}(sQ_{23},Q_{22})\).
`solve_ik` reports such families parametrically; `IK` signals feasible
continuous families with `InfiniteSolutions`. Sources:
`src/task3_ik.py`, `tests/test_task3_ik.py`.

"""
    output = re.sub(r"(?m)^(#+)(?= )", lambda m: "#" + m.group(1),
                    captured(ik_report).split("\n", 2)[2])
    images = []
    for side, filename in (("right", "robot_model.json"), ("left", "robot_model_left.json")):
        robot = load_robot(ROOT / filename)
        from task2_fk import FK
        target = FK(np.deg2rad(DUAL_DEG), robot)
        branches = solve_ik(target, robot).feasible
        for i, q in enumerate(branches, 1):
            fig, _ = draw_arm(q, robot, ROOT / f"figures/task3_{side}_branch{i}.png",
                              f"{side} arm: target two_branches, IK branch {i}")
            plt.close(fig)
            images.append(f"![{side} branch {i}](figures/task3_{side}_branch{i}.png)")
    return derivation + output + "\n" + "\n\n".join(images) + "\n\n"


def task5_section():
    robot = load_robot(ROOT / "robot_model.json")
    rows = []
    generic_matrix = ""
    for name, degrees in (("generic", CONFIGURATIONS_DEG["generic"]),
                          ("near singular", (0, 0, 0, 0, 89.9, 0))):
        q = np.deg2rad(degrees)
        A, B, C = DifferentialKinematics(q, robot), jacobian_product(q, robot), jacobian_fd(q, robot)
        rows.append(f"| {name} | {list(degrees)} | {np.linalg.norm(A-B):.3e} | "
                    f"{np.linalg.norm(A-C):.3e} | {np.linalg.norm(B-C):.3e} |")
        if name == "generic":
            generic_matrix = np.array2string(A, formatter={"float_kind": lambda x: f"{x: .9f}"})
    return ("## Task 5 — differential kinematics\n\n"
            "Right arm, b=+1, same generic q as Task 2. Base-frame spatial twist "
            "[v;ω]=J(q)q̇ (linear m/s, angular rad/s). First derivation: for each "
            "joint axis aᵢ through pᵢ, Jᵥᵢ=aᵢ×(p_tool−pᵢ), Jωᵢ=aᵢ. "
            "Independent product derivation: differentiate each joint factor "
            "in T=F₁…Fₙ, dT/dqᵢ=F₁…Fᵢ₋₁ GᵢFᵢ…Fₙ; extract "
            "vᵢ=dp/dqᵢ and ωᵢ=vee((dR/dqᵢ)Rᵀ). "
            "FK finite difference: dp/dqᵢ≈[p(q+h eᵢ)−p(q−h eᵢ)]/(2h), "
            "ωᵢ≈vee(([R(q+h eᵢ)−R(q−h eᵢ)]/(2h))Rᵀ), h=10⁻⁶ rad. "
            "Pairwise Frobenius tolerance 10⁻⁸.\n\n"
            "| case | q (degrees) | geometric vs product | geometric vs FD | product vs FD |\n"
            "|---|---|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
            "Geometric J at generic q (top rows m/rad; bottom rows dimensionless):\n"
            f"```text\n{generic_matrix}\n```\n\n"
            "Sources: `src/task5_jacobian.py`, `tests/test_tasks4_to_6.py`.\n\n")


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    test = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    print(test.stdout, end="")
    if test.returncode:
        print(test.stderr, file=sys.stderr)
        raise SystemExit(test.returncode)
    (ROOT / "figures").mkdir(exist_ok=True)
    test_summary = re.search(r"\d+ passed(?:, \d+ subtests passed)?", test.stdout)
    report = ("# AR-601M Arm Kinematics — Technical Report\n\n"
              "MIPT Introduction to Robotics course project. "
              "[Assignment and mechanism data](https://github.com/Lupasic/In2Ro_MIPT_2026/blob/main/slides/homework/W01/resources/hw01_handout.pdf). "
              "All results below are regenerated by `pixi run check`; automated tests: "
              + test_summary.group() + ".\n\n"
              + task1_section() + task2_section() + task3_section()
              + numerical_report() + "\n" + task5_section() + singularity_report())
    (ROOT / "REPORT.md").write_text(report, encoding="utf-8")
    print("Regenerated REPORT.md and figures/.")


if __name__ == "__main__":
    main()
