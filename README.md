# AR-601M — Homework 1

From a clean checkout with [Pixi](https://pixi.sh) installed:

```bash
pixi run check
```

This runs the tests and regenerates [REPORT.md](REPORT.md) and every image in `figures/`. The report covers Tasks 1–6 in order. Joint angles in the Python API are radians; CLI angles are degrees; lengths are metres.

- `robot_model.json`, `robot_model_left.json`: six-joint models for right/left arms.
- `src/kinematics.py`: required `FK`, `IK`, `DifferentialKinematics` interface; `src/task*.py`: solvers and analyses; `tests/`: automated checks.
- `task1_interactive_3d.html`: rotatable frame diagram; `src/visualize.py`: Task 2/3 angle and pose visualizer. Examples:

```bash
pixi run python src/visualize.py --q 20 -35 10 25 40 -15 --show
pixi run python src/visualize.py --fk-target 10 5 -30 -40 -30 20 --side left --show
pixi run python src/visualize.py --pose 0.2 -0.3 0.2 0 0 0 --side right --show
```

The `--pose` arguments are `x y z roll pitch yaw` in ZYX convention. Omit `--show` to save PNGs. The project adapts the [course starter files](https://github.com/Lupasic/In2Ro_MIPT_2026/tree/main/slides/homework/W01/input_data); the report records the chosen transformation chain and tests.
