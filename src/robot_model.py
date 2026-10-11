"""Load one of the two human-readable arm models."""

from pathlib import Path

from task2_fk import load_robot


def load(side="right"):
    if side not in ("right", "left"):
        raise ValueError("side must be right or left")
    filename = "robot_model.json" if side == "right" else "robot_model_left.json"
    return load_robot(Path(__file__).resolve().parents[1] / filename)
