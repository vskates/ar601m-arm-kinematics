"""Check that the on-disk AR-601M model matches Homework Task 1."""

import json
import unittest
from pathlib import Path


class TestTask1Model(unittest.TestCase):
    def test_loaded_model(self):
        model = json.loads((Path(__file__).resolve().parents[1] / "robot_model.json").read_text(encoding="utf-8"))
        chain = model["chain"]
        self.assertEqual((model["arm_side"], model["b"], model["translation_frame"]),
                         ("right", 1, "current_local"))
        self.assertEqual([step["name"] for step in chain],
                         ["q1", "l1", "alpha", "q2", "l2", "q3", "l3", "q4", "q5", "q6", "l4"])

        joints = [step for step in chain if step["kind"] == "joint"]
        expected = [("q1", "z", [-90, 90]), ("q2", "y", [-105, 105]),
                    ("q3", "x", [-45, 45]), ("q4", "z", [-90, 90]),
                    ("q5", "y", [-90, 90]), ("q6", "x", [-90, 90])]
        self.assertEqual(len(joints), 6)
        self.assertEqual([(j["name"], j["type"], j["axis"], j["limits_deg"]) for j in joints],
                         [(name, "revolute", axis, limits) for name, axis, limits in expected])
        self.assertEqual([(s["name"], s["direction"], s["length_m"])
                          for s in chain if s["kind"] == "offset"],
                         [("l1", "+z", 0.40), ("l2", "-y", 0.18),
                          ("l3", "-z", 0.30), ("l4", "+x", 0.12)])
        self.assertEqual([s for s in chain if s["kind"] == "fixed_rotation"],
                         [{"kind": "fixed_rotation", "name": "alpha", "axis": "z", "angle_deg": 10}])

    def test_left_model_mirrors_right(self):
        root = Path(__file__).resolve().parents[1]
        right = json.loads((root / "robot_model.json").read_text(encoding="utf-8"))
        left = json.loads((root / "robot_model_left.json").read_text(encoding="utf-8"))
        self.assertEqual((left["arm_side"], left["b"]), ("left", -1))
        self.assertEqual([s["name"] for s in left["chain"]],
                         [s["name"] for s in right["chain"]])
        for a, b in zip(right["chain"], left["chain"]):
            if a["name"] == "l2":
                self.assertEqual((a["direction"], b["direction"]), ("-y", "+y"))
            elif a["name"] == "alpha":
                self.assertEqual((a["angle_deg"], b["angle_deg"]), (10, -10))
            else:
                self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
