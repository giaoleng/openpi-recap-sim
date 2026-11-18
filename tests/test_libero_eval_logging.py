"""只使用标准库验证 LIBERO 评测记录的最小正确性。"""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/libero/eval_logging.py"
spec = importlib.util.spec_from_file_location("libero_eval_logging", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvaluationLoggingTests(unittest.TestCase):
    def recorder(self, root):
        return module.EvaluationRecorder(
            output_dir=root, model_label="Mixed SFT", suite="libero_spatial",
            seed=7, max_steps=280, replan_steps=5,
        )

    def test_csv_json_and_error_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with module.EvaluationRecorder(
                output_dir=tmp, model_label="Mixed SFT", suite="libero_spatial",
                seed=7, max_steps=280, replan_steps=5,
                run_metadata={
                    "checkpoint_path": "/weights/sft/step_12000",
                    "checkpoint_step": 12000, "git_commit": "a" * 40,
                    "action_horizon": 15,
                },
            ) as r:
                success_video = r.video_path(tmp, 0, 0, "success")
                other_video = r.video_path(tmp, 0, 1, "failure")
                self.assertNotEqual(success_video.name, other_video.name)
                self.assertIn("task00_ep000", success_video.name)
                r.record(task_id=0, episode_id=0, init_state_index=0,
                         status="success", steps=42, video_path=str(success_video))
                r.record(task_id=0, episode_id=1, init_state_index=1,
                         status="failure", steps=280)
                r.record(task_id=0, episode_id=2, init_state_index=2,
                         status="error", steps=12, error=RuntimeError("test outage"))
                csv_path, json_path = r.csv_path, r.summary_path
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["status"] for row in rows], ["success", "failure", "error"])
            self.assertEqual(rows[-1]["error_type"], "RuntimeError")
            self.assertEqual(rows[0]["init_state_index"], "0")
            summary = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "incomplete_errors")
            self.assertEqual(summary["run_metadata"]["checkpoint_step"], 12000)
            self.assertEqual(summary["run_metadata"]["action_horizon"], 15)
            self.assertEqual(summary["total"]["successful"], 1)
            self.assertEqual(summary["total"]["failed"], 1)
            self.assertEqual(summary["total"]["errors"], 1)
            self.assertEqual(summary["total"]["valid_episodes"], 2)
            self.assertEqual(summary["total"]["success_rate_valid"], 0.5)
            self.assertEqual(summary["total"]["success_rate_all_attempts"], 1/3)

    def test_interruption_is_not_marked_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                with self.recorder(tmp) as r:
                    r.record(task_id=0, episode_id=0, init_state_index=0, status="success", steps=4)
                    path = r.summary_path
                    raise RuntimeError("unexpected interruption")
            summary = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(summary["total"]["attempted"], 1)

    def test_invalid_row_not_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.recorder(tmp) as r:
                with self.assertRaises(ValueError):
                    r.record(task_id=0, episode_id=0, init_state_index=0,
                             status="error", steps=0)

    def test_main_code_is_parseable(self):
        ast.parse((ROOT / "examples/libero/main.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
