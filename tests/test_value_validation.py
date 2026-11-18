"""轻量验证Episode分组与Value验证集必填参数，不下载GPU模型。"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("split_value_episodes", ROOT / "scripts/split_value_episodes.py")
split = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split)


class ValueSplitTests(unittest.TestCase):
    def test_task_stratified_split_has_no_episode_overlap(self):
        rows = ([{"episode_index": i, "tasks": ["task-a"]} for i in range(12)] +
                [{"episode_index": i, "tasks": ["task-b"]} for i in range(12, 20)])
        a, b = split.split_episode_ids(rows, 0.2, 42)
        self.assertEqual(set(a) | set(b), set(range(20)))
        self.assertFalse(set(a) & set(b))
        self.assertEqual((a, b), split.split_episode_ids(rows, 0.2, 42))
        self.assertEqual(sum(x >= 12 for x in b), 2)

    def test_reject_invalid_ratio_and_duplicate_id(self):
        rows = [{"episode_index": 0, "tasks": ["t"]}, {"episode_index": 0, "tasks": ["t"]}]
        with self.assertRaisesRegex(ValueError, "重复"):
            split.split_episode_ids(rows, 0.2, 7)
        with self.assertRaises(ValueError):
            split.split_episode_ids([], 1, 7)

    def test_value_training_requires_validation_for_formal_runs(self):
        script = ROOT / "scripts/train_value.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_training_args")
        isolated = ast.Module(body=[definition], type_ignores=[])
        namespace = {"Path": Path, "json": json}
        exec(compile(ast.fix_missing_locations(isolated), str(script), "exec"), namespace)
        verify = namespace["validate_training_args"]
        args = SimpleNamespace(batch_size=4, num_train_steps=5, log_interval=1, save_interval=5,
                               data_dir="/data/train", val_data_dir=None, allow_no_val=False, val_interval=1)
        with self.assertRaisesRegex(ValueError, "val_data_dir"):
            verify(args)
        args.allow_no_val = True
        verify(args)
        args.val_data_dir = "/data/train"
        with self.assertRaises(ValueError):
            verify(args)
        args.val_data_dir = "/data/val"
        args.val_interval = 0
        with self.assertRaises(ValueError):
            verify(args)

    def test_metadata_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "train").mkdir()
            (Path(tmp) / "val").mkdir()
            manifest = {"train_episode_ids": [1, 2], "val_episode_ids": [2]}
            (Path(tmp) / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            script = ROOT / "scripts/train_value.py"
            tree = ast.parse(script.read_text(encoding="utf-8"))
            definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_training_args")
            namespace = {"Path": Path, "json": json}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])), str(script), "exec"), namespace)
            args = SimpleNamespace(batch_size=4, num_train_steps=5, log_interval=1, save_interval=5,
                                   data_dir=str(Path(tmp) / "train"), val_data_dir=str(Path(tmp) / "val"),
                                   allow_no_val=False, val_interval=1)
            with self.assertRaisesRegex(ValueError, "Episode"):
                namespace["validate_training_args"](args)


if __name__ == "__main__":
    unittest.main()
