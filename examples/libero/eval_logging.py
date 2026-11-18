"""LIBERO 逐 Episode 评测记录：标准库实现，避免依赖仿真环境。"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

FIELDS = (
    "run_id", "model", "suite", "task_id", "episode_id", "init_state_index",
    "seed", "max_steps", "replan_steps", "status", "success", "steps",
    "error_type", "error_message", "video_path",
)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")[:64]
    if not safe:
        raise ValueError("model_label/suite 不能为空或只含非法字符")
    return safe


class EvaluationRecorder:
    """每次评测创建唯一 CSV + JSON；异常单独记录，不混入普通失败。"""

    def __init__(self, *, output_dir: str, model_label: str, suite: str,
                 seed: int, max_steps: int, replan_steps: int):
        self.model = _safe_name(model_label)
        self.suite = _safe_name(suite)
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.replan_steps = int(replan_steps)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
        prefix = f"{self.model}_{self.suite}_{self.run_id}"
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.csv_path = root / f"{prefix}.csv"
        self.summary_path = root / f"{prefix}.json"
        self._file = self.csv_path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self.rows: list[dict] = []
        self._closed = False

    def video_path(self, directory: str, task_id: int, episode_id: int, status: str) -> Path:
        if status not in {"success", "failure", "error"}:
            raise ValueError(f"无效的 Episode 状态: {status}")
        filename = (
            f"{self.model}_{self.suite}_{self.run_id}"
            f"_task{task_id:02d}_ep{episode_id:03d}_{status}.mp4"
        )
        return Path(directory) / filename

    def record(self, *, task_id: int, episode_id: int, init_state_index: int,
               status: str, steps: int, error: Exception | None = None,
               video_path: str = "") -> None:
        if self._closed:
            raise RuntimeError("评测记录已关闭")
        if status not in {"success", "failure", "error"}:
            raise ValueError(f"无效的 Episode 状态: {status}")
        if status == "error" and error is None:
            raise ValueError("异常 Episode 必须附带异常信息")
        row = {
            "run_id": self.run_id, "model": self.model, "suite": self.suite,
            "task_id": task_id, "episode_id": episode_id,
            "init_state_index": init_state_index, "seed": self.seed,
            "max_steps": self.max_steps, "replan_steps": self.replan_steps,
            "status": status, "success": int(status == "success"), "steps": steps,
            "error_type": type(error).__name__ if error else "",
            "error_message": str(error).replace("\r", " ").replace("\n", " ")[:500] if error else "",
            "video_path": video_path,
        }
        self._writer.writerow(row)
        self._file.flush()
        self.rows.append(row)

    @staticmethod
    def _stats(rows: list[dict]) -> dict:
        counts = {state: sum(row["status"] == state for row in rows)
                  for state in ("success", "failure", "error")}
        valid = counts["success"] + counts["failure"]
        return {
            "attempted": len(rows),
            "successful": counts["success"],
            "failed": counts["failure"],
            "errors": counts["error"],
            "valid_episodes": valid,
            "success_rate_valid": counts["success"] / valid if valid else None,
            "success_rate_all_attempts": counts["success"] / len(rows) if rows else None,
        }

    def close(self, *, interrupted: bool = False) -> None:
        if self._closed:
            return
        self._file.close()
        self._closed = True
        task_ids = sorted({int(row["task_id"]) for row in self.rows})
        summary = {
            "run_id": self.run_id, "model": self.model, "suite": self.suite,
            "seed": self.seed, "max_steps": self.max_steps,
            "replan_steps": self.replan_steps,
            "csv_path": str(self.csv_path),
            "note": "success_rate_valid excludes errors; any errors make this run incomplete.",
            "status": ("incomplete_errors" if any(row["status"] == "error" for row in self.rows)
                       else "interrupted" if interrupted else "completed"),
            "total": self._stats(self.rows),
            "tasks": {str(task_id): self._stats([row for row in self.rows if row["task_id"] == task_id])
                      for task_id in task_ids},
        }
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def __enter__(self) -> "EvaluationRecorder":
        return self

    def __exit__(self, exc_type: object, *_exc: object) -> None:
        self.close(interrupted=exc_type is not None)
