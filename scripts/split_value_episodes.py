#!/usr/bin/env python3
"""按完整Episode分层切分 LeRobot v2 parquet Value 数据，避免相邻帧泄漏。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re

PARQUET_NAME = re.compile(r"^episode_(\d+)\.parquet$")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def split_episode_ids(episode_rows: list[dict], ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """固定随机种子，按任务名分层，完整Episode不跨集合。"""
    if not 0 < ratio < 1:
        raise ValueError("val_ratio必须在(0,1)之间")
    groups: dict[str, list[int]] = defaultdict(list)
    ids: set[int] = set()
    for row in episode_rows:
        ep = int(row["episode_index"])
        if ep in ids:
            raise ValueError(f"重复Episode编号: {ep}")
        ids.add(ep)
        names = row.get("tasks") or ["__unknown_task__"]
        groups[str(names[0])].append(ep)
    rng = random.Random(seed)
    train, val = [], []
    for task in sorted(groups):
        episodes = sorted(groups[task])
        rng.shuffle(episodes)
        n_val = min(len(episodes) - 1, max(1, round(len(episodes) * ratio))) if len(episodes) > 1 else 0
        val.extend(episodes[:n_val])
        train.extend(episodes[n_val:])
    train, val = sorted(train), sorted(val)
    if not train or not val:
        raise ValueError("无法构造非空train/val：至少需要一个任务包含两个Episode")
    assert len(set(train) & set(val)) == 0
    assert set(train) | set(val) == ids
    return train, val


def _write_split(src: Path, dest: Path, original_ids: list[int], rows: dict[int, dict],
                 parquet_files: dict[int, Path], info: dict, tasks: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (dest / "data").mkdir(parents=True)
    (dest / "meta").mkdir()
    chunks_size = int(info.get("chunks_size", 1000))
    if chunks_size <= 0:
        raise ValueError("source chunks_size无效")
    global_frame_index = 0
    result_rows, stats_rows = [], []
    for new_episode_idx, src_episode_idx in enumerate(original_ids):
        original = rows[src_episode_idx]
        table = pq.read_table(parquet_files[src_episode_idx])
        length = table.num_rows
        if length <= 0:
            raise ValueError(f"Episode {src_episode_idx} 是空轨迹")
        # 不修改图像/动作/奖励等样本值，只为独立数据根目录重建索引。
        for col, data in (
            ("episode_index", [new_episode_idx] * length),
            ("frame_index", list(range(length))),
            ("index", list(range(global_frame_index, global_frame_index + length))),
        ):
            arr = pa.array(data, type=pa.int64())
            pos = table.schema.get_field_index(col)
            table = table.set_column(pos, col, arr) if pos >= 0 else table.append_column(col, arr)
        chunk = new_episode_idx // chunks_size
        parquet_path = dest / "data" / f"chunk-{chunk:03d}" / f"episode_{new_episode_idx:06d}.parquet"
        parquet_path.parent.mkdir(exist_ok=True)
        pq.write_table(table, parquet_path)
        result_rows.append({"episode_index": new_episode_idx, "tasks": original.get("tasks", []), "length": length})
        stats_rows.append({"episode_index": new_episode_idx, "stats": {}})
        global_frame_index += length

    out_info = dict(info)
    out_info.update({
        "total_episodes": len(original_ids), "total_frames": global_frame_index,
        "total_chunks": math.ceil(len(original_ids) / chunks_size),
        "total_tasks": len(tasks), "total_videos": 0,
        "chunks_size": chunks_size,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "splits": {"train": f"0:{len(original_ids)}"},
    })
    (dest / "meta" / "info.json").write_text(json.dumps(out_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(dest / "meta" / "episodes.jsonl", result_rows)
    _write_jsonl(dest / "meta" / "episodes_stats.jsonl", stats_rows)
    _write_jsonl(dest / "meta" / "tasks.jsonl", tasks)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="已合并的LeRobot v2 parquet数据根目录")
    parser.add_argument("--output", type=Path, required=True, help="全新的输出目录，生成train/和val/")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("输出目录已经存在，拒绝覆盖: " + str(output))
    if source == output or source in output.parents or output in source.parents:
        parser.error("输出目录不得与输入根目录嵌套")
    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    if str(info.get("codebase_version", "")).startswith("v3"):
        parser.error("此脚本仅支持逐Episode的LeRobot v2 parquet数据")
    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(source / "meta" / "tasks.jsonl")
    rows = {int(r["episode_index"]): r for r in episodes}
    if len(rows) != len(episodes):
        raise ValueError("重复Episode编号")
    parquet_files = {}
    for path in (source / "data").glob("chunk-*/episode_*.parquet"):
        match = PARQUET_NAME.match(path.name)
        if match:
            key = int(match.group(1))
            if key in parquet_files:
                raise ValueError(f"多个parquet对应Episode {key}")
            parquet_files[key] = path
    missing = set(rows) - set(parquet_files)
    if missing:
        raise FileNotFoundError(f"缺少{len(missing)}个Episode parquet: {sorted(missing)[:5]}")
    train_ids, val_ids = split_episode_ids(episodes, args.val_ratio, args.seed)
    # 先验证所需输入，避免创建空的结果目录。
    output.mkdir(parents=True)
    _write_split(source, output / "train", train_ids, rows, parquet_files, info, tasks)
    _write_split(source, output / "val", val_ids, rows, parquet_files, info, tasks)
    manifest = {
        "source": str(source), "seed": args.seed, "val_ratio": args.val_ratio,
        "train_episode_ids": train_ids, "val_episode_ids": val_ids,
        "note": "IDs refer to original source Episode indices; no overlapping Episodes.",
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Episode split complete: train={len(train_ids)}, val={len(val_ids)}; output={output}")


if __name__ == "__main__":
    main()
