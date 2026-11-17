#!/usr/bin/env python3
"""根据真实训练日志 CSV 绘制 SFT / RECAP / Value Loss 曲线。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> tuple[list[int], list[float], list[float | None]]:
    steps: list[int] = []
    train: list[float] = []
    valid: list[float | None] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"step", "train_loss"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}: CSV 必须包含 step,train_loss 两列")
        for row_number, row in enumerate(reader, 2):
            try:
                step_str = (row.get("step") or "").strip()
                train_str = (row.get("train_loss") or "").strip()
                if not step_str or not train_str:
                    raise ValueError("step / train_loss 不能为空")
                step = int(step_str)
                loss = float(train_str)
                val_text = (row.get("val_loss") or "").strip()
                val = float(val_text) if val_text else None
            except ValueError as exc:
                raise ValueError(f"{path}: 第 {row_number} 行无效：{exc}") from exc
            import math
            if not math.isfinite(loss) or (val is not None and not math.isfinite(val)):
                raise ValueError(f"{path}: 第 {row_number} 行包含非有限数值")
            steps.append(step)
            train.append(loss)
            valid.append(val)
    if not steps:
        raise ValueError(f"{path}: 数据为空")
    if len(set(steps)) != len(steps):
        raise ValueError(f"{path}: step 存在重复")
    return steps, train, valid


def plot_one(input_path: Path, output_path: Path, title: str, window: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, train, valid = read_csv(input_path)
    pairs = sorted(zip(steps, train, valid), key=lambda item: item[0])
    xs = [x for x, _, _ in pairs]
    ys = [y for _, y, _ in pairs]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(xs, ys, linewidth=0.8, alpha=0.45, label="Train loss (raw)")
    if window > 1 and len(ys) >= window:
        rolling = [sum(ys[i - window + 1 : i + 1]) / window for i in range(window - 1, len(ys))]
        ax.plot(xs[window - 1 :], rolling, linewidth=1.8, label=f"Train loss (MA {window})")
    valid_points = [(x, val) for x, _, val in pairs if val is not None]
    if valid_points:
        ax.plot([p[0] for p in valid_points], [p[1] for p in valid_points],
                marker="o", markersize=3, linewidth=1.3, label="Validation loss")
    ax.set(title=title, xlabel="Training step", ylabel="Loss")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"已生成 {output_path}（{len(xs)} 个 Train 点，{len(valid_points)} 个 Val 点）")


def main() -> None:
    parser = argparse.ArgumentParser(description="从真实 CSV 日志生成训练曲线，不生成示意数据")
    parser.add_argument("--input-dir", type=Path, required=True, help="包含 sft_loss.csv / recap_loss.csv / value_loss.csv 的目录")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--smooth-window", type=int, default=20)
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("--smooth-window 必须 >= 1")
    items = [
        ("sft_loss.csv", "sft_loss.png", "Pi 0.5 SFT training loss"),
        ("recap_loss.csv", "recap_loss.png", "RECAP policy training loss"),
        ("value_loss.csv", "value_loss.png", "Value model training / validation loss"),
    ]
    count = 0
    for name, out, title in items:
        source = args.input_dir / name
        if not source.is_file():
            print(f"跳过 {name}：未找到真实日志")
            continue
        plot_one(source, args.output_dir / out, title, args.smooth_window)
        count += 1
    if not count:
        parser.error("未找到任何训练 CSV。需要 step,train_loss，可选 val_loss；不会自动生成曲线。")


if __name__ == "__main__":
    main()
