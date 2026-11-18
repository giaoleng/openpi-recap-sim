# LIBERO 评测协议

本文定义本项目 SFT、Mixed-Data SFT 与 RECAP 风格策略的**评测协议**，不等同于某次实验的实际执行记录。

## 评测对象

- **SFT**：以 π₀.₅ 基础权重为起点，在 LIBERO 专家示范上微调得到的策略。
- **Mixed-Data SFT（B）**：以已训练的 SFT 策略为初始化，在相同混合数据上继续训练，但关闭 Advantage 文本条件（`--stage mixed_sft`）。
- **RECAP（C）**：以已训练的 SFT 策略为初始化，采用与 B 相同的混合数据，通过 Advantage Conditioning 继续训练（`--stage recap`）。
- **SFT（A）**：使用专家示范，关闭 Advantage 文本条件（`--stage sft`）；A/B/C 保留相同 PiStar 参数架构。
- 评测前分别指定对应 Checkpoint，并加载与训练数据标识一致的动作/状态归一化统计。Checkpoint 路径与训练步数应由实际产物确定，不以固定命名代替验证。

## 建议评测设置

| 项目 | 约定 |
| --- | --- |
| Benchmark | LIBERO |
| 套件 | `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` |
| 每套件任务数 | 10 |
| 每任务 Episode 数 | 50（建议协议，非既有实测声明） |
| 初始随机种子 | 7（建议协议；每次 Episode 的初态按固定规则生成和记录） |
| 对比方法 | A：SFT、B：Mixed-Data SFT、C：RECAP |
| 成功定义 | 环境给出的任务成功标志 |
| 比较原则 | 对齐任务 ID、初始状态/随机种子、动作执行设置及终止条件 |

每种策略按此协议评测：4 套件 × 10 任务 × 50 Episode = **2,000 Episode**；A/B/C 三种策略合计 **6,000 Episode**。此计算仅为**计划评测量**，并不表示这些 Episode 已执行。

## 指标定义

- **任务成功率**：该任务成功 Episode 数 / 有效 Episode 总数（成功 + 正常失败），推理或环境异常记为 `error` 且不混作正常失败；存在错误的评测标记为未完成，不能直接与完整评测比较。
- **套件成功率**：在每任务 Episode 数相同的情况下，取 10 个任务成功率的算术平均；若每任务次数不同，同时公开加权方式及各任务样本量。
- **四套件宏平均**：四个套件成功率的算术平均。
- **提升值**：分别报告 B − A、C − B、C − A，单位为百分点（pp）。

## 实验记录建议字段

`examples/libero/main.py` 每次评测默认导出 `results/evaluation/runs/` 下的独立 CSV 和 JSON：逐 Episode 包含 `run_id`、`model`、`suite`、`task_id`、`episode_id`、`init_state_index`、`seed`、`status`（success/failure/error）、`success`、`steps`、`error_type`、`video_path`、`replan_steps`、`max_steps`；JSON 汇总成功率、异常数和完成状态。还应在实验元数据中另行保存 `checkpoint`、`checkpoint_step`、`git_commit`、`libero_commit`、`action_horizon`，避免不同版本的结果混用。

## 训练曲线

仓库提供 `scripts/plot_training_curves.py`，可读取真实 CSV 日志生成以下图片：

- `sft_loss.png`：SFT 训练损失。
- `recap_loss.png`：RECAP 策略训练损失。
- `value_loss.png`：Value Model 训练损失与验证损失（仅绘制有记录的验证点）。

**绘图工具不会生成训练日志或填补缺失数值；图形本身只有在导入实际日志后才是实验结果。**
