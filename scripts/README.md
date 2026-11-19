# 训练与数据处理脚本

本目录包含项目训练、数据准备、评测辅助和检查工具。

| 脚本 | 职责 |
| --- | --- |
| `libero_pipeline.py` | 统一执行 LIBERO 数据归一化统计、SFT/混合数据/RECAP 训练及策略服务 |
| `compute_norm_stats.py` | 根据训练配置计算 state/action 归一化统计 |
| `train.py` | 执行策略模型训练 |
| `train_value.py` | 使用 `value_label` 训练 VLM Value Model |
| `label_advantage_from_vlm.py` | Value 推理、N-step Advantage 计算及 `adv_ind` 标注 |
| `merge_datasets.py` | 合并字段已经对齐的 LeRobot 数据集，并重建全局索引和元数据 |
| `check_value_data.py` | 检查 Value batch 的数值范围、异常值和图像张量 |
| `add_value_labels.py` | 为 LeRobot Episode 生成归一化 `value_label` |
| `filter_success_episodes.py` | 按 reward 或 value 标签筛选成功 Episode |
| `serve_policy.py` | 启动策略 WebSocket 服务 |
| `plot_training_curves.py` | 从实际 CSV 日志绘制训练曲线 |
| `verify_source.py` | 无需 GPU 的源码静态检查 |

LIBERO 任务执行与自主 Rollout 采集入口位于 `examples/libero/main.py`；RLDS 专家示范转换入口位于 `examples/libero/pistar_rlds_demo_processing.py`。

合并脚本只处理已经对齐字段的 LeRobot 数据集，不负责补默认标签、修正图像布局或推断 Demo/Rollout 类型。执行前应检查字段、动作维度、图像分辨率和任务元数据。

各脚本参数可通过对应命令的 `--help` 查看。完整流程见根目录 README。
