# 训练与数据处理脚本

本目录包含项目实际维护的训练、数据统计和评测辅助入口。

| 脚本 | 职责 |
| --- | --- |
| `libero_pipeline.py` | 统一执行 LIBERO 数据归一化统计、SFT/混合数据/RECAP 训练及策略服务 |
| `compute_norm_stats.py` | 根据训练配置计算 state/action 归一化统计，供训练与推理使用 |
| `train.py` | 执行策略模型训练 |
| `train_value.py` | 使用 `value_label` 训练 VLM Value Model |
| `label_advantage_from_vlm.py` | Value 推理、N-step Advantage 计算及 `adv_ind` 标注 |
| `plot_training_curves.py` | 从实际 CSV 日志绘制训练曲线 |
| `verify_source.py` | 无需 GPU 的源码静态检查 |

LIBERO 评测与自主 Rollout 采集入口位于 `examples/libero/main.py`；RLDS 专家示范转换入口位于 `examples/libero/pistar_rlds_demo_processing.py`。

当前仓库没有提供通用 LeRobot 数据集合并脚本。合并专家示范与自主 Rollout 时，需使用与项目锁定 LeRobot 版本兼容的工具，并确保两类数据的字段、图像布局、动作维度及任务元数据一致。

各脚本参数可通过对应命令的 `--help` 查看。完整流程见根目录 README。
