# 训练与数据处理脚本

本目录包含从专家示范处理到优势条件策略训练的主要入口。

| 脚本 | 职责 |
| --- | --- |
| `train.py` | 根据训练配置执行策略 SFT/条件微调 |
| `train_value.py` | 基于 `value_label` 训练 VLM Value Model |
| `label_advantage_from_vlm.py` | Value 推理、N-step Advantage 计算与 `adv_ind` 标注 |
| `check_value_data.py` | 检查 Value batch 的数值范围与图像张量 |
| `merge_datasets.py` | 合并 Schema 对齐的专家示范和 Rollout |
| `compute_norm_stats.py` | 计算训练所需的数据归一化统计 |
| `serve_policy.py` | 启动策略推理服务 |
| `filter_success_episodes.py` | 按成功条件筛选 Episode |
| `add_value_labels.py` | 构造 Value 学习所需监督字段 |
| `verify_source.py` | 源码静态检查 |

各脚本的参数可通过对应命令的 `--help` 查看。完整训练流程与数据准备方式见根目录 README。
