# 项目状态

本项目围绕 π₀.₅ VLA 在 LIBERO 仿真环境中的策略学习展开，包含专家示范微调、自主 Rollout 采集、VLM Value Model 训练、N-step Advantage 估计、优势条件策略再训练及仿真评测。

## 实现模块

- **策略模型与训练：** JAX/Flax 训练框架、π₀.₅ 策略模型及 SFT/优势条件训练入口。
- **仿真交互：** LIBERO 任务执行、策略服务、Episode 评测与自主轨迹采集。
- **价值估计：** VLM Value Model 训练及状态价值预测。
- **优势标注：** N-step Advantage 计算与 `adv_ind` 标签生成。
- **数据处理：** RLDS 专家示范转换、LeRobot 数据集字段整理与专家/Rollout 轨迹合并。
- **评测：** LIBERO Spatial、Object、Goal 和 LIBERO-10 四个任务套件。

## 训练与评测入口

- 统一训练、归一化和策略服务：`scripts/libero_pipeline.py`
- Value Model 训练：`scripts/train_value.py`
- Advantage 标注：`scripts/label_advantage_from_vlm.py`
- 数据集合并：`scripts/merge_datasets.py`
- LIBERO 仿真客户端：`examples/libero/main.py`
- 评测协议配置：`results/evaluation/eval_config.yaml`

## 实验结果

四套件实验结果、对照设置和统计口径以根目录 README 的实验结果章节为准。实验配置、Checkpoint、数据集及逐 Episode 记录按实际运行环境保存；公开仓库不包含模型权重和原始数据集。

## 验证方式

仓库通过 GitHub Actions 执行源码静态检查与轻量数据接口回归。模型训练和 LIBERO 仿真评测使用独立运行环境。
