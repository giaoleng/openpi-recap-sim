# 核心算法与训练框架

这里集中放置 VLA 模型、策略适配、训练数据管线和通信服务。

| 子目录 | 主要功能 |
| --- | --- |
| `models/` | π₀.₅、双专家网络、LoRA、SigLIP、Gemma、Value Model |
| `training/` | 训练配置、DataLoader、优化器、归一化统计、Checkpoint 与参数加载 |
| `policies/` | LIBERO 观测/动作适配和策略输入输出 |
| `serving/` | WebSocket 策略服务 |
| `shared/` | 图像、JAX/Flax、数据与兼容性工具 |
| `models_pytorch/` | PyTorch 实现与相关 Transformer 模块 |

主要数据路径：LIBERO 观测 → 策略输入变换 → 模型输出动作 → 环境执行；训练端通过 `value_label` / `reward_label` → Value → `adv_ind` 支持优势条件训练。
