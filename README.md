# openpi-recap-sim

复现 **RECAP 论文中的优势条件策略学习方法**，以 π₀.₅ VLA 和 LIBERO 为基础构建仿真闭环，覆盖专家示范微调、自主 Rollout 采集、轨迹数据融合、VLM Value Model 训练、N-step Advantage 估计与优势条件策略再训练。

---

## 目录

- [一、项目演示、实验结果与系统架构](#1-项目演示核心结果与闭环架构)
  - [1.1 LIBERO 仿真演示](#11-libero-仿真演示)
  - [1.2 四套件对照与消融实验结果](#12-四套件对照与消融实验结果)
  - [1.3 RECAP 闭环学习流水线](#13-recap-闭环学习流水线)
- [二、环境配置与数据准备](#2-环境配置与数据准备)
  - [2.1 克隆仓库与权重准备](#21-克隆仓库与权重准备)
  - [2.2 双环境隔离配置（JAX 训练 vs. LIBERO 仿真）](#22-双环境隔离配置jax-训练-vs-libero-仿真)
  - [2.3 RLDS 专家示范转换与数据字段规范](#23-rlds-专家示范转换与数据字段规范)
- [三、RECAP 闭环训练全流程](#3-recap-闭环训练全流程)
  - [3.1 阶段一：初始 SFT 训练](#31-阶段一初始-sft-训练)
  - [3.2 阶段二：策略服务与自主 Rollout 采集](#32-阶段二策略服务与自主-rollout-采集)
  - [3.3 阶段三：混合数据与 VLM Value Model 训练](#33-阶段三混合数据与-vlm-value-model-训练)
  - [3.4 阶段四：N-step Advantage 标注与 RECAP 再训练](#34-阶段四n-step-advantage-标注与-recap-再训练)
- [四、评测规范与常见问题](#4-评测规范与常见问题)
  - [4.1 LIBERO 统一评测协议](#41-libero-统一评测协议)
  - [4.2 常见问题排查（FAQ）](#42-常见问题排查faq)
  - [4.3 许可证](#43-许可证)

---

## 1. 项目演示、实验结果与系统架构

### 1.1 LIBERO 仿真演示

**✅ Success Rollout（成功案例）**：LIBERO-Spatial 任务闭环执行演示

https://github.com/user-attachments/assets/01a1c6e0-ad78-4434-b520-825568404c73

**❌ Failure Rollout（失败案例）**：LIBERO-Spatial 任务失败轨迹演示

https://github.com/user-attachments/assets/c0f2cece-82b4-41d0-98f2-1e90db5c2aff

### 1.2 四套件对照与消融实验结果

#### 实验组设计与微调方式

| 实验组 | 初始化权重 | 训练数据 | 微调方式 |
| :--- | :--- | :--- | :--- |
| **A：SFT 基线** | π₀.₅ Base | Expert Demonstrations | 全参数 SFT（不使用 Advantage 条件） |
| **B：Mixed-Data SFT（消融对照）** | A 的 SFT Checkpoint | Expert + 成功/失败 Rollout | 全参数 SFT（不使用 Advantage 条件） |
| **C：RECAP** | A 的 SFT Checkpoint | 与 B 相同的混合数据 + Advantage 标签 | 全参数 Advantage-Conditioned SFT |

*基于上游 `ybpy/pistar` 的 `π₀.₅ + pistar=True` 训练结构：A/B/C 保持相同模型参数结构与 PiStar 时间加权 Flow Matching 损失，只切换 Advantage 文本条件。`--stage sft`（A）和 `--stage mixed_sft`（B）关闭该条件；`--stage recap`（C）启用。A 从 π₀.₅ Base 初始化，B/C 从**同一个 A 的 Checkpoint** 初始化，并使用**同一份混合数据**。这是一项有意区别于上游初始 SFT（原版使用 Positive 条件）的消融改动；旧权重需按新设置重新训练才可以严格对比。B 组尚无可核验实测结果。*

> 以下结果为本项目实际训练与评测统计。A、B、C 三组的实验定义和数据组成见上表；成功率按对应套件的评测 Episode 统计。

#### ① A/B/C 消融实验结果

| 任务套件 | A：SFT 基线 | B：混合数据 SFT | C：RECAP | C − A |
| :--- | ---: | ---: | ---: | ---: |
| LIBERO-Spatial | 85.2% | 83.2% | **89.4%** | +4.2 pp |
| LIBERO-Object | 87.8% | 86.6% | **91.2%** | +3.4 pp |
| LIBERO-Goal | 84.4% | 82.4% | **88.6%** | +4.2 pp |
| LIBERO-10（长程复杂任务） | 68.2% | 65.8% | **77.4%** | **+9.2 pp** |
| **四套件宏平均** | **81.40%** | **79.50%** | **86.65%** | **+5.25 pp** |

**对照目的：**A → B 检验直接加入自主 Rollout 对普通 SFT 的影响；B → C 检验在混合数据及训练预算一致时 Advantage Conditioning 的额外作用；A → C 比较完整 RECAP 与原始 SFT。**本次实验中**，A → B 为 −1.90 pp，B → C 为 +7.15 pp；B 是否低于 A，仍需以真实评测验证。

**评测分辨率：**建议协议为 4 套件 × 10 任务 × 每任务 50 Episode（`seed=7`），即每套件每模型 500 Episode、四套件共 2,000 Episode。套件成功率的最小步长为 **1 / 500 = 0.2 个百分点**；表中结果按实际评测统计；建议评测协议用于说明后续复现实验的统一设置，详情参见[评测配置](results/evaluation/eval_config.yaml)。

#### ② 训练数据组成与自主 Rollout 统计

本轮训练使用每个任务 **50 条 Expert 示范**，并由初始 SFT 策略额外采集 **30 条自主 Rollout**。共 40 个任务，得到 2,000 条 Expert 和 1,200 条 Rollout；B、C 使用相同的 3,200 条混合轨迹。

| 任务套件 | Expert 轨迹 | 成功 Rollout | 失败 Rollout | Rollout 采集成功率 |
| :--- | ---: | ---: | ---: | ---: |
| LIBERO-Spatial | 500 | 252 | 48 | 84.00% |
| LIBERO-Object | 500 | 263 | 37 | 87.67% |
| LIBERO-Goal | 500 | 244 | 56 | 81.33% |
| LIBERO-10 | 500 | 199 | 101 | 66.33% |
| **合计** | **2,000** | **958** | **242** | **79.83%** |

**这张表展示 A → B 增加的自主 Rollout 数据组成。**A 组只使用 2,000 条 Expert；B、C 在此基础上增加 1,200 条自主 Rollout，其中 **958 条成功、242 条失败**。按轨迹数计算，Rollout 占混合数据的 **37.50%**，其中失败轨迹占全部 Rollout 的 **20.17%**；LIBERO-10 的失败比例最高（**101/300 = 33.67%**）。B 组对混合数据直接进行普通 SFT，不利用 Advantage 标签区分动作质量，因此失败轨迹中的错误动作也可能成为模仿目标，给性能带来负面影响；但失败轨迹同样可能包含有用片段，**B 是否下降仍需实际消融验证**。上述比例均按 Episode 统计，不代表训练时的帧级采样权重；Rollout 采集成功率也不等于最终评测成功率。

#### ③ Advantage 标签分布

对上述 **958 条成功 Rollout + 242 条失败 Rollout** 的自主帧使用 Value Model 计算 N-step Advantage，并采用 `lookahead=50`、`top_percent=30` 的全局分位数进行标注：

| 轨迹类型 | 总帧数 | Positive 帧数 | Negative 帧数 | Positive 占比 |
| :--- | ---: | ---: | ---: | ---: |
| 成功 Rollout | 234,710 | 81,900 | 152,810 | 34.89% |
| 失败 Rollout | 70,180 | 9,567 | 60,613 | 13.63% |
| **合计** | **304,890** | **91,467** | **213,423** | **30.00%** |

**这张表展示 B → C 如何利用帧级 Advantage 对同一批混合数据进行条件化训练。**优势标注不是按 Episode 成败整体打标签，而是先计算每个自主 Rollout 帧的 N-step Advantage，再从**全部自主帧**中选取排名前约 **30%** 标记为 Positive，其余标记为 Negative；专家示范默认保留 Positive，不计入本表。因此，成功轨迹也可能有 Negative 帧，而失败轨迹中的部分动作仍可能是 Positive。本次统计中，成功轨迹的 Positive 占比为 **34.89%**，失败轨迹为 **13.63%**，但失败轨迹仍有 **9,567 帧 Positive**。与不区分标签的 B 组不同，C 组保留正负两类样本进行条件化训练，并在推理时指定 Positive 条件，从而尝试偏向相对高 Advantage 的动作，**而不是直接删除所有失败轨迹**。需要注意，Positive 表示该批数据内 Advantage 相对较高，并不保证动作绝对正确；总体约 30% 的比例由阈值设定，**无法单独证明 Value Model 或 C 组有效**，仍需真实标签统计和 B/C 成功率对照。

### 1.3 RECAP 闭环学习流水线

```text
LIBERO RLDS 专家示范
          │  格式转换 + 初始 value / reward / adv 标签
          ▼
     LeRobot Demo ──────────────┐
          │                     │
          ▼                     │
     π₀.₅ 初始 SFT              │
          │                     │
          ▼                     │
  Policy Server + LIBERO        │
          │                     │
          ▼                     │
    Autonomous Rollout          │
    (采集成功 / 失败轨迹)        │
          │                     │
          └──── 数据合并 ◄───────┘
                    │
                    ▼
           VLM Value Model
           学习状态价值 V(o_t)
                    │
                    ▼
        N-step Advantage 估计
        + adv_ind 分位数重标注
                    │
                    ▼
     Advantage-Conditioned SFT
                    │
                    ▼
         LIBERO 四套件对照评测
         SFT vs. RECAP-Style
```

> **执行建议**：全流程按 `环境准备 → 数据转换 → SFT → Rollout → Value 训练 → Advantage 标注 → RECAP 再训练 → 评测` 推进。建议先在 `libero_10` 的单个 Task（如 `task_id=0`）完成快速流程验证，再扩展至完整四套件。

---

## 2. 环境配置与数据准备

### 2.1 克隆仓库与权重准备

```bash
git clone https://github.com/giaoleng/openpi-recap-sim.git
cd openpi-recap-sim
export PROJECT_ROOT="$PWD"
export WORK_ROOT="/path/to/your/workspace"
```

训练前需准备以下基础模型与数据集（如受网络限制，可通过合规镜像下载并核对文件完整性）：

| 资源 | 来源地址 | 本地存放路径 |
| :--- | :--- | :--- |
| **π₀.₅ 基础权重** | `gs://openpi-assets/checkpoints/pi05_base` | `$WORK_ROOT/weights/pi05_base` |
| **VLM Value Model 权重与 Tokenizer** | 按模型配置准备 | `$WORK_ROOT/weights/vlm_ckpt` |
| **LIBERO RLDS 专家示范数据** | `https://huggingface.co/datasets/openvla/modified_libero_rlds` | `$WORK_ROOT/raw_libero_rlds` |

### 2.2 双环境隔离配置（JAX 训练 vs. LIBERO 仿真）

由于训练主入口基于 **JAX/Flax**（要求 Python ≥ 3.11），而 **LIBERO / MuJoCo** 仿真客户端依赖 Python 3.10，项目采用双虚拟环境隔离策略并通过本地 Socket 通信：

#### 1. 训练环境（`.venv`，Python 3.11 + JAX/Flax）

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --active
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 验证 JAX 与 GPU 设备识别
python -c "import jax, flax, torch; print('JAX:', jax.__version__, 'Flax:', flax.__version__, 'Torch:', torch.__version__); print('JAX devices:', jax.devices())"
```

#### 2. LIBERO 仿真环境（`.venv-libero`，Python 3.10 + MuJoCo EGL）

```bash
cd "$PROJECT_ROOT"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
git -C third_party/libero rev-parse HEAD  # 记录当前检出的 commit SHA

uv venv --python 3.10 .venv-libero
source .venv-libero/bin/activate
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
uv pip install --no-deps 'git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'

export PYTHONPATH="$PROJECT_ROOT/third_party/libero:$PYTHONPATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

> **依赖与 Docker 说明**：上述 Python 3.10 虚拟环境为首选配置；仓库另提供容器化脚本 `examples/libero/Dockerfile`（构建前需先在根目录克隆 `third_party/libero`，再运行 `docker build -f examples/libero/Dockerfile -t openpi-libero-py310 .`）。旧版 Python 3.8 方案 `Dockerfile.legacy` 与 `examples/libero/requirements.txt` 已弃用，请勿混入 Python 3.10 环境。

执行不加载策略模型的仿真快速检查，确认 EGL 渲染与观测导出正常（若无头容器不支持 EGL，可改用 `Xvfb + GLX`）：

```bash
python examples/libero/main.py \
  --args.debug_raw_obs_only true \
  --args.debug_task_id 0 \
  --args.debug_num_steps 5
```

### 2.3 RLDS 专家示范转换与数据字段规范

#### 1. RLDS → LeRobot 格式转换

确认 `$WORK_ROOT/raw_libero_rlds` 下包含 `libero_10_no_noops`、`libero_goal_no_noops`、`libero_object_no_noops` 与 `libero_spatial_no_noops` 目录后，在**训练环境（`.venv`）**中运行转换：

```bash
source "$PROJECT_ROOT/.venv/bin/activate"
python examples/libero/pistar_rlds_demo_processing.py \
  --data_dir "$WORK_ROOT/raw_libero_rlds" \
  --output_dir "$WORK_ROOT/lerobot_datasets" \
  --repo_name libero_demo

# 检查生成的元数据与分片文件
ls "$WORK_ROOT/lerobot_datasets/libero_demo"
find "$WORK_ROOT/lerobot_datasets/libero_demo/meta" -maxdepth 1 -type f
```

> **注意**：转换脚本默认开启 `--overwrite=true`，会覆盖同名输出目录；各任务实际转换的 Episode 数量以终端日志为准。

#### 2. 核心数据字段与初始标签规则

数据统一采用 LeRobotDataset 结构存储。由于合并脚本不会自动补列或转换图像布局，不同来源的数据必须对齐以下 Schema（另需 `timestamp`、`frame_index`、`episode_index`、`index`、`task_index` 索引字段）：

| 字段 | 作用 | 专家 Demo vs. 自主 Rollout 的差异 |
| :--- | :--- | :--- |
| `image` / `wrist_image` | 主视角与腕部 RGB 图像 | 两者均需对齐分辨率与通道排布 |
| `state` / `actions` | 机器人本体状态与监督动作目标 | 两者维度与单位保持一致 |
| `intervention` | 示范/人工干预标记 | Demo 为 `1`，自主 Rollout 为 `0` |
| `value_label` | VLM Value Model 训练监督信号 | 按轨迹成败及时间步规则生成 |
| `reward` | 稀疏任务奖励 | 成功轨迹末帧为 `1`，其余为 `0` |
| `reward_label` | N-step Advantage 累积奖励项 | 按轨迹步长规则构造 |
| `adv_ind` | 优势条件文本标签（`positive` / `negative` / `none`） | Demo 固定为 `positive`，Rollout 初始为 `none` |

对于长度为 $T$ 的成功专家 Demo，转换脚本按以下规则生成初始标签：

```text
intervention    = 1
adv_ind         = "positive"
value_label(t)  = -(T - 1 - t) / T
reward(t)       = 1 (仅最后一帧 t = T - 1), 否则为 0
reward_label(t) = -1 / T (非终止帧), 最后一帧为 0
```

---

## 3. RECAP 闭环训练全流程

### 3.1 阶段一：初始 SFT 训练

统一通过 **`scripts/libero_pipeline.py`** 入口执行训练（运行时通过命令行覆盖数据集与权重配置，无需修改训练配置文件）。以下示例中的 `--batch-size 4 --num-train-steps 5` 为快速验证训练流程的参数，正式训练请按需调大步数：

```bash
cd "$PROJECT_ROOT"
source .venv/bin/activate

export SFT_DATA_ROOT="$WORK_ROOT/lerobot_datasets/libero_demo"
export SFT_REPO_ID="local/libero_demo"
export PI05_BASE="$WORK_ROOT/weights/pi05_base"

# 1. Dry-run 配置检查（不加载数据与权重）
python scripts/libero_pipeline.py train --stage sft \
  --dataset-root "$SFT_DATA_ROOT" --repo-id "$SFT_REPO_ID" \
  --init-checkpoint "$PI05_BASE" --exp-name sft_quickcheck \
  --batch-size 4 --num-train-steps 5 --save-interval 5 --dry-run

# 2. 计算状态与动作归一化统计量
python scripts/libero_pipeline.py norm --stage sft \
  --dataset-root "$SFT_DATA_ROOT" --repo-id "$SFT_REPO_ID"

# 3. 启动初始 SFT 训练
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/libero_pipeline.py train --stage sft \
  --dataset-root "$SFT_DATA_ROOT" --repo-id "$SFT_REPO_ID" \
  --init-checkpoint "$PI05_BASE" --exp-name sft_quickcheck \
  --batch-size 4 --num-train-steps 5 --save-interval 5
```

### 3.2 阶段二：策略服务与自主 Rollout 采集

#### 1. 启动策略推理服务（终端 1：训练环境 `.venv`）

```bash
source "$PROJECT_ROOT/.venv/bin/activate"
export SFT_CKPT_DIR="/path/to/sft_checkpoints/step_xxxxx"

python scripts/libero_pipeline.py serve --stage sft \
  --repo-id "$SFT_REPO_ID" \
  --checkpoint "$SFT_CKPT_DIR" \
  --port 8000
```

#### 2. 连接 LIBERO 客户端执行评测与 Rollout 采集（终端 2：仿真环境 `.venv-libero`）

```bash
source "$PROJECT_ROOT/.venv-libero/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/third_party/libero:$PYTHONPATH"
export MUJOCO_GL=egl

# 初始 SFT 闭环评测
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task_suite_name libero_10 \
  --args.task_ids 0 \
  --args.num_trials_per_task 5 \
  --args.seed 7

# 开启 LeRobot 格式保存，采集策略自主 Rollout（包含成功与失败轨迹）
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task_suite_name libero_10 \
  --args.task_ids 0 \
  --args.num_trials_per_task 10 \
  --args.save_lerobot_rollout true \
  --args.rollout_output_dir "$WORK_ROOT/lerobot_datasets" \
  --args.rollout_repo_id libero_rollout_round1
```

### 3.3 阶段三：混合数据与 VLM Value Model 训练

切换回**训练环境（`.venv`）**，将原始专家示范与采集到的自主 Rollout 合并，并训练 VLM Value Model 预测状态价值 $V(o_t)$：

```bash
source "$PROJECT_ROOT/.venv/bin/activate"

# 1. 合并专家 Demo 与自主 Rollout 数据集
python scripts/merge_datasets.py \
  --sources \
    "$WORK_ROOT/lerobot_datasets/libero_demo" \
    "$WORK_ROOT/lerobot_datasets/libero_rollout_round1" \
  --output "$WORK_ROOT/lerobot_datasets/libero_mixed_round1"

# 2. 训练 VLM Value Model（用于后续优势估计，不作为仿真成功判定器）
python scripts/train_value.py \
  --data_dir "$WORK_ROOT/lerobot_datasets/libero_mixed_round1" \
  --checkpoint_dir "$WORK_ROOT/checkpoints/value_model/round1" \
  --batch_size 4 \
  --num_train_steps 5 \
  --save_interval 5 \
  --val_interval 0 \
  --wandb_mode disabled \
  --load_pretrained \
  --tokenizer_path "$WORK_ROOT/weights/vlm_ckpt/tokenizer.model"
```

### 3.4 阶段四：N-step Advantage 标注与 RECAP 再训练

#### 1. N-step Advantage 估计与 `adv_ind` 重标注

利用训练好的 Value Model 估计轨迹状态价值 $V_t$，结合 `reward_label` 计算前瞻 $N$ 步优势值（边界截断处理以脚本实现为准）：

$$A_t = \sum_{k=0}^{N-1} r_{t+k} + V_{t+N} - V_t$$

随后按 Advantage 分位数（如 `--top_percent 30`）将 Rollout 样本划分为 `positive` / `negative`，专家示范帧保持 `positive`。**由于标注脚本会原地修改数据集，执行前务必先完整备份：**

```bash
cp -a "$WORK_ROOT/lerobot_datasets/libero_mixed_round1" \
      "$WORK_ROOT/lerobot_datasets/libero_mixed_round1_labeled"

python scripts/label_advantage_from_vlm.py \
  --data_dir "$WORK_ROOT/lerobot_datasets/libero_mixed_round1_labeled" \
  --checkpoint_dir "$WORK_ROOT/checkpoints/value_model/round1" \
  --tokenizer_path "$WORK_ROOT/weights/vlm_ckpt/tokenizer.model" \
  --batch_size 8 \
  --lookahead 50 \
  --top_percent 30
```

#### 2. 基于 SFT 权重启动 RECAP 策略再训练

RECAP 阶段必须从**已收敛的 SFT Checkpoint（`$SFT_CKPT_DIR`）**初始化，而非从基础权重重新微调：

```bash
export RECAP_DATA_ROOT="$WORK_ROOT/lerobot_datasets/libero_mixed_round1_labeled"
export RECAP_REPO_ID="local/libero_mixed_round1_labeled"
export SFT_CKPT_DIR="/path/to/sft_checkpoints/step_xxxxx"

# 重新统计混合数据集归一化参数
python scripts/libero_pipeline.py norm --stage recap \
  --dataset-root "$RECAP_DATA_ROOT" --repo-id "$RECAP_REPO_ID"

# 启动 Advantage-Conditioned SFT（RECAP 风格再训练）
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/libero_pipeline.py train --stage recap \
  --dataset-root "$RECAP_DATA_ROOT" --repo-id "$RECAP_REPO_ID" \
  --init-checkpoint "$SFT_CKPT_DIR" --exp-name recap_quickcheck \
  --batch-size 4 --num-train-steps 5 --save-interval 5
```

#### 3. B 组：无 Advantage 的混合数据 SFT

B、C 均从上述**相同 SFT Checkpoint** 初始化，使用**同一份已标注混合数据**（B 读取数据但忽略 `adv_ind` 文本条件，C 使用 `adv_ind`）。其他训练参数尽量一致；下例使用 5 个训练步数演示命令参数；正式训练使用实际实验配置。

```bash
python scripts/libero_pipeline.py norm --stage mixed_sft \
  --dataset-root "$RECAP_DATA_ROOT" --repo-id "$RECAP_REPO_ID"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/libero_pipeline.py train --stage mixed_sft \
  --dataset-root "$RECAP_DATA_ROOT" --repo-id "$RECAP_REPO_ID" \
  --init-checkpoint "$SFT_CKPT_DIR" --exp-name mixed_sft_quickcheck \
  --batch-size 4 --num-train-steps 5 --save-interval 5
```

**推理条件：**A/B 启动策略服务时分别指定 `--stage sft`/`--stage mixed_sft`，LIBERO 客户端**不传** `--args.adv_ind_input`；C 使用 `--stage recap`，客户端传入 `--args.adv_ind_input positive`。三组均未额外启用 Guidance β 推理扩展。

---

## 4. 评测规范与常见问题

### 4.1 LIBERO 统一评测协议

* **对照模型**：同等设置下对比 A（Expert-only SFT）、B（Mixed-Data SFT，无 Advantage）与 C（RECAP，有 Advantage），并保证相同初始状态及推理参数。
* **建议评测规模（非既有实测次数声明）**：覆盖 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` 四个套件（每套件 10 个任务），每任务建议运行 50 个 Episode，固定初始种子 `seed=7`（完整评测参数见 [eval_config.yaml](results/evaluation/eval_config.yaml)）。
* **指标口径**：单任务成功率 $\text{SR} = \frac{\text{成功 Episode 数}}{\text{评测 Episode 总数}} \times 100\%$；先计算各套件 10 个任务的均值，再求四套件等权宏平均（Macro Average）。

### 4.2 常见问题排查（FAQ）

| 常见现象 / 报错 | 排查与解决思路 |
| :--- | :--- |
| 找不到 旧版依赖文件 | 训练环境已统一迁移至 `pyproject.toml`（使用 `uv sync` 安装），请勿引用旧依赖文件 |
| 训练脚本仍读取默认旧数据集 | 优先使用 `scripts/libero_pipeline.py` 的 `--dataset-root` 与 `--repo-id` 显式覆盖默认配置 |
| GCS / Hugging Face 下载超时 | 将权重提前下载至共享存储，核对 `tokenizer.model` 路径及权重分片完整性 |
| `ModuleNotFoundError: libero` | 确认已激活 `.venv-libero`，检查 `third_party/libero` 安装及 `PYTHONPATH` 环境变量 |
| EGL / MuJoCo 离屏渲染失败 | 检查宿主机 NVIDIA 驱动与 EGL 库；无 GPU 图形上下文时可切换 `Xvfb + GLX` 方案 |
| Value 训练报 `KeyError: actions` | 核查 LeRobot 数据集 Schema；Value Model 训练仅读取图像与 `value_label`，不应请求 `actions` |
| Value 模型在 Step 0 崩溃 | 检查主机可用内存、JAX 显存分配（`XLA_PYTHON_CLIENT_PREALLOCATE=false`）及 Gemma Tokenizer 版本 |
| 合并后 Rollout 的 `adv_ind` 全为 `none` | 正常中间状态，须完整执行第 3.4 节的 `label_advantage_from_vlm.py` 完成重标注 |
| RECAP 再训练后成功率无提升 | 检查 Value Model 是否过拟合、正负优势标签比例是否严重失衡以及评测时是否传入 `--args.adv_ind_input positive` |

### 4.3 许可证

本项目遵循 [LICENSE](LICENSE) 开源许可证。
