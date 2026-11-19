# LIBERO 仿真评测与 Rollout

本目录提供 LIBERO 策略评测客户端，支持按任务套件和 Task ID 执行评测，并导出 LeRobot 格式的自主 Rollout 轨迹。

## 任务套件

| 套件 | 主要任务类型 |
| --- | --- |
| `libero_spatial` | 空间关系 |
| `libero_object` | 物体变化 |
| `libero_goal` | 目标条件变化 |
| `libero_10` | 长时序任务 |

每套件包含 10 个任务。评测时分别统计每个任务的成功次数、Episode 总数和成功率。

## 准备仿真环境

LIBERO 本体作为外部依赖安装，不随本仓库分发。在项目根目录执行：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
uv venv --python 3.10 .venv-libero
source .venv-libero/bin/activate
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH="$PWD/third_party/libero:$PYTHONPATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## 评测与数据采集

先启动对应 Checkpoint 的策略服务，再运行 LIBERO 客户端：

```bash
python examples/libero/main.py \
  --args.adv_ind_input positive \
  --args.task_suite_name libero_10 \
  --args.num_trials_per_task 5
```

只选择部分任务：

```bash
python examples/libero/main.py \
  --args.task_suite_name libero_spatial \
  --args.task_ids 0 3 7 \
  --args.num_trials_per_task 10
```

保存自主 Rollout：

```bash
python examples/libero/main.py \
  --args.adv_ind_input positive \
  --args.task_suite_name libero_10 \
  --args.num_trials_per_task 5 \
  --args.save_lerobot_rollout true \
  --args.rollout_output_dir outputs/libero_rollouts \
  --args.rollout_repo_id libero_rollout_round1 \
  --args.rollout_overwrite true
```

`adv_ind_input=positive` 是推理时送给策略的条件；新导出的自主 Rollout 可先以 `adv_ind=none` 保存，之后再使用 Value Model 计算 Advantage 并完成标注。

## 输出与评测记录

评测时固定 Checkpoint、任务 ID、随机种子、每任务 Episode 数和成功判定方式。成功率定义为成功 Episode 数除以有效评测 Episode 总数。开启 Rollout 导出时，检查 LeRobot 数据集的 `meta/`、`data/` 与 `videos/` 目录。
