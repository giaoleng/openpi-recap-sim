#!/usr/bin/env bash
# 真实 GPU + LIBERO 端到端冒烟测试：训练一小步、保存、加载策略服务、运行一个Episode。
# 不包含正式评测，不会生成/替代 README 成功率。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${SFT_DATA_ROOT:?请设置SFT_DATA_ROOT为本地LeRobot专家示范数据目录}"
: "${SFT_REPO_ID:?请设置SFT_REPO_ID为对应归一化资产ID}"
: "${PI05_BASE:?请设置PI05_BASE为包含params/的π0.5预训练Checkpoint根目录}"

TRAIN_PYTHON="${TRAIN_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-$PROJECT_ROOT/.venv-libero/bin/python}"
PORT="${SMOKE_PORT:-8000}"
RUN_TAG="gpu_smoke_$(date -u +%Y%m%dT%H%M%SZ)_$$"
OUT_DIR="${SMOKE_OUTPUT_DIR:-$PROJECT_ROOT/outputs/gpu_smoke/$RUN_TAG}"
EXP_NAME="$RUN_TAG"
mkdir -p "$OUT_DIR"

for executable in "$TRAIN_PYTHON" "$LIBERO_PYTHON"; do
  [[ -x "$executable" ]] || { echo "找不到Python环境: $executable" >&2; exit 1; }
done

# 显式确认真实GPU设备，不将CPU上的轻量测试冒充GPU验收。
"$TRAIN_PYTHON" - <<'PY'
import jax
devices = jax.devices()
print("JAX devices:", devices)
if not any(d.platform in ("gpu", "cuda") for d in devices):
    raise SystemExit("当前JAX环境没有GPU设备，停止冒烟测试")
PY

export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "1/4 计算归一化参数..."
"$TRAIN_PYTHON" scripts/libero_pipeline.py norm --stage sft \
  --dataset-root "$SFT_DATA_ROOT" --repo-id "$SFT_REPO_ID" \
  >"$OUT_DIR/norm.log" 2>&1

echo "2/4 执行2步真实SFT训练并保存Checkpoint..."
"$TRAIN_PYTHON" scripts/libero_pipeline.py train --stage sft \
  --dataset-root "$SFT_DATA_ROOT" --repo-id "$SFT_REPO_ID" \
  --init-checkpoint "$PI05_BASE" --exp-name "$EXP_NAME" \
  --batch-size 1 --num-train-steps 2 --warmup-steps 0 --save-interval 1 \
  --num-workers 0 >"$OUT_DIR/train.log" 2>&1

CKPT_RUN_DIR="$PROJECT_ROOT/checkpoints/pi05_star_libero_sft/$EXP_NAME"
[[ -d "$CKPT_RUN_DIR" ]] || { echo "未找到训练输出: $CKPT_RUN_DIR" >&2; exit 1; }
CKPT_DIR="$(find "$CKPT_RUN_DIR" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
[[ -n "$CKPT_DIR" && -d "$CKPT_DIR/params" ]] || {
  echo "训练后未产生可加载的params目录；请查看$OUT_DIR/train.log" >&2
  exit 1
}
echo "Checkpoint: $CKPT_DIR"

echo "3/4 启动Checkpoint对应的策略服务..."
"$TRAIN_PYTHON" scripts/libero_pipeline.py serve --stage sft \
  --repo-id "$SFT_REPO_ID" --checkpoint "$CKPT_DIR" --port "$PORT" \
  >"$OUT_DIR/serve.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 90); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "策略服务提前退出，请查看$OUT_DIR/serve.log" >&2
    exit 1
  fi
  if "$TRAIN_PYTHON" - "$PORT" <<'PY'
import socket
import sys
try:
    with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
  then
    READY=1
    break
  fi
  sleep 2
done
[[ "$READY" -eq 1 ]] || { echo "策略服务未成功开放端口$PORT" >&2; exit 1; }

echo "4/4 使用真实LIBERO执行1个Episode，验证服务端推理..."
export PYTHONPATH="$PROJECT_ROOT/third_party/libero:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
"$LIBERO_PYTHON" examples/libero/main.py \
  --args.host 127.0.0.1 --args.port "$PORT" \
  --args.task_suite_name libero_spatial --args.task_ids 0 \
  --args.num_trials_per_task 1 --args.seed 7 \
  --args.model_label "$RUN_TAG" --args.checkpoint_path "$CKPT_DIR" \
  --args.action_horizon 15 --args.eval_output_dir "$OUT_DIR/evaluation" \
  >"$OUT_DIR/libero.log" 2>&1

echo "GPU和LIBERO冒烟测试完成：$OUT_DIR"
echo "注意：单Episode演示仅验证执行链路，不代表模型性能或正式成功率。"
