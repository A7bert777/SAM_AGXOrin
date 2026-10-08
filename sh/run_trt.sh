#!/usr/bin/env bash
# SAM_AGXOrin 用 TensorRT engine 推理的运行入口
#
# 用法（与 run.sh 提示参数基本一致，需已构建 engine）：
#   ./sh/run_trt.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_trt_point.png
#   ./sh/run_trt.sh --image inputimage/0000.jpg --box 100 100 800 600 --out outputimage/0000_trt_box.png
#   ./sh/run_trt.sh --image inputimage/0000.jpg --point 500 400 --bench
#
# 前提：已导出 ONNX 并构建 engine：
#   ./venv310/bin/python py/export_onnx.py --model vit_h
#   ./sh/build_engine.sh models/sam_vit_h_encoder.onnx
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
SP="$VENV/lib/python3.10/site-packages"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[错误] 未找到 venv：$VENV"
  echo "       请先执行 ./sh/setup.sh 初始化环境"
  exit 1
fi

# TensorRT 运行库（Python 绑定在 venv 或系统，均可能需要系统库路径）
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
# 注意：不要设置 NVIDIA_TF32_OVERRIDE。TensorRT 要求「构建 engine」与「执行 engine」
# 时该环境变量一致，否则报:
#   [E] ICudaEngine::createExecutionContext ... Inconsistent setting of
#       NVIDIA_TF32_OVERRIDE env var at build -1 and at execution 1
# build_engine.sh 未设置（即默认 -1），故此处也必须保持不设置。
# 如需 TF32 控制，用 py 里的 --no-tf32，或构建/推理同时显式设同一值。
if [ -n "${NVIDIA_TF32_OVERRIDE:-}" ]; then
  echo "[提示] 检测到 NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE}，"
  echo "       请确保构建 engine 时使用相同的值，否则 TensorRT 会拒绝加载。"
fi
unset NVIDIA_TF32_OVERRIDE

exec "$VENV/bin/python" "$ROOT/py/sam_trt_infer.py" "$@"