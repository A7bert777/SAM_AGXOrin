#!/usr/bin/env bash
# SAM_AGXOrin 运行入口：设置环境后调用客户端 sam_client.py
#
# 用法（统一标志式，与 SAM3_AGXOrin 一致）：
#   ./sh/run.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_point.png
#   ./sh/run.sh --image inputimage/0000.jpg --box 100 100 800 600 --out outputimage/0000_box.png
#   ./sh/run.sh --image inputimage/0000.jpg --auto --auto-points 16 --out outputimage/0000_auto.png
#   ./sh/run.sh --image inputimage/0000.jpg --point 500 400 --bench
#   ./sh/run.sh --check
#
# 行为：
#   若已执行 ./sh/serve.sh start（常驻服务在线）-> 请求转发给服务，约 0.5~1s 返回
#   否则                                      -> 本地加载模型单次运行（约 6s）
#   两种方式输出完全一致。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SAM_PY:-$ROOT/venv310}"
SP="$VENV/lib/python3.10/site-packages"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[错误] 未找到 venv：$VENV"
  echo "       请先执行 ./sh/setup.sh 初始化环境（或用 SAM_PY=<python路径> 指定）"
  exit 1
fi

# cuSPARSELt 等 NVIDIA 运行库（Jetson 上 torch 依赖）
export LD_LIBRARY_PATH="$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
# Jetson 上避免碎片化；如显存紧张可改为 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# PyTorch 后端：客户端连 run/sam.sock（由 serve.sh start 启动），
# 连不上则回退到 sam_infer.py 本地单次运行。
export SAM_BACKEND="torch"
# 注意：不要设 NVIDIA_TF32_OVERRIDE。若同一进程/环境里还要加载 TensorRT engine，
# TensorRT 要求该值与构建时一致，否则拒绝创建 context。PyTorch 侧的 TF32 由
# sam_infer.py 里的 torch.backends.*.allow_tf32 控制，无需此环境变量。

exec "$VENV/bin/python" "$ROOT/py/sam_client.py" "$@"
