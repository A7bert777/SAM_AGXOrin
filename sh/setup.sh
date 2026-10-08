#!/usr/bin/env bash
# SAM_AGXOrin 一键环境初始化（Jetson AGX Orin / JetPack, aarch64, Python 3.10）
#
# 步骤：
#   1. 环境自检（Jetson / python3.10 / CUDA / 磁盘）
#   2. 创建 venv310（若已存在则复用）
#   3. 确保 pip 可用
#   4. 安装 NVIDIA 定制 PyTorch + torchvision（Jetson 专用 wheel）
#   5. 安装 segment-anything 及其依赖
#   6. 检查可选的 TensorRT 组件
#
# 用法：./sh/setup.sh
# 说明：若已有可用的 venv310（内含 torch），本脚本会直接复用，不会重新下载 PyTorch。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
PY="$VENV/bin/python"
SP="$VENV/lib/python3.10/site-packages"
LOG="$ROOT/logs"

# 版本必须成对匹配 (torch <-> torchvision)
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
# Jetson AI Lab 的 PEP503 源（专为 JetPack 6 / CUDA 12.6 构建的 aarch64 wheel）
JETSON_PYPI_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"

mkdir -p "$LOG" "$ROOT/models" "$ROOT/assets" \
         "$ROOT/inputimage" "$ROOT/outputimage" "$ROOT/outputs" "$ROOT/run"

echo "=============================================="
echo " SAM_AGXOrin 环境初始化"
echo " 项目根目录: $ROOT"
echo " torch: $TORCH_VERSION  /  torchvision: $TORCHVISION_VERSION"
echo "=============================================="

# ---------- 1. 环境自检 ----------
echo "[1/6] 环境自检 ..."
if [ ! -f /etc/nv_tegra_release ]; then
  echo "  [警告] 未检测到 /etc/nv_tegra_release，当前可能不是 Jetson 设备"
fi
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "  [错误] 未找到 python3.10，请先: sudo apt install python3.10 python3.10-venv"
  exit 1
fi
echo "  Python 3.10 : $(python3.10 --version 2>&1)"
if command -v nvcc >/dev/null 2>&1; then
  echo "  CUDA        : $(nvcc --version | grep -oE 'release [0-9.]+')"
else
  echo "  [警告] 未找到 nvcc，CUDA 可能未正确安装"
fi
AVAIL_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
echo "  可用磁盘    : ${AVAIL_GB}GB"
[ "$AVAIL_GB" -lt 10 ] && echo "  [警告] 磁盘可用空间不足 10GB，安装可能失败"

# ---------- 2. venv ----------
if [ -x "$PY" ]; then
  echo "[2/6] 复用已有 venv: $VENV"
else
  echo "[2/6] 创建 venv310 ..."
  if python3.10 -m venv "$VENV" 2>/dev/null; then
    echo "      虚拟环境已创建"
  else
    echo "      标准 venv 失败，改用 --without-pip 方式 ..."
    rm -rf "$VENV"
    python3.10 -m venv --without-pip "$VENV"
  fi
fi

export LD_LIBRARY_PATH="$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"

# ---------- 3. pip ----------
echo "[3/6] 检查 pip ..."
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "      venv 内缺少 pip，正在引导安装 ..."
  if [ -f "$ROOT/get-pip.py" ]; then
    "$PY" "$ROOT/get-pip.py" --no-cache-dir
  else
    curl -sSL -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
    "$PY" /tmp/get-pip.py --no-cache-dir
  fi
fi
echo "      pip : $("$PY" -m pip --version)"
"$PY" -m pip install --upgrade --no-cache-dir pip setuptools wheel >/dev/null

# ---------- 4. torch / torchvision ----------
echo "[4/6] 安装 PyTorch ${TORCH_VERSION} + torchvision ${TORCHVISION_VERSION} ..."
if "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "      torch 已就绪: $("$PY" -c 'import torch; print(torch.__version__)')，跳过"
else
  echo "      源: $JETSON_PYPI_INDEX（约 1GB，请耐心等待）"
  "$PY" -m pip install --no-cache-dir --index-url "$JETSON_PYPI_INDEX" \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    > "$LOG/torch_install.log" 2>&1 || { tail -20 "$LOG/torch_install.log"; exit 1; }
fi

# 验证 CUDA
"$PY" - <<'PYEOF'
import sys
import torch
print(f"  torch       : {torch.__version__}")
try:
    import torchvision
    print(f"  torchvision : {torchvision.__version__}")
except Exception as e:
    print(f"  [警告] 无法导入 torchvision: {e}")
print(f"  CUDA 可用   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU 设备    : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"  计算能力    : sm_{cap[0]}{cap[1]}")
else:
    print("  [错误] CUDA 不可用，请检查 JetPack / CUDA 安装")
    sys.exit(1)
PYEOF

# ---------- 5. segment-anything 及依赖 ----------
echo "[5/6] 安装 segment-anything 及其依赖 ..."
# 关键: numpy 必须 < 2（NVIDIA 定制 torch 按 numpy 1.x 编译）
"$PY" -m pip install --no-cache-dir \
  "numpy<2" "opencv-python-headless==4.10.0.84" \
  matplotlib pillow tqdm pycocotools \
  > "$LOG/deps.log" 2>&1 || { tail -20 "$LOG/deps.log"; exit 1; }
echo "      依赖安装完成（日志：logs/deps.log）"

if "$PY" -c "import segment_anything" >/dev/null 2>&1; then
  echo "      segment-anything 已安装，跳过"
else
  "$PY" -m pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/segment-anything.git" \
    > "$LOG/sam_install.log" 2>&1 || { tail -20 "$LOG/sam_install.log"; exit 1; }
fi

# ---------- 6. 可选加速组件 ----------
echo "[6/6] 检查可选加速组件 ..."
for pkg in tensorrt onnx onnxruntime; do
  if "$PY" -c "import ${pkg//-/_}" >/dev/null 2>&1; then
    echo "      ${pkg}: 已安装"
  else
    echo "      ${pkg}: 未安装（可选，如需 TensorRT 加速可后续安装）"
  fi
done

# ---------- 自检 ----------
echo
echo "[自检] 运行 environment selfcheck ..."
"$PY" "$ROOT/py/selfcheck.py" || true

echo
echo "=============================================="
echo " 完成！下一步："
echo "   ./sh/download.sh                 # 下载 ViT-B（375MB，推荐先用）"
echo "   ./sh/download.sh sam_vit_h        # 或下载 ViT-H（2.4GB，精度最高）"
echo
echo " 然后运行推理："
echo "   ./sh/run.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_point.png"
echo "=============================================="