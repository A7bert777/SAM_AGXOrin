#!/usr/bin/env bash
# SAM_AGXOrin 一键环境初始化（Jetson AGX Orin / JetPack, aarch64, Python 3.10）
#
# 步骤：
#   1. 环境自检（Jetson / python3.10 / CUDA / 磁盘）
#   2. 创建 venv310（若已存在则复用）
#   3. 确保 pip 可用
#   4. 安装 NVIDIA 定制 PyTorch + torchvision（Jetson 专用 wheel）
#   5. 安装 segment-anything 及其依赖
#   6. 安装 ONNX 相关组件（onnx 为 ONNX 导出必需；保持 numpy<2）
#   7. 下载 SAM 权重（默认 ViT-B）
#
# 用法：./sh/setup.sh
# 说明：若已有可用的 venv310（内含 torch），本脚本会直接复用，不会重新下载 PyTorch。
#       权重选择（可选）：
#         SAM_WEIGHTS=sam_vit_h ./sh/setup.sh   # 下载 ViT-H（2.4GB）
#         SAM_WEIGHTS=all       ./sh/setup.sh   # 下载全部权重
#         SAM_SKIP_WEIGHTS=1    ./sh/setup.sh   # 跳过权重下载
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
echo "[1/7] 环境自检 ..."
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
  echo "[2/7] 复用已有 venv: $VENV"
else
  echo "[2/7] 创建 venv310 ..."
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
echo "[3/7] 检查 pip ..."
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
echo "[4/7] 安装 PyTorch ${TORCH_VERSION} + torchvision ${TORCHVISION_VERSION} ..."
if "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "      torch 已就绪: $("$PY" -c 'import torch; print(torch.__version__)')，跳过"
else
  echo "      源: $JETSON_PYPI_INDEX（约 1GB，请耐心等待）"

  # 4.0 确保 tqdm 可用（下载器的实时进度条）
  if ! "$PY" -c "import tqdm" >/dev/null 2>&1; then
    echo "      安装 tqdm（用于实时下载进度条）..."
    "$PY" -m pip install --no-cache-dir tqdm >/dev/null 2>&1 \
      || echo "      [警告] tqdm 安装失败，将使用内置进度条"
  fi

  # 4.1 用带 tqdm 进度条的下载器抓取 wheel 到本地 wheels/
  #     （断点续传 + sha256 校验；重复运行会自动跳过已完成的文件）
  WHEELS="$ROOT/wheels"
  mkdir -p "$WHEELS"
  echo "      下载 wheel（tqdm 进度条，支持断点续传）..."
  "$PY" "$ROOT/py/fetch_wheels.py" \
    --index-url "$JETSON_PYPI_INDEX" --dest "$WHEELS" \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    || { echo "      [错误] wheel 下载失败，请重试（会从断点续传）"; exit 1; }

  # 4.2 磁盘空间预检（torch 解压约需 2GB+）
  AVAIL_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
  AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
  if [ "$AVAIL_GB" -lt 4 ] && [ "${SAM_SKIP_DISK_CHECK:-0}" != "1" ]; then
    echo "      [提示] 磁盘仅剩 ${AVAIL_GB}GB，先清理 pip 缓存 ..."
    "$PY" -m pip cache purge >/dev/null 2>&1 || true
    find "$WHEELS" -name '*.part' -delete 2>/dev/null || true
    AVAIL_KB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
    AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
  fi
  if [ "$AVAIL_GB" -lt 4 ] && [ "${SAM_SKIP_DISK_CHECK:-0}" != "1" ]; then
    echo "      [错误] 磁盘可用空间不足（仅 ${AVAIL_GB}GB，安装 torch 需 ≥4GB）。"
    echo "             请先释放空间后重试，例如："
    echo "               du -sh /home/*/.cache/* 2>/dev/null | sort -h | tail"
    echo "               sudo apt clean; pip cache purge; rm -rf ~/.cache/pip"
    echo "             如确认空间足够，可跳过本检查：SAM_SKIP_DISK_CHECK=1 ./sh/setup.sh"
    exit 1
  fi

  # 4.3 用本地 wheel 离线安装 torch / torchvision 本体
  #     关键：必须用 --no-index，否则 pip 在同版本下会优先从索引重新下载（白下一遍）！
  #     注意：此处用 --no-deps，torch 的依赖（typing-extensions 等）在 4.4 才安装，
  #           所以这里只校验 pip 退出码，不能校验 `import torch`（此时依赖尚不完整）。
  echo "      离线安装本地 wheel（--no-index，不再联网重下）..."
  set +e
  "$PY" -m pip install --no-cache-dir --no-index --find-links "$WHEELS" \
    --no-deps --progress-bar on \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    2>&1 | tee "$LOG/torch_install.log"
  pip_rc=${PIPESTATUS[0]}
  set -e
  if [ "$pip_rc" -ne 0 ]; then
    echo "      [错误] 本地 wheel 安装失败，日志：logs/torch_install.log"
    tail -20 "$LOG/torch_install.log"
    exit 1
  fi

  # 4.4 安装 torch / torchvision 的依赖（体积小，从默认 PyPI 获取）
  #     此时 torch 本体已装好，pip 不会再下载大 wheel
  echo "      安装运行依赖（体积小，pip 进度条已开启）..."
  set +e
  "$PY" -m pip install --no-cache-dir --progress-bar on \
    filelock "typing-extensions>=4.10.0" "sympy>=1.13.3" \
    "networkx>=2.5.1" jinja2 "fsspec>=0.8.5" "numpy<2" pillow \
    2>&1 | tee -a "$LOG/torch_install.log"
  dep_rc=${PIPESTATUS[0]}
  set -e
  if [ "$dep_rc" -ne 0 ]; then
    echo "      [错误] 依赖安装失败，日志：logs/torch_install.log"
    tail -20 "$LOG/torch_install.log"
    exit 1
  fi
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
echo "[5/7] 安装 segment-anything 及其依赖 ..."
# 关键: numpy 必须 < 2（NVIDIA 定制 torch 按 numpy 1.x 编译）
# 实时显示进度并同时写日志（依赖体积较小）
set +e
"$PY" -m pip install --no-cache-dir --progress-bar on \
  "numpy<2" "opencv-python-headless==4.10.0.84" \
  matplotlib pillow tqdm pycocotools \
  2>&1 | tee "$LOG/deps.log"
dep_rc=${PIPESTATUS[0]}
set -e
if [ "$dep_rc" -ne 0 ]; then
  echo "      [错误] 依赖安装失败，日志：logs/deps.log"
  tail -20 "$LOG/deps.log"
  exit 1
fi
echo "      依赖安装完成（日志：logs/deps.log）"

if "$PY" -c "import segment_anything" >/dev/null 2>&1; then
  echo "      segment-anything 已安装，跳过"
else
  set +e
  "$PY" -m pip install --no-cache-dir --progress-bar on \
    "git+https://github.com/facebookresearch/segment-anything.git" \
    2>&1 | tee "$LOG/sam_install.log"
  sam_rc=${PIPESTATUS[0]}
  set -e
  if [ "$sam_rc" -ne 0 ]; then
    echo "      [错误] segment-anything 安装失败，日志：logs/sam_install.log"
    tail -20 "$LOG/sam_install.log"
    exit 1
  fi
fi

# ---------- 6. ONNX / TensorRT 组件 ----------
echo "[6/7] 安装 ONNX 相关组件 ..."
# torch.onnx.export 依赖 onnx 包，缺失会报 "Module onnx is not installed!"。
# 关键：必须与 numpy<2 一起装！onnx 默认会把 numpy 升到 2.x，而 NVIDIA 定制
# torch 是按 NumPy 1.x 编译的，升级后会报
# "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"，
# 导致 import torch 异常、ONNX 导出结果损坏。
if "$PY" -c "import onnx" >/dev/null 2>&1; then
  echo "      onnx: 已安装 ($("$PY" -c 'import onnx; print(onnx.__version__)'))"
else
  echo "      安装 onnx（ONNX 导出必需，体积小）..."
  set +e
  "$PY" -m pip install --no-cache-dir --progress-bar on "numpy<2" onnx \
    2>&1 | tee "$LOG/onnx_install.log"
  onnx_rc=${PIPESTATUS[0]}
  set -e
  if [ "$onnx_rc" -ne 0 ]; then
    echo "      [警告] onnx 安装失败（ONNX 导出将不可用），日志：logs/onnx_install.log"
  else
    echo "      onnx: $("$PY" -c 'import onnx; print(onnx.__version__)')"
  fi
fi

# 兜底：确认 numpy 仍是 1.x（防止 pip 解析依赖时再次升级）
if ! "$PY" -c "import numpy; raise SystemExit(0 if numpy.__version__.split('.')[0]=='1' else 1)" >/dev/null 2>&1; then
  echo "      [警告] numpy 被升级到 2.x，回退到 <2 ..."
  "$PY" -m pip install --no-cache-dir "numpy<2" >/dev/null 2>&1 || true
fi

# 可选：tensorrt（JetPack 自带）、onnxruntime（CPU 校验用）
for pkg in tensorrt onnxruntime; do
  if "$PY" -c "import ${pkg//-/_}" >/dev/null 2>&1; then
    echo "      ${pkg}: 已安装"
  else
    echo "      ${pkg}: 未安装（可选）"
  fi
done

# ---------- 7. SAM 权重 ----------
echo "[7/7] 下载 SAM 权重 ..."
# 通过环境变量选择权重（默认 ViT-B）：sam_vit_b / sam_vit_l / sam_vit_h / all
# 设置 SAM_SKIP_WEIGHTS=1 可跳过本步骤
declare -A SAM_FILES=(
  [sam_vit_b]="sam_vit_b_01ec64.pth|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
  [sam_vit_l]="sam_vit_l_0b3195.pth|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
  [sam_vit_h]="sam_vit_h_4b8939.pth|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

download_weight() {
  local key="$1"
  local entry="${SAM_FILES[$key]:-}"
  if [ -z "$entry" ]; then
    echo "      [错误] 未知权重: $key (可选: sam_vit_b / sam_vit_l / sam_vit_h / all)"
    return 1
  fi
  local fname url dest
  fname="${entry%%|*}"
  url="${entry#*|}"
  dest="$MODELS_DIR/$fname"

  if [ -f "$dest" ]; then
    echo "      [跳过] $fname 已存在 ($(du -h "$dest" | cut -f1))"
    return 0
  fi

  echo "      下载 $key -> $dest"
  # -C -：断点续传；--retry：失败自动重试；进度条由 curl 实时显示
  if ! curl -L --fail --retry 5 --retry-delay 3 -C - -o "$dest" "$url"; then
    echo "      [错误] 下载失败: $url"
    return 1
  fi
  echo "      [完成] $fname ($(du -h "$dest" | cut -f1))"
  return 0
}

if [ "${SAM_SKIP_WEIGHTS:-0}" = "1" ]; then
  echo "      SAM_SKIP_WEIGHTS=1，跳过权重下载"
else
  SAM_WEIGHTS_TARGET="${SAM_WEIGHTS:-sam_vit_b}"
  echo "      目标: $SAM_WEIGHTS_TARGET（可用 SAM_WEIGHTS 选择: sam_vit_b / sam_vit_l / sam_vit_h / all）"
  if [ "$SAM_WEIGHTS_TARGET" = "all" ]; then
    for k in sam_vit_b sam_vit_l sam_vit_h; do
      download_weight "$k" || true
      echo ""
    done
  else
    download_weight "$SAM_WEIGHTS_TARGET" || true
  fi
  echo "      当前已下载的权重:"
  ls -lh "$MODELS_DIR"/*.pth 2>/dev/null | awk '{printf "        %-40s %s\n", $9, $5}' || echo "        (无)"
fi

# ---------- 自检 ----------
echo
echo "[自检] 运行 environment selfcheck ..."
"$PY" "$ROOT/py/selfcheck.py" || true

echo
echo "=============================================="
echo " 完成！环境与权重均已就绪。"
echo
echo " 如需下载其它权重："
echo "   ./sh/download.sh sam_vit_l        # ViT-L（1.2GB）"
echo "   ./sh/download.sh sam_vit_h        # ViT-H（2.4GB，精度最高）"
echo
echo " 运行推理："
echo "   ./sh/run.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_point.png"
echo "=============================================="
