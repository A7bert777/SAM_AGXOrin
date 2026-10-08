#!/usr/bin/env bash
# 环境自检：确认 ONNX / TensorRT / onnxruntime 的可用性与版本
set -u
cd "$(dirname "$0")/.."

PY=./venv310/bin/python
export LD_LIBRARY_PATH="$PWD/venv310/lib/python3.10/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"

echo "==================== 环境自检 ===================="
echo "--- machine ---"
if [ -f /etc/nv_tegra_release ]; then cat /etc/nv_tegra_release; fi
echo

echo "--- python 包 ---"
$PY - <<'PYEOF'
import importlib

def show(name):
    try:
        m = importlib.import_module(name)
        print(f"  {name:16s} OK    {getattr(m, '__version__', '?')}")
        return m
    except Exception as e:
        print(f"  {name:16s} MISS  {type(e).__name__}: {e}")
        return None

show("torch")
show("torchvision")
show("segment_anything")
show("onnx")
ort = show("onnxruntime")
if ort is not None:
    print(f"  onnxruntime providers = {ort.get_available_providers()}")
trt = show("tensorrt")
if trt is not None and hasattr(trt, "Builder"):
    try:
        print(f"  tensorrt builder_available = {trt.Builder(trt.Logger(trt.Logger.ERROR)) is not None}")
    except Exception as e:
        print(f"  tensorrt builder init failed: {e}")
PYEOF
echo

echo "--- trtexec ---"
TRTEXEC=""
for c in "$(command -v trtexec 2>/dev/null || true)" \
         /usr/src/tensorrt/bin/trtexec \
         /usr/local/tensorrt/bin/trtexec \
         /opt/tensorrt/bin/trtexec; do
  if [ -n "$c" ] && [ -x "$c" ]; then TRTEXEC="$c"; break; fi
done
if [ -n "$TRTEXEC" ]; then
  echo "  trtexec = $TRTEXEC"
  "$TRTEXEC" --version 2>&1 | head -5 | sed 's/^/    /'
else
  echo "  trtexec = 未找到（将回退到 TensorRT Python API）"
fi
echo

echo "--- 已导出的 ONNX / engine ---"
ls -lh models/*.onnx models/*.engine 2>/dev/null | sed 's/^/  /' || echo "  (无)"
echo

echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  nvidia-smi 不可用（Jetson 统一内存，属正常）"
echo "=================================================="