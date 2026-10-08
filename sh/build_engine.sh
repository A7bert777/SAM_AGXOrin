#!/usr/bin/env bash
# 用 trtexec 把 SAM 图像编码器 ONNX 构建成 TensorRT engine
#
# 用法:
#   ./sh/build_engine.sh <onnx文件>              # 默认 fp16
#   ./sh/build_engine.sh <onnx文件> tf32         # TF32（Ampere+ 默认精度，精度高、速度快）
#   ./sh/build_engine.sh <onnx文件> fp32         # 纯 FP32（禁用 TF32，最精确、最慢）
#   ./sh/build_engine.sh <onnx文件> fp16         # FP16（最快、内存最小）
#   ./sh/build_engine.sh <onnx文件> <精度> [额外 trtexec 参数...]
#
# 输出:
#   <同名>_<精度>.engine    与 onnx 同目录（默认在 models/）
#
# 三档精度的实现差异（TensorRT 语义）:
#   - TF32 是 FP32 的一种「加速模式」，默认就开启，不是独立开关：
#       tf32 -> 不加任何精度 flag（既不加 --fp16，也不加 --noTF32）
#               卷积 / matmul 用 TF32 计算，权重仍为 FP32 存储 → 精度接近 FP32、速度接近 FP16
#       fp32 -> 加 --noTF32（显式禁用 TF32，纯 FP32 计算，最精确但最慢）
#       fp16 -> 加 --fp16（权重与计算 FP16，最快、显存占用最小，精度略降）
#
# 说明:
#   - 构建只做一次，之后推理直接加载 .engine（几秒即可）
#   - engine 绑定本机 GPU 型号 + TensorRT 版本，换设备需重建
#   - SAM 是 Transformer，不要加 --useDLACore（DLA 不支持 attention）
set -euo pipefail
cd "$(dirname "$0")/.."

ONNX="${1:-}"
PRECISION="${2:-fp16}"
# 按“实际传入的参数个数”位移，不能无条件 shift 2：
# 当只传 1 个参数时，bash 的 `shift 2` 会失败且不位移，导致 $@ 仍含 ONNX 路径，
# 被下面的 EXTRA="$*" 当作额外参数追加到 trtexec 末尾，触发 "Unknown option"。
if [ "$#" -ge 2 ]; then
  shift 2
elif [ "$#" -ge 1 ]; then
  shift 1
fi
EXTRA="$*"

TRTEXEC="/usr/src/tensorrt/bin/trtexec"
[ -x "$TRTEXEC" ] || TRTEXEC="$(command -v trtexec || true)"
if [ -z "${TRTEXEC:-}" ] || [ ! -x "$TRTEXEC" ]; then
  echo "[错误] 找不到 trtexec，请检查 TensorRT 安装"
  exit 1
fi

if [ -z "$ONNX" ]; then
  echo "[错误] 用法: ./sh/build_engine.sh <onnx文件> [fp16|tf32|fp32] [额外 trtexec 参数]"
  echo "  例: ./sh/build_engine.sh models/sam_vit_h_encoder.onnx tf32"
  exit 1
fi

if [ ! -f "$ONNX" ]; then
  echo "[错误] ONNX 文件不存在: $ONNX"
  echo "       请先运行: ./venv310/bin/python py/export_onnx.py --model vit_h"
  exit 1
fi

ONNX_ABS="$(cd "$(dirname "$ONNX")" && pwd)/$(basename "$ONNX")"
ENGINE="${ONNX_ABS%.onnx}_${PRECISION}.engine"
LOG="logs/engine_$(basename "${ONNX%.onnx}")_${PRECISION}.log"
mkdir -p logs

echo "============================================================"
echo " 构建 TensorRT engine"
echo "============================================================"
echo " ONNX     : $ONNX_ABS"
echo " 精度     : $PRECISION"
echo " 输出     : $ENGINE"
echo " trtexec  : $TRTEXEC"
echo " 日志     : $LOG"
echo "============================================================"
echo ""
echo "[提示] 构建过程可能耗时数分钟（TensorRT 会为每层做 kernel 自动调优）"
echo ""

# 精度 -> trtexec flag 映射
#   tf32 : 不传任何精度 flag（TF32 是 fp32 的默认加速模式，本来就开着）
#   fp32 : --noTF32 显式关闭 TF32，得到纯 fp32
#   fp16 : --fp16 启用半精度
case "$PRECISION" in
  fp16|FP16) PREC_FLAG="--fp16"; PREC_DESC="FP16（最快，精度略降）" ;;
  tf32|TF32) PREC_FLAG="";       PREC_DESC="TF32（默认：FP32 精度 + Ampere 张量核加速）" ;;
  fp32|FP32) PREC_FLAG="--noTF32"; PREC_DESC="FP32（关闭 TF32，最精确、最慢）" ;;
  *) echo "[错误] 未知精度: $PRECISION (可选 fp16 / tf32 / fp32)"; exit 1 ;;
esac
echo " 精度说明 : $PREC_DESC"
echo ""

START=$(date +%s)
echo " 开始构建 ..."
# --memPoolSize=workspace:4096 : 给 TensorRT 4GB 工作区做 kernel 调优
# shellcheck disable=SC2086
"$TRTEXEC" \
  --onnx="$ONNX_ABS" \
  --saveEngine="$ENGINE" \
  $PREC_FLAG \
  --memPoolSize=workspace:4096 \
  $EXTRA \
  > "$LOG" 2>&1 || {
    echo "❌ 构建失败，日志尾部:"
    tail -40 "$LOG"
    exit 1
  }
END=$(date +%s)
DUR=$((END - START))

echo ""
if [ -f "$ENGINE" ]; then
  echo "============================================================"
  echo " ✅ engine 构建成功"
  echo "============================================================"
  echo " 文件 : $ENGINE"
  echo " 大小 : $(du -h "$ENGINE" | cut -f1)"
  echo " 耗时 : ${DUR}s"
  echo " 日志 : $LOG"
  echo ""
  echo " 关键耗时信息 (来自日志):"
  grep -iE "Engine built|Total Host Walltime|Throughput|GPU Compute Time" "$LOG" | head -8 || true
  echo ""
  echo " 下一步：用 TensorRT 推理"
  echo "   ./sh/run_trt.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_trt.png"
else
  echo "============================================================"
  echo " ❌ engine 构建失败，请查看日志: $LOG"
  echo "============================================================"
  exit 1
fi