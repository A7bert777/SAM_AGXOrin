#!/usr/bin/env bash
# 下载 SAM 权重（支持断点续传，重复执行即可）。
#
# 用法:
#   ./sh/download.sh                # 默认下载 ViT-B（375MB）
#   ./sh/download.sh sam_vit_b      # ViT-B  375MB  最快，推荐入门
#   ./sh/download.sh sam_vit_l      # ViT-L  1.2GB  平衡
#   ./sh/download.sh sam_vit_h      # ViT-H  2.4GB  精度最高
#   ./sh/download.sh all            # 下载全部
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

# 权重清单: key -> 文件名|URL
declare -A SAM_FILES=(
  [sam_vit_b]="sam_vit_b_01ec64.pth|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
  [sam_vit_l]="sam_vit_l_0b3195.pth|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
  [sam_vit_h]="sam_vit_h_4b8939.pth|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)

download_one() {
  local key="$1"
  local entry="${SAM_FILES[$key]:-}"
  if [ -z "$entry" ]; then
    echo "  [错误] 未知权重: $key  (可选: sam_vit_b / sam_vit_l / sam_vit_h / all)"
    return 1
  fi

  local fname url dest
  fname="${entry%%|*}"
  url="${entry#*|}"
  dest="$MODELS_DIR/$fname"

  if [ -f "$dest" ]; then
    echo "  [跳过] $fname 已存在 ($(du -h "$dest" | cut -f1))"
    return 0
  fi

  echo "  正在下载 $key ..."
  echo "    -> $dest"
  if ! curl -L --fail --retry 5 --retry-delay 3 -C - -o "$dest" "$url"; then
    echo "  [错误] 下载失败: $url"
    return 1
  fi
  echo "  [完成] $fname ($(du -h "$dest" | cut -f1))"
  return 0
}

TARGET="${1:-sam_vit_b}"

echo "============================================================"
echo " SAM 权重下载"
echo "============================================================"
echo " 权重目录 : $MODELS_DIR"
echo " 目标     : $TARGET"
echo "============================================================"

if [ "$TARGET" = "all" ]; then
  for k in sam_vit_b sam_vit_l sam_vit_h; do
    download_one "$k" || true
    echo ""
  done
else
  download_one "$TARGET"
fi

echo
echo "============================================================"
echo " 当前已下载的权重:"
ls -lh "$MODELS_DIR"/*.pth 2>/dev/null | awk '{printf "   %-40s %s\n", $9, $5}' || echo "   (无)"
echo "============================================================"
echo " 下一步: ./sh/run.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_point.png"
echo "============================================================"