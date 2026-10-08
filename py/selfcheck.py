#!/usr/bin/env python3
"""环境自检：确认 SAM 在 Jetson 上可用。

检查项：
  1. 关键模块可导入（torch / torchvision / segment_anything）
  2. CUDA 可用、设备名称、显存
  3. 本地权重存在且大小合理
  4. torchvision NMS 是否可用（不可用会自动打纯 PyTorch 补丁）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
ok = True


def check(desc: str, passed: bool, detail: str = "") -> None:
    global ok
    mark = "✅" if passed else "❌"
    print(f"  {mark} {desc}" + (f"  —— {detail}" if detail else ""))
    if not passed:
        ok = False


print("=" * 46)
print(" SAM_AGXOrin 环境自检")
print("=" * 46)

# 1. 模块导入
try:
    import torch
    import torchvision
    import segment_anything
    from segment_anything import sam_model_registry, SamPredictor  # noqa: F401
    check("导入 torch / torchvision / segment_anything", True,
          f"torch={torch.__version__} tv={torchvision.__version__}")
except Exception as e:  # noqa: BLE001
    check("导入 torch / torchvision / segment_anything", False, repr(e))
    print("\n[提示] 请先执行： ./sh/setup.sh")
    sys.exit(1)

# 2. CUDA
try:
    cuda = torch.cuda.is_available()
    detail = torch.cuda.get_device_name(0) if cuda else "不可用"
    if cuda:
        free, total = torch.cuda.mem_get_info()
        detail += f" | 显存 {total / 2**30:.1f} GiB（空闲 {free / 2**30:.1f} GiB）"
    check("CUDA 可用", cuda, detail)
except Exception as e:  # noqa: BLE001
    check("CUDA 可用", False, repr(e))

# 3. torchvision NMS（auto 模式依赖）
try:
    _ = torchvision.ops.nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([1.0]), 0.5)
    check("torchvision C++ NMS 可用", True)
except Exception as e:  # noqa: BLE001
    check("torchvision C++ NMS 可用", False,
          f"{type(e).__name__}；sam_infer.py 会自动使用纯 PyTorch NMS 替代")

# 4. 本地权重
any_ckpt = False
for m, f in (("vit_h", "sam_vit_h_4b8939.pth"),
             ("vit_l", "sam_vit_l_0b3195.pth"),
             ("vit_b", "sam_vit_b_01ec64.pth")):
    ckpt = ROOT / "models" / f
    if ckpt.is_file():
        check(f"权重 {f}", True, f"{ckpt.stat().st_size / 2**20:.0f} MB")
        any_ckpt = True
if not any_ckpt:
    check("本地权重", False, "缺失，请执行 ./sh/download.sh")

print("=" * 46)
if ok:
    print(" 结论：环境就绪，可以开始推理。")
else:
    print(" 结论：存在未通过项，请按上述提示修复。")
sys.exit(0 if ok else 1)