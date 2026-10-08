#!/usr/bin/env python3
"""将 SAM 的「图像编码器」(Image Encoder, ViT) 导出为 ONNX，供 TensorRT 加速。

为什么要只导出编码器？
    SAM 推理分两段:
      ① 图像编码器 (ViT)       —— 重！1024x1024 输入，占约 95% 耗时
      ② 提示编码器 + 掩码解码器 —— 轻，仅十几 ms
    TensorRT 的收益几乎全部来自 ①，且 ① 是固定尺寸、无动态 shape，
    最适合 TensorRT 优化。解码器保持 PyTorch 即可（已足够快）。

用法::

    # 默认输出到 models/sam_<type>_encoder.onnx
    ./venv310/bin/python py/export_onnx.py --model vit_h

    # 自定义输出
    ./venv310/bin/python py/export_onnx.py --model vit_h --output models/sam_h_encoder.onnx

导出后构建 engine::

    ./sh/build_engine.sh models/sam_vit_h_encoder.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import torch

import sam_infer as S


class EncoderWrapper(torch.nn.Module):
    """包一层，确保「单输入、单输出」，避免 ONNX 图里出现多余节点。"""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, image):
        return self.enc(image)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="导出 SAM 图像编码器为 ONNX (供 TensorRT 加速)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default=None, choices=list(S.CKPT_MAP), help="模型类型")
    p.add_argument("--ckpt", default=None, help=".pth 权重路径（默认按模型推断）")
    p.add_argument("--output", default=None,
                   help="输出 onnx 路径（默认 models/sam_<type>_encoder.onnx）")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    p.add_argument("--image-size", type=int, default=1024, help="编码器输入边长")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.model = S.resolve_model(args)

    ckpt = args.ckpt or str(ROOT / "models" / S.CKPT_MAP[args.model])
    if not Path(ckpt).is_file():
        sys.exit(f"[错误] 找不到权重文件: {ckpt}，请先执行 ./sh/download.sh")

    from segment_anything import sam_model_registry

    out = args.output or str(ROOT / "models" / f"sam_{args.model}_encoder.onnx")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    print("=" * 68)
    print(" SAM 图像编码器  ->  ONNX")
    print("=" * 68)
    print(f" 权重      : {ckpt}")
    print(f" 模型类型  : {args.model}")
    print(f" 输出      : {out}")
    print(f" opset     : {args.opset}")
    print(f" 输入尺寸  : 1 x 3 x {args.image_size} x {args.image_size}")
    print("=" * 68)

    # ---- 1. 加载 PyTorch 模型 ----
    print("\n[1/3] 加载 PyTorch 模型 ...")
    t0 = time.time()
    sam = sam_model_registry[args.model](checkpoint=ckpt)
    sam.eval()
    print(f"      加载耗时 : {time.time() - t0:.2f}s")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("      [警告] CUDA 不可用，回退 CPU")
        device = "cpu"

    encoder = sam.image_encoder.float().to(device).eval()
    print(f"      device   : {device}")

    # ---- 2. 导出 ONNX ----
    wrapper = EncoderWrapper(encoder).eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)

    print("\n[2/3] 导出 ONNX ...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            out,
            input_names=["image"],
            output_names=["image_embeddings"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,          # 固定尺寸 -> 最利于 TensorRT 优化
            export_params=True,
        )
    print(f"      导出耗时 : {time.time() - t0:.2f}s")
    print(f"      文件大小 : {os.path.getsize(out) / 1024**2:.1f} MB")

    # ---- 3. 校验 ----
    print("\n[3/3] 校验 ONNX ...")
    try:
        import onnx
        model = onnx.load(out)
        onnx.checker.check_model(model)
        ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
               for i in model.graph.input]
        outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                for o in model.graph.output]
        print("      ✅ 校验通过")
        print(f"      输入 : {ins}")
        print(f"      输出 : {outs}")
    except ImportError:
        print("      (未安装 onnx，跳过校验)")
    except Exception as e:  # noqa: BLE001
        print(f"      [警告] 校验异常: {e}")

    # ---- 下一步提示 ----
    engine_path = os.path.abspath(out).replace(".onnx", "_fp16.engine")
    print("\n" + "=" * 68)
    print(" 下一步：构建 TensorRT engine")
    print("=" * 68)
    print(f"  ./sh/build_engine.sh {out}")
    print("\n 或手动执行 trtexec:")
    print(f"  /usr/src/tensorrt/bin/trtexec --onnx={out} \\")
    print(f"      --saveEngine={engine_path} \\")
    print("      --fp16 --memPoolSize=workspace:4096")
    return 0


if __name__ == "__main__":
    sys.exit(main())