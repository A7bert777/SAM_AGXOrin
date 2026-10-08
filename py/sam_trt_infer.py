#!/usr/bin/env python3
"""SAM + TensorRT 推理（AGX Orin）：图像编码器走 TensorRT，解码器走 PyTorch。

思路
----
① 图像编码器 (ViT, 最重)  ->  用 TensorRT engine 执行   ★提速点
② 提示编码器 + 掩码解码器  ->  仍用 PyTorch (仅十几 ms，不值得转)

这是业界常见做法（如 NVIDIA 的 SAM ONNX 示例），只把「固定尺寸、无动态 shape、
耗时占比 95%」的编码器交给 TensorRT。

用法::

    ./sh/run_trt.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_trt_point.png
    ./sh/run_trt.sh --image inputimage/0000.jpg --box 100 100 800 600 --out outputimage/0000_trt_box.png
    ./sh/run_trt.sh --image inputimage/0000.jpg --point 500 400 --bench

依赖: tensorrt（Python 绑定，JetPack 自带或从 deb 装入 venv）。
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

_T_LAUNCH = time.time()

import numpy as np
import torch

import sam_infer as S
from trt_runner import TRTEngine


def default_engine(model: str) -> Path:
    """按模型类型给出默认 engine 路径（FP16）。"""
    return ROOT / "models" / f"sam_{model}_encoder_fp16.engine"


def build_predictor_with_trt_encoder(sam, engine):
    """返回一个 SamPredictor，但其 image_encoder 被替换成 TensorRT 版本。

    这样 set_image() 就会走 TensorRT，而 predict() 保持原样。
    """
    predictor = S.build_predictor(sam)

    class TRTEncoderWrapper(torch.nn.Module):
        """伪装成 SAM 的 image_encoder，输出同样的 embedding。"""

        def __init__(self, eng):
            super().__init__()
            self.eng = eng

        def forward(self, x):
            out = self.eng({"image": x.contiguous()})
            return out[self.eng.output_names[0]]

    predictor.model.image_encoder = TRTEncoderWrapper(engine)
    return predictor


def preprocess_image(image_rgb, image_size=1024):
    """RGB HWC uint8 -> 归一化 NCHW float32 (1,3,S,S)，复刻 SAM 的 ResizeLongestSide。"""
    import cv2

    h, w = image_rgb.shape[:2]
    scale = image_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h, pad_w = image_size - nh, image_size - nw
    padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    x = padded.astype(np.float32) / 255.0
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32) / 255.0
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32) / 255.0
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(x.astype(np.float32))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM + TensorRT 推理",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--image", required=True, help="输入图像")
    p.add_argument("--engine", default=None, help="TensorRT engine 路径（默认按模型推断）")
    p.add_argument("--model", default=None, choices=list(S.CKPT_MAP), help="SAM 模型类型")
    p.add_argument("--ckpt", default=None, help="SAM 权重（默认按模型推断）")
    p.add_argument("--point", nargs="+", type=float, default=None, metavar="X Y", help="点提示")
    p.add_argument("--point-label", nargs="+", type=int, default=None, help="每点标签")
    p.add_argument("--box", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                   help="框提示 xyxy")
    p.add_argument("--out", default=None, help="输出可视化图像")
    p.add_argument("--mask-dir", default=None, help="导出每个掩码 PNG 到此目录")
    p.add_argument("--bench", action="store_true", help="对比 PyTorch 与 TensorRT 编码器耗时")
    p.add_argument("--repeat", type=int, default=5, help="基准重复次数")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    p.add_argument("--no-tf32", action="store_true", help="禁用 TF32")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not Path(args.image).is_file():
        sys.exit(f"[错误] 找不到图像: {args.image}")
    if args.point is None and args.box is None:
        sys.exit("[错误] 必须提供 --point 或 --box")

    args.model = S.resolve_model(args)
    engine_path = Path(args.engine) if args.engine else default_engine(args.model)
    if not engine_path.is_file():
        sys.exit(f"[错误] 找不到 engine: {engine_path}\n"
                 f"       请先执行: ./venv310/bin/python py/export_onnx.py --model {args.model}\n"
                 f"                  ./sh/build_engine.sh models/sam_{args.model}_encoder.onnx")

    ckpt = args.ckpt or str(ROOT / "models" / S.CKPT_MAP[args.model])
    if not Path(ckpt).is_file():
        sys.exit(f"[错误] 找不到权重: {ckpt}，请先执行 ./sh/download.sh")

    sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (lambda: None)

    print("=" * 68)
    print(" SAM + TensorRT 推理")
    print("=" * 68)
    print(f" engine   : {engine_path}")
    print(f" 权重     : {Path(ckpt).name}")
    print(f" 图片     : {args.image}")
    print("=" * 68)

    image = S.load_image(args.image, announce=True)

    # ---- 加载 SAM（为了拿到 prompt encoder + mask decoder）----
    print("\n[1/4] 加载 SAM 权重 ...")
    t0 = time.time()
    sam, device = S.build_model(argparse.Namespace(
        model=args.model, ckpt=args.ckpt, device=args.device, no_tf32=args.no_tf32))
    args.device = device
    print(f"      耗时 {time.time() - t0:.2f}s")

    # ---- 加载 TensorRT engine ----
    print("\n[2/4] 加载 TensorRT engine ...")
    t0 = time.time()
    engine = TRTEngine(engine_path)
    print(engine.info())
    print(f"      耗时 {time.time() - t0:.2f}s")

    predictor = build_predictor_with_trt_encoder(sam, engine)

    # ---- 基准对比 ----
    if args.bench:
        print(f"\n[基准] 对比编码器耗时（各跑 {args.repeat} 次）...")
        arr = preprocess_image(image)
        t = torch.from_numpy(arr).to(device)

        torch_enc = copy.deepcopy(sam.image_encoder).to(device).eval()
        with torch.no_grad():
            torch_enc(t)  # 预热
            sync()
            ts = []
            for _ in range(args.repeat):
                sync(); t0 = time.time(); torch_enc(t); sync()
                ts.append((time.time() - t0) * 1000)

        engine({"image": t})  # 预热
        trs = []
        for _ in range(args.repeat):
            sync(); t0 = time.time(); engine({"image": t}); sync()
            trs.append((time.time() - t0) * 1000)

        pt, trt = np.mean(ts), np.mean(trs)
        print(f"      PyTorch 编码器  : {pt:8.1f} ms")
        print(f"      TensorRT 编码器 : {trt:8.1f} ms")
        print(f"      加速比          : {pt/trt:8.2f} x")

    # ---- set_image（走 TensorRT）----
    print("\n[3/4] 图像编码（TensorRT）...")
    args.point_pairs, args.point_labels_p = S.parse_points(args)
    t0 = time.time()
    predictor.set_image(image)
    sync()
    print(f"      set_image（TRT 前向）: {(time.time() - t0)*1000:.1f} ms")

    # ---- 提示 ----
    print("\n[4/4] 提示分割 ...")
    masks, scores, _, _ = S.run_once(predictor, image, args)
    if args.point is not None:
        desc = f"point{args.point_pairs}"
    else:
        desc = f"box{list(args.box)}"
    S.report(image, masks, scores, args, desc + " @TRT")

    print(f"[计时] 端到端(含初始化): {time.time() - _T_LAUNCH:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())