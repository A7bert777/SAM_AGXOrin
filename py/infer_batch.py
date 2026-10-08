#!/usr/bin/env python3
"""SAM 批量推理：inputimage/ -> outputimage/

功能:
  读取 inputimage/ 下所有图片，逐张做「点提示分割」（默认取图像中心点），
  结果保存到 outputimage/，命名格式为: <输入图像原名><后缀>
  例如: inputimage/0000.jpg -> outputimage/0000_out.png

设计要点:
  * 模型只加载一次，循环处理所有图片（避免每张图重复加载权重）
  * 复用 sam_infer.py 的 build_model / run_image，保证可视化风格一致

用法::

    ./venv310/bin/python py/infer_batch.py --model vit_h
    ./venv310/bin/python py/infer_batch.py --model vit_b --suffix _seg.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import sam_infer as S


def collect_images(in_dir: Path):
    """收集目录下的所有图片，去重并按文件名排序。"""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp",
            "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.WEBP")
    imgs = []
    for e in exts:
        imgs.extend(in_dir.glob(e))
    return sorted(set(imgs), key=lambda p: p.name)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM 批量推理: inputimage/ -> outputimage/",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default=None, choices=list(S.CKPT_MAP),
                   help="SAM 模型类型（不给则自动选用本机权重）")
    p.add_argument("--ckpt", default=None, help="自定义权重路径")
    p.add_argument("--input-dir", default=str(ROOT / "inputimage"), help="输入目录")
    p.add_argument("--output-dir", default=str(ROOT / "outputimage"), help="输出目录")
    p.add_argument("--suffix", default="_out.png", help="输出文件名后缀")
    p.add_argument("--auto", action="store_true", help="改用全自动分割（较慢）")
    p.add_argument("--auto-points", type=int, default=16, help="自动分割每边采样点数")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    p.add_argument("--no-tf32", action="store_true", help="禁用 TF32")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.model = S.resolve_model(args)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    S.print_env_info()

    if not in_dir.is_dir():
        sys.exit(f"[错误] 输入目录不存在: {in_dir}")
    imgs = collect_images(in_dir)
    if not imgs:
        sys.exit(f"[错误] 输入目录中没有图片: {in_dir}")

    print("=" * 68)
    print(f" SAM 批量推理  |  模型: {args.model}  |  共 {len(imgs)} 张")
    print(f" 输入: {in_dir}")
    print(f" 输出: {out_dir}   (命名: <原名>{args.suffix})")
    print("=" * 68)

    sam, device = S.build_model(args)
    predictor = S.build_predictor(sam)

    ok, fail = 0, 0
    for n, img_path in enumerate(imgs, 1):
        print("\n" + "-" * 60)
        print(f"[{n}/{len(imgs)}] 处理: {img_path.name}")
        print("-" * 60)

        if args.auto:
            req = argparse.Namespace(
                image=str(img_path), model=args.model, ckpt=args.ckpt, device=device,
                auto=True, auto_points=args.auto_points, no_box=False,
                out=str(out_dir / f"{img_path.stem}{args.suffix}"), mask_dir=None,
                alpha=0.5, max_masks=20)
        else:
            import cv2
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[跳过] 无法读取图片: {img_path}")
                fail += 1
                continue
            h, w = bgr.shape[:2]
            req = argparse.Namespace(
                image=str(img_path), model=args.model, ckpt=args.ckpt, device=device,
                point=[w // 2, h // 2], point_pairs=[(w // 2, h // 2)],
                point_labels_p=[1], box=None, auto=False,
                out=str(out_dir / f"{img_path.stem}{args.suffix}"), mask_dir=None,
                max_masks=20)

        try:
            S.run_image(predictor, req, warmup=False, announce=False)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[失败] {img_path.name}: {e}")
            fail += 1

    print("\n" + "=" * 68)
    print(f" 全部完成!  成功 {ok} 张 / 失败 {fail} 张")
    print(f" 结果目录: {out_dir}")
    for f in sorted(out_dir.glob(f"*{args.suffix}")):
        print(f"   {f.name}   ({f.stat().st_size // 1024} KB)")
    print("=" * 68)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())