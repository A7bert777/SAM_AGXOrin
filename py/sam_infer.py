#!/usr/bin/env python3
"""SAM 图像分割推理（AGX Orin）：点提示 / 框提示 / 全自动分割。

官方接口（segment-anything）::

    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(image)                       # 图像编码（每张图只做一次）
    masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbls)

用法::

    ./sh/run.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_point.png
    ./sh/run.sh --image inputimage/0000.jpg --box 100 100 800 600 --out outputimage/0000_box.png
    ./sh/run.sh --image inputimage/0000.jpg --auto --auto-points 16 --out outputimage/0000_auto.png
    ./sh/run.sh --image inputimage/0000.jpg --bench --repeat 5

本模块同时被 :mod:`sam_server`（常驻服务）复用：服务的每次请求都会调用
:func:`prepare_args` + :func:`run_image`，因此「命令行单次运行」与「常驻服务推理」
走的是**同一套代码**，输出完全一致。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_T_LAUNCH = time.time()          # 进程启动时刻（初始化耗时对比基准）

import numpy as np
import torch

_T_IMPORT = time.time() - _T_LAUNCH   # numpy/torch 导入完成耗时

# 权重文件名映射（与官方发布一致，前缀即官方 MD5 前 6 位）
CKPT_MAP = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
MODEL_ORDER = ("vit_h", "vit_l", "vit_b")   # 未指定时自动选用顺序


# ---------------------------------------------------------------------------
# torchvision C++ ops 兼容补丁（Jetson 定制 torch 与 pip torchvision ABI 不匹配）
# ---------------------------------------------------------------------------
# 本机 NVIDIA 定制版 torch 与 pip 装到的 torchvision 的 C++ 扩展可能不匹配，
# torchvision.ops.nms / batched_nms 会抛:
#     RuntimeError: Couldn't load custom C++ ops
# 而 SAM 的「全自动分割」内部需要 torchvision.ops.batched_nms 做 NMS 去重。
# 这里用纯 PyTorch 实现替换，功能等价，且不依赖 C++ 扩展。
# 点提示 / 框提示不使用 NMS，因此原本就不受影响。
# ---------------------------------------------------------------------------
def _nms_pure(boxes, scores, iou_threshold):
    """纯 PyTorch 实现的 NMS（等价于 torchvision.ops.nms）"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def _batched_nms_pure(boxes, scores, idxs, iou_threshold):
    """纯 PyTorch 实现的 batched NMS（等价于 torchvision.ops.batched_nms）"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    return _nms_pure(boxes + offsets[:, None], scores, iou_threshold)


def patch_torchvision_ops():
    """检测 torchvision C++ ops；不可用则替换为纯 PyTorch 实现。"""
    try:
        import torchvision
        import torchvision.ops
        import torchvision.ops.boxes
    except ImportError:
        return

    # 先探测原生实现是否可用
    try:
        torchvision.ops.nms(
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([1.0]), 0.5
        )
        return  # 原生可用，无需打补丁
    except Exception:  # noqa: BLE001
        pass

    for mod in (torchvision.ops, torchvision.ops.boxes):
        if hasattr(mod, "nms"):
            setattr(mod, "nms", _nms_pure)
        if hasattr(mod, "batched_nms"):
            setattr(mod, "batched_nms", _batched_nms_pure)

    print("[兼容] torchvision C++ ops 不可用，已启用纯 PyTorch NMS 替代实现")


def _ensure_nms_patched():
    """防御性补丁：确保 segment_anything 内部引用的 NMS 也是纯 PyTorch 版"""
    try:
        import segment_anything.automatic_mask_generator as _amg  # noqa
    except Exception:  # noqa: BLE001
        return
    if getattr(_amg, "batched_nms", None) is not _batched_nms_pure:
        try:
            _amg.batched_nms = _batched_nms_pure
        except Exception:  # noqa: BLE001
            pass


patch_torchvision_ops()


# ---------------------------------------------------------------------------
# 环境信息
# ---------------------------------------------------------------------------
def print_env_info():
    print("=" * 68)
    print(" 运行环境信息")
    print("=" * 68)
    print(f" Python       : {sys.version.split()[0]}  ({sys.executable})")
    print(f" PyTorch      : {torch.__version__}")
    print(f" CUDA (torch) : {torch.version.cuda}")
    print(f" CUDA 可用    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" GPU          : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f" 计算能力     : sm_{cap[0]}{cap[1]}")
        props = torch.cuda.get_device_properties(0)
        print(f" 显存总量     : {props.total_memory / 1024**3:.1f} GB")
        print(f" 多处理器数量 : {props.multi_processor_count}")

    try:
        import torchvision
        print(f" torchvision  : {torchvision.__version__}")
    except ImportError:
        print(" torchvision  : 未安装")

    try:
        import segment_anything
        print(f" segment-anything: {getattr(segment_anything, '__version__', 'installed')}")
    except ImportError:
        print(" segment-anything: 未安装")

    # Jetson 特有信息
    for f, label in (
        ("/etc/nv_tegra_release", "L4T"),
        ("/proc/device-tree/model", "设备型号"),
    ):
        if os.path.exists(f):
            with open(f, "r") as fp:
                print(f" {label:<12} : {fp.read().strip()}")

    try:
        import subprocess
        r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            print(f" 功耗模式     : {r.stdout.strip()}")
    except Exception:  # noqa: BLE001
        pass

    print("=" * 68)


# ---------------------------------------------------------------------------
# 参数解析与校验
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SAM 图像分割（点 / 框 / 全自动）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", default=None, help="输入图像路径")
    p.add_argument("--model", default=None, choices=list(CKPT_MAP),
                   help="SAM 模型类型（不给则自动选用本机已下载的权重，优先 vit_h）")
    p.add_argument("--ckpt", default=None, help="自定义权重路径")
    p.add_argument("--point", nargs="+", type=float, default=None, metavar="X Y",
                   help="点提示：像素坐标，成对给出，可多对，如 --point 500 400 320 240")
    p.add_argument("--point-label", nargs="+", type=int, default=None,
                   help="每个点的标签：1 正样本 / 0 负样本，数量需与点一致（默认全 1）")
    p.add_argument("--box", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                   help="框提示：左上角与右下角像素坐标（xyxy）")
    p.add_argument("--auto", action="store_true", help="全自动分割（网格撒点）")
    p.add_argument("--auto-points", type=int, default=32, help="自动分割每边采样点数（调小可提速）")
    p.add_argument("--bench", action="store_true", help="打印耗时基准")
    p.add_argument("--repeat", type=int, default=5, help="基准重复次数")
    p.add_argument("--check", action="store_true", help="仅打印环境信息")
    p.add_argument("--no-box", action="store_true", help="auto 模式只画掩码，不画目标框")
    p.add_argument("--out", default=None, help="输出可视化图像")
    p.add_argument("--mask-dir", default=None, help="导出每个掩码 PNG 到此目录")
    p.add_argument("--alpha", type=float, default=0.5, help="掩码叠加透明度")
    p.add_argument("--max-masks", type=int, default=20, help="最多可视化前 N 个")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    p.add_argument("--no-tf32", action="store_true", help="禁用 TF32")
    return p.parse_args(argv)


def resolve_model(args) -> str:
    """确定使用的模型类型：显式指定 > --ckpt 推断 > 本机已有权重自动选用。"""
    if args.ckpt:
        ckpt = Path(args.ckpt)
        if not ckpt.is_file():
            sys.exit(f"[错误] 找不到权重: {args.ckpt}")
        if args.model:
            return args.model
        for m, f in CKPT_MAP.items():
            if f == ckpt.name:
                return m
        sys.exit("[错误] 无法从 --ckpt 文件名推断模型类型，请同时指定 --model")

    if args.model:
        f = ROOT / "models" / CKPT_MAP[args.model]
        if not f.is_file():
            sys.exit(f"[错误] 找不到权重 {f}\n       请先执行 ./sh/download.sh sam_{args.model}")
        return args.model

    for m in MODEL_ORDER:
        if (ROOT / "models" / CKPT_MAP[m]).is_file():
            print(f"[信息] 未指定 --model，自动选用本机可用的 {m}")
            return m
    sys.exit("[错误] 本机没有任何 SAM 权重，请先执行 ./sh/download.sh")


def parse_points(args):
    """解析 --point 的扁平坐标列表 -> ([(x, y), ...], [label, ...])。"""
    if args.point is None:
        return [], []
    coords = list(args.point)
    if len(coords) < 2 or len(coords) % 2 != 0:
        sys.exit("[错误] --point 需要成对的 x y 坐标，如 --point 500 400 320 240")
    pairs = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    labels = list(args.point_label) if args.point_label else [1] * len(pairs)
    if len(labels) != len(pairs):
        sys.exit(f"[错误] --point-label 数量({len(labels)})与点数({len(pairs)})不一致")
    return pairs, labels


def prepare_args(argv=None):
    """解析命令行并做基本校验（本地运行与常驻服务共用）。

    失败时抛 ``SystemExit``（携带中文错误信息），由调用方决定是退出进程还是回给客户端。
    """
    args = parse_args(argv)
    if args.check:
        return args
    if not args.image:
        sys.exit("[错误] 必须提供 --image")
    if not Path(args.image).is_file():
        sys.exit(f"[错误] 找不到图像: {args.image}")
    if args.point is None and args.box is None and not args.auto:
        sys.exit("[错误] 必须提供 --point、--box 或 --auto 之一")
    args.model = resolve_model(args)
    args.point_pairs, args.point_labels_p = parse_points(args)
    return args


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def build_model(args):
    """加载 SAM 模型，返回 (sam, device)。"""
    from segment_anything import sam_model_registry

    ckpt = args.ckpt or str(ROOT / "models" / CKPT_MAP[args.model])
    if not Path(ckpt).is_file():
        sys.exit(f"[错误] 找不到权重 {ckpt}，请先执行 ./sh/download.sh")

    if not args.no_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，回退到 CPU")
        device = "cpu"

    t0 = time.time()
    sam = sam_model_registry[args.model](checkpoint=ckpt)
    sam.to(device=device)
    sam.eval()
    print(f"[信息] 模型 {args.model} 加载 {time.time() - t0:.1f}s | 设备 {device} | 权重 {Path(ckpt).name}")
    return sam, device


def build_predictor(sam):
    from segment_anything import SamPredictor
    return SamPredictor(sam)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def save_triple(mask, image, out_path, title=""):
    """保存单掩码三联图（原图 / 掩码 / 叠加）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image); axes[0].set_title("Original")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Mask")
    axes[2].imshow(image); axes[2].imshow(mask, alpha=0.5, cmap="jet")
    axes[2].set_title(f"Overlay {title}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[输出] 可视化 -> {out_path}")


def save_multi(masks, image, out_path, alpha=0.5, max_masks=20, draw_box=False):
    """保存多掩码叠加图（自动分割用）。masks: list[dict]（键 segmentation/area/bbox）。"""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    rng = np.random.default_rng(42)
    items = sorted(masks, key=lambda x: x["area"], reverse=True)[:max_masks]
    for m in items:
        color = np.concatenate([rng.random(3), [alpha]])
        h, w = m["segmentation"].shape
        mask_img = np.ones((h, w, 4))
        mask_img[:, :, :3] = color[:3]
        mask_img[:, :, 3] = m["segmentation"] * alpha
        ax.imshow(mask_img)
        if draw_box:
            x0, y0, bw, bh = m["bbox"]
            ax.add_patch(plt.Rectangle((x0, y0), bw, bh, fill=False,
                                       edgecolor=color[:3], linewidth=1.5))

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[输出] 可视化 -> {out_path}   (共 {len(masks)} 个掩码)")


def load_image(path, announce=True):
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit(f"[错误] 无法读取图片: {path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if announce:
        print(f"[信息] 图像 {path}  {image.shape[1]}x{image.shape[0]}")
    return image


# ---------------------------------------------------------------------------
# 提示分割（点 / 框）
# ---------------------------------------------------------------------------
def run_once(predictor, image, args):
    """一次「图像编码 + 提示前向」，返回 (masks, scores, 编码秒, 提示秒)。"""
    sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (lambda: None)

    t0 = time.time()
    predictor.set_image(image)
    sync()
    t_enc = time.time() - t0

    t0 = time.time()
    if args.point is not None:
        pts = np.array(args.point_pairs, dtype=np.float32).reshape(-1, 2)
        lbls = np.array(args.point_labels_p, dtype=np.int32)
        masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbls,
                                             multimask_output=True)
        best = int(np.argmax(scores))          # 三个候选里取分数最高
        masks, scores = masks[best:best + 1], scores[best:best + 1]
    else:
        box = np.array(args.box, dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    sync()
    return masks, scores, t_enc, time.time() - t0


def run_auto(sam, image, args):
    """全自动分割，返回掩码列表（dict: segmentation/area/bbox）。"""
    from segment_anything import SamAutomaticMaskGenerator
    _ensure_nms_patched()

    print(f"\n[自动分割] points_per_side={args.auto_points}  (点越多越慢)")
    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.auto_points,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100,
    )
    t0 = time.time()
    masks = gen.generate(image)
    print(f"[自动分割] 耗时 {time.time() - t0:.2f}s | 掩码数量 {len(masks)}")
    if masks:
        areas = sorted([m["area"] for m in masks], reverse=True)[:5]
        print(f"[自动分割] 前5大掩码面积: {areas}")
    return masks


# ---------------------------------------------------------------------------
# 结果打印与保存
# ---------------------------------------------------------------------------
def report(image, masks, scores, args, desc):
    print(f"\n[结果] 提示={desc} | 掩码数 {len(scores)}")
    for i in range(min(len(scores), args.max_masks)):
        print(f"       #{i}: score={scores[i]:.4f} 面积={int(masks[i].sum())}px")

    if args.mask_dir:
        import cv2
        d = Path(args.mask_dir); d.mkdir(parents=True, exist_ok=True)
        stem = Path(args.image).stem
        for i in range(len(masks)):
            cv2.imwrite(str(d / f"{stem}_mask{i}.png"),
                        (masks[i].astype(np.uint8) * 255))
        print(f"[输出] 掩码 -> {d}/")

    if args.out:
        out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
        save_triple(masks[0], image, str(out_path), f"({desc})")


# ---------------------------------------------------------------------------
# 单图入口（本地运行与常驻服务共用）
# ---------------------------------------------------------------------------
def run_image(predictor, args, warmup=False, announce=True):
    """读图 -> 推理 -> 打印/保存。

    warmup=True: 跑两次并分别打印「首次 / 热身后」耗时（本地单次运行用）。
    warmup=False: 只跑一次，打印「推理」耗时（常驻服务用，模型已预热）。
    """
    image = load_image(args.image, announce)

    if args.auto:
        masks = run_auto(predictor.model, image, args)
        if args.mask_dir:
            import cv2
            d = Path(args.mask_dir); d.mkdir(parents=True, exist_ok=True)
            stem = Path(args.image).stem
            for i, m in enumerate(masks):
                cv2.imwrite(str(d / f"{stem}_mask{i}.png"),
                            (m["segmentation"].astype(np.uint8) * 255))
            print(f"[输出] 掩码 -> {d}/")
        if args.out:
            out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
            save_multi(masks, image, str(out_path), args.alpha, args.max_masks,
                       draw_box=not args.no_box)
        return masks

    if args.point is not None:
        desc = f"point{args.point_pairs}"
    else:
        desc = f"box{list(args.box)}"

    if warmup:
        _, _, e1, p1 = run_once(predictor, image, args)
        print(f"[计时] 首次推理:   图像编码 {e1*1000:7.1f}ms | 提示前向 {p1*1000:7.1f}ms "
              f"| 合计 {(e1 + p1)*1000:7.1f}ms")
        masks, scores, e2, p2 = run_once(predictor, image, args)
        print(f"[计时] 热身后推理: 图像编码 {e2*1000:7.1f}ms | 提示前向 {p2*1000:7.1f}ms "
              f"| 合计 {(e2 + p2)*1000:7.1f}ms")
    else:
        masks, scores, e, p = run_once(predictor, image, args)
        print(f"[计时] 推理:       图像编码 {e*1000:7.1f}ms | 提示前向 {p*1000:7.1f}ms "
              f"| 合计 {(e + p)*1000:7.1f}ms")

    report(image, masks, scores, args, desc)
    return masks


# ---------------------------------------------------------------------------
# 性能基准
# ---------------------------------------------------------------------------
def run_bench(predictor, image, args):
    print(f"\n{'='*68}")
    print(f" 性能基准测试  (model={args.model}, repeat={args.repeat})")
    print(f"{'='*68}")
    h, w = image.shape[:2]
    print(f" 输入图像尺寸 : {w}x{h}")

    sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (lambda: None)

    # --- 编码器基准 ---
    enc = []
    for _ in range(args.repeat):
        sync(); t0 = time.time(); predictor.set_image(image); sync()
        enc.append(time.time() - t0)
    enc_arr = np.array(enc) * 1000
    print("\n [图像编码器 / ViT]")
    print(f"   set_image    : 平均 {enc_arr.mean():7.1f} ms   "
          f"最小 {enc_arr.min():7.1f} ms   最大 {enc_arr.max():7.1f} ms")
    print(f"   编码器 FPS   : {1000/enc_arr.mean():.2f}")

    predictor.set_image(image)
    print("\n [掩码解码器 / Mask Decoder]")
    for n_pts in (1, 4, 16):
        pts = np.random.uniform([0, 0], [w, h], size=(n_pts, 2)).astype(np.float32)
        lbls = np.ones(n_pts, dtype=np.int32)
        predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
        dec = []
        for _ in range(args.repeat):
            sync(); t0 = time.time()
            predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
            sync(); dec.append(time.time() - t0)
        dec_arr = np.array(dec) * 1000
        print(f"   {n_pts:2d} 个点    : 平均 {dec_arr.mean():6.1f} ms   "
              f"(解码 FPS {1000/dec_arr.mean():6.2f})")

    print("\n [端到端吞吐]")
    print(f"   单张图 + 单次提示 : {enc_arr.mean():.1f} ms  =>  {1000/enc_arr.mean():.2f} FPS")

    if args.device == "cuda" and torch.cuda.is_available():
        print("\n [显存占用]")
        print(f"   已分配       : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   峰值分配     : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"   缓存保留     : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"{'='*68}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = prepare_args(argv)

    if args.check:
        print_env_info()
        return 0

    sam, device = build_model(args)
    args.device = device
    predictor = build_predictor(sam)

    print(f"[计时] 初始化: 导入三方库 {_T_IMPORT:.1f}s | 初始化总计 {time.time() - _T_LAUNCH:.1f}s")

    if args.bench:
        image = load_image(args.image, announce=True)
        run_bench(predictor, image, args)
        return 0

    run_image(predictor, args, warmup=True, announce=True)
    print(f"[计时] 端到端(含初始化): {time.time() - _T_LAUNCH:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())