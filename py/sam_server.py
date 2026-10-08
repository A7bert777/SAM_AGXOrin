#!/usr/bin/env python3
"""SAM 常驻推理服务：模型只加载一次，之后每次推理秒级返回。

为什么需要它
------------
单次运行 ``./sh/run.sh ...`` 时，大部分时间花在**导入三方库 + 加载权重**
（ViT-H 约 6s，ViT-B 约 1s），真正的推理只有编码器 600ms + 解码器十几 ms。
本服务把模型常驻在内存里，后续每次请求只做「图像编码 + 提示前向」。

工作方式
--------
* 监听一个 **Unix domain socket**（默认 ``run/sam.sock``）。
* 协议：一行一个 JSON 请求，一行一个 JSON 响应（stdout 文本放在 "stdout" 字段）。
    - ``{"cmd": "ping"}``                     -> 探活
    - ``{"cmd": "stop"}``                     -> 释放模型并退出
    - ``{"cmd": "infer", "argv": [...]}``     -> 执行一次推理（argv 同 sam_infer.py）
* 客户端 :mod:`sam_client` 由 ``sh/run.sh`` 自动调用；服务未启动时客户端会
  **自动回退**到本地单次运行，因此不会因为忘记启动服务而失败。

释放显存
--------
* ``./sh/serve.sh stop``  -> 发 stop 请求（或 kill），进程退出即释放全部显存
* ``--idle-timeout N``    -> 空闲 N 秒后自动退出（默认 0 = 不自动退出）
* Ctrl+C / SIGTERM        -> 优雅退出并释放

TensorRT engine 预加载（可选）
-----------------------------
默认加载 PyTorch 权重（``.pth``）。若指定 ``--engine`` 或 ``--precision``，
则改为加载 TensorRT engine 作为图像编码器（其余提示/解码仍走 PyTorch），
即「engine 构建一次 + 服务常驻」，之后每次请求最快::

    ./sh/serve.sh start --engine models/sam_vit_h_encoder_tf32.engine
    ./sh/serve.sh start --model vit_h --precision tf32     # 自动推断 engine 路径

指定后照常用 ``./sh/run.sh ...`` 请求即可（客户端参数完全不变）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

_T_LAUNCH = time.time()

import torch  # noqa: E402

import sam_infer as S  # noqa: E402


def default_socket() -> Path:
    """默认 socket 路径（客户端与服务端共用同一规则）。"""
    env = os.environ.get("SAM_SOCKET")
    if env:
        return Path(env)
    return ROOT / "run" / "sam.sock"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM 常驻推理服务",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--socket", default=str(default_socket()), help="Unix socket 路径")
    p.add_argument("--model", default=None, help="SAM 模型类型（不给则自动选用本机权重）")
    p.add_argument("--ckpt", default=None, help="权重路径")
    p.add_argument("--device", default="cuda", help="cuda / cpu")
    p.add_argument("--idle-timeout", type=float, default=0.0,
                   help="空闲多少秒后自动退出并释放显存（0 = 不自动退出）")
    p.add_argument("--no-tf32", action="store_true", help="禁用 TF32")
    p.add_argument("--engine", default=None,
                   help="TensorRT engine 路径（指定则用 TRT 编码器替代 PyTorch）")
    p.add_argument("--precision", default=None, choices=["tf32", "fp16", "fp32"],
                   help="engine 精度（自动推断 engine 路径，指定则用 TRT 编码器）")
    p.add_argument("--warmup", action="store_true",
                   help="启动时用一张内置图预热（首帧更稳，但启动慢一点）")
    return p.parse_args(argv)


def _resolve_engine(args, model: str):
    """根据 --engine / --precision 确定 engine 路径；未指定则返回 None。"""
    if getattr(args, "engine", None):
        return Path(args.engine)
    if getattr(args, "precision", None):
        import sam_trt_infer as T
        return T.default_engine(model, args.precision)
    return None


def build_predictor(args):
    """加载模型 + 构造 SamPredictor（只做一次）。

    若指定 --engine / --precision，则把图像编码器替换成 TensorRT engine，
    其余（提示编码 + 掩码解码）仍使用 PyTorch。
    """
    model_args = argparse.Namespace(
        model=args.model, ckpt=args.ckpt, device=args.device, no_tf32=args.no_tf32)
    model_args.model = S.resolve_model(model_args)
    sam, device = S.build_model(model_args)

    engine_path = _resolve_engine(args, model_args.model)
    if engine_path is not None:
        if not engine_path.is_file():
            sys.exit(f"[错误] 找不到 engine: {engine_path}\n"
                     f"       请先执行: ./sh/build_engine.sh "
                     f"models/sam_{model_args.model}_encoder.onnx [fp16|tf32|fp32]")
        # TensorRT 要求「构建 engine」与「执行 engine」时 NVIDIA_TF32_OVERRIDE 一致。
        # build_engine.sh 未设置（默认 -1），此处也清掉，避免环境里残留值导致加载失败。
        os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
        import sam_trt_infer as T
        engine = T.TRTEngine(engine_path)
        print(engine.info())
        predictor = T.build_predictor_with_trt_encoder(sam, engine)
        print(f"[服务] 已启用 TensorRT 编码器：{engine_path.name}")
        return predictor, device

    predictor = S.build_predictor(sam)
    return predictor, device


# ---------------------------------------------------------------------- #
# 请求处理
# ---------------------------------------------------------------------- #
def handle_infer(predictor, device, req, model=None) -> dict:
    """执行一次推理请求，把子函数打印的内容收集到返回值里。"""
    argv = list(req.get("argv") or [])
    if not argv:
        return {"ok": False, "error": "缺少 argv（应形如 ['--image','a.jpg','--point','1','2']）"}

    # 模型已在服务启动时固定加载：把服务端模型名注入 argv，避免 prepare_args 里的
    # resolve_model 按本机权重重新推断并打印出与实际不符的模型名
    # （如服务加载 vit_b，却因本机有 vit_h 权重而打印 "自动选用 vit_h"）。
    if model and "--model" not in argv:
        argv = ["--model", model] + argv

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            args = S.prepare_args(argv)          # SystemExit -> 由外层捕获
            args.device = device
            if model:
                args.model = model
            S.run_image(predictor, args, warmup=False, announce=True)
        return {"ok": True, "stdout": buf.getvalue()}
    except SystemExit as e:                       # argparse / 校验错误
        text = buf.getvalue()
        msg = str(e.code) if e.code not in (None, 0) else "参数错误"
        return {"ok": False, "error": msg, "stdout": text}
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": traceback.format_exc(),
                "stdout": buf.getvalue()}


def serve_forever(predictor, device, sock_path: Path, idle_timeout: float,
                  model=None) -> None:
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():                       # 清理上次残留
        try:
            sock_path.unlink()
        except OSError:
            pass

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(8)
    os.chmod(sock_path, 0o600)
    if idle_timeout > 0:
        srv.settimeout(idle_timeout)

    print(f"[服务] 就绪 socket={sock_path}")
    print(f"[服务] 设备={device} | 空闲超时={idle_timeout or '不退出'}s "
          f"| 启动+加载总耗时 {time.time() - _T_LAUNCH:.1f}s")
    print("[服务] Ctrl+C 或 ./sh/serve.sh stop 可释放模型与显存", flush=True)

    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                print(f"[服务] 空闲 {idle_timeout:.0f}s，自动退出并释放显存")
                break

            with conn, conn.makefile("rwb") as f:
                for raw in f:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    try:
                        req = json.loads(line)
                    except json.JSONDecodeError as e:
                        f.write((json.dumps({"ok": False, "error": f"JSON 解析失败: {e}"}) + "\n").encode())
                        f.flush()
                        continue

                    cmd = req.get("cmd", "infer")
                    if cmd == "ping":
                        resp = {"ok": True, "stdout": "", "device": str(device),
                                "uptime_s": round(time.time() - _T_LAUNCH, 1)}
                    elif cmd == "stop":
                        f.write((json.dumps({"ok": True, "stdout": "[服务] 正在退出...\n"}) + "\n").encode())
                        f.flush()
                        print("[服务] 收到 stop 请求，退出并释放显存")
                        return
                    else:
                        t0 = time.time()
                        resp = handle_infer(predictor, device, req, model)
                        resp["elapsed_s"] = round(time.time() - t0, 3)
                        first = (req.get("argv") or ["?"])[0:3]
                        print(f"[服务] 推理 {first} | ok={resp['ok']} | "
                              f"{resp['elapsed_s']}s", flush=True)

                    f.write((json.dumps(resp, ensure_ascii=False) + "\n").encode())
                    f.flush()
    finally:
        srv.close()
        if sock_path.exists():
            try:
                sock_path.unlink()
            except OSError:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[服务] 已释放，进程退出")


def main(argv=None):
    args = parse_args(argv)
    sock_path = Path(args.socket).resolve()

    predictor, device = build_predictor(args)

    if args.warmup:
        # 用项目自带图预热一次，排除首帧 CUDA 内核初始化开销
        imgs = sorted((ROOT / "inputimage").glob("*"))
        if imgs:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    a = S.prepare_args(["--image", str(imgs[0]), "--auto", "--auto-points", "4"])
                    a.device = device
                    S.run_image(predictor, a, warmup=False, announce=False)
                print(f"[服务] 预热完成（{imgs[0].name}）")
            except Exception as e:                # noqa: BLE001 预热失败不影响服务
                print(f"[服务] 预热跳过：{e}")

    # 把服务实际加载的模型名传下去（供日志/展示使用）
    model_args = argparse.Namespace(
        model=args.model, ckpt=args.ckpt, device=args.device, no_tf32=args.no_tf32)
    model_args.model = S.resolve_model(model_args)
    serve_forever(predictor, device, sock_path, args.idle_timeout, model_args.model)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[服务] 收到 Ctrl+C，退出并释放显存")
        sys.exit(0)