# SAM_AGXOrin

在 **NVIDIA Jetson AGX Orin (64GB)** 上部署 **Meta SAM**（Segment Anything Model），
面向**可提示的通用图像分割**：给出点或框提示即可得到像素级掩码，也支持无提示的
全自动分割；并提供常驻服务与 TensorRT 加速能力。

本项目在官方 `segment-anything` 包之上做了三类工程化工作：

1. **Jetson 适配** —— 解决 NVIDIA 定制版 torch 与 torchvision 的 C++ ops ABI 不匹配（自动回退纯 PyTorch NMS）；
2. **三种提示 + 结果后处理** —— 点 / 框 / 全自动分割统一命令行，点提示自动选取得分最高的掩码；
3. **常驻服务与 TensorRT 加速** —— 模型只加载一次，后续每次推理约 **0.5~1s**；编码器可选 TensorRT。

---

## 目录

1. [硬件与环境](#1-硬件与环境)
2. [快速开始](#2-快速开始)
3. [三种提示方式](#3-三种提示方式)
4. [常驻服务（推荐）](#4-常驻服务推荐)
5. [TensorRT 加速](#5-tensorrt-加速)
6. [参数速查](#6-参数速查)
7. [项目结构](#7-项目结构)
8. [实现要点](#8-实现要点)
9. [Jetson 关键适配](#9-jetson-关键适配)
10. [性能实测](#10-性能实测)
11. [常见问题](#11-常见问题)
12. [许可](#12-许可)

---

## 1. 硬件与环境

| 项目 | 值 |
|------|-----|
| 设备 | Jetson AGX Orin 64GB（统一内存架构，CPU/GPU 共享） |
| 系统 | Ubuntu 24.04 aarch64 / kernel 5.15.148-tegra（JetPack 6.2 / L4T R36.4.7） |
| Python | 3.10.12（`venv310/`） |
| PyTorch | 2.8.0（NVIDIA 定制 aarch64 wheel） |
| torchvision | 0.23.0 |
| CUDA | 12.6 可用 |
| TensorRT | 10.3（用于可选的 engine 加速） |
| segment-anything | 1.0（Meta 官方包，GitHub 安装） |
| 权重 | `models/sam_vit_{b,l,h}_*.pth`（375MB / 1.2GB / 2.4GB） |

从零初始化（新机器）：

```bash
cd /home/jetson/zhangtianqi/SAM_AGXOrin
./sh/setup.sh        # 建 venv + 装依赖 + 自检
./sh/download.sh     # 下载 ViT-B 权重（375MB）
./sh/check_env.sh    # 检查 ONNX / TensorRT / onnxruntime 可用性
```

环境自检：

```bash
./venv310/bin/python py/selfcheck.py
```

---

## 2. 快速开始

> 所有命令都在项目根目录执行：`cd /home/jetson/zhangtianqi/SAM_AGXOrin`

```bash
# 点提示分割（在 (500,400) 处点击）
./sh/run.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_point.png

# 框提示分割（xyxy）
./sh/run.sh --image inputimage/0000.jpg --box 100 100 800 600 --out outputimage/0000_box.png

# 全自动分割（网格撒点，较慢）
./sh/run.sh --image inputimage/0000.jpg --auto --auto-points 16 --out outputimage/0000_auto.png

# 导出每个掩码为独立 PNG
./sh/run.sh --image inputimage/0000.jpg --point 500 400 \
            --out outputimage/0000_point.png --mask-dir outputs/0000_masks

# 批量处理（模型只加载一次）
./venv310/bin/python py/infer_batch.py --model vit_b \
            --input-dir inputimage --output-dir outputimage

# 性能基准
./sh/run.sh --image inputimage/0000.jpg --point 500 400 --bench --repeat 5
```

不指定 `--model` 时，脚本会自动选用本机已下载的权重（优先 `vit_h > vit_l > vit_b`）。

---

## 3. 三种提示方式

三种提示任选其一（点提示优先于框提示，若同时给出则按 `--point` > `--box` 处理）。

### 3.1 点提示 `--point`

坐标是**像素坐标**，成对给出，可给多个点，用 `--point-label` 指定正/负样本
（`1` 正 / `0` 负，默认全 `1`）：

```bash
# 单点
./sh/run.sh --image inputimage/0000.jpg --point 500 400 --out out.png

# 多点：同时对多个物体分割
./sh/run.sh --image inputimage/0000.jpg --point 500 400 800 300 --out out.png

# 正 + 负点：保留正点目标，排除负点所在区域
./sh/run.sh --image inputimage/0000.jpg --point 500 400 800 300 \
            --point-label 1 0 --out out.png
```

点提示会输出 SAM 的 **3 个候选掩码**，本项目自动**选取得分最高**的一个作为结果。

### 3.2 框提示 `--box`

格式为 `X1 Y1 X2 Y2`（左上角与右下角像素坐标，xyxy）：

```bash
./sh/run.sh --image inputimage/0000.jpg --box 100 100 800 600 --out out.png
```

### 3.3 全自动分割 `--auto`

网格撒点，自动输出全图所有物体掩码，返回多个掩码（按面积从大到小着色叠加）。
`--auto-points` 控制每边采样点数，越大越慢、越细：

```bash
./sh/run.sh --image inputimage/0000.jpg --auto --auto-points 16 --out out.png
./sh/run.sh --image inputimage/0000.jpg --auto --auto-points 32 --no-box --out out.png
```

> `--no-box` 只绘制掩码，不绘制目标框。

### 3.4 可视化说明

- **点提示 / 框提示**：输出三联图（原图 / 掩码 / 叠加）。
- **全自动分割**：输出彩色多掩码叠加图（颜色来自内置调色板，`--alpha` 调透明度）。

---

## 4. 常驻服务（推荐）

### 4.1 为什么需要

单次运行 `./sh/run.sh ...` 时，**大部分时间花在加载模型**，真正的推理只有十几毫秒到几百毫秒：

```
[计时] 初始化: 导入三方库 1.5s | 初始化总计 3.0s
[计时] 首次推理:   图像编码 1415.0ms | 提示前向  25.0ms | 合计 1440.0ms
[计时] 热身后推理: 图像编码  600.0ms | 提示前向  14.0ms | 合计  614.0ms
```

常驻服务把模型留在内存里，后续每次请求只做「图像编码 + 提示前向」。

### 4.2 用法

```bash
# 启动服务（模型加载一次）
./sh/serve.sh start
./sh/serve.sh start --warmup               # 启动时预热一张图，首帧更快
./sh/serve.sh start --idle-timeout 1800    # 空闲 30 分钟自动退出释放显存

# 之后正常推理 —— 命令完全不变
./sh/run.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_out.png

# 管理
./sh/serve.sh status      # 查看服务状态 + 显存占用
./sh/serve.sh log         # 跟踪服务日志
./sh/serve.sh stop        # 停止并释放模型与显存
./sh/serve.sh restart     # 重启
./sh/serve.sh fore        # 前台启动（Ctrl+C 即释放）
```

### 4.3 工作原理

```
./sh/run.sh ...  ->  py/sam_client.py
                        ├─ run/sam.sock 存在且服务在线 -> 转发（约 0.5~1s）
                        └─ 否则                        -> 本地加载单次运行（约 6s）
```

- **通信**：Unix domain socket（默认 `run/sam.sock`），一行 JSON 请求 / 一行 JSON 响应；
- **协议**：`{"cmd":"infer","argv":[...]}`、`{"cmd":"ping"}`、`{"cmd":"stop"}`；
- **对使用者透明**：`run.sh` 的参数与用法**完全不变**；
- **可选优化**：不启动服务也不会报错，客户端会自动回退到本地运行；
- **同一套代码**：服务端与本地运行都调用 `py/sam_infer.py` 的
  `prepare_args()` / `run_image()`，因此**输出必然一致**。

### 4.4 释放显存的三种方式

| 方式 | 命令 | 说明 |
|------|------|------|
| 主动释放 | `./sh/serve.sh stop` | 优雅通知服务退出，释放显存并清理 socket/pid |
| 自动释放 | `./sh/serve.sh start --idle-timeout 1800` | 空闲超时后自动退出 |
| 手动中断 | `./sh/serve.sh fore` 后按 `Ctrl+C` | 前台模式 |

环境变量：

| 变量 | 说明 |
|------|------|
| `SAM_SOCKET` | 自定义 socket 路径（客户端与服务端须一致） |
| `SAM_NO_SERVER` | 设为 `1` 强制本地运行（绕过服务） |

---

## 5. TensorRT 加速

将图像编码器（ViT）导出 ONNX 并构建 TensorRT engine，可绕过部分 PyTorch 运行时开销。

```bash
# 1) 导出 ONNX（生成 models/sam_<type>_encoder.onnx）
./venv310/bin/python py/export_onnx.py --model vit_h

# 2) 构建 FP16 engine（生成 models/sam_<type>_encoder_fp16.engine）
./sh/build_engine.sh models/sam_vit_h_encoder.onnx

# 3) 用 engine 推理
./sh/run_trt.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_trt.png

# A/B 对比：$bench 会打印 PyTorch 与 TensorRT 编码器耗时
./sh/run_trt.sh --image inputimage/0000.jpg --point 500 400 --bench
```

**说明**：SAM 的瓶颈在 ViT 编码器的访存，TensorRT 首次运行有引擎加载与
CUDA 图初始化开销，日常使用建议直接用 PyTorch 路径（`run.sh`）。

> engine 与 ONNX 体积较大，若磁盘紧张可只保留 `*.engine`。

---

## 6. 参数速查

### 6.1 推理参数（`sh/run.sh` 与 `sh/run_trt.sh` 通用）

| 参数 | 说明 | 默认 |
|------|------|------|
| `--image` | 输入图像路径（必需） | — |
| `--model` | 模型类型 `vit_b` / `vit_l` / `vit_h` | 自动选用本机权重 |
| `--ckpt` | 自定义权重路径 | — |
| `--point X Y [X Y ...]` | 点提示：像素坐标，成对给出，可多对 | — |
| `--point-label L [L ...]` | 每点标签：`1` 正 / `0` 负，数量须与点一致 | 全 `1` |
| `--box X1 Y1 X2 Y2` | 框提示：xyxy 像素坐标 | — |
| `--auto` | 全自动分割 | 关 |
| `--auto-points` | 自动分割每边采样点数 | `32` |
| `--no-box` | auto 模式只绘制掩码，不绘制目标框 | 关 |
| `--out` | 可视化输出路径 | — |
| `--mask-dir` | 每个掩码单独导出 PNG 到此目录 | — |
| `--alpha` | 掩码叠加透明度 | `0.5` |
| `--max-masks` | 最多可视化前 N 个 | `20` |
| `--bench` / `--repeat` | 打印耗时基准 / 重复次数 | 关 / `5` |
| `--device` | `cuda` 或 `cpu` | `cuda` |
| `--no-tf32` | 关闭 TF32（默认开启加速） | 关 |
| `--check` | 仅打印环境信息 | 关 |

> 查看参数帮助请用：`./venv310/bin/python py/sam_infer.py --help`
> （`./sh/run.sh --help` 缺少 `--image` 会被参数校验拦截。）

### 6.2 服务参数（`sh/serve.sh`）

| 子命令 | 说明 |
|--------|------|
| `start [参数]` | 后台启动服务 |
| `stop` | 停止并释放显存 |
| `restart [参数]` | 重启 |
| `status` | 查看状态与显存占用 |
| `log` | 跟踪日志（`run/sam.log`） |
| `fore [参数]` | 前台启动 |

可透传给服务的参数：`--idle-timeout N`、`--model vit_h`、`--warmup`、
`--socket PATH`、`--device cuda|cpu`、`--ckpt`、`--no-tf32`。

### 6.3 批量参数（`py/infer_batch.py`）

| 参数 | 说明 |
|------|------|
| `--input-dir` | 输入目录（默认 `inputimage/`） |
| `--output-dir` | 输出目录（默认 `outputimage/`） |
| `--suffix` | 输出文件名后缀（默认 `_out.png`） |
| `--model` | 模型类型（默认自动选用本机权重） |
| `--auto` | 改用全自动分割（较慢） |

---

## 7. 项目结构

```
SAM_AGXOrin/
├── README.md
├── requirements.txt
│
├── sh/                           # Shell 入口（只负责环境变量 + 调度）
│   ├── run.sh                    # ⭐ 主入口（自动选择服务或本地）
│   ├── serve.sh                  # ⭐ 常驻服务管理（start/stop/status/log/fore）
│   ├── run_trt.sh                # TensorRT 推理入口
│   ├── setup.sh                  # 一键环境初始化
│   ├── download.sh               # 下载权重（断点续传）
│   ├── build_engine.sh           # ONNX -> TensorRT engine
│   └── check_env.sh              # ONNX/TensorRT 环境检查
│
├── py/                           # Python 实现
│   ├── sam_infer.py              # 推理核心（三提示 + 可视化 + 基准）
│   ├── sam_client.py             # 客户端：服务优先，自动回退本地
│   ├── sam_server.py             # 常驻服务（Unix socket + JSON 协议）
│   ├── sam_trt_infer.py          # TensorRT 版推理
│   ├── trt_runner.py             # TRT engine 加载与执行封装
│   ├── infer_batch.py            # 批量推理
│   ├── export_onnx.py            # 导出图像编码器 ONNX
│   └── selfcheck.py              # 环境自检
│
├── models/                       # 权重与引擎（需下载/构建）
│   ├── sam_vit_b_01ec64.pth           # 375 MB
│   ├── sam_vit_l_0b3195.pth           # 1.2 GB
│   ├── sam_vit_h_4b8939.pth           # 2.4 GB
│   └── sam_<type>_encoder_fp16.engine # 可选（TRT）
│
├── assets/                       # 示例图片
├── inputimage/                   # 输入图像（0000.jpg 等）
├── outputimage/                  # 可视化输出
├── outputs/                      # 掩码等中间产物
├── run/                          # 服务运行时文件（socket / pid / log）
├── logs/                         # 安装与构建日志
└── venv310/                      # Python 虚拟环境（含 NVIDIA 定制 torch）
```

---

## 8. 实现要点

### 8.1 使用官方 Python API

推理接口源自 `segment-anything` 官方示例：

```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

predictor.set_image(image)                       # 视觉编码（每张图只做一次）
masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbls)
```

**关键：视觉编码与提示前向是分离的。** 同一张图换不同提示时，
`set_image()` 的结果可复用，只需重跑提示前向。这也是「图像编码」与
「提示前向」在计时中分开统计的原因。

### 8.2 点提示选取得分最高的掩码

`predict(multimask_output=True)` 会返回 **3 个候选掩码**（不同粒度）。
本项目在 `py/sam_infer.py::run_once()` 中取 `argmax(scores)` 对应的掩码，
即 SAM 自评置信度最高的那个。

### 8.3 全自动分割与 NMS

`SamAutomaticMaskGenerator` 内部用 `torchvision.ops.batched_nms` 去重。
在 Jetson 上该 C++ 扩展可能因 ABI 不匹配而不可用，本项目在
`py/sam_infer.py` 顶部自动探测并替换为**纯 PyTorch NMS**（见第 9 节）。

### 8.4 权重

| 模型 | 文件名 | 体积 |
|------|--------|------|
| ViT-B | `sam_vit_b_01ec64.pth` | 375 MB |
| ViT-L | `sam_vit_l_0b3195.pth` | 1.2 GB |
| ViT-H | `sam_vit_h_4b8939.pth` | 2.4 GB |

文件名前缀即官方 MD5 前 6 位，可用 `md5sum` 校验完整性。均可用 `--ckpt` 覆盖。

### 8.5 安装踩坑记录

1. **必须用 Python 3.10**：Jetson AI Lab 的 aarch64 torch wheel 仅提供 cp310 构建。
2. **`numpy` 必须 < 2**：torch 2.8 for Jetson 按 numpy 1.x 编译。
3. **`opencv-python-headless` 固定 `4.10.0.84`**：更高版本会要求 numpy>=2。
4. **不要设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**：
   Jetson 统一内存架构下会触发 `CUDA driver error: out of memory`。
5. **`LD_LIBRARY_PATH` 需含 cuSPARSELt**：
   `venv310/lib/python3.10/site-packages/nvidia/cusparselt/lib`，
   否则 `import torch` 可能失败。`sh/run.sh` 已自动设置。

---

## 9. Jetson 关键适配

### 9.1 问题：`RuntimeError: Couldn't load custom C++ ops`

NVIDIA 定制版 torch 与 pip 安装的 torchvision 的 **C++ 扩展 ABI 不匹配**，
`torchvision.ops.nms` / `batched_nms` 会抛
`undefined symbol: ...`。SAM 的「全自动分割」内部需要 `batched_nms` 做 NMS 去重，
因此只有 `--auto` 模式会失败；点提示 / 框提示不受影响。

### 9.2 方案

`py/sam_infer.py::patch_torchvision_ops()` 在导入时：

1. 先**探测**原生 `torchvision.ops.nms` 是否可用；
2. 不可用则用**纯 PyTorch 实现的 NMS**（`_nms_pure` / `_batched_nms_pure`）替换，
   并在 `segment_anything.automatic_mask_generator` 中一并替换引用。

替换后 auto 模式可正常通过。这是**幂等、透明**的：原生可用时不做任何改动。

### 9.3 影响范围

| 能力 | Jetson |
|------|--------|
| 点提示图像分割 | ✅ |
| 框提示图像分割 | ✅ |
| 全自动分割 | ✅（纯 PyTorch NMS 兜底） |
| TensorRT 编码器加速 | ✅（需自行构建 engine） |

---

## 10. 性能实测

### 10.1 编码器 / 解码器（输入 640×480）

| 模型 | 权重体积 | 加载耗时 | 显存占用 | 编码器 set_image | 解码器 predict |
|------|---------|---------|---------|-----------------|---------------|
| **ViT-B** | 375 MB | 0.88 s | 0.35 GB | ~625 ms | ~12–14 ms |
| **ViT-L** | 1.2 GB | 2.44 s | 1.16 GB | ~1009 ms | ~14 ms |
| **ViT-H** | 2.4 GB | 4.74 s | 2.45 GB | ~1415 ms | ~14.8 ms |

> 编码器耗时含首次 CUDA 预热，重复执行 `set_image` 稳定在 600ms 左右（ViT-B）。

### 10.2 常驻服务（推荐）

| 阶段 | 耗时 |
|------|------|
| 服务启动（加载模型 + 预热，**仅一次**） | 数秒 |
| 后续每次推理 | **约 0.5~1 s** |

**核心收益**：省去每次运行的模型加载开销，输出与本地运行完全一致。

### 10.3 各部分耗时占比

| 阶段 | 耗时 | 可复用性 |
|------|------|---------|
| 导入三方库 | ~1.5 s | 常驻后省去 |
| 模型加载 | ~1–5 s | 常驻后省去 |
| 图像编码（ViT @1024²） | ~600–1400 ms | 同图多提示可复用 |
| 提示前向 | ~15–30 ms | 每次提示必做 |

### 10.4 基准命令

```bash
./sh/run.sh --image inputimage/0000.jpg --point 500 400 --bench --repeat 5
```

输出分别给出**图像编码器**、**掩码解码器**（1/4/16 个点）与**端到端**耗时，以及显存占用。

---

## 11. 常见问题

**Q1. `Couldn't load custom C++ ops`**
本项目已自动用纯 PyTorch NMS 兜底，无需处理。若仍出现，运行
`./venv310/bin/python py/selfcheck.py` 查看 NMS 项状态。

**Q2. `import torch` 报找不到 `.so`**
`LD_LIBRARY_PATH` 没设。用 `./sh/run.sh`，或手动
`export LD_LIBRARY_PATH=$PWD/venv310/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH`

**Q3. `./sh/run.sh --help` 提示「参数错误」**
`run.sh` 现在经过客户端，缺少 `--image` 会被校验拦截。
查看参数请用：`./venv310/bin/python py/sam_infer.py --help`。

**Q4. 找不到权重**
不指定 `--model` 时会自动选用本机已有权重；若都没有，执行
`./sh/download.sh`（ViT-B）或 `./sh/download.sh all`。

**Q5. 显存不足（OOM）**
调小 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`；确认无其他进程占用
（`sudo tegrastats`）；或 `./sh/serve.sh stop` 释放常驻服务；`--device cpu` 兜底（很慢）。

**Q6. 服务启动失败 / `run.sh` 卡住**
`./sh/serve.sh log` 看日志；`./sh/serve.sh stop` 清理后重试。
若 socket 残留但服务已死，客户端会**自动回退本地运行**，不会卡住。

**Q7. 全自动分割很慢**
`--auto` 用网格撒点，耗时随 `--auto-points` 平方增长。入门用 `16`，需要细粒度再调大。

---

## 12. 许可

模型与代码来自 Meta，遵循其
[Apache 2.0 License](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE)。
本项目仅为 Jetson 部署适配。