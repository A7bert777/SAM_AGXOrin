#!/usr/bin/env bash
# SAM 常驻服务管理：模型加载一次，之后每次推理 ~0.3-1s。
#
# 用法：
#   ./sh/serve.sh start [参数]       # 后台启动（按参数自动判定后端）
#   ./sh/serve.sh stop [--trt|--torch]   # 停止（默认两个后端都停）
#   ./sh/serve.sh restart [参数]     # 重启
#   ./sh/serve.sh status             # 查看状态与显存（默认看两个后端）
#   ./sh/serve.sh log [--trt|--torch]    # 跟踪服务日志
#   ./sh/serve.sh fore [参数]        # 前台启动（Ctrl+C 退出）
#
# 两种后端（各自独立的 socket / pid / log，互不干扰）：
#   默认（不带 --engine/--precision）  -> PyTorch 后端，供 ./sh/run.sh 使用
#      socket=run/sam.sock      pid=run/sam.pid      log=run/sam.log
#   带 --engine 或 --precision          -> TensorRT 后端，供 ./sh/run_trt.sh 使用
#      socket=run/sam_trt.sock  pid=run/sam_trt.pid  log=run/sam_trt.log
#
# 可选参数（透传给 sam_server.py）：
#   --idle-timeout 1800     空闲 30 分钟自动退出并释放显存
#   --model vit_h           指定模型（默认自动选用本机权重）
#   --warmup                启动时预热一张图（首帧更快）
#
# TensorRT engine 预加载（需先 sh/build_engine.sh 构建 engine）：
#   --engine models/sam_vit_h_encoder_tf32.engine   直接指定 engine 文件
#   --precision tf32                                按模型+精度自动推断 engine 路径
#
#   示例（常驻 tf32 engine，之后 ./sh/run_trt.sh 请求最快）：
#     ./sh/serve.sh start --model vit_h --precision tf32 --warmup
#     ./sh/serve.sh start --engine models/sam_vit_h_encoder_tf32.engine
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv310"
SP="$VENV/lib/python3.10/site-packages"
RUN_DIR="$ROOT/run"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[错误] 未找到 venv：$VENV"
  echo "       请先执行 ./sh/setup.sh 初始化环境"
  exit 1
fi

# 与 run.sh / run_trt.sh 保持一致的环境变量
# （含 /usr/lib/aarch64-linux-gnu，供 TensorRT Python 绑定加载 libnvinfer.so）
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:$SP/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
# 不要设 NVIDIA_TF32_OVERRIDE：TensorRT engine 要求构建与执行时该值一致
# （build_engine.sh 未设置），sam_server.py 也会主动清除它。

mkdir -p "$RUN_DIR"

sock_of() { echo "$RUN_DIR/$1.sock"; }
pid_of()  { echo "$RUN_DIR/$1.pid"; }
log_of()  { echo "$RUN_DIR/$1.log"; }

# 从参数判断后端 suffix：出现 --engine/--precision（含 = 形式）即 TensorRT
backend_suffix() {
  local a
  for a in "$@"; do
    case "$a" in
      --engine|--engine=*|--precision|--precision=*) echo "sam_trt"; return ;;
    esac
  done
  echo "sam"
}

# 读取 pidfile 里存活的 PID（存活则输出，否则返回 1）
alive_pid() {  # $1 = suffix
  local pf pid
  pf="$(pid_of "$1")"
  [[ -f "$pf" ]] || return 1
  pid="$(cat "$pf" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

# 通过 socket 发一条 JSON 请求并回显响应（无服务时静默返回）
sock_send() {  # $1 = socket 路径, $2 = JSON 字符串
  [[ -S "$1" ]] || return 0
  "$VENV/bin/python" - "$1" "$2" <<'PY' 2>/dev/null || true
import json, socket, sys
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(sys.argv[1])
    with s, s.makefile("rwb") as f:
        f.write((sys.argv[2] + "\n").encode()); f.flush()
        sys.stdout.write(f.readline().decode())
except OSError:
    pass
PY
}

cmd_start() {  # $1 = suffix, 其余参数透传给 sam_server.py
  local suffix="$1"; shift
  local sock pidfile logfile pid
  sock="$(sock_of "$suffix")"; pidfile="$(pid_of "$suffix")"; logfile="$(log_of "$suffix")"

  if pid="$(alive_pid "$suffix")"; then
    echo "[服务] 已在运行（后端=$suffix，PID=$pid，socket=$sock）"
    return 0
  fi
  rm -f "$sock" "$pidfile"

  if [[ "$suffix" == "sam_trt" ]]; then export SAM_BACKEND="trt"; else export SAM_BACKEND="torch"; fi

  echo "[服务] 正在后台启动（后端=$suffix，首次需加载模型约 6s）..."
  nohup "$VENV/bin/python" "$ROOT/py/sam_server.py" --socket "$sock" "$@" >"$logfile" 2>&1 &
  pid=$!
  echo "$pid" >"$pidfile"

  # 等待 socket 就绪（最多 120s）
  for _ in $(seq 1 240); do
    if [[ -S "$sock" ]]; then
      echo "[服务] 就绪（后端=$suffix，PID=$pid）"
      echo "       日志：./sh/serve.sh log"
      if [[ "$suffix" == "sam_trt" ]]; then
        echo "       现在可直接用：./sh/run_trt.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_trt_point.png"
      else
        echo "       现在可直接用：./sh/run.sh --image inputimage/0000.jpg --point 500 400 --out outputimage/0000_point.png"
      fi
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[错误] 服务启动失败，日志末尾："
      tail -n 30 "$logfile" || true
      return 1
    fi
    sleep 0.5
  done
  echo "[警告] 等待 socket 超时，请查看日志：./sh/serve.sh log"
}

cmd_stop_one() {  # $1 = suffix
  local suffix="$1" sock pidfile pid
  sock="$(sock_of "$suffix")"; pidfile="$(pid_of "$suffix")"

  if ! pid="$(alive_pid "$suffix")"; then
    rm -f "$sock" "$pidfile"
    echo "[服务] $suffix 未在运行"
    return 0
  fi

  echo "[服务] 正在停止 $suffix（PID=$pid）..."
  sock_send "$sock" '{"cmd": "stop"}'   # 优先优雅退出，让服务自己释放显存

  # 等待退出（最多 20s），超时则强杀
  for _ in $(seq 1 40); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[服务] $suffix 优雅退出超时，强制结束"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$sock" "$pidfile"
  echo "[服务] $suffix 已停止，模型与显存已释放"
}

cmd_status_one() {  # $1 = suffix
  local suffix="$1" sock pid
  sock="$(sock_of "$suffix")"
  if pid="$(alive_pid "$suffix")"; then
    echo "[服务] $suffix 运行中（PID=$pid）"
    [[ -S "$sock" ]] && echo "       socket: $sock" || echo "       socket: 缺失（可能正在启动）"
    local resp; resp="$(sock_send "$sock" '{"cmd": "ping"}')"
    [[ -n "$resp" ]] && echo "       $resp"
  else
    echo "[服务] $suffix 未运行"
  fi
}

cmd_log() {  # $1 = suffix
  local suffix="$1" logfile
  logfile="$(log_of "$suffix")"
  [[ -f "$logfile" ]] || { echo "[服务] 暂无日志（$logfile）"; return 0; }
  tail -n 40 -f "$logfile"
}

cmd_fore() {  # $1 = suffix, 其余参数透传
  local suffix="$1"; shift
  local sock pidfile pid
  sock="$(sock_of "$suffix")"; pidfile="$(pid_of "$suffix")"
  if pid="$(alive_pid "$suffix")"; then
    echo "[错误] $suffix 服务已在运行（PID=$pid），请先 ./sh/serve.sh stop"
    return 1
  fi
  rm -f "$sock" "$pidfile"
  if [[ "$suffix" == "sam_trt" ]]; then export SAM_BACKEND="trt"; else export SAM_BACKEND="torch"; fi
  echo "[服务] 前台启动（后端=$suffix），Ctrl+C 退出并释放显存"
  exec "$VENV/bin/python" "$ROOT/py/sam_server.py" --socket "$sock" "$@"
}

# ------------------------------ 命令分发 ------------------------------ #
cmd="${1:-}"; shift || true

case "$cmd" in
  start)
    cmd_start "$(backend_suffix "$@")" "$@"
    ;;
  fore)
    cmd_fore "$(backend_suffix "$@")" "$@"
    ;;
  restart)
    sfx="$(backend_suffix "$@")"
    cmd_stop_one "$sfx"
    cmd_start "$sfx" "$@"
    ;;
  stop)
    case "${1:-}" in
      --trt)   cmd_stop_one sam_trt ;;
      --torch) cmd_stop_one sam ;;
      *)       cmd_stop_one sam; cmd_stop_one sam_trt ;;
    esac
    ;;
  status)
    case "${1:-}" in
      --trt)   cmd_status_one sam_trt ;;
      --torch) cmd_status_one sam ;;
      *)       cmd_status_one sam; cmd_status_one sam_trt ;;
    esac
    # 显存占用
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "--- nvidia-smi ---"
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
    fi
    ;;
  log)
    case "${1:-}" in
      --trt)   cmd_log sam_trt ;;
      --torch) cmd_log sam ;;
      *)       echo "[服务] 提示：未指定后端，默认跟踪 PyTorch 日志（run/sam.log）。"
               echo "       查看 TensorRT 日志请用：./sh/serve.sh log --trt"
               cmd_log sam ;;
    esac
    ;;
  ""|-h|--help|help)
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    echo "[错误] 未知命令：$cmd"
    echo "       可用：start | stop | restart | status | log | fore"
    exit 1
    ;;
esac