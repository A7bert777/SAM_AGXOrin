#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM_AGXOrin wheel 下载器（PEP 503 simple index）

功能
  * 从 PEP 503 simple 索引（如 https://pypi.jetson-ai-lab.io/jp6/cu126）
    解析并下载匹配 aarch64 / cp310 的 wheel
  * tqdm 实时进度条（百分比 / 速度 / 剩余时间）；未安装 tqdm 时自动降级为内置进度条
  * 断点续传（HTTP Range）
  * sha256 完整性校验（取自索引页 URL 的 #sha256= 片段）
  * 已存在且校验通过的文件自动跳过

用法
  python3 py/fetch_wheels.py \\
      --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \\
      --dest wheels \\
      torch==2.8.0 torchvision==0.23.0

退出码
  0 = 全部 wheel 就绪；1 = 失败
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "SAM_AGXOrin-setup/1.0"
CHUNK_SIZE = 1 << 18          # 256 KB
PROGRESS_ENV = os.environ.get("FETCH_WHEELS_NO_PROGRESS", "") == "1"

# --------------------------------------------------------------------------- #
# 进度条：优先 tqdm，缺失时使用内置兜底实现（接口保持一致）
# --------------------------------------------------------------------------- #
try:
    from tqdm import tqdm as _tqdm_cls
except Exception:                                  # pragma: no cover
    _tqdm_cls = None


class _SimpleBar:
    """当环境缺少 tqdm 时的兜底进度条（输出到 stderr，语义与 tqdm 接近）。"""

    def __init__(self, total, desc):
        self.total = total or 0
        self.desc = desc
        self.n = 0
        self.t0 = time.time()
        self.width = 30
        self.last = 0.0

    def update(self, n):
        self.n += n
        now = time.time()
        if now - self.last < 0.2 and (not self.total or self.n < self.total):
            return
        self.last = now
        self._render()

    def _render(self):
        elapsed = max(time.time() - self.t0, 1e-6)
        speed = self.n / elapsed
        if self.total:
            frac = min(self.n / self.total, 1.0)
            done = int(frac * self.width)
            bar = "#" * done + "-" * (self.width - done)
            msg = (f"\r{self.desc}: |{bar}| "
                   f"{self.n / 1048576:7.1f}/{self.total / 1048576:.1f} MB "
                   f"{frac * 100:5.1f}% {speed / 1048576:6.2f} MB/s")
        else:
            msg = (f"\r{self.desc}: {self.n / 1048576:.1f} MB "
                   f"{speed / 1048576:6.2f} MB/s")
        sys.stderr.write(msg)
        sys.stderr.flush()

    def close(self):
        self._render()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_bar(total, desc):
    """创建进度条对象（统一的 update/close/上下文管理器接口）。"""
    if PROGRESS_ENV:
        return _NullBar()
    if _tqdm_cls is not None:
        return _tqdm_cls(total=total or None, unit="B", unit_scale=True,
                         unit_divisor=1024, desc=desc, ncols=100,
                         miniters=1, mininterval=0.3, leave=True,
                         dynamic_ncols=True, ascii=True)
    return _SimpleBar(total, desc)


class _NullBar:
    def update(self, n):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


# --------------------------------------------------------------------------- #
# simple index 解析
# --------------------------------------------------------------------------- #
_LINK_RE = re.compile(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S)


def _strip_tags(text):
    return re.sub(r"<[^>]+>", "", text).strip()


def fetch_index(index_url, package):
    """下载并返回包的 simple index HTML。"""
    base = index_url.rstrip("/") + "/"
    url = urllib.parse.urljoin(base, package.lower().replace("_", "-") + "/")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", "replace")


def parse_links(html):
    """返回 [(绝对 href, 文件名, sha256|None), ...]。"""
    out = []
    for href, text in _LINK_RE.findall(html):
        raw, _, frag = href.partition("#")
        sha = None
        if frag.startswith("sha256="):
            sha = frag[len("sha256="):].strip().lower()
        fname = urllib.parse.unquote(raw.rstrip("/").split("/")[-1])
        out.append((raw, fname, sha))
    return out


def pick_wheel(html, package, version, py_tag="cp310", plat_tag="linux_aarch64"):
    """从索引页选出 name-version-...{py_tag}...{plat_tag}.whl。"""
    prefix = f"{package.lower().replace('-', '_')}-{version}-"
    prefix_alt = f"{package.lower()}-{version}-"
    best = None
    for raw, fname, sha in parse_links(html):
        low = fname.lower()
        if not low.endswith(".whl"):
            continue
        if not (low.startswith(prefix) or low.startswith(prefix_alt)):
            continue
        if py_tag not in low or plat_tag not in low:
            continue
        best = (raw, fname, sha)
        if "cp310-cp310" in low:                   # 精确匹配 cp310 优先
            break
    return best


# --------------------------------------------------------------------------- #
# URL 解析（兼容 Jetson AI Lab 的非标准相对链接）
# --------------------------------------------------------------------------- #
def _http_ok(url, timeout=20):
    """用 Range: bytes=0-0 探测 URL 是否可下载。"""
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            return code in (200, 206)
    except Exception:
        return False


def resolve_href(index_url, package, raw):
    """把 simple index 中的 href 解析为可下载的绝对 URL。

    pypi.jetson-ai-lab.io 的 href 形如 "../../+f/xx/<wheel>"，其真实基准
    是索引根（而非包页），直接 urljoin 会得到 404。这里生成多个候选并
    用 Range 探测选取可用地址。
    """
    if raw.startswith(("http://", "https://")):
        return raw
    root = index_url.rstrip("/")
    pkg = package.lower().replace("_", "-")
    rel = re.sub(r"^(\.\./)+", "", raw).lstrip("/")
    candidates = [
        f"{root}/{rel}",                                  # Jetson AI Lab：/jp6/cu126/+f/...
        urllib.parse.urljoin(f"{root}/{pkg}/", raw),      # 标准 PEP503
        urllib.parse.urljoin(root + "/", raw),
        urllib.parse.urljoin(f"{root}/{pkg}/", rel),
    ]
    seen, uniq = set(), []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    for u in uniq:
        if _http_ok(u):
            return u
    return uniq[0]


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url, dest_path, sha256=None, desc=None, retries=3):
    """下载 url 到 dest_path，返回最终路径。"""
    desc = desc or os.path.basename(dest_path)
    part = dest_path + ".part"

    # 已完成且校验通过 → 直接复用
    if os.path.exists(dest_path):
        if not sha256 or _hash_file(dest_path) == sha256:
            print(f"      [跳过] {os.path.basename(dest_path)} 已存在且校验通过")
            return dest_path
        os.remove(dest_path)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            hasher = hashlib.sha256()
            resume = os.path.getsize(part) if os.path.exists(part) else 0
            headers = {"User-Agent": USER_AGENT}
            if resume:
                headers["Range"] = f"bytes={resume}-"
                with open(part, "rb") as f:        # 续传：先补齐已有部分的哈希
                    while True:
                        b = f.read(1 << 20)
                        if not b:
                            break
                        hasher.update(b)
                if desc:
                    sys.stderr.write(f"      [续传] {desc} 从 "
                                     f"{resume / 1048576:.1f} MB 继续\n")

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if resume and status != 206:       # 服务端不支持 Range → 重下
                    resume = 0
                    hasher = hashlib.sha256()
                    sys.stderr.write("      [提示] 服务端不支持断点续传，重新下载\n")
                mode = "ab" if resume else "wb"
                length = int(resp.headers.get("Content-Length") or 0)
                total = (length + resume) if length else resume
                with open(part, mode) as out, make_bar(total, desc) as bar:
                    if resume:
                        bar.update(resume)
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        hasher.update(chunk)
                        bar.update(len(chunk))

            if sha256 and hasher.hexdigest() != sha256:
                raise RuntimeError(
                    f"sha256 校验失败: 期望 {sha256[:16]}...，实际 "
                    f"{hasher.hexdigest()[:16]}...")
            os.replace(part, dest_path)
            return dest_path
        except Exception as e:                     # noqa: BLE001
            last_err = e
            sys.stderr.write(f"      [警告] 第 {attempt} 次下载失败: {e}\n")
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"下载失败: {url} → {last_err}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_spec(spec):
    if "==" not in spec:
        raise ValueError(f"包规格必须形如 name==version，收到: {spec!r}")
    name, _, version = spec.partition("==")
    return name.strip(), version.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="从 PEP503 simple index 下载 aarch64/cp310 wheel（tqdm 进度条）")
    ap.add_argument("--index-url", required=True, help="simple index 根地址")
    ap.add_argument("--dest", required=True, help="wheel 保存目录")
    ap.add_argument("--py-tag", default="cp310", help="Python 标签（默认 cp310）")
    ap.add_argument("--plat-tag", default="linux_aarch64",
                    help="平台标签（默认 linux_aarch64）")
    ap.add_argument("specs", nargs="+", help="包规格，如 torch==2.8.0")
    args = ap.parse_args(argv)

    os.makedirs(args.dest, exist_ok=True)
    if _tqdm_cls is None and not PROGRESS_ENV:
        sys.stderr.write("      [提示] 未检测到 tqdm，使用内置进度条\n")

    results = []
    for spec in args.specs:
        name, version = parse_spec(spec)
        print(f"      查询 {name}=={version} ...")
        html = fetch_index(args.index_url, name)
        picked = pick_wheel(html, name, version, args.py_tag, args.plat_tag)
        if not picked:
            raise SystemExit(f"      [错误] 在 {args.index_url} 未找到匹配的 "
                             f"{name}=={version} ({args.py_tag}/{args.plat_tag}) wheel")
        raw, fname, sha = picked
        url = resolve_href(args.index_url, name, raw)
        dest = os.path.join(args.dest, fname)
        print(f"      下载 {fname}")
        print(f"      地址 {url}")
        download(url, dest, sha256=sha, desc=fname)
        results.append(dest)

    print("      已就绪:")
    for p in results:
        size = os.path.getsize(p) / 1048576
        print(f"        {p}  ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stderr.write("\n      已取消\n")
        sys.exit(130)
