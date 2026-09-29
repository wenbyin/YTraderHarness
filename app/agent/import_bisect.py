"""
导入崩溃二分工具（开发用）。

    python -m app.agent.import_bisect

背景：`from app.main import app` 偶发 segfault（Windows 上是
0xC0000005 访问违规），没有 Python traceback。这类崩溃只可能来自
C/Rust 扩展，且往往是间歇性的。

做法：把每个候选导入放到**独立子进程**里跑多次，统计崩溃率 ——
崩掉的子进程不会拖垮整体，这样才能定位到具体是哪个包。

退出码约定（Windows）：
    0xC0000005 = 3221225477   访问违规（segfault 的 Windows 形式）
    139 / -11                  在 POSIX 语义下的 segfault
"""

from __future__ import annotations

import subprocess
import sys
from typing import Final

#: 每个候选跑几次。间歇性崩溃需要多跑才能暴露。
ROUNDS: Final[int] = 5

#: 逐个测试的导入语句。按「原生代码比重」从高到低排。
CANDIDATES: Final[tuple[tuple[str, str], ...]] = (
    ("pydantic_core", "import pydantic_core"),
    ("ormsgpack", "import ormsgpack"),
    ("numpy", "import numpy"),
    ("grpc (pymilvus 依赖)", "import grpc"),
    ("pymilvus", "import pymilvus"),
    ("psycopg", "import psycopg"),
    ("psycopg_pool", "import psycopg_pool"),
    ("sqlalchemy", "import sqlalchemy"),
    ("langgraph", "import langgraph"),
    ("langchain_core", "import langchain_core"),
    ("langgraph checkpoint postgres", "from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver"),
    ("elasticsearch", "import elasticsearch"),
    ("fastapi", "import fastapi"),
    ("uvicorn", "import uvicorn"),
    ("zstandard", "import zstandard"),
    ("xxhash", "import xxhash"),
)

#: 整条链
CHAIN: Final[str] = "from app.main import app"


def _run(code: str) -> tuple[int, str]:
    """在子进程里跑一段导入，返回 (退出码, stderr 摘要)。"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
    )

    tail = (proc.stderr or "").strip().splitlines()
    summary = tail[-1][:90] if tail else ""

    return proc.returncode, summary


def _report(label: str, rc: int, summary: str) -> bool:
    """返回是否崩溃。"""

    # Windows 访问违规 / POSIX segfault
    crashed = rc in (
        3221225477,      # 0xC0000005 access violation
        -1073741819,     # 同一个码的有符号形式
        139,             # SIGSEGV
        -11,
    )

    if crashed:
        print(f"  [崩溃] {label:<32} rc={rc}")
        return True

    if rc != 0:
        print(f"  [异常] {label:<32} rc={rc}  {summary}")
        return False

    return False


def main() -> int:
    print(f"Python {sys.version.split()[0]}  —  每个候选跑 {ROUNDS} 次")
    print()

    print("=== 单个包 ===")
    crashed_packages: list[str] = []

    for label, code in CANDIDATES:
        hits = 0
        for _ in range(ROUNDS):
            rc, summary = _run(code)
            if _report(label, rc, summary):
                hits += 1

        if hits == 0:
            print(f"  [ OK ] {label}")
        else:
            crashed_packages.append(label)

    print()
    print("=== 整条导入链 ===")
    chain_hits = 0
    for _ in range(ROUNDS):
        rc, summary = _run(CHAIN)
        if _report("app.main", rc, summary):
            chain_hits += 1

    print(f"  崩溃 {chain_hits}/{ROUNDS}")

    print()

    if crashed_packages:
        print(f"崩溃的包：{crashed_packages}")
        return 1

    if chain_hits:
        print(
            "单个包都不崩，但整条链崩 —— 说明是**组合**触发的"
            "（两个扩展的符号冲突 / ABI 不兼容），"
            "需要进一步按 app 的分层逐个加包定位。"
        )
        return 1

    print("全部通过，没能复现。可能是更低频的时序问题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
