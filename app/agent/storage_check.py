"""
LocalFileStorage 的路径安全与长路径自检（开发用）。

    python -m app.agent.storage_check

覆盖三件容易出错的事：

    1. Windows 扩展长度路径前缀（\\\\?\\）导致的误判
       —— Path.resolve() 在路径超 MAX_PATH 时会加这个前缀，
          与未加前缀的根路径直接比较会误报"逃逸出根目录"

    2. 路径穿越防护有效（加了上面的修复之后不能松掉）

    3. 绝对路径被拒绝

第 1 条是实际踩过的：产物曾用
`pipeline/{version_id}/{kind}/{sha256}` 作为 key，路径深度
很容易超过 MAX_PATH，于是 resolve() 加上 \\\\?\\ 前缀，
relative_to() 误判为逃逸，写入失败。

产物现在改走 File 模块（key 是 `{file_id}/source{ext}`，短得多），
但 `_resolve_storage_path` 仍会对任意 key 调用（get / exists /
delete），所以这道修复与自检都要留着。
"""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

from app.services.file.storage.local import (
    LocalFileStorage,
    _strip_extended_prefix,
)

BASE = "./data/_storage_check"


def _check_prefix_stripping() -> list[str]:
    problems: list[str] = []

    cases = [
        ("\\\\?\\C:\\a\\b", "C:\\a\\b"),
        ("\\\\?\\UNC\\srv\\share\\x", "\\\\srv\\share\\x"),
        ("C:\\a\\b", "C:\\a\\b"),
        ("/tmp/x", "/tmp/x"),
    ]

    for raw, expected in cases:
        got = _strip_extended_prefix(raw)
        status = "OK" if got == expected else "FAIL"
        print(f"  [{status}] {raw!r} -> {got!r}")
        if got != expected:
            problems.append(
                f"prefix strip: {raw!r} -> {got!r}, want {expected!r}"
            )

    return problems


def _check_long_path() -> list[str]:
    """
    深路径必须被接受。

    这是修复前的失败点：resolve() 加上 \\\\?\\ 前缀后，
    与未加前缀的 root 比较会被误判为逃逸。
    """

    problems: list[str] = []

    shutil.rmtree(BASE, ignore_errors=True)
    storage = LocalFileStorage(base_path=BASE)

    # 模拟旧产物那种深路径：pipeline/<uuid>/<kind>/<sha256>
    deep_key = (
        f"pipeline/{uuid.uuid4()}/chunks_enriched/{'a' * 64}"
    )

    try:
        resolved = storage._resolve_storage_path(deep_key)
        print(f"  [OK]   深路径被接受（{len(str(resolved))} 字符）")

    except ValueError as exc:
        print(f"  [FAIL] 深路径被误判为逃逸：{exc}")
        problems.append("deep path rejected as escape")

    shutil.rmtree(BASE, ignore_errors=True)

    return problems


async def _check_round_trip() -> list[str]:
    problems: list[str] = []

    shutil.rmtree(BASE, ignore_errors=True)
    storage = LocalFileStorage(base_path=BASE)

    file_id = uuid.uuid4()
    payload = b"x" * 64

    stored = await storage.put(
        file_id=file_id,
        file_name="report.pdf",
        content=payload,
        content_type="application/pdf",
    )

    got = await storage.get(stored.storage_key)
    exists = await storage.exists(stored.storage_key)

    ok = got == payload and exists and stored.file_size == len(payload)

    print(
        f"  [{'OK' if ok else 'FAIL'}] put/get/exists "
        f"key={stored.storage_key}"
    )

    if not ok:
        problems.append("round trip failed")

    # put 对同一个 file_id 必须拒绝重复（一个 file_id 一个源文件）
    try:
        await storage.put(
            file_id=file_id,
            file_name="report.pdf",
            content=payload,
            content_type="application/pdf",
        )
        print("  [FAIL] 重复 put 未被拒绝")
        problems.append("duplicate put not rejected")

    except FileExistsError:
        print("  [OK]   重复 put 被拒绝（一个 file_id 一个源文件）")

    shutil.rmtree(BASE, ignore_errors=True)

    return problems


async def _check_traversal() -> list[str]:
    problems: list[str] = []

    shutil.rmtree(BASE, ignore_errors=True)
    storage = LocalFileStorage(base_path=BASE)

    escape = ".." + "/" + ".." + "/" + "outside.txt"

    attack_vectors = [
        ("相对穿越", escape),
        ("绝对路径", str(Path("C:/Windows/win.ini"))),
        ("空 key", ""),
        ("裸 ..", ".."),
    ]

    for label, key in attack_vectors:
        try:
            await storage.get(key)
            print(f"  [FAIL] {label} 未被拒绝：{key!r}")
            problems.append(f"traversal not blocked: {label}")

        except ValueError as exc:
            print(f"  [OK]   {label} 被拒：{str(exc)[:58]}")

        except FileNotFoundError:
            print(
                f"  [FAIL] {label} 通过了校验（只是文件不存在）"
            )
            problems.append(
                f"traversal passed validation: {label}"
            )

    shutil.rmtree(BASE, ignore_errors=True)

    return problems


async def _main() -> int:
    problems: list[str] = []

    print("=== 扩展路径前缀剥离 ===")
    problems += _check_prefix_stripping()

    print()
    print("=== 深路径（长路径回归）===")
    problems += _check_long_path()

    print()
    print("=== put / get / exists ===")
    problems += await _check_round_trip()

    print()
    print("=== 路径穿越防护 ===")
    problems += await _check_traversal()

    print()

    if problems:
        print("自检失败：")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
