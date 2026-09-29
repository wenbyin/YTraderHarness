"""
数据集落盘。

为什么不用项目里的 artifact_store
---------------------------------
那套是**内容寻址**的产物存储：按 (tenant, checksum) 去重，落在
`./data/files/<uuid>/source`，文件名是 `clean_text` 这种通用名。它服务
的是"管线中间产物"—— 谁都不需要找到它们，只要能用 id 取回来。

数据集不一样：它是给人看的、要给 ragas 读的、要能直接丢进别的工具。
所以需要一个**稳定可预期的文件名**放在一个**固定的目录**里。

为什么文件名带时间戳而不是覆盖
------------------------------
同一份文档会被反复生成（换模型、改 prompt、调参数），每次都覆盖就
丢掉了"上一版长什么样"这个信息 —— 而比较两版数据集正是调参的主要
手段。

时间戳取**调用方传入**的值而不是 `datetime.now()`：这样同一批次的
多个分片能共用一个时间戳，命名可预期，也让这个模块保持纯函数
（不依赖时钟，可测）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "dataset_path",
    "write_dataset",
]


def dataset_path(
        output_dir: str | Path,
        document_version_id: str,
        stamp: str,
) -> Path:
    """
    拼出数据集文件名。

    形如 `<version_id>__<stamp>.json` —— 前缀是 version，所以按版本
    排序/筛选时一眼能看出属于谁；双下划线分隔，避免与 UUID 里的连字符
    混淆。
    """

    return Path(output_dir) / f"{document_version_id}__{stamp}.json"


def write_dataset(
        cases: list[dict[str, Any]],
        *,
        output_dir: str | Path,
        document_version_id: str,
        stamp: str,
        meta: dict[str, Any] | None = None,
) -> Path:
    """
    把数据集写成 JSON，返回落盘路径。

    **顶层是对象而不是裸数组**：ragas 可能要吃裸数组（调用方自己取
    `cases` 字段即可），但裸数组装不下 `meta` —— 而"这批数据是谁、用
    什么模型、什么时候、哪些窗口失败了"这些信息一旦丢掉，日后拿到一份
    数据集就无法判断它可不可信。

    `ensure_ascii=False` 是必须的：问题与答案都是中文，转义成 \\uXXXX
    之后文件体积翻几倍，而且人眼没法直接看。

    `indent=2` 同理 —— 这是给人看的文件，不是传输格式。
    """

    path = dataset_path(output_dir, document_version_id, stamp)

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "document_version_id": document_version_id,
        "generated_at": stamp,
        "case_count": len(cases),
        "meta": meta or {},
        "cases": cases,
    }

    # 先写临时文件再 replace：中途失败不会留下半截 JSON。
    # 与 LocalFileStorage 的 _write_atomic 同一个理由 —— 一份读了一半
    # 的 JSON 比没有文件更难排查。
    temporary = path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary.replace(path)

    return path
