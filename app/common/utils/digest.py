"""
内容摘要：规范化 JSON + sha256。

"把一份结构规范化成字符串再取 sha256"这件事在本项目有四处
（拓扑指纹、工具清单指纹、缓存键、chunk 身份），它们必须**稳**：
同一个结构每次都要得到同一个摘要，否则指纹会在两次运行之间无缘
无故地变，或者缓存永远不命中。

稳的前提是 `json.dumps` 的几个参数一个都不能少：

    sort_keys=True        字典的键序不参与 —— 否则同样的结构，
                          因构造顺序不同就得到不同摘要
    separators=(",", ":") 去掉默认的空格 —— 否则摘要取决于
                          json 模块的默认格式，而那是实现细节
    ensure_ascii=False    中文不转义成 \\uXXXX —— 否则摘要取决于
                          字符集展开方式

这三条是本项目踩出来的口径，四处必须一致。

⚠️ 有一处刻意**不同**：`default=str`
--------------------------------------
`app/infra/graph/ids.py` 的拓扑指纹**没有** `default=str`，另外三处有。

差别在输入含非 JSON 可序列化对象时：
    有 default=str    静默转成 str(...)，得到一个摘要
    没有              抛 TypeError

ids.py 那份保持"抛"是有意的：拓扑指纹是对外契约（有 golden 校验），
输入本该全是 str/int；真混进别的东西时，报错比悄悄按 repr 摘要要好
（repr 会随对象状态变，摘要跟着变，而没人知道为什么）。

因此 `canonical_json` 把这个差异做成**显式参数**，默认与多数一致
（`default_str=True`），ids.py 那边显式传 `False` —— 一个开关摆
在调用点上，比两处实现长得几乎一样却不相同要安全。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["canonical_json", "sha256_text"]


def canonical_json(
        value: Any,
        *,
        default_str: bool = True,
) -> str:
    """
    把结构规范化成**稳定**的 JSON 字符串。

    Parameters
    ----------
    default_str
        True（默认）：非 JSON 可序列化的对象用 `str()` 兜底。
        False：让它抛 TypeError —— 见模块 docstring，拓扑指纹
        需要这个行为。
    """

    kwargs: dict[str, Any] = {
        "sort_keys": True,
        "separators": (",", ":"),
        "ensure_ascii": False,
    }

    if default_str:
        kwargs["default"] = str

    return json.dumps(value, **kwargs)


def sha256_text(text: str) -> str:
    """
    文本的 sha256 十六进制摘要。

    ⚠️ 只接受 str，不接受 bytes —— 本项目对**文件内容**取摘要是
    另一回事（`hashlib.sha256(content).hexdigest()`，就地一行，
    各处自己处理），两者不要混：这里要的是"结构指纹"，编码口径
    由 `canonical_json` 统一钉死为 utf-8。
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
