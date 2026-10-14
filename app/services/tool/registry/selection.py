"""
按 key 选工具：`tool_keys -> 可用的 ToolSpec`。

它是**唯一**的一处"哪些工具能用"的判定
--------------------------------------
同一个 `tool_keys` 有两个消费方，它们必须得到**完全相同**的集合：

    给模型看什么     CapabilityProcessor -> ModelRequest.tools -> bind_tools
    允许执行什么     tools_node 的允许集（模型幻觉出一个没暴露的名字要挡住）

这两处一旦不一致，症状是"模型调了一个自己看得见的工具，却被拒绝"
（或反过来：它调了一个从未暴露的工具，居然执行了）——两种都极难查，
因为两侧各自都"工作正常"。

所以它们不是靠约定对齐，而是**调同一个纯函数、喂同一个输入**：
集合恒等是算出来的，不是维护出来的。

工具名也必须两边一致（`wire_name` / `tool_key_of`）
------------------------------------------------
`tool_key` 形如 `source_ref.method`（见 decorator.py 的推导），**带点号**；
而 OpenAI 兼容端点要求函数名匹配 `^[a-zA-Z0-9_-]+$` —— 实测 deepseek：

    400 Invalid 'tools[0].function.name': string does not match pattern

所以发给 provider 的名字要把点号转掉，模型回传的也是转过的名字。这一对
转换与上面那个集合是同一类东西：**出站与入站必须用同一套规则**，否则
"模型调了它看得见的工具，却在允许集里对不上"。因此这两个函数与
`select_specs` 同住一处。

⚠️ `tool_key` 本身**不变**：它是对外契约字段（tool.yaml），也是
`tool_definition` 表里的列。转的只是"发给 provider 的那个字符串"。

为什么"能执行"要求描述符非空
----------------------------
一个 key 在注册表里、但没有 `ToolDescriptor`，意味着这条来源
（MCP / OpenAPI …）的执行入口还没落地。此时：
    - 给模型看没有意义（它不是 langchain 工具，bind 不出去）
    - 允许执行更危险（执行不了）
因此这类 key 一律**不进**结果集 —— 两个消费方一起看不到它。

⚠️ 但不静默：跳过要打 WARNING。一台机器上"能力挂着 12 个工具、
实际只暴露了 9 个"是必须能被看见的事实（与"静默取其一比报错危险"
同一立场，只是这里的正确处置是跳过 + 留痕，而不是让整轮对话失败）。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from .registry import ToolRegistry, get_tool_registry
from .spec import ToolSpec

__all__ = ["select_specs", "tool_key_of", "wire_name"]

logger = logging.getLogger(__name__)


#: provider 允许的函数名字符集（OpenAI 兼容端点）。
#:
#: 不在里面的字符一律换成下划线。**只是字符替换，不做长度截断** ——
#: 截断会让两个不同的 key 撞成同一个名字，而"撞了之后模型调的是谁"
#: 没法从名字上恢复。真撞上长度上限时，那是该改 `tool_key` 的命名，
#: 不是在这里悄悄砍一刀。
_WIRE_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]")


def wire_name(tool_key: str) -> str:
    """
    `tool_key` -> 发给 provider 的**函数名**。

        capability.list_capabilities -> capability_list_capabilities

    只改发送的那个字符串，`tool_key` 本身（契约字段、库里的列、注册表的
    键）一个字都不动。模型看到的、回传的都是这里转过的名字，所以它必须
    与 `tool_key_of` 是一对 —— 两者住在本模块，与 `select_specs` 同一份
    纪律（见模块 docstring）。
    """

    return _WIRE_UNSAFE.sub("_", tool_key)


def tool_key_of(
        wire: str,
        tool_keys: Sequence[str],
) -> str | None:
    """
    provider 函数名 -> `tool_key`。**只在给定的 `tool_keys` 里找。**

    为什么必须限定在给定的集合里：`wire_name` 是**多对一**的 ——
    `source_ref` 与名称里都可能带下划线，`a_b.c` 与 `a.b_c` 都转成
    `a_b_c`。拿一个名字去扫全注册表就可能认错工具，而"认错"的后果是
    **执行了另一个工具**。所以反向查找只在"这次真的暴露给模型的那一批"
    里做 —— 模型只可能调那一批。

    集合内撞名（两个不同的 key 转出同一个名字）返回 None 并打 WARNING：
    那是注册表里真的有两个工具在抢同一个名字，静默取其一比让这次调用
    落进"未暴露"更难查。

    返回 None = 这不是本次暴露的工具（模型幻觉，或名字被篡改）。调用方
    按"未暴露"处理，不要在这里兜底成一个猜测值。
    """

    matches = [key for key in tool_keys if wire_name(key) == wire]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        logger.warning(
            "provider 函数名 %r 对应多个 tool_key（%s）—— 注册表里有两个"
            "工具转出了同一个名字，本次调用按未暴露处理。",
            wire,
            "、".join(sorted(matches)),
        )

    return None


def select_specs(
        tool_keys: Sequence[str] | None,
        *,
        registry: ToolRegistry | None = None,
) -> list[ToolSpec]:
    """
    `tool_keys` -> 代码里真实存在、可绑可执行的 `ToolSpec`（去重保序）。

    保序是刻意的：工具在 `ModelRequest.tools` 里的顺序会传到 provider
    的请求体里。按调用方给的顺序（通常来自一次确定性查询）比按
    注册表的字典序更贴"这次为什么是这些工具"。

    注册表默认取进程级单例。显式传 `registry` 只给测试用 —— 与
    `ToolDiscoverer.default()` 同一个形状。
    """

    source = registry if registry is not None else get_tool_registry()

    selected: list[ToolSpec] = []
    seen: set[str] = set()

    for raw in tool_keys or ():

        key = str(raw).strip()

        # 空 key 是上游拼出来的（能力表里一条空的 tool_key 之类），
        # 与"注册表里没有"不是一回事，值得单独说一句。
        if not key:
            logger.warning("select_specs 收到空的 tool_key，已跳过。")
            continue

        # 去重放在查表**之前**：同一个 key 在 capabilities 里被挂到
        # 两个能力下是常事，重复查一次没有意义。
        if key in seen:
            continue

        seen.add(key)

        spec = source.get(key)

        if spec is None:
            logger.warning(
                "select_specs：tool_key %r 不在代码注册表里（能力可能"
                "挂着一条已消失的工具，或同步还没跑过），已跳过。",
                key,
            )
            continue

        if spec.descriptor is None:
            logger.warning(
                "select_specs：tool_key %r（%s）没有 ToolDescriptor，"
                "执行入口尚未落地，已跳过。",
                key,
                spec.entrypoint,
            )
            continue

        selected.append(spec)

    return selected
