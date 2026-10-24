"""
消息工具：本项目里**唯一**一处知道"消息长什么样"的地方。

为什么集中在这里
--------------
上下文流水线里到处要处理消息：

    计数（budget）      要拿到可计数的文本
    历史选择（history）  要判角色、要配 tool_call / tool_result
    组装（assembler）    要判"当前用户消息在不在里面"
    校验（validator）    要判"工具调用有没有配对的结果"

如果每处各自 `getattr(message, "content", "")`，那么 langchain 换一次
消息结构就要改四处，而漏改一处的表现是**静默少算 token 或错判角色**。

⚠️ 本模块只做**鸭子类型**判定，不 import langchain
------------------------------------------------
判角色用 `type(message).__name__` 与"有没有某些属性"，而不是
`isinstance(msg, HumanMessage)`。理由与 `app/agent/model` 那边一致：
"上下文构建"的契约层不该被具体消息库绑死，而且测试里可以喂一个
SimpleNamespace 就走完流程（不必造真的 langchain 消息）。
真的需要 langchain 的地方只有组装器（它要**产出**真消息）。

⚠️ content 有三种形态
-------------------
langchain v1 的 `content` 可以是：

    str                 普通文本
    list[dict | str]    多模态块（含文本块与图片块）
    dict                单个内容块

所有取文本的地方都必须走 `message_text()`。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "is_human_message",
    "is_reasoning_block",
    "is_system_message",
    "is_tool_message",
    "message_identifier",
    "message_kind",
    "message_reasoning",
    "message_text",
    "message_type",
    "repair_tool_pairs",
    "tool_call_ids",
    "tool_calls",
    "tool_message_call_id",
    "tool_name",
]


# ============================================================
# 角色判定
# ============================================================


def message_type(message: Any) -> str:
    """
    消息类型名（小写）：`human` / `ai` / `tool` / `system` / 别的。

    优先读 `type` / `role` 属性（langchain 的消息都带 `type`），
    没有就退回类名推断 —— 从而对 SimpleNamespace 与自定义消息同样有效。
    """

    for attribute in ("type", "role"):

        value = getattr(message, attribute, None)

        if isinstance(value, str) and value:
            return value.lower()

    return type(message).__name__.lower()


def _type_matches(message: Any, *tokens: str) -> bool:

    current = message_type(message)

    return any(token in current for token in tokens)


def is_human_message(message: Any) -> bool:

    return _type_matches(message, "human", "user")


def message_kind(message: Any) -> str | None:
    """
    消息自带的业务 kind（`additional_kwargs["kind"]`）。

    只有任务指令 / 任务结果会带它；普通消息没有，返回 None。它不参与
    角色判定 —— 这是给 agent_message 落库分类用的标记。
    """

    kwargs = getattr(message, "additional_kwargs", None)

    if not isinstance(kwargs, Mapping):
        return None

    value = kwargs.get("kind")

    if not isinstance(value, str):
        return None

    text = value.strip()

    return text or None


def is_tool_message(message: Any) -> bool:

    return _type_matches(message, "tool")


def is_system_message(message: Any) -> bool:

    return _type_matches(message, "system")


# ============================================================
# 文本
# ============================================================


def message_text(message: Any) -> str:
    """取一条消息的**可计数文本**（三种 content 形态都认）。"""

    content = getattr(message, "content", message)

    return content_text(content)


def content_text(content: Any) -> str:

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, Mapping):

        for key in ("text", "content", "value"):

            value = content.get(key)

            if isinstance(value, str):
                return value

        return json.dumps(content, ensure_ascii=False, default=str)

    if isinstance(content, (list, tuple)):

        parts: list[str] = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, Mapping):

                # 非文本块（图片等）也要占额度：它们同样是请求体的一部分，
                # 只是没法按字符估。给一个固定占位，避免"看起来 0 token"。
                text = item.get("text")

                parts.append(
                    text if isinstance(text, str)
                    else f"[{item.get('type', 'block')}]"
                )
                continue

            parts.append(str(item))

        return "\n".join(parts)

    return str(content)


# ============================================================
# 思维链
# ============================================================
#
# ⚠️ 与 `content_text` 是**两件事，不要合并**
# ---------------------------------------
# `content_text` 取的是"要计进 token 预算的文本"，它对推理块故意只留一个
# `[reasoning]` 占位符（那样非文本块也占额度，避免"看起来 0 token"）。
# 这里取的是**推理原文**，给 `agent_message.reasoning_content` 那一列用。
#
# 一个要占位、一个要原文，所以同一条消息走两个函数 —— 把 `content_text`
# 改成返回推理原文会让所有按它数 token 的地方（budget / history / 摘要）
# 一起漂。

#: 推理块的类型名。与 langchain 的块类型、以及 projector 侧的 delta 类型
#: 同名（上游词汇，不自造 `thinking` / `cot`）。
REASONING_BLOCK_TYPE = "reasoning"


def is_reasoning_block(block: Any) -> bool:
    """一个内容块是不是推理块。非 Mapping 一律不是。"""

    return (
        isinstance(block, Mapping)
        and block.get("type") == REASONING_BLOCK_TYPE
    )


def message_reasoning(message: Any) -> str | None:
    """
    一条消息的**思维链原文**。没有就是 `None`。

    ⚠️ 返回 None 而不是空串
    --------------------
    那一列是可空的，而 NULL 与空串在这件事上是两种意思："这次没有思维链"
    与"有思维链但是空的"。后者不存在，所以空串只能是"没取到"的伪装 ——
    它会写进库、进契约、被前端当成"有推理"渲染出一个空框。

    ⚠️ 三个来源，先取到非空者胜
    -------------------------
    同一个模型在不同 provider / 不同 langchain 版本下把思维链放在三个
    不同的地方，实测与本仓库相关的两种：

        content 里 type=reasoning 的块      provider 自己写的结构
        additional_kwargs["reasoning_content"]  DeepSeek / Ollama / XAI / Groq
                                            （本仓库真实的 DeepSeek 路径）
        content_blocks 属性                  langchain 归一化后的公共出口

    顺序是刻意的，不是随便排的：前两个是**声明式**的（provider 就这么
    给的），第三个是**推导式**的（langchain 依版本翻译出来的）。推导的
    那份只在声明都没给出东西时才用 —— 反过来会让"上游某天改了归一化
    规则"覆盖掉 provider 原文。

    前两个都**不 import langchain**（只读属性），第三个是 langchain 专有
    且版本相关，因此包在 try 里：取不到就是"没有这个来源"，不是错误。
    这也是本模块能喂 SimpleNamespace 跑测试的前提。
    """

    for extract in (
        _reasoning_from_content,
        _reasoning_from_additional_kwargs,
        _reasoning_from_content_blocks,
    ):

        text = extract(message)

        # 空白串按"没有"处理（有些 provider 会回一个纯换行的推理块），
        # 但返回的是**原文**：首尾空白是真实 token，与 `_chunk_text`
        # 不 strip 是同一条立场。
        if text and text.strip():
            return text

    return None


# ============================================================
# 标识
# ============================================================


def message_identifier(message: Any, index: int) -> str:
    """
    一条消息的稳定标识。

    langchain 的消息不一定有 id（在图里现造的那些往往没有），因此
    缺失时用 `#<序号>` 兜底 —— 快照要的是"能不能指到同一条"，而不是
    一个全局唯一的 id。
    """

    identifier = getattr(message, "id", None)

    if isinstance(identifier, str) and identifier:
        return identifier

    return f"#{index}"


def message_identity(message: Any) -> str:
    """
    去重用的身份。

    有 id 用 id；没有就退回**对象身份**（`id()`）—— 两个不同对象就算
    内容一样也不该被判成同一条消息（它们的 tool_call_id 之类可能不同）。
    """

    identifier = getattr(message, "id", None)

    if isinstance(identifier, str) and identifier:
        return f"id:{identifier}"

    return f"obj:{id(message)}"


# ============================================================
# 工具调用 / 结果
# ============================================================


def tool_calls(message: Any) -> list[Mapping[str, Any]]:
    """这条消息要求的工具调用（只认 dict 形态的那些）。"""

    raw = getattr(message, "tool_calls", None)

    if not isinstance(raw, (list, tuple)):
        return []

    return [item for item in raw if isinstance(item, Mapping)]


def tool_call_ids(message: Any) -> tuple[str, ...]:
    """这条消息要求的工具调用 id（缺 id 的调用用空串占位，便于计数）。"""

    return tuple(
        str(call.get("id") or "") for call in tool_calls(message)
    )


def tool_message_call_id(message: Any) -> str | None:

    value = getattr(message, "tool_call_id", None)

    return str(value) if value else None


def tool_name(tool: Any) -> str:
    """一个工具（BaseTool / dict / 别的）的名字。"""

    name = getattr(tool, "name", None)

    if isinstance(name, str) and name:
        return name

    if isinstance(tool, Mapping):

        value = tool.get("name")

        if isinstance(value, str) and value:
            return value

        function = tool.get("function")

        if isinstance(function, Mapping):
            inner = function.get("name")
            if isinstance(inner, str) and inner:
                return inner

    return type(tool).__name__


# ============================================================
# 工具配对修复
# ============================================================


def repair_tool_pairs(
        messages: Sequence[Any],
) -> tuple[list[Any], int]:
    """
    去掉**不成对**的工具调用与结果。

    规则（规格 §十五）：
        Assistant 的 tool_call 必须在同一段里找到它的 ToolMessage
        孤立的 ToolMessage（没有对应的 tool_call）也要去掉

    为什么必须有这一步：一次裁剪（按轮次或按 token）原则上不会切断
    一轮内部，但**上游**给过来的历史本身就可能从半路开始（比如
    `summarized_until_message_id` 恰好落在一次工具往返中间，或者图
    state 的历史被别的机制截断过）。provider 对不成对的工具调用会
    直接 400，而那种报错的指向是 provider，不是我们。

    Returns
    -------
    (messages, dropped)
        清理后的列表，以及被丢掉的条数（进观测：这个数不为 0 说明
        上游的历史形状有问题，值得看见）。
    """

    current = list(messages)
    dropped_total = 0

    for _ in range(3):  # 丢 AI 消息会产生孤立结果，再清一轮；3 轮足够收敛

        satisfied: set[str] = set()

        for message in current:

            if is_tool_message(message):

                call_id = tool_message_call_id(message)

                if call_id:
                    satisfied.add(call_id)

        wanted: set[str] = set()
        survivors: list[Any] = []

        for message in current:

            calls = tool_call_ids(message)

            if calls:

                wanted.update(call_id for call_id in calls if call_id)

                if any(call_id and call_id not in satisfied for call_id in calls):

                    # 这次工具往返没有完整结果 -> 整条 AI 消息去掉
                    # （留着一个"要求调用工具但没有任何结果"的轮次，
                    #   provider 会拒绝；而只留结果去掉请求同样非法）。
                    dropped_total += 1
                    continue

            if is_tool_message(message):

                call_id = tool_message_call_id(message)

                # 孤立结果：它的请求已经被丢掉（或被上游切掉了）
                if call_id and call_id not in wanted:
                    dropped_total += 1
                    continue

            survivors.append(message)

        if len(survivors) == len(current):
            return survivors, dropped_total

        current = survivors

    return current, dropped_total


# ============================================================
# 思维链：三个来源各自的取值（internals）
# ============================================================
#
# 三个都返回**空串表示"这个来源没有"**，由 `message_reasoning` 决定要不要
# 往下走。返回 None 会让调用方多一处判空，而那处判空迟早会漏一个。


def _join_reasoning_blocks(blocks: Any) -> str:
    """
    块列表里的推理块原文，`\\n\\n` 连接。

    用空行而不是直接拼：多个推理块是模型分段的几段思考，连在一起读起来
    像一句话分成两半。前端只渲染成一个折叠块，空行在那里也是对的。

    ⚠️ 取 `reasoning` 键，退回 `text`
    ------------------------------
    langchain 的推理块两种键都存在过（Anthropic 的翻译给 `reasoning`，
    有的 provider 直接沿用文本块的 `text`）。只认一种的症状是**整段静默
    取不到**，而"这条消息没有思维链"与"键名看错了"在界面上完全一样。
    """

    parts: list[str] = []

    for block in blocks if isinstance(blocks, (list, tuple)) else ():

        if not is_reasoning_block(block):
            continue

        text = block.get("reasoning")

        if not isinstance(text, str) or not text:
            text = block.get("text")

        if isinstance(text, str) and text:
            parts.append(text)

    return "\n\n".join(parts)


def _reasoning_from_content(message: Any) -> str:
    """`content` 里的推理块（三种 content 形态都认）。"""

    content = getattr(message, "content", None)

    if isinstance(content, Mapping):
        return _join_reasoning_blocks([content])

    return _join_reasoning_blocks(content)


def _reasoning_from_additional_kwargs(message: Any) -> str:
    """
    `additional_kwargs["reasoning_content"]`。

    本仓库真实的 DeepSeek 路径（实测：`content=''` 而推理在这里）。键名与
    langchain 的 `_extract_reasoning_from_additional_kwargs` 一致，那边
    注明这条覆盖 Ollama / DeepSeek / XAI / Groq。
    """

    kwargs = getattr(message, "additional_kwargs", None)

    if not isinstance(kwargs, Mapping):
        return ""

    value = kwargs.get("reasoning_content")

    return value if isinstance(value, str) else ""


def _reasoning_from_content_blocks(message: Any) -> str:
    """
    langchain 归一化后的 `content_blocks`（最后一个来源）。

    ⚠️ 包在 try 里，且是**兜底**而不是首选
    ------------------------------------
    它是 AIMessage 的属性（langchain 依 `model_provider` 选翻译器算出来的），
    而本模块按约定不 import langchain：非 langchain 的鸭子对象上没有它，
    某些版本上访问它还可能抛（content 形状不认得时）。两种都只是"这个
    来源没有"，不该让取思维链这件事把调用方炸掉。

    放最后是因为它**会把 `additional_kwargs` 里那份也一起翻译进来**
    （langchain 就是这么设计的）—— 前两个来源都空时它才有增量价值。
    """

    try:
        blocks = getattr(message, "content_blocks", None)

    except Exception:       # noqa: BLE001 —— 见上面说明
        return ""

    return _join_reasoning_blocks(blocks)
