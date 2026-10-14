"""
两个「记忆分层」处理器共用的内里：一次预览 -> 可注入 / 可保活的 system 分节。

谁用它
------
    profile.py   L3 画像 + L2 场景索引   每轮现读（纯 SQL，便宜）
    memory.py    L0 原话 + L1 原子记忆   一个运行只召回一次（贵：embedder + 向量检索）

⚠️ 为什么单独一个模块，而不是各留一份
-----------------------------
两个处理器吃的是**同一份预览响应**（`preview.sections`，按 `block` 取名），
而"怎么渲染、怎么配来源 id、形状守卫怎么写"必须只有一处：

    各留一份  一处改了另一处不改**不会报错** —— 只会让两段 system 分节的
              排版、或者快照上的 `memory_ids` 慢慢分叉，而那种不一致
              正是原规格那句"预览里看到的与模型拿到的不会漂"要挡的东西

⚠️ 本模块**不是**处理器
-------------------
这里没有 `ContextProcessor` 子类，发现器按子类收集，所以它只是被 import
的一个普通模块（与 `__init__.py` 同级同在 `processors/` 包内，便于"这两个
文件的东西放一起"）。

⚠️ 分节名为什么要跨处理器共用一个
---------------------------
快照的 `memory_ids` 是按 **section 名**收的
（`_section_ids(sections, "memory")`，见 processor/context.py 的
`context_memory_items`）。L2/L3 拆到另一个处理器之后仍然用同一个名字 ——
四段合起来才是"这一轮注入了几条记忆"，拆成两个名字会让那个指标只数到
一半，而它看起来还是"在报数"。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.agent.prompt.prompts.memory_retrieval import (
    BLOCK_CONVERSATIONS,
    BLOCK_FACTS,
    BLOCK_PERSONA,
    BLOCK_SCENES,
    render_blocks,
)

from ..context import ContextBuildContext
from ..dependencies import dependencies_of

__all__ = [
    "SECTION_NAME",
    "blocks_from",
    "first",
    "inject",
    "items",
    "mapping",
    "source_ids",
]


#: 四段共用的 section 名（见模块 docstring 最后一段）。
SECTION_NAME = "memory"


def blocks_from(preview: Any) -> list[dict[str, Any]]:
    """
    预览 -> 可注入、可保活的分节（**只含这次预览真的取到的那几层**）。

    ⚠️ 排版来自 `render_blocks`（提示词模块），**不是** `preview.prompt.user`
    ------------------------------------------------------------------
    后者是"四段拼成一整串"的结果，管理端的预览端点在用它。主模型这边四段
    各占一段 system 分节、标题由 `add_section(label=...)` 给 —— 若用那个串，
    每段会自带一个 `## 标题`，再被 `system_message()` 套一层，渲染成两层
    标题。

    两个消费者共用同一处排版（`render_blocks`），所以管理端看到的与主模型
    看到的**不会漂**。

    ⚠️ 哪几层在预览里由**调用方**决定（`retrieve` 那几个开关）
    ---------------------------------------------------
    本函数只渲染"手里这份响应有的段"：L0/L1 关掉时那两段根本不在响应里，
    于是这里天然只产出画像 / 场景两条 —— 不需要在这里再判一遍开关（两处
    判同一个条件，一处改了另一处不改就是静默的错位）。

    ⚠️ 段标识由 `render_blocks` 一并给出
    ---------------------------------
    调用方要按"这是哪一段"接回 `source_ids`。自己再数一遍"哪几层非空"来
    对齐下标的话，两处条件一旦分叉就**静默**错位（id 挂到隔壁那段上，
    没人会去核）。
    """

    sections = {section.block: section for section in preview.sections}

    scenes = sections.get(BLOCK_SCENES)

    rendered = render_blocks(
        persona=first(sections.get(BLOCK_PERSONA)),
        scenes=items(scenes),
        scenes_total=int(getattr(scenes, "scenes_total", 0) or 0),
        conversations=items(sections.get(BLOCK_CONVERSATIONS)),
        facts=items(sections.get(BLOCK_FACTS)),
    )

    return [
        {
            "label": title,
            "text": body,
            "source_ids": list(source_ids(block, sections.get(block))),
        }
        for block, title, body in rendered
    ]


def inject(
        context: ContextBuildContext,
        blocks: Sequence[Mapping[str, Any]],
) -> None:
    """
    把分节铺成 system 片段。

    各段共用 section 名（见 `SECTION_NAME`），标题走 label ——
    `system_message()` 会把每段渲染成 `## {label}`。
    """

    counter = dependencies_of(context).resolved_token_counter()

    for block in blocks:

        text = str(block.get("text") or "")

        context.add_section(
            name=SECTION_NAME,
            label=str(block.get("label") or ""),
            text=text,
            source_ids=tuple(
                str(item) for item in (block.get("source_ids") or ())
            ),
            # 重算而不是把存下来的那份也带上：计数器是纯函数（同一段
            # 文本永远同一个数），而少存一个字段就少一处会失配的形状。
            token_estimate=counter.count_text(text),
        )


def source_ids(block: str, section: Any) -> tuple[str, ...]:
    """
    一段的来源 id：画像是文件名，场景是文件名，L0 是 message_id，
    L1 是 record_id。

    它们进快照的 `memory_ids`（`_section_ids(sections, "memory")`），
    也就是 span 上的 `context_memory_items` —— "这一轮注入了几条"。
    """

    entries = items(section)

    if block in (BLOCK_PERSONA, BLOCK_SCENES):
        return tuple(str(item.filename) for item in entries)

    if block == BLOCK_CONVERSATIONS:
        return tuple(str(item.message_id) for item in entries)

    if block == BLOCK_FACTS:
        return tuple(str(item.record_id) for item in entries)

    return ()


def mapping(raw: Any) -> dict[str, Any]:
    """任意值 -> 一个永远可用的 dict。"""

    return dict(raw) if isinstance(raw, Mapping) else {}


def items(section: Any) -> list[Any]:
    """一段里的条目（段不存在时是空表）。"""

    entries = getattr(section, "items", None)

    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        return list(entries)

    return []


def first(section: Any) -> Any | None:
    """一段里的第一条（画像至多一条）。"""

    entries = items(section)

    return entries[0] if entries else None
