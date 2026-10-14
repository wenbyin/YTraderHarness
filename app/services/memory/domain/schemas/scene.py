"""
L2（memory_scene）整合产物的解析模型与解析函数。

它回答什么
--------
    parse_consolidation   这一次模型回复要写哪些文件、有没有触发画像更新

⚠️ `parse_meta` **不在这里**（2026-10-11 搬走的）
------------------------------------------
它解析的是 `content` 里那个 `-----META-START-----` 块（摘要 / 热度），
而那个块的**格式**由 L2 提示词的模板定义 —— 所以解析器搬到了
`app/agent/prompt/prompts/memory_scene_consolidate.py`，与模板住在一起。
搬的理由是"改模板忘了改解析器"的代价太隐蔽：症状是清单里每个场景的
摘要都变空、模型从此选不准该 UPDATE 进哪一个，**不报错**。

本文件剩下的这一件事是"**这一轮**的模型输出" -> 变更清单，它是服务侧
要**执行**的东西（写库 / 删库），与"读回来给模型看的元数据"不是一类。

⚠️ 与 atom.py 同一套纪律
-------------------
`extra="ignore"`（模型多带字段不该让整次整合失败）、解析失败**抛**
（抛了才看得见）、剥围栏与顶层类型校验走 `_model_output.py` 的共用实现。

⚠️ 为什么是"对象"不是"数组"
----------------------
L1 两份提示词的顶层都是数组。L2 不是：它要同时表达"写哪些文件"与
"有没有触发画像更新"这两件事，所以顶层是一个对象（`operations` +
`persona_update_request`）。顶层形状由各自的 `load_array` / `load_object`
挑，见 `_model_output.py`。

⚠️ 文件语义在这里落地成字符串比较
---------------------------
一个操作是"写"还是"删"，判据是 `content` 是否**恰好**等于 `[DELETED]`
（原项目的约定：sandbox 没有删除工具，只能靠写一个标记让工程侧真删）。
两边都要 strip —— 模型常在标记前后带换行或空格，而 `" [DELETED] "`
严格比较是不相等的，症状是"该删的没删、反而写进了一篇只写着 [DELETED]
的场景文件"（不报错）。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ._model_output import load_object, validate_items

__all__ = [
    "DELETED_MARKER",
    "ConsolidationOperation",
    "ConsolidationParseError",
    "ConsolidationResult",
    "is_deletion",
    "parse_consolidation",
]


#: ”删除这个文件“的唯一表达方式。见模块 docstring 最后一段。
DELETED_MARKER = "[DELETED]"


class ConsolidationParseError(ValueError):
    """
    模型输出没法解析成一次整合。

    与 `ExtractionParseError` 同一立场：继承 ValueError（"输入不合法"那
    一类），因此既有的 `except ValueError` 接得住，任务层把它当**可重试**
    错误 —— 模型的这一次输出不合法不代表下一次也是。
    """


class ConsolidationOperation(BaseModel):
    """要在某个场景文件上做的一件事 —— 提示词输出里 `operations[]` 的一项。"""

    model_config = ConfigDict(extra="ignore")

    path: str = Field(
        min_length=1,
        description=(
            "场景文件名（相对路径）。**原样来自模型**，所以可能带空格、"
            "括号、甚至目录分隔符 —— 归一化在服务侧做（见 "
            "application/memory_scene_service_impl.py 的 "
            "`_normalize_filename`），这里只保证它是非空字符串。\n\n"
            "⚠️ 不在这里归一化：这一层的职责是”模型到底说了什么“，而"
            "归一化会**合并**两个不同的 path，失败时就再也看不出模型"
            "原本写了什么。"
        ),
    )

    content: str = Field(
        description=(
            "该文件的**完整内容**（整篇重写）。内容恰好是 `[DELETED]` 时"
            "表示删除该文件（见模块 docstring）。\n\n"
            "允许空串（模型偶尔会给一个空 content）—— 服务侧会把它当"
            "**无效操作丢掉**，而不是写一行空内容进库（`content <> ''` "
            "那条 CHECK 会在写入那一刻炸，离原因很远）。"
        ),
    )


class ConsolidationResult(BaseModel):
    """一次整合的完整结果。"""

    model_config = ConfigDict(extra="ignore")

    operations: list[ConsolidationOperation] = Field(
        default_factory=list,
        description=(
            "本次要落盘的全部变更。**只含真的改动的文件** —— 提示词明写"
            "“没改动的文件不要出现”。没有变更时是空数组（合法且常见："
            "这一批新记忆全都是已有的场景已经讲过的事）。"
        ),
    )

    persona_update_request: bool = Field(
        default=False,
        description=(
            "模型主动请求更新 L3 画像 —— 一个 **true / false** 标记"
            "（原项目的 `[PERSONA_UPDATE_REQUEST]` 标记，这里改成 JSON 字段）。\n\n"
            "⚠️ 曾经是**理由原文**（`str | None`），2026-10-13 改成布尔\n"
            "------------------------------------------------\n"
            "原因：它是 L3 侧的**免跳过信号**，消费方只问「要不要」，从不读"
            "那段文字；而「填一个字符串」这种要求模型给得很不稳定（实测"
            "一次都没给过）。压成布尔之后，提示词里那句判断也更好写。\n\n"
            "代价说清楚：L3 的提示词里原本会把这段理由当作「触发信息」透传，"
            "现在没有了 —— 那一格改成陈述「L2 请求了更新」这个事实。"
        ),
    )


def is_deletion(content: str) -> bool:
    """
    这个 content 是不是"删除该文件"的标记。

    两边 strip 之后**严格相等**才算删除 —— `[DELETED] 这个场景已经废弃`
    不是删除，是一篇正文（模型可能这么写，那我们就照写，它是有意义的
    文本）。

    ⚠️ 少了 strip 的症状
    -----------------
    模型常在标记前后带换行或空格，而 `" [DELETED] "` 严格比较是不相等的
    —— 症状是"该删的没删、反而写进了一篇只写着 [DELETED] 的场景文件"，
    而且**不报错**。
    """

    return content.strip() == DELETED_MARKER


def parse_consolidation(text: str) -> ConsolidationResult:
    """
    整合提示词的输出 -> 一次整合。

    Raises:
        ConsolidationParseError: 不是合法 JSON、顶层不是对象、`operations`
                                 缺失或不是数组、或某一项不符合形状。
    """

    payload = load_object(text, error=ConsolidationParseError)

    raw_operations = payload.get("operations")

    # ⚠️ 缺失**不是**"没有变更"
    # ----------------------
    # 提示词明确要求这个字段。给一个缺字段的输出当"什么都不用做"，症状是
    # "这一批新记忆一条都没进画像"，而它与"确实没有值得改的"长得一样 ——
    # 正是 atom.py 那段"解析失败要抛"要挡的事。空变更的正确写法是空数组。
    if not isinstance(raw_operations, list):
        raise ConsolidationParseError(
            "模型输出里没有 `operations` 数组（提示词要求一个 JSON 对象，"
            f"其中 operations 是数组）。实际：{raw_operations!r}。"
            "没有变更时要给空数组，不能省略这个字段。"
        )

    return ConsolidationResult(
        operations=validate_items(
            raw_operations,
            ConsolidationOperation,
            error=ConsolidationParseError,
        ),
        persona_update_request=_optional_flag(
            payload.get("persona_update_request")
        ),
    )


# ============================================================
# internals
# ============================================================


def _optional_flag(value: object) -> bool:
    """
    `persona_update_request` -> 布尔。**认不出一律 False**。

    ⚠️ 这里**宽容**（与 operations 的严格相反）
    -------------------------------------
    它不是落库内容，只是"顺带告诉我一声"。为一个可选的提示字段失败掉
    整次整合（几条场景的写入全没了）代价完全不对等。所以除下面几种
    "真"的写法之外一律当 False —— **包括模型按老提示词回的长文本**：
    那是旧字段的残留，不是"要更新"（老契约里那段文字是**理由**，
    只有非空才代表举手；新契约里直接是 true/false）。

    ⚠️ 为什么连 `"false"` 都要显式挡
    ----------------------------
    `bool("false")` 是 **True** —— 字符串非空。模型偶尔会把布尔写成字符串，
    而这里一旦按 truthy 判，`"false"` 就变成"要更新画像"，且**不报错**。
    所以字符串走白名单，不走 `bool()`。
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")

    return False


