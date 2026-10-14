"""
L3（memory_persona）生成产物的解析模型与解析函数。

它回答什么
--------
    parse_persona   这一次模型回复要不要改画像、改成什么

⚠️ 与 L2 那份（scene.py）的分工
--------------------------
L2 要表达"**哪些文件**改成什么"（`operations[]`），所以顶层是一个数组的
容器。L3 只有**一篇**，不需要指认目标，所以形状退化成一个二选一：

    { "no_change": true }                        不改（常见且合法）
    { "no_change": false, "content": "整篇 md" } 改，content 是全文

⚠️ 为什么保留一个显式的 `no_change`，而不是只比 content_md5
---------------------------------------------------
两种做法都能省下一次写入，但**省的不是同一件事**：

    md5 短路     模型仍然把整篇重写了一遍（那次模型调用已经付了钱）
    no_change    模型自己判断"这一轮没有值得抽象的"，可以**不吐全文**，
                 于是输出短、也不会有一次"重写却逐字相同"的浪费

而"不改"在原提示词里本来就是一等的决策（增量更新策略里的「不改：新增内容
只有项目状态、普通任务或低层事实时，不更新 L3」）。抹掉这个信号等于把一条
提示词规则变成没有任何落点的空话。

⚠️ md5 短路**仍然留着**（两层各挡一件事）
----------------------------------
`no_change: false` 但内容与库里的逐字相同，是**允许**的（模型判断该重写、
写出来一样）。服务侧在那一档照样不写库、不推 version —— 否则"版本号"会
变成"跑过几轮"的同义词，乐观锁的基线失去意义（同 L2 `_apply` 里那段）。

⚠️ `no_change: true` 时**不看** `content`
------------------------------------
模型偶尔会两边都给（说不用改、又附了一段）。取"不改"是因为**写**的那一
侧代价更大：万一那段 content 是它写了一半的东西，落库就是拿一份残缺的
画像覆盖掉好的。与 L2 的 `_apply` 同一条取舍（宁可多留、不猜意图）。

⚠️ 与 atom.py / scene.py 同一套纪律
-----------------------------
`extra="ignore"`（模型多带字段不该让整次生成失败）、解析失败**抛**（抛了
才看得见）、剥围栏与顶层类型校验走 `_model_output.py` 的共用实现。

⚠️ 内容长度**不在这里挡**
-------------------
提示词要求个人 ≤2000 字、团队 ≤1200 字，那是**给模型看**的约束（自然语言
写的）。工程侧不设硬上限：模型偶尔超一点，砍掉尾巴等于毁掉一份完整文档，
而超长只是"这一篇有点啰嗦"——下次生成会压回来。要观测就记一条 warning，
不要在这里造一个解析失败。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ._model_output import load_object

__all__ = [
    "PersonaParseError",
    "PersonaResult",
    "parse_persona",
]


class PersonaParseError(ValueError):
    """
    模型输出没法解析成一次画像生成。

    与 `ConsolidationParseError` 同一立场：继承 ValueError（"输入不合法"
    那一类），因此既有的 `except ValueError` 接得住，任务层把它当**可重试**
    错误 —— 模型的这一次输出不合法不代表下一次也是。
    """


class PersonaResult(BaseModel):
    """一次画像生成的完整结果。"""

    model_config = ConfigDict(extra="ignore")

    no_change: bool = Field(
        default=False,
        description=(
            "模型判断这一轮**不需要**改画像。True 时 `content` 一律为 None"
            "（即使模型顺手给了一段，也不采用 —— 见模块 docstring）。"
        ),
    )

    content: str | None = Field(
        default=None,
        description=(
            "画像的**整篇新内容**。`no_change` 为 False 时必须有且非空"
            "（少了它就不是“不改”，是“模型没按格式答”，那要重试）。"
        ),
    )


def parse_persona(text: str) -> PersonaResult:
    """
    画像提示词的输出 -> 一次生成结果。

    Raises:
        PersonaParseError: 不是合法 JSON、顶层不是对象、`no_change` 不是
                           布尔、或 `no_change` 为假但 `content` 缺失/空白。
    """

    payload = load_object(text, error=PersonaParseError)

    raw_no_change = payload.get("no_change", False)

    # ⚠️ 宽松地把 "true" / "false" 字符串认下来
    # -------------------------------------
    # 模型偶尔把布尔写成字符串（`"no_change": "false"`）。为一个**开关**
    # 失败掉整次生成不划算 —— 而且这个错法有方向性：把 "false" 当成真值
    # 会静默丢掉一次本该发生的更新（那是"画像停止演进"，最难发现的一类）。
    # 所以识别字符串形式，但**认不出就抛**（比如给了个对象）。
    if isinstance(raw_no_change, str):
        lowered = raw_no_change.strip().lower()
        if lowered in ("true", "false"):
            no_change = lowered == "true"
        else:
            raise PersonaParseError(
                f"`no_change` 是字符串但既不是 true 也不是 false："
                f"{raw_no_change!r}"
            )
    elif isinstance(raw_no_change, bool):
        no_change = raw_no_change
    else:
        raise PersonaParseError(
            f"`no_change` 必须是布尔（或 'true'/'false'），实际："
            f"{type(raw_no_change).__name__}。"
        )

    if no_change:
        return PersonaResult(no_change=True, content=None)

    content = payload.get("content")

    # ⚠️ 缺失 / 空串**不是**"不改"
    # ------------------------
    # 提示词明确要求：不改时把 no_change 设成 true；要改就给整篇 content。
    # 给一段空的 content 当"不用改"，症状是"这一轮什么都没写进画像"，而它
    # 与"确实没有值得抽象的"长得一样 —— 正是 atom.py 那段"解析失败要抛"
    # 要挡的事。
    if not isinstance(content, str) or not content.strip():
        raise PersonaParseError(
            "`no_change` 为假，但 `content` 缺失或为空。要改就给出整篇"
            " markdown；不改要把 `no_change` 设成 true。"
            f"实际 content：{content!r}"
        )

    return PersonaResult(no_change=False, content=content.strip())
